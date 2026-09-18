"""
Model definitions for GANular, including the
generator, discriminator, and projection modules.
"""

import pathlib
from os import PathLike
from warnings import warn
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from tqdm.auto import tqdm, trange

from . import params
from .artifacts import save_voxel_npy, save_voxel_stl
from .runtime import recommend_accelerator_batch_size


def sample_latent_noise(size):
    """Generate a tensor of Gaussian noise for the generator input.

    Args:
        size: Shape of the noise tensor, typically ``(batch_size, noise_dim)``.

    Returns:
        A ``torch.Tensor`` sampled from ``N(0, 1)``.
    """
    return torch.randn(size, dtype=torch.float32)


class VoxelGenerator(nn.Module):
    """3D voxel generator network, adapted from diaGAN (Coiffier et al. 2020).

    Maps a 1-D latent noise vector to a ``64x64x64`` binary voxel grid through
    a linear embedding followed by three trilinear-upsampling + Conv3D blocks.

    Args:
        noise_size: Dimensionality of the input noise vector.
            Defaults to ``params.noise_size``.
    """

    def __init__(self, noise_size: Optional[int] = None) -> None:
        super().__init__()

        if noise_size:
            self.noise_size = noise_size
        else:
            self.noise_size = params.noise_size

        self.embed_shape = (1, params.embed_size, params.embed_size, params.embed_size)

        self.upscale = lambda x: nn.functional.interpolate(
            x, scale_factor=2, mode="trilinear", align_corners=True
        )

        self.lin_embed = nn.Sequential(
            nn.Linear(self.noise_size, params.embed_size**3),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=3, padding=1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a 3D voxel grid from a noise vector.

        Args:
            x: Latent noise tensor of shape ``(B, noise_size)``.

        Returns:
            Voxel grid of shape ``(B, 1, 64, 64, 64)`` with values in ``[0, 1]``.
        """
        x = self.lin_embed(x)
        x = x.view((x.size(0),) + self.embed_shape)
        # 16x16x16

        x = self.conv1(x)
        x = self.upscale(x)
        # 32x32x32

        x = self.conv2(x)
        x = self.upscale(x)
        # 64x64x64

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class ProjectionDiscriminator(nn.Module):
    """2D projection discriminator with spectral and instance normalisation.

    An improved discriminator using spectral-normalised Conv2D layers with
    instance normalisation and LeakyReLU activations. Weights are initialised
    from ``N(0, 0.02)``.

    Args:
        img_channels: Number of input image channels.
        features_d: Base number of feature maps (doubled at each block).
    """

    def __init__(self, img_channels=1, features_d=64):
        super().__init__()
        self.disc = nn.Sequential(
            # Input: N x channels x 64 x 64
            nn.Conv2d(img_channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Block 1
            self._block(features_d, features_d * 2, 4, 2, 1),
            # Block 2
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # Block 3
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=0),
        )
        self.initialize_weights()

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """Build a spectral-normalised Conv2D block with instance norm and LeakyReLU.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides of the input.

        Returns:
            ``nn.Sequential`` block.
        """
        return nn.Sequential(
            # Apply Spectral Norm to every Conv layer
            spectral_norm(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False
                )
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def initialize_weights(self):
        """Initialise convolutional layer weights from ``N(0, 0.02)``."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        """Compute the discriminator score for a batch of 2D projection images.

        Args:
            x: Image tensor of shape ``(B, img_channels, H, W)``.

        Returns:
            Score tensor of shape ``(B, 1, 1, 1)``.
        """
        return nn.functional.adaptive_avg_pool2d(self.disc(x), (1, 1))


class BinaryThresholdSTE(torch.autograd.Function):
    """Straight-through estimator for binary thresholding.

    Applies a hard threshold in the forward pass (producing binary output)
    while passing gradients through unchanged in the backward pass, enabling
    gradient-based optimisation of upstream layers.
    """

    @staticmethod
    def forward(ctx, input, threshold=0.5):
        """Binarise the input at *threshold*.

        Args:
            ctx: Autograd context (unused).
            input: Continuous-valued input tensor.
            threshold: Cut-off value for binarisation.

        Returns:
            Binary tensor with the same shape as *input*.
        """
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Pass gradients through unchanged (straight-through estimator).

        Args:
            ctx: Autograd context (unused).
            *grad_outputs: Upstream gradients.

        Returns:
            Tuple of (gradient for *input*, ``None`` for *threshold*).
        """
        return grad_outputs[0], None


class VoxelProjector(nn.Module):
    """Differentiable 3D-to-2D projection module.

    Renders multi-view 2D projections of 3D voxel grids by applying random
    3D rotations and taking a softmax-weighted maximum along the depth axis.
    Used during training so the discriminator can compare generated
    projections with real 2D particle images.

    Args:
        volume_size: Spatial size of the cubic voxel grid.
        threshold: Binarisation threshold (applied via
            :class:`BinaryThresholdSTE`).
        num_views: Number of random viewing angles per sample.
    """

    def __init__(
        self, volume_size: int = 64, threshold: float = 0.5, num_views: int = 2
    ) -> None:
        super().__init__()
        self.volume_size = volume_size
        self.threshold = threshold
        self.num_views = num_views
        self.binarize = BinaryThresholdSTE.apply

    def get_rotation_mat(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate random 3D rotation matrices for the full batch of views.
        Uses Shoemake's method to uniformly sample rotations from SO(3) via quaternions.

        Args:
            batch_size: Number of samples in the batch (before view expansion).
            device: Torch device for tensor allocation.

        Returns:
            Affine matrices of shape ``(batch_size * num_views, 3, 4)``.
        """
        # (Batch * Num_Views) total rotation matrices
        total_samples = batch_size * self.num_views

        u1, u2, u3 = torch.rand(3, total_samples, device=device)

        qw = torch.sqrt(1 - u1) * torch.sin(2 * np.pi * u2)
        qx = torch.sqrt(1 - u1) * torch.cos(2 * np.pi * u2)
        qy = torch.sqrt(u1) * torch.sin(2 * np.pi * u3)
        qz = torch.sqrt(u1) * torch.cos(2 * np.pi * u3)

        # Convert quaternions to rotation matrices
        xx, yy, zz = qx * qx, qy * qy, qz * qz
        xy, xz, yz = qx * qy, qx * qz, qy * qz
        xw, yw, zw = qw * qx, qw * qy, qw * qz

        rot_mats = torch.stack(
            [
                torch.stack([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], dim=1),
                torch.stack([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], dim=1),
                torch.stack([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], dim=1),
            ],
            dim=2,
        )

        translation = torch.zeros((total_samples, 3, 1), device=device)
        return torch.cat([rot_mats, translation], dim=2)

    def forward(
        self,
        voxels: torch.Tensor,
        elev=None,
        azim=None,
        rotation_matrices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project 3D voxel grids into 2D images from random viewing angles.

        Args:
            voxels: Voxel grids of shape ``(B, C, D, H, W)``.
            elev: Unused. Reserved for fixed elevation angles.
            azim: Unused. Reserved for fixed azimuth angles.
            rotation_matrices: Optional fixed affine matrices with shape
                ``(B * num_views, 3, 4)`` for reproducible evaluation.

        Returns:
            2D projection images of shape ``(B * num_views, C, D, H)``.
        """
        # voxels: (B, C, D, H, W)
        B, C, D, H, W = voxels.shape
        device = voxels.device

        # Expand voxels to match the number of views
        expanded_voxels = voxels.unsqueeze(1).repeat(1, self.num_views, 1, 1, 1, 1)
        expanded_voxels = expanded_voxels.view(-1, C, D, H, W)

        # Get rotations for all B*N items
        theta = (
            rotation_matrices.to(device)
            if rotation_matrices is not None
            else self.get_rotation_mat(B, device)
        )
        if theta.shape != (B * self.num_views, 3, 4):
            raise ValueError(
                "rotation_matrices must have shape "
                f"{(B * self.num_views, 3, 4)}, got {tuple(theta.shape)}"
            )

        grid = nn.functional.affine_grid(
            theta, expanded_voxels.size(), align_corners=False
        )
        rotated_voxels = nn.functional.grid_sample(
            expanded_voxels,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # beta=10 closely approximates max pooling for occupancies in [0, 1]
        # while allowing gradients to reach more than one voxel per ray.
        ray_weights = torch.softmax(10.0 * rotated_voxels, dim=4)
        raw_proj = torch.sum(ray_weights * rotated_voxels, dim=4)

        return raw_proj


def generate_from_checkpoint(
    checkpoint_path: str,
    n_samples: int,
    output_dir: PathLike,
    device: str = None,
    out_filename: Optional[str] = "sample",
    dump_npy: bool = True,
    dump_stl: bool = False,
    batch_limit: Optional[int] = None,
    threshold: float = 0.5,
    progress: bool = True,
):
    """Generate synthetic 3D particles from a pre-trained generator checkpoint.

    Loads a saved :class:`VoxelGenerator` state dict and produces *n_samples*
    voxel grids. Each grid can be saved as a ``.npy`` file and, when
    ``dump_stl=True``, as an STL mesh. Generation is batched to limit GPU
    memory usage.

    Args:
        checkpoint_path: Path to a ``.pt`` or ``.pth`` checkpoint file.
        n_samples: Total number of particles to generate.
        output_dir: Directory where outputs are written (created if absent).
        device: Torch device string. The current accelerator is auto-detected
            when ``None``, with CPU as the fallback.
        out_filename: Base name for output files (without extension). Each
            file is suffixed with an incrementing index.
        dump_npy: Whether to also save voxels as ``.npy`` files.
        dump_stl: Whether to save each thresholded voxel grid as an STL mesh.
        batch_limit: Maximum batch size per forward pass. When ``None``,
            accelerator memory is probed to select a conservative value capped
            at 32.

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
        ValueError: If the checkpoint file does not have a ``.pt`` or
            ``.pth`` extension.
    """
    if not device:
        device = str(
            torch.accelerator.current_accelerator(check_available=True) or "cpu"
        )

    if not isinstance(output_dir, pathlib.Path):
        output_dir = pathlib.Path(output_dir)

    if n_samples < 1:
        raise ValueError("n_samples must be positive")

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if not pathlib.Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    elif pathlib.Path(checkpoint_path).suffix not in {".pt", ".pth"}:
        raise ValueError(f"Checkpoint must be a .pt or .pth file: {checkpoint_path}")

    if "." in out_filename:
        out_filename = out_filename.rsplit(".", 1)[0]

    checkpoint = torch.load(checkpoint_path, map_location=device)

    gen = VoxelGenerator().to(device)

    if "state_dict" in checkpoint:
        gen.load_state_dict(checkpoint["state_dict"])
    else:
        gen.load_state_dict(checkpoint)

    gen.eval()

    if batch_limit is None:
        batch_limit = min(32, n_samples)
        if torch.device(device).type != "cpu":
            with torch.inference_mode():
                batch_limit = recommend_accelerator_batch_size(
                    lambda size: gen(
                        sample_latent_noise((size, params.noise_size)).to(device)
                    ),
                    max_batch=batch_limit,
                )
    elif batch_limit < 1:
        raise ValueError("batch_limit must be positive")

    with torch.inference_mode():
        progress_bar = tqdm(
            total=n_samples,
            desc="Generating batches",
            unit="sample",
            disable=not progress,
        )
        i = 0
        while i < n_samples:
            current_batch_size = min(batch_limit, n_samples - i)
            noise = sample_latent_noise((current_batch_size, params.noise_size)).to(
                device
            )
            try:
                voxel_collection = gen(noise)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or current_batch_size == 1:
                    raise
                batch_limit = max(1, current_batch_size // 2)
                _empty_device_cache(device)
                warn(
                    f"Device memory exhausted; reducing batch_limit={batch_limit}",
                    stacklevel=2,
                )
                continue

            for j in trange(current_batch_size, leave=False):
                voxel = voxel_collection[j]
                voxel_np = voxel.squeeze().cpu().numpy()

                if dump_npy:
                    save_path = output_dir / f"{out_filename}_{i + j + 1}.npy"
                    save_voxel_npy(voxel_np, save_path)
                if dump_stl:
                    stl_save_path = output_dir / f"{out_filename}_{i + j + 1}.stl"
                    save_voxel_stl(voxel, stl_save_path, threshold=threshold)
            i += current_batch_size
            progress_bar.update(current_batch_size)
        progress_bar.close()


def _empty_device_cache(device: str) -> None:
    if torch.device(device).type != "cpu":
        torch.accelerator.memory.empty_cache()
