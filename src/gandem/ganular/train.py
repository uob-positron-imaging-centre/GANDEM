"""
Training loop for GANular.
"""

import os
import glob
import json
import math
from typing import Optional
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
from tqdm.auto import tqdm

from .artifacts import (
    save_voxel_plot,
    save_loss_plot,
    save_projection_plot,
    save_voxel_npy,
)
from .datasets import ProjectionDataset
from .model import (
    ProjectionDiscriminator,
    VoxelGenerator,
    VoxelProjector,
    sample_latent_noise,
)
from . import params
from .quality import (
    FEATURES_2D,
    QualityChecks,
    append_jsonl,
    assess_generated_samples,
    check_training_health,
    feature_table,
    load_uct_features,
    shape_metrics_2d,
)

_TRAINING_CHECKPOINT_KEYS = {
    "epoch",
    "global_step",
    "generator",
    "discriminator",
    "generator_optimizer",
    "discriminator_optimizer",
    "torch_rng_state",
    "numpy_rng_state",
}


def _gradient_norm(module: torch.nn.Module) -> float:
    return float(
        np.sqrt(
            sum(
                parameter.grad.detach().pow(2).sum().item()
                for parameter in module.parameters()
                if parameter.grad is not None
            )
        )
    )


def compute_gradient_penalty(
    disc, real_samples, fake_samples, device, return_gradient_norms=False
):
    """Calculate the WGAN-GP gradient penalty."""
    alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(
        True
    )
    scores = disc(interpolates)
    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradient_norms = gradients.view(gradients.size(0), -1).norm(2, dim=1)
    penalty = ((gradient_norms - 1) ** 2).mean()
    if return_gradient_norms:
        return penalty, gradient_norms.detach()
    return penalty


def _image_paths(directory):
    paths = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        paths.extend(glob.glob(os.path.join(directory, extension)))
    return sorted(paths)


def _reference_2d_features(dataset, sample_size, threshold, seed):
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(dataset), size=sample_size, replace=len(dataset) < sample_size
    )
    images = [dataset[int(index)].numpy() for index in indices]
    return feature_table(images, shape_metrics_2d, threshold, FEATURES_2D)


def _generate_quality_samples(gen, proj, noise, rotations, batch_size, n_views, device):
    volumes, projections = [], []
    was_training = gen.training
    gen.eval()
    with torch.inference_mode():
        for start in range(0, len(noise), batch_size):
            end = min(start + batch_size, len(noise))
            voxel_batch = gen(noise[start:end].to(device))
            rotation_batch = rotations[start * n_views : end * n_views]
            projection_batch = proj(voxel_batch, rotation_matrices=rotation_batch)
            volumes.append(voxel_batch.cpu().numpy())
            projections.append(projection_batch.cpu().numpy())
    gen.train(was_training)
    return np.concatenate(volumes), np.concatenate(projections)


def _mean_object_score(scores, object_batch_size, views_per_object):
    """Average critic scores over views first, then over generated objects."""
    return scores.reshape(object_batch_size, views_per_object, -1).mean(dim=1).mean()


def _select_quality_samples(
    volumes, projections, object_sample_size, projection_sample_size, views_per_object
):
    """Select fixed object and projection counts, taking one view per object first."""
    projection_shape = projections.shape[1:]
    projections = (
        projections.reshape(-1, views_per_object, *projection_shape)
        .swapaxes(0, 1)
        .reshape(-1, *projection_shape)
    )
    return volumes[:object_sample_size], projections[:projection_sample_size]


def _fixed_rotations(batch_size, n_views, rng):
    total = batch_size * n_views
    u1, u2, u3 = rng.random((3, total))
    qw = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
    qx = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
    qy = np.sqrt(u1) * np.sin(2 * np.pi * u3)
    qz = np.sqrt(u1) * np.cos(2 * np.pi * u3)
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    xw, yw, zw = qw * qx, qw * qy, qw * qz
    rotations = np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)], axis=1),
            np.stack([2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)], axis=1),
            np.stack([2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)], axis=1),
        ],
        axis=2,
    )
    translations = np.zeros((total, 3, 1))
    return torch.from_numpy(
        np.concatenate([rotations, translations], axis=2).astype(np.float32)
    )


def _training_checkpoint(
    epoch,
    global_step,
    gen,
    disc,
    opt_gen,
    opt_disc,
    d_losses,
    g_losses,
    hyperparameters,
    last_training_health,
):
    """Return all mutable state needed to continue at the next epoch."""
    checkpoint = {
        "checkpoint_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "generator": gen.state_dict(),
        "discriminator": disc.state_dict(),
        "generator_optimizer": opt_gen.state_dict(),
        "discriminator_optimizer": opt_disc.state_dict(),
        "discriminator_losses": d_losses,
        "generator_losses": g_losses,
        "hyperparameters": hyperparameters,
        "last_training_health": last_training_health,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        checkpoint["mps_rng_state"] = torch.mps.get_rng_state()
    return checkpoint


def _restore_training_checkpoint(checkpoint, gen, disc, opt_gen, opt_disc):
    """Restore a full training checkpoint and return loop bookkeeping."""
    missing = _TRAINING_CHECKPOINT_KEYS.difference(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint cannot resume training; missing: " + ", ".join(sorted(missing))
        )

    gen.load_state_dict(checkpoint["generator"])
    disc.load_state_dict(checkpoint["discriminator"])
    opt_gen.load_state_dict(checkpoint["generator_optimizer"])
    opt_disc.load_state_dict(checkpoint["discriminator_optimizer"])
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    np.random.set_state(checkpoint["numpy_rng_state"])
    if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state"]]
        )
    if "mps_rng_state" in checkpoint and torch.backends.mps.is_available():
        torch.mps.set_rng_state(checkpoint["mps_rng_state"].cpu())

    return (
        checkpoint["epoch"],
        checkpoint["global_step"],
        list(checkpoint.get("discriminator_losses", [])),
        list(checkpoint.get("generator_losses", [])),
        checkpoint.get("last_training_health"),
    )


def train(
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    output_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    n_views: Optional[int] = None,
    seed: Optional[int] = None,
    quality_checks: Optional[QualityChecks] = None,
    learning_rate: Optional[float] = None,
    lambda_gp: Optional[float] = None,
    lambda_binary: Optional[float] = None,
    critic_iterations: Optional[int] = None,
    object_batch_size: Optional[int] = None,
    views_per_object: Optional[int] = None,
    updates_per_epoch: Optional[int] = None,
    resume_from: Optional[str] = None,
):
    """Train the GANular 3D particle generator using WGAN-GP.

    Trains a projection-based GAN that learns to generate 3D voxel
    representations of particles from 2D projection images. Uses a
    Wasserstein loss with gradient penalty and configurable critic updates.

    Model checkpoints, loss plots, voxel visualisations, and projection
    images are saved periodically to ``output_dir``.

    Args:
        batch_size: Deprecated alias for ``object_batch_size``.
        epochs: Number of training epochs. When resuming, this is the number of
            additional epochs. Defaults to ``params.epochs``.
        output_dir: Root directory for all outputs. Must not already exist for
            a new run. When resuming, defaults to the checkpoint's run directory.
        data_dir: Path to the directory of 2D particle images.
            Defaults to ``params.data_dir``.
        n_views: Deprecated alias for ``views_per_object``.
        seed: Random seed for reproducibility. When set, seeds all devices and
            configures deterministic cuDNN behavior.
        quality_checks: Configuration for the five advisory quality checks.
            Training health, projection quality, and topology work without uCT
            data; set ``uct_data_dir`` to a directory of raw STL meshes
            to enable volume fidelity and coverage checks.
        learning_rate: Adam optimiser learning rate. Defaults to
            ``params.learning_rate``.
        lambda_gp: Gradient penalty coefficient. Defaults to 10.
        lambda_binary: Weight for the differentiable voxel-binarity penalty.
            Defaults to 0.1. Set to 0 to disable it.
        critic_iterations: Critic updates per generator update. Defaults to 5.
        object_batch_size: Independently generated 3D objects per update.
            Defaults to ``params.object_batch_size``.
        views_per_object: Projection views rendered per generated object.
            Defaults to ``params.views_per_object``.
        updates_per_epoch: Generator updates in each epoch, independent of the
            real dataset size and ``views_per_object``. Defaults to
            ``params.updates_per_epoch``.
        resume_from: Full ``training_epoch_*.pth`` checkpoint to continue from.
            Only load checkpoints that you trust, because PyTorch checkpoints
            use Python pickle internally.

    Raises:
        FileExistsError: If a new run's ``output_dir`` already exists, or
            resuming would overwrite a later checkpoint.
        FileNotFoundError: If ``data_dir`` or ``resume_from`` does not exist.
        ValueError: If a checkpoint is invalid or no images exist in ``data_dir``.
    """

    device = params.device
    checkpoint = None
    full_resume = False
    if resume_from:
        if not os.path.isfile(resume_from):
            raise FileNotFoundError(f"Checkpoint not found: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must contain a state dictionary")
        full_resume = checkpoint.get("checkpoint_version") == 1
        if full_resume:
            missing = _TRAINING_CHECKPOINT_KEYS.difference(checkpoint)
            if missing:
                raise ValueError(
                    "Checkpoint cannot resume training; missing: "
                    + ", ".join(sorted(missing))
                )
    saved_hyperparams = checkpoint.get("hyperparameters", {}) if full_resume else {}

    if seed is None:
        seed = saved_hyperparams.get("seed")

    # Reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if object_batch_size is not None and batch_size is not None:
        if object_batch_size != batch_size:
            raise ValueError("object_batch_size and batch_size disagree")
    if views_per_object is not None and n_views is not None:
        if views_per_object != n_views:
            raise ValueError("views_per_object and n_views disagree")
    if object_batch_size is None:
        object_batch_size = (
            batch_size
            if batch_size is not None
            else saved_hyperparams.get("object_batch_size", params.object_batch_size)
        )
    if views_per_object is None:
        views_per_object = (
            n_views
            if n_views is not None
            else saved_hyperparams.get("views_per_object", params.views_per_object)
        )
    if updates_per_epoch is None:
        updates_per_epoch = saved_hyperparams.get(
            "updates_per_epoch", params.updates_per_epoch
        )
    if critic_iterations is None:
        critic_iterations = saved_hyperparams.get("critic_iterations", 5)
    if lambda_gp is None:
        lambda_gp = saved_hyperparams.get("lambda_gp", 10)
    if lambda_binary is None:
        lambda_binary = saved_hyperparams.get("lambda_binary", 0.1)
    if lambda_binary < 0:
        raise ValueError("lambda_binary must be non-negative")
    if (
        min(object_batch_size, views_per_object, updates_per_epoch, critic_iterations)
        < 1
    ):
        raise ValueError("batch, view, update, and critic counts must be positive")
    projection_batch_size = object_batch_size * views_per_object

    quality_checks = quality_checks or QualityChecks(
        **saved_hyperparams.get("quality_checks", {})
    )
    quality_checks.validate()

    if not output_dir:
        if full_resume:
            checkpoint_dir = os.path.dirname(os.path.abspath(resume_from))
            output_dir = (
                os.path.dirname(checkpoint_dir)
                if os.path.basename(checkpoint_dir) == "model_dump"
                else checkpoint_dir
            )
        else:
            output_dir = os.path.join("outputs", "GANular_run")

    if os.path.exists(output_dir) and not full_resume:
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    os.makedirs(output_dir, exist_ok=full_resume)

    if not data_dir:
        data_dir = saved_hyperparams.get("data_dir", params.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    voxel_dir = os.path.join(output_dir, "gan_gens")
    if not os.path.exists(voxel_dir):
        os.makedirs(voxel_dir)

    proj_dir = os.path.join(output_dir, "gan_projections")
    if not os.path.exists(proj_dir):
        os.makedirs(proj_dir)

    npy_dir = os.path.join(output_dir, "gan_npy")
    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    model_dump_dir: str = os.path.join(output_dir, "model_dump")
    if not os.path.exists(model_dump_dir):
        os.makedirs(model_dump_dir)

    start_epoch = checkpoint["epoch"] if full_resume else 0
    if full_resume:
        next_checkpoint = os.path.join(
            model_dump_dir, f"training_epoch_{start_epoch + 1}.pth"
        )
        if os.path.exists(next_checkpoint):
            raise FileExistsError(
                "Refusing to overwrite later training output: " + next_checkpoint
            )

    quality_dir = os.path.join(output_dir, "quality_reports")
    if quality_checks.enabled:
        os.makedirs(quality_dir, exist_ok=full_resume)

    if not epochs:
        epochs = params.epochs
    if learning_rate is None:
        learning_rate = saved_hyperparams.get("learning_rate", params.learning_rate)

    total_epochs = start_epoch + epochs
    print(f"Using device: {device}")

    # Save hyperparameters for reproducibility
    hyperparams = {
        "seed": seed,
        "object_batch_size": object_batch_size,
        "views_per_object": views_per_object,
        "projection_batch_size": projection_batch_size,
        "updates_per_epoch": updates_per_epoch,
        "projection_sample_size": quality_checks.projection_sample_size,
        "epochs": total_epochs,
        "learning_rate": learning_rate,
        "noise_size": params.noise_size,
        "embed_size": params.embed_size,
        "img_resolution": params.img_resolution,
        "device": str(device),
        "data_dir": data_dir,
        "critic_iterations": critic_iterations,
        "lambda_gp": lambda_gp,
        "lambda_binary": lambda_binary,
        "quality_checks": quality_checks.to_dict(),
    }
    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    img_paths = _image_paths(data_dir)

    if not img_paths:
        raise ValueError(f"No images found in {data_dir}")

    print(f"Found {len(img_paths)} images.")

    image_transform = transforms.Compose(
        [
            transforms.Resize(
                (params.img_resolution, params.img_resolution),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
        ]
    )
    dataset = ProjectionDataset(
        img_paths=img_paths,
        size=(params.img_resolution, params.img_resolution),
        transform=image_transform,
    )

    real_sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=True,
        num_samples=updates_per_epoch * critic_iterations * projection_batch_size,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=projection_batch_size,
        sampler=real_sampler,
        num_workers=0,
    )

    quality_seed = quality_checks.evaluation_seed
    if quality_checks.enabled:
        reference_dataset = dataset
        if quality_checks.projection_reference_dir:
            reference_paths = _image_paths(quality_checks.projection_reference_dir)
            if not reference_paths:
                raise ValueError(
                    "No projection images found in "
                    f"{quality_checks.projection_reference_dir}"
                )
            reference_dataset = ProjectionDataset(
                img_paths=reference_paths,
                size=(params.img_resolution, params.img_resolution),
                transform=image_transform,
            )
        real_2d_features = _reference_2d_features(
            reference_dataset,
            quality_checks.projection_sample_size,
            quality_checks.projection_threshold,
            quality_seed,
        )
        real_3d_features = load_uct_features(quality_checks)
        with open(os.path.join(quality_dir, "reference_summary.json"), "w") as f:
            json.dump(
                {
                    "2d_features": FEATURES_2D,
                    "valid_2d_samples": len(real_2d_features),
                    "projection_sample_size": quality_checks.projection_sample_size,
                    "valid_3d_samples": (
                        0 if real_3d_features is None else len(real_3d_features)
                    ),
                    "uct_data_dir": quality_checks.uct_data_dir,
                    "projection_reference_dir": (
                        quality_checks.projection_reference_dir or data_dir
                    ),
                    "projection_reference_is_training_data": (
                        quality_checks.projection_reference_dir is None
                    ),
                },
                f,
                indent=2,
            )

    # Initialize models
    gen = VoxelGenerator().to(device)
    disc = ProjectionDiscriminator(img_channels=1, features_d=64).to(device)
    proj = VoxelProjector(
        volume_size=params.img_resolution, num_views=views_per_object
    ).to(device)

    if quality_checks.enabled:
        rng = np.random.default_rng(quality_seed)
        quality_object_count = max(
            quality_checks.sample_size,
            math.ceil(quality_checks.projection_sample_size / views_per_object),
        )
        quality_noise = torch.from_numpy(
            rng.normal(0.0, 0.5, (quality_object_count, params.noise_size)).astype(
                np.float32
            )
        )
        quality_rotations = _fixed_rotations(
            quality_object_count, views_per_object, rng
        )

    gen.train()
    disc.train()
    proj.train()

    # Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    d_losses = []
    g_losses = []
    global_step = 0
    last_training_health = None

    if full_resume:
        (
            start_epoch,
            global_step,
            d_losses,
            g_losses,
            last_training_health,
        ) = _restore_training_checkpoint(checkpoint, gen, disc, opt_gen, opt_disc)
        print(f"Resuming training after epoch {start_epoch}.")
    elif checkpoint is not None:
        state_dict = checkpoint.get("state_dict", checkpoint)
        gen.load_state_dict(state_dict)
        discriminator_path = os.path.join(
            os.path.dirname(resume_from),
            os.path.basename(resume_from).replace("generator_", "discriminator_"),
        )
        if discriminator_path != resume_from and os.path.isfile(discriminator_path):
            disc.load_state_dict(
                torch.load(discriminator_path, map_location=device, weights_only=True)
            )
        print("Warm-starting from model weights with fresh optimiser state.")

    print("Starting training...")
    for epoch in range(start_epoch, total_epochs):
        real_batches = iter(dataloader)
        for step in tqdm(
            range(updates_per_epoch), desc=f"Epoch {epoch + 1}/{total_epochs}"
        ):
            # --- Train Discriminator ---
            critic_stats = []
            for _ in range(critic_iterations):
                opt_disc.zero_grad()

                # Independent real projections; no repeated images within objects.
                real_imgs = next(real_batches).to(device)
                d_real = disc(real_imgs)

                # Correlated multi-view projections from independent objects.
                noise = sample_latent_noise((object_batch_size, params.noise_size)).to(
                    device
                )
                voxels = gen(noise)

                fake_imgs = proj(voxels)

                d_fake = disc(fake_imgs.detach())
                fake_score = _mean_object_score(
                    d_fake, object_batch_size, views_per_object
                )

                # Gradient Penalty
                gp, gp_gradient_norms = compute_gradient_penalty(
                    disc,
                    real_imgs,
                    fake_imgs.detach(),
                    device,
                    return_gradient_norms=True,
                )

                # WGAN Loss
                d_loss = -(d_real.mean() - fake_score) + lambda_gp * gp
                d_loss.backward()
                critic_gradient_norm = _gradient_norm(disc)
                opt_disc.step()
                critic_stats.append(
                    {
                        "critic_loss": d_loss.item(),
                        "critic_real": d_real.mean().item(),
                        "critic_fake": fake_score.item(),
                        "gradient_penalty": gp.item(),
                        "gp_gradient_norm": gp_gradient_norms.mean().item(),
                        "critic_gradient_norm": critic_gradient_norm,
                    }
                )

            # --- Train Generator ---
            opt_gen.zero_grad()

            # Fresh noise for Generator update to prevent graph retention issues
            noise_g = sample_latent_noise((object_batch_size, params.noise_size)).to(
                device
            )
            voxels_g = gen(noise_g)

            fake_imgs_g = proj(voxels_g)
            d_fake_g = disc(fake_imgs_g)

            adversarial_loss = -_mean_object_score(
                d_fake_g, object_batch_size, views_per_object
            )
            binarity_loss = (4.0 * voxels_g * (1.0 - voxels_g)).mean()
            g_loss = adversarial_loss + lambda_binary * binarity_loss
            g_loss.backward()
            generator_gradient_norm = _gradient_norm(gen)
            opt_gen.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            if quality_checks.enabled:
                mean_critic = {
                    name: float(np.mean([item[name] for item in critic_stats]))
                    for name in critic_stats[0]
                }
                fake_values = fake_imgs_g.detach()
                optimization_stats = {
                    **mean_critic,
                    "generator_loss": g_loss.item(),
                    "generator_adversarial_loss": adversarial_loss.item(),
                    "voxel_binarity_loss": binarity_loss.item(),
                    "ambiguous_voxel_fraction": (
                        ((voxels_g > 0.05) & (voxels_g < 0.95))
                        .float()
                        .mean()
                        .item()
                    ),
                    "generator_gradient_norm": generator_gradient_norm,
                    "projection_saturation": max(
                        (fake_values < 0.01).float().mean().item(),
                        (fake_values > 0.99).float().mean().item(),
                    ),
                    "voxel_mean": voxels_g.detach().mean().item(),
                }
                last_training_health = check_training_health(
                    optimization_stats, global_step, quality_checks
                )
                append_jsonl(
                    os.path.join(quality_dir, "training_health.jsonl"),
                    {
                        "epoch": epoch + 1,
                        "step": step,
                        "global_step": global_step,
                        **last_training_health,
                    },
                )

            if step % 10 == 0:
                tqdm.write(
                    f"Epoch [{epoch + 1}/{total_epochs}] "
                    f"Step [{step}/{updates_per_epoch}] "
                    f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}"
                    + (
                        f" Training health: {last_training_health['status']}"
                        if last_training_health
                        else ""
                    )
                )
            global_step += 1

        # Plot losses at the end of each epoch
        save_loss_plot(
            d_losses, g_losses, os.path.join(plots_dir, f"training_loss{epoch + 1}.png")
        )

        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            torch.save(
                gen.state_dict(),
                os.path.join(model_dump_dir, f"generator_epoch_{epoch + 1}.pth"),
            )
            torch.save(
                disc.state_dict(),
                os.path.join(model_dump_dir, f"discriminator_epoch_{epoch + 1}.pth"),
            )

            # Save visualization
            save_voxel_plot(
                voxels,
                os.path.join(voxel_dir, f"voxel_epoch_{epoch + 1}.png"),
                save_dir=voxel_dir,
            )

            save_voxel_npy(
                voxels, os.path.join(npy_dir, f"voxel_epoch_{epoch + 1}.npy")
            )

            save_projection_plot(
                fake_imgs, os.path.join(proj_dir, f"proj_epoch_{epoch + 1}.png")
            )

        if (
            quality_checks.enabled
            and (epoch + 1) % quality_checks.evaluation_interval == 0
        ):
            quality_volumes, quality_projections = _generate_quality_samples(
                gen,
                proj,
                quality_noise,
                quality_rotations,
                object_batch_size,
                views_per_object,
                device,
            )
            quality_volumes, quality_projections = _select_quality_samples(
                quality_volumes,
                quality_projections,
                quality_checks.sample_size,
                quality_checks.projection_sample_size,
                views_per_object,
            )
            sample_assessment = assess_generated_samples(
                real_2d_features,
                quality_projections,
                quality_volumes,
                quality_checks,
                real_3d_features,
                quality_seed,
            )
            report = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "training_health": last_training_health,
                **sample_assessment,
            }
            append_jsonl(os.path.join(quality_dir, "sample_quality.jsonl"), report)
            tqdm.write(
                "Quality checks: "
                + ", ".join(
                    f"{name}={result.get('passed', result['status'])}"
                    for name, result in report.items()
                    if isinstance(result, dict)
                )
            )

        torch.save(
            _training_checkpoint(
                epoch + 1,
                global_step,
                gen,
                disc,
                opt_gen,
                opt_disc,
                d_losses,
                g_losses,
                hyperparams,
                last_training_health,
            ),
            os.path.join(model_dump_dir, f"training_epoch_{epoch + 1}.pth"),
        )

        torch.save(
            _training_checkpoint(
                epoch + 1,
                global_step,
                gen,
                disc,
                opt_gen,
                opt_disc,
                d_losses,
                g_losses,
                hyperparams,
                last_training_health,
            ),
            os.path.join(model_dump_dir, f"training_epoch_{epoch + 1}.pth"),
        )
