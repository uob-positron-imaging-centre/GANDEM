import torch
import cv2
from torch.utils import data


class ProjectionDataset(data.Dataset):
    """PyTorch dataset for loading 2D particle projection images.

    Images are loaded as grayscale, normalised to ``[0, 1]``, colour-inverted
    (so particles become white-on-black), and binarised at a 0.5 threshold.

    Args:
        img_paths: File paths to the real 2D projection images.
        size: Target spatial dimensions, e.g. ``(64, 64)``.
        transform: Optional ``torchvision`` transform applied after loading.
    """

    def __init__(self, img_paths: list, size: tuple, transform=None):
        self.img_paths = img_paths
        self.sample_size = size  # (64, 64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx) -> torch.Tensor:
        path = self.img_paths[idx]
        img_arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img_arr is None:
            raise ValueError(f"Could not load image at {path}")

        img = torch.from_numpy(img_arr).float() / 255.0

        # White and black inversion
        img = 1.0 - img

        # Add channel dim: (H, W) -> (1, H, W)
        img = img.unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        img = (img > 0.5).float()

        return img
