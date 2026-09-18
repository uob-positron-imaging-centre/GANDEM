"""Default training hyperparameters and device configuration for GANular.

Attributes:
    epochs: Number of training epochs.
    object_batch_size: Number of independently generated objects per update.
    views_per_object: Number of projections rendered per generated object.
    updates_per_epoch: Number of generator updates in each epoch.
    learning_rate: Adam optimiser learning rate.
    noise_size: Dimensionality of the latent noise vector fed to the generator.
    embed_size: Spatial size of the 3D embedding cube produced by the generator's
        linear layer (i.e. each side of the ``embed_size^3`` cube).
    img_resolution: Pixel resolution of 2D projection images (square).
        Must be divisible by 16.
    batch_size: Deprecated alias for ``object_batch_size``.
    num_views: Deprecated alias for ``views_per_object``.
    device: Automatically selected accelerator ``torch.device``, or CPU.
    data_dir: Default path to the directory of cropped particle images.
"""

import os
import torch

# Training parameters
epochs = 500
object_batch_size = 2
views_per_object = 2
updates_per_epoch = 100
learning_rate = 0.0002

noise_size = 256
embed_size = 16

img_resolution = 64  # Size in pixels (e.g., 64x64)
# Backward-compatible aliases.
batch_size = object_batch_size
num_views = views_per_object

assert (
    img_resolution % 16 == 0
), "img_resolution should be divisible by 16 for 4 downsampling layers"

device = torch.accelerator.current_accelerator(check_available=True) or torch.device(
    "cpu"
)


data_dir = os.path.join("cropped_particles", "MCC_Vivapur_102")
