import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

# Channel-wise dataset statistics for CIFAR-100.
#
# Each tuple contains the per-channel mean and standard deviation computed from
# the CIFAR-100 training split.  We subtract the mean and divide by the standard
# deviation in the transform pipeline (see ``T.Normalize`` below) so the network
# receives inputs that are zero-centered with unit variance.  When we need to
# convert tensors back into images for saving, the ``_denormalize`` helper uses
# the same values to undo the transformation.
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAugmenter:
    """Class to handle CIFAR-style image augmentation operations."""

    def __init__(
        self,
        augmentations_per_image: int = 5,
        seed: int = 42,
        save_original: bool = True,
        image_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ):
        """
        Initialize the ImageAugmenter.

        Args:
            augmentations_per_image: Number of augmented versions per original image.
            seed: Random seed for reproducibility.
            save_original: Whether to save the original image with prefix 'orig_'.
            image_extensions: Tuple of valid image file extensions.
        """
        self.augmentations_per_image = augmentations_per_image
        self.seed = seed
        self.save_original = save_original
        self.image_extensions = image_extensions

        self._set_seed()

        # Define torchvision-based augmentation pipeline so that
        # offline augmentations are consistent with the on-the-fly
        # augmentations used during training.
        self._to_pil = T.ToPILImage()
        self.transform = T.Compose(
            [
                T.RandomCrop(32, padding=4, padding_mode="reflect"),
                T.RandomHorizontalFlip(p=0.5),
                T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
                T.ToTensor(),
                T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ]
        )
        self.random_erasing = T.RandomErasing(
            p=0.25, value="random", scale=(0.02, 0.33), ratio=(0.3, 3.3)
        )

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def augment_image(self, image: Image.Image) -> Image.Image:
        """
        Apply augmentation transforms to a single image.

        Args:
            image: PIL Image to augment.

        Returns:
            Augmented PIL Image.
        """
        # Apply torchvision transform pipeline.
        tensor = self.transform(image)
        tensor = self.random_erasing(tensor)

        # Denormalize before converting back to PIL for saving.
        tensor = self._denormalize(tensor)
        tensor = torch.clamp(tensor, 0.0, 1.0)
        return self._to_pil(tensor)

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert a normalized tensor back to the [0, 1] range."""
        mean = torch.tensor(CIFAR100_MEAN, device=tensor.device).view(-1, 1, 1)
        std = torch.tensor(CIFAR100_STD, device=tensor.device).view(-1, 1, 1)
        return tensor * std + mean

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Augment all images in input directory and save to output directory.

        Preserves folder structure. Skips files that fail to load.

        Args:
            input_dir: Path to input directory with class subfolders.
            output_dir: Path to output directory for augmented images.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        count = 0

        image_files = self._find_image_files(input_path)

        logger.info(f"Found {len(image_files)} images to augment.")

        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue

            # Determine output subdirectory
            rel_dir = img_path.parent.relative_to(input_path)
            target_dir = output_path / rel_dir
            if not target_dir.exists():
                target_dir.mkdir(parents=True, exist_ok=True)

            # Save original if requested
            if self.save_original:
                orig_name = f"orig_{img_path.name}"
                image.save(target_dir / orig_name)

            # Generate and save augmented versions
            for i in range(self.augmentations_per_image):
                augmented = self.augment_image(image.copy())
                aug_name = f"aug_{i}_{img_path.name}"
                augmented.save(target_dir / aug_name)
                count += 1

        logger.info(
            f"Augmentation of {count} images completed. Output saved to: {output_dir}"
        )

    def _find_image_files(self, root: Path) -> List[Path]:
        """
        Recursively find all image files in directory.

        Args:
            root: Root directory path.

        Returns:
            List of image file paths.
        """
        files = []
        for ext in self.image_extensions:
            files.extend(root.rglob(f"*{ext}"))
        return files


def augment_dataset(
    input_dir: str,
    output_dir: str,
    augmentations_per_image: int = 5,
    seed: int = 42,
) -> None:
    """
    Backward-compatible wrapper for legacy code.

    Args:
        input_dir: Directory containing cleaned images (organized by class).
        output_dir: Directory to save augmented images.
        augmentations_per_image: Number of augmented versions per original image.
        seed: Random seed for reproducibility.
    """
    augmenter = ImageAugmenter(
        augmentations_per_image=augmentations_per_image, seed=seed, save_original=True
    )
    augmenter.process_directory(input_dir, output_dir)
