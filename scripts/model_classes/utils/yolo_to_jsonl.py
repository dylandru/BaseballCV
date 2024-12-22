import random
import logging
from typing import List, Dict, Any
from PIL import Image, ImageEnhance
import os
import json
from torch.utils.data import Dataset
import numpy as np

class JSONLDetection(Dataset):
    def __init__(self, entries, image_directory_path, logger: logging.Logger, augment=True):
        """
        Initialize the JSONLDetection dataset.

        Args:
            entries: List of entries (annotations) for the dataset.
            image_directory_path: Path to the directory containing images.
            logger: Logger instance for logging.
            augment: Whether to apply data augmentation.
        """
        self.entries = entries
        self.image_directory_path = image_directory_path
        self.augment = augment
        self.transforms = [self.random_color_jitter, self.random_blur, self.random_noise] if augment else []
        self.logger = logger

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.entries)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing the prefix, suffix, and image.
        """
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path)

        if self.augment and random.random() > 0.5:
            for transform in self.transforms:
                image = transform(image)

        return image, entry
    
    @staticmethod
    def random_color_jitter(image):
        """Apply random color jittering to the image."""
        factors = {
            'brightness': random.uniform(0.8, 1.2),
            'contrast': random.uniform(0.8, 1.2),
            'color': random.uniform(0.8, 1.2)
        }

        for enhance_type, factor in factors.items():
            if random.random() > 0.5:
                if enhance_type == 'brightness':
                    image = ImageEnhance.Brightness(image).enhance(factor)
                elif enhance_type == 'contrast':
                    image = ImageEnhance.Contrast(image).enhance(factor)
                elif enhance_type == 'color':
                    image = ImageEnhance.Color(image).enhance(factor)
        return image

    @staticmethod
    def random_blur(image):
        """Apply random Gaussian blur to the image."""
        if random.random() > 0.8:
            from PIL import ImageFilter
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        return image

    @staticmethod
    def random_noise(image):
        """Apply random noise to the image."""
        if random.random() > 0.8:
            img_array = np.array(image)
            noise = np.random.normal(0, 2, img_array.shape)
            noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_img)
        return image
    
    @staticmethod
    def load_jsonl_entries(jsonl_file_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
        """
        Load entries from a JSONL file.

        Args:
            jsonl_file_path: Path to the JSONL file.
            logger: Logger instance for logging.
        Returns:

            List of entries loaded from the JSONL file.
        """
        entries = []
        try:
            with open(jsonl_file_path, 'r') as file:
                for line in file:
                    try:
                        data = json.loads(line.strip())
                        if isinstance(data, dict):
                            entries.append(data)
                        else:
                            logger.warning(f"Skipping invalid entry: not dictionary - {data}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON line: {e}")
                        continue

            if not entries:
                logger.error(f"No valid entries found in {jsonl_file_path}")
                raise ValueError(f"No valid entries found in {jsonl_file_path}")

            logger.info(f"Loaded {len(entries)} valid entries from {jsonl_file_path}")
            return entries

        except Exception as e:
            logger.error(f"Error loading entries from {jsonl_file_path}: {str(e)}")
            raise
