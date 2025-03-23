import random
import logging
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageEnhance
import os
import json
from torch.utils.data import Dataset
import numpy as np

class JSONLDetection(Dataset):
    '''
    A class to load YOLO datasets into a JSONL format that can be used for training and validation in PyTorch.
    '''
    def __init__(self, entries: List[Dict[str, Any]], image_directory_path: str, logger: logging.Logger, augment: bool = True) -> None:
        """
        Initialize the JSONLDetection dataset.

        Args:
            entries (List[Dict[str, Any]]): List of entries (annotations) for the dataset.
            image_directory_path (str): Path to the directory containing images.
            logger (logging.Logger): Logger instance for logging.
            augment (bool): Whether to apply data augmentation.
        """

        self.entries = entries
        self.image_directory_path = image_directory_path
        self.augment = augment
        self.transforms = [self.random_color_jitter, self.random_blur, self.random_noise] if augment else []
        self.logger = logger

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            image (PIL.Image.Image): The image.
            entry (Dict[str, Any]): The entry.
        """
        entry = self.entries[idx]
        image_path = os.path.join(self.image_directory_path, entry['image'])
        image = Image.open(image_path)

        if self.augment and random.random() > 0.5:
            for transform in self.transforms:
                image = transform(image)

        return image, entry
    
    @staticmethod
    def random_color_jitter(image: Image.Image) -> Image.Image:
        """
        Apply random color jittering to image.

        Args:
            image (PIL.Image.Image): The image to apply the transformation to.

        Returns:
            image (PIL.Image.Image): The transformed image.
        """
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
    def random_blur(image: Image.Image) -> Image.Image:
        """
        Apply random Gaussian blur to the image.

        Args:
            image (PIL.Image.Image): The image to apply the transformation to.

        Returns:
            image (PIL.Image.Image): The transformed image.
        """
        if random.random() > 0.8:
            from PIL import ImageFilter
            return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
        return image

    @staticmethod
    def random_noise(image: Image.Image) -> Image.Image:
        """
        Apply random noise to the image.

        Args:
            image (PIL.Image.Image): The image to apply the transformation to.

        Returns:
            image (PIL.Image.Image): The transformed image.
        """
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
            jsonl_file_path (str): Path to the JSONL file.
            logger (logging.Logger): Logger instance for logging.

        Returns:
            List[Dict[str, Any]]: List of entries loaded from the JSONL file.
        
        Raises:
            ValueError: If no valid entries are found in the JSONL file.
            Exception: If there is an error loading entries from the JSONL file.
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
