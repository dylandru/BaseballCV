import random
import shutil
import logging
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageEnhance
import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset

class YOLOToJSONLDetection(Dataset):
    def __init__(self, parent, entries, image_directory_path, logger: logging.Logger, augment=True):
        """
        Initialize the YOLOToJSONL dataset.

        Args:
            parent: The parent model class instance.
            entries: List of entries (annotations) for the dataset.
            image_directory_path: Path to the directory containing images.
            augment: Whether to apply data augmentation.
            logger: Logger instance for logging.
        """
        self.parent = parent
        self.entries = entries
        self.image_directory_path = image_directory_path
        self.augment = augment
        self.transforms = parent.get_augmentation_transforms() if augment else []
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
        image, data = self.parent.get_jsonl_item(self.entries, idx, self.image_directory_path)

        if self.augment and random.random() > 0.5:
            for transform in self.transforms:
                image = transform(image)

        prefix = data['prefix']
        suffix = data['suffix']
        return prefix, suffix, image

    def get_unaugmented_item(self, idx):
        """
        Get a sample from the dataset without augmentation.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing the image and data.
        """
        return self.parent.get_jsonl_item(self.entries, idx, self.image_directory_path)
    
    def load_jsonl_entries(self, jsonl_file_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
        """
        Load entries from a JSONL file.

        Args:
            jsonl_file_path: Path to the JSONL file.

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
                            self.logger.warning(f"Skipping invalid entry: not dictionary - {data}")
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse JSON line: {e}")
                        continue

            if not entries:
                self.logger.error(f"No valid entries found in {jsonl_file_path}")
                raise ValueError(f"No valid entries found in {jsonl_file_path}")

            self.logger.info(f"Loaded {len(entries)} valid entries from {jsonl_file_path}")
            return entries

        except Exception as e:
            self.logger.error(f"Error loading entries from {jsonl_file_path}: {str(e)}")
            raise

    def get_jsonl_item(self, entries: List[Dict[str, Any]], idx: int,
                image_directory_path: str) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Get an item from the JSONL entries.

        Args:
            entries: List of entries.
            idx: Index of the entry to retrieve.
            image_directory_path: Path to the directory containing images.

        Returns:
            Tuple containing the image and entry data.
        """
        try:
            entry = entries[idx]
            if not isinstance(entry, dict):
                raise TypeError(f"Entry must be a dictionary, got {type(entry)}")

            image_name = entry['image']
            if not isinstance(image_name, (str, bytes, os.PathLike)):
                raise TypeError(f"Image name must be a string or path-like object, got {type(image_name)}")

            image_path = os.path.join(image_directory_path, image_name)

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image, entry

        except Exception as e:
            self.logger.error(f"Error processing item at index {idx}: {str(e)}")
            raise

    def convert_annotations(self, base_path: str, split: str, dict_classes: Dict[int, str]):
        """
        Convert annotations to the required format.

        Args:
            base_path: Base path to the dataset.
            split: The split to process (train, test, valid).
            dict_classes: Dictionary mapping class IDs to class names.
        """
        annotations_dir = os.path.join(base_path, split, "labels")
        output_file = os.path.join(base_path, split, "images", f"{split}_annotations.json")

        annotations = []
        files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]

        for filename in tqdm(files, desc=f"Converting {split} annotations"):
            annotation_file = os.path.join(annotations_dir, filename)
            with open(annotation_file, 'r') as f:
                lines = f.readlines()

            image_name = os.path.basename(annotation_file).replace('.txt', '.jpg')
            suffix_lines = []

            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                x1 = int((x_center - width/2) * 1000)
                y1 = int((y_center - height/2) * 1000)
                x2 = int((x_center + width/2) * 1000)
                y2 = int((y_center + height/2) * 1000)

                class_name = dict_classes.get(class_id, f"Unknown Class {class_id}")
                suffix_line = f"{class_name}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"
                suffix_lines.append(suffix_line)

            annotations.append({
                "image": image_name,
                "prefix": "<OD>",
                "suffix": "".join(suffix_lines)
            })

        with open(output_file, 'w') as f:
            for annotation in annotations:
                f.write(json.dumps(annotation) + '\n')

    def prepare_dataset(self, base_path: str, dict_classes: Dict[int, str],
                     train_test_split: Tuple[int, int, int] = (80, 10, 10)):
        """
        Prepare the dataset by splitting it into train, test, and validation sets.

        Args:
            base_path: Base path to the dataset.
            dict_classes: Dictionary mapping class IDs to class names.
            train_test_split: Tuple specifying the train, test, and validation split ratios.

        Returns:
            Tuple containing the paths to the train and validation datasets.
        """
        existing_split = all(
            os.path.exists(os.path.join(base_path, split))
            for split in ["train", "test", "valid"]
        )

        if existing_split:
            self.logger.info("Found existing train/test/valid split. Using existing split.")
            train_files = [f for f in os.listdir(os.path.join(base_path, "train", "images"))
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            test_files = [f for f in os.listdir(os.path.join(base_path, "test", "images"))
                        if f.endswith(('.jpg', '.png', '.jpeg'))]
            valid_files = [f for f in os.listdir(os.path.join(base_path, "valid", "images"))
                          if f.endswith(('.jpg', '.png', '.jpeg'))]

            for split, files in [("train", train_files), ("test", test_files), ("valid", valid_files)]:
                label_dir = os.path.join(base_path, split, "labels")
                os.makedirs(label_dir, exist_ok=True)

                for img_file in files:
                    base_name = os.path.splitext(img_file)[0]
                    label_name = f"{base_name}.txt"
                    src_label = os.path.join(base_path, label_name)
                    dst_label = os.path.join(label_dir, label_name)

                    if os.path.exists(src_label) and not os.path.exists(dst_label):
                        shutil.copy2(src_label, dst_label)

            self.logger.info(f"Train: {len(train_files)} images, Test: {len(test_files)} images, Valid: {len(valid_files)} images")
        else:
            self.logger.info("No existing split found. Creating new train/test/valid split.")
            for split in ["train", "test", "valid"]:
                os.makedirs(os.path.join(base_path, split, "images"), exist_ok=True)
                os.makedirs(os.path.join(base_path, split, "labels"), exist_ok=True)

            image_files = [f for f in os.listdir(base_path)
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            total_images = len(image_files)

            random.shuffle(image_files)
            train_count = int(train_test_split[0] * total_images / 100)
            test_count = int(train_test_split[1] * total_images / 100)

            train_files = image_files[:train_count]
            test_files = image_files[train_count:train_count + test_count]
            valid_files = image_files[train_count + test_count:]

            splits = [("train", train_files), ("test", test_files), ("valid", valid_files)]
            for split_name, files in splits:
                for file_name in tqdm(files, desc=f"Processing {split_name}"):
                    src_image = os.path.join(base_path, file_name)
                    dst_image = os.path.join(base_path, split_name, "images", file_name)
                    shutil.copy2(src_image, dst_image)

                    label_name = os.path.splitext(file_name)[0] + ".txt"
                    src_label = os.path.join(base_path, label_name)
                    dst_label = os.path.join(base_path, split_name, "labels", label_name)

                    if os.path.exists(src_label):
                        shutil.copy2(src_label, dst_label)

        for split in ["train", "valid", "test"]:
            self.convert_annotations(base_path, split, dict_classes)

        return os.path.join(base_path, "train", "images/"), os.path.join(base_path, "valid", "images/")

    def return_clean_text_output(self, results: Dict) -> str:
        """
        Return clean text output from the results.

        Args:
            results: Dictionary containing the results.

        Returns:
            Clean text output.
        """
        return next(iter(results.values())).strip()
    
    def _get_augmentation_transforms(self):
        """
        Get data augmentation transforms.

        Returns:
            List of augmentation transform functions.
        """
        def random_color_jitter(image):
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

        def random_blur(image):
            if random.random() > 0.8:
                from PIL import ImageFilter
                return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0)))
            return image

        def random_noise(image):
            if random.random() > 0.8:
                import numpy as np
                img_array = np.array(image)
                noise = np.random.normal(0, 2, img_array.shape)
                noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                return Image.fromarray(noisy_img)
            return image

        return [random_color_jitter, random_blur, random_noise]