import logging
import os
import shutil
import random
from tqdm import tqdm
from typing import Dict, Tuple
import json

class DataProcessor:

    def __init__(self, logger: logging.Logger):
        self.logger = logger


    def prepare_dataset(self, base_path: str, dict_classes: Dict[int, str],
                     train_test_split: Tuple[int, int, int] = None,
                     dataset_type: str = "yolo") -> Tuple[str, str, str, str, str]:
        """
        Prepare the dataset by splitting it into train, test, and validation sets.

        Args:
            base_path: Base path to the dataset.
            dict_classes: Dictionary mapping class IDs to class names.
            train_test_split: Tuple specifying the train, test, and validation split ratios.
            dataset_type: Type of dataset (yolo or paligemma). Default is yolo.
            
        Returns:
            Tuple containing the paths to the train and validation datasets.
        """
        existing_split = all(
            os.path.exists(os.path.join(base_path, split))
            for split in ["train", "test", "valid"]
        )
        if dataset_type == "yolo":
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

            train_file_path = self.convert_annotations(base_path, "train", dict_classes)
            test_file_path = self.convert_annotations(base_path, "test", dict_classes)
            valid_file_path = self.convert_annotations(base_path, "valid", dict_classes)
            
            return os.path.join(base_path, "train", "images/"), os.path.join(base_path, "valid", "images/"), train_file_path, test_file_path, valid_file_path
    
        elif dataset_type == "paligemma":
            jsonl_files = {
                split: next(f for f in os.listdir(base_path) if split in f.lower() and f.endswith('.jsonl'))
                for split in ["train", "test", "valid"]
            }
            
            train_file_path = jsonl_files["train"]
            test_file_path = jsonl_files["test"] 
            valid_file_path = jsonl_files["valid"]

            for split in ["train", "test", "valid"]:
                os.makedirs(os.path.join(base_path, split, "images"), exist_ok=True)

            for split_name, jsonl_file in jsonl_files.items():
                split_image_dir = os.path.join(base_path, split_name, "images")
                
                with open(os.path.join(base_path, jsonl_file), 'r') as f:
                    for line in f:
                        entry = json.loads(line)
                        
                        for class_name in [part.split("<loc")[0].strip() 
                                         for part in entry["suffix"].split(" ; ")]:
                            if class_name and class_name not in dict_classes.values():
                                raise ValueError(f"Class '{class_name}' from dataset not in dict_classes... Check your classes.")
                        
                        image_name = entry["image"]
                        src_image = os.path.join(base_path, image_name)
                        dst_image = os.path.join(split_image_dir, image_name)
                        
                        if os.path.exists(src_image) and not os.path.exists(dst_image):
                            shutil.copy2(src_image, dst_image)

            return os.path.join(base_path, "train", "images/"), os.path.join(base_path, "valid", "images/"), train_file_path, test_file_path, valid_file_path
        
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
    
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
                
        return output_file