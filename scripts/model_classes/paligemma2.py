import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import ImageEnhance
import torch.backends
from tqdm import tqdm
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import random
import shutil
from typing import List, Dict, Any, Tuple
import logging
import seaborn as sns
import torch.multiprocessing as mp
from datetime import datetime
import supervision as sv
from supervision.metrics import MeanAveragePrecision, MetricTarget

class YOLOToPaliGemma2(Dataset):
    def __init__(self, parent, entries, image_directory_path, augment=True):
        """
        Initialize the YOLOToPaliGemma2 dataset.

        Args:
            parent: The parent PaliGemma2 instance.
            entries: List of entries (annotations) for the dataset.
            image_directory_path: Path to the directory containing images.
            augment: Whether to apply data augmentation.
        """
        self.parent = parent
        self.entries = entries
        self.image_directory_path = image_directory_path
        self.augment = augment
        self.transforms = parent._get_augmentation_transforms() if augment else []

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
        image, data = self.parent._get_jsonl_item(self.entries, idx, self.image_directory_path)

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
        return self.parent._get_jsonl_item(self.entries, idx, self.image_directory_path)

class PaliGemma2:
    def __init__(self, model_id: str = 'google/paligemma2-3b-pt-448', model_run_path: str = f'paligemma2_run_{datetime.now().strftime("%Y%m%d")}', batch_size: int = 1):
        """
        Initialize the PaliGemma2 model.

        Args:
            model_id: The identifier of the model to use.
            model_run_path: The path to save model run information.
            batch_size: The batch size for training and inference.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                  else "mps" if torch.backends.mps.is_available()
                                  else "cpu")
        self.model_id = model_id
        self.batch_size = batch_size
        self.model = None
        self.processor = None
        self.peft_model = None
        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.logger = None
        self.model_run_path = model_run_path
        self._init_model()
        self._orig_logging()

    def _orig_logging(self):
        """Set up logging for the model."""
        log_dir = os.path.join(self.model_run_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"paligemma2_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"PALIGEMMA2_({self.model_id})")
        self.logger.info(f"Initializing PaliGemma2 model with Batch Size: {self.batch_size}")
        self.logger.info(f"Device: {self.device}")

    def _init_model(self):
        """Initialize the model and processor."""
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, trust_remote_code=True).to(self.device)
        self.processor = PaliGemmaProcessor.from_pretrained(
            self.model_id, trust_remote_code=True)

    def _load_jsonl_entries(self, jsonl_file_path: str) -> List[Dict[str, Any]]:
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

    def _get_jsonl_item(self, entries: List[Dict[str, Any]], idx: int,
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

    def _create_detection_dataset(self, jsonl_file_path: str, image_directory_path: str,
                           augment: bool = True) -> Dataset:
        """
        Create a detection dataset from a JSONL file.

        Args:
            jsonl_file_path: Path to the JSONL file.
            image_directory_path: Path to the directory containing images.
            augment: Whether to apply data augmentation.

        Returns:
            An instance of the YOLOToPaliGemma2 dataset.
        """
        entries = self._load_jsonl_entries(jsonl_file_path)

        return YOLOToPaliGemma2(self, entries, image_directory_path, augment)

    def _collate_fn(self, batch):
        """
        Collate function for DataLoader.

        Args:
            batch: List of tuples, where each tuple contains prefix, suffix, and image.

        Returns:
            Tuple containing the processed inputs and suffixes.
        """
        prefixes, suffixes, images = zip(*batch)
        inputs = self.processor(
            text=list(prefixes),
            images=list(images),
            return_tensors="pt",
            padding=True
        ).to(self.device)
        return inputs, suffixes

    def _prepare_dataset(self, base_path: str, dict_classes: Dict[int, str],
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
            self._convert_annotations(base_path, split, dict_classes)

        return os.path.join(base_path, "train", "images/"), os.path.join(base_path, "valid", "images/")

    def _convert_annotations(self, base_path: str, split: str, dict_classes: Dict[int, str]):
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

    def _setup_data_loaders(self, train_path: str, valid_path: str, num_workers: int = 0):
        """
        Set up data loaders for training and validation.

        Args:
            train_path: Path to the training dataset.
            valid_path: Path to the validation dataset.
            num_workers: Number of worker processes for data loading.
        """
        self.train_dataset = self._create_detection_dataset(
            jsonl_file_path=f"{train_path}train_annotations.json",
            image_directory_path=train_path,
            augment=True
        )
        self.val_dataset = self._create_detection_dataset(
            jsonl_file_path=f"{valid_path}valid_annotations.json",
            image_directory_path=valid_path,
            augment=False
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            shuffle=True,
            persistent_workers=False if num_workers == 0 else True,
            pin_memory=True if self.device == 'cuda' else False
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            persistent_workers=False if num_workers == 0 else True,
            pin_memory=True if self.device == 'cuda' else False
        )

    def _setup_peft(self, r: int = 8, alpha: int = 8, dropout: float = 0.05, create_peft_config: bool = True):
        """
        Set up Parameter-Efficient Fine-Tuning (PEFT).

        Args:
            r: Rank for LoRA.
            alpha: Scaling factor for LoRA.
            dropout: Dropout rate for LoRA.
            create_peft_config: Whether to create a new PEFT configuration.

        Returns:
            Information about trainable parameters.
        """
        if create_peft_config:
            if hasattr(self.model, 'peft_config'):
                self.logger.info("Existing PEFT configuration found. Removing old configuration...")
                peft_core_attrs = ['peft_config', 'base_model_prepare_inputs']

                for attr in peft_core_attrs:
                    if hasattr(self.model, attr):
                        delattr(self.model, attr)

                self.logger.info("PEFT configuration successfully removed - Adding new configuration...")

            config = LoraConfig(
                r=r,
                lora_alpha=alpha,
                target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
                lora_dropout=dropout,
                bias="none",
                inference_mode=False,
                use_rslora=True,
                init_lora_weights="gaussian"
            )

        else:
            if hasattr(self, 'peft_config'):
                self.logger.info("PEFT model already exists.. Skipping configuration.")

        self.peft_model = get_peft_model(self.model, config)
        trainable_params_info = self.peft_model.print_trainable_parameters()
        return trainable_params_info

    def _save_training_plots(self, vis_path: str, metrics: Dict[str, List[float]], epoch: int):
        """
        Save training plots.

        Args:
            vis_path: Path to save the visualizations.
            metrics: Dictionary containing training metrics.
            epoch: Current epoch number.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(metrics['train_losses'], label='Training Loss')
        plt.plot(metrics['val_losses'], label='Validation Loss')
        plt.title(f'Training and Validation Loss - Epoch {epoch}')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(vis_path, f'loss_curves.png'))
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(metrics['learning_rates'], label='Learning Rate')
        plt.title(f'Learning Rate Schedule - Epoch {epoch}')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(vis_path, f'learning_rate.png'))
        plt.close()

        if 'confusion_matrix' in metrics:
            plt.figure(figsize=(10, 8))
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Epoch {epoch}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(vis_path, f'confusion_matrix.png'))
            plt.close()

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

    def _return_clean_text_output(self, results: Dict) -> str:
        """
        Return clean text output from the results.

        Args:
            results: Dictionary containing the results.

        Returns:
            Clean text output.
        """
        return next(iter(results.values())).strip()

    def _visualize_results(self, image: Image.Image, results: Dict, save_viz_dir: str = 'visualizations'):
        """
        Visualize the results.

        Args:
            image: The input image.
            results: Dictionary containing the results.
            save_viz_dir: Directory to save the visualizations.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        ax = plt.gca()

        for bbox, label in zip(results['bboxes'], results['labels']):
            xmin, ymin, xmax, ymax = bbox
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor='red',
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(
                xmin,
                ymin - 2,
                label,
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=12,
                color='white'
            )

        plt.axis('off')
        plt.show()
        os.makedirs(save_viz_dir, exist_ok=True)
        plt.savefig(os.path.join(self.model_run_path, save_viz_dir, 'result.png'))

    def inference(self, image_path: str, task: str = "<OD>", text_input: str = None):
        """
        Perform inference on an image.

        Args:
            image_path: Path to the input image.
            task: The task to perform (e.g., object detection).
            text_input: Optional text input for the task.

        Returns:
            The result of the inference.
        """
        image = Image.open(image_path)
        prompt = task + (text_input if text_input else "")

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        self.model.eval()

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )

        text_output = parsed_answer[task]

        if task == "<OD>":
            self._visualize_results(image, text_output)

        if task == "CAPTION_TO_PHASE_GROUNDING" or task == "<OPEN_VOCABULARY_DETECTION>":
            if text_input is not None:
                if task == "CAPTION_TO_PHASE_GROUNDING":
                    self._visualize_results(image, text_output)
                else:
                    boxes = text_output.get('bboxes', [])
                    labels = text_output.get('bboxes_labels', [])
                    results = {
                        'bboxes': boxes,
                        'labels': labels
                    }
                    self._visualize_results(image, results)
            else:
                raise ValueError("Text input is needed for this type of task")

        if task == "<CAPTION>" or task == "<DETAILED_CAPTION>" or task == "<MORE_DETAILED_CAPTION>":
            print(self._return_clean_text_output(text_output))

        return text_output

    def finetune(self, dataset: str, classes: Dict[int, str],
                 train_test_split: Tuple[int, int, int] = (80, 10, 10),
                 epochs: int = 20, lr: float = 4e-6, save_dir: str = "model_checkpoints",
                 num_workers: int = 4, lora_r: int = 8, lora_scaling: int = 8, patience: int = 5,
                 lora_dropout: float = 0.05, warmup_epochs: int = 1, lr_schedule: str = "cosine", create_peft_config: bool = True):
        """
        Fine-tune the model on the given dataset.

        Args:
            dataset: Path to the dataset.
            classes: Dictionary mapping class IDs to class names.
            train_test_split: Tuple specifying the train, test, and validation split ratios.
            epochs: Number of epochs to train.
            lr: Learning rate.
            save_dir: Directory to save the model checkpoints.
            num_workers: Number of worker processes for data loading.
            lora_r: Rank for LoRA.
            lora_scaling: Scaling factor for LoRA.
            patience: Number of epochs to wait for improvement before early stopping.
            lora_dropout: Dropout rate for LoRA.
            warmup_epochs: Number of warmup epochs.
            lr_schedule: Learning rate schedule.
            create_peft_config: Whether to create a new PEFT configuration.

        Returns:
            Dictionary containing training metrics.
        """
        self.logger.info(f"Finetuning {self.model_id} on {dataset} for {epochs} epochs...")

        save_dir = os.path.join(self.model_run_path, save_dir)
        vis_path = os.path.join(
            self.model_run_path,
            'training_visualizations',
            self.model_id.replace('/', '_'),
            'finetuning_info',
            os.path.basename(dataset.rstrip('/'))
        )
        os.makedirs(vis_path, exist_ok=True)

        metrics = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'best_val_loss': float('inf'),
            'best_epoch': -1
        }

        try:

            train_path, valid_path = self._prepare_dataset(
                base_path=dataset,
                dict_classes=classes,
                train_test_split=train_test_split
            )

            self.logger.info(f"Dataset Preparation Complete - Train Path: {train_path}, Valid Path: {valid_path}")

            self._setup_data_loaders(train_path, valid_path, num_workers=num_workers)

            self.logger.info(f"Data Loader Setup Complete")

            trainable_params_info = self._setup_peft(r=lora_r, alpha=lora_scaling, dropout=lora_dropout, create_peft_config=create_peft_config)

            self.logger.info(trainable_params_info)

            self.logger.info(f"PEFT Setup Complete w/ Specified Params: \n"
                        f"LoRA r: {lora_r}, Scaling: {lora_scaling}, Dropout: {lora_dropout}")

            config = {
                'model_id': self.model_id,
                'dataset': dataset,
                'classes': classes,
                'epochs': epochs,
                'learning_rate': lr,
                'batch_size': self.batch_size,
                'lora_config': {
                    'r': lora_r,
                    'scaling': lora_scaling,
                    'dropout': lora_dropout
                }
            }
            with open(os.path.join(vis_path, 'training_config.json'), 'w') as f:
                json.dump(config, f, indent=4)

            optimizer = AdamW(self.peft_model.parameters(), lr=lr, weight_decay=0.01)
            num_steps = epochs * len(self.train_loader)
            warmup_steps = warmup_epochs * len(self.train_loader)
            lr_scheduler = get_scheduler(
                lr_schedule,
                optimizer=optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_steps
            )

            patience_counter = 0

            self.logger.info(f"Beginning Training Loop w/ Specified Params: \n"
                        f"Optimizer: AdamW, Learning Rate: {lr}, Scheduler: {lr_schedule}, "
                        f"Warmup Epochs: {warmup_epochs}, Number of Steps: {num_steps}, "
                        f"Patience: {patience}")

            for epoch in range(epochs):
                self.peft_model.train()
                train_loss = 0

                for batch_idx, (inputs, answers) in enumerate(tqdm(self.train_loader,
                                                                desc=f"Training Epoch {epoch + 1}/{epochs}")):
                    try:
                        input_ids = inputs["input_ids"]
                        pixel_values = inputs["pixel_values"]
                        labels = self.processor.tokenizer(
                            text=answers,
                            return_tensors="pt",
                            padding=True,
                            return_token_type_ids=False
                        ).input_ids.to(self.device)

                        outputs = self.peft_model(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            labels=labels
                        )

                        loss = outputs.loss
                        loss.backward()

                        nn.utils.clip_grad_norm_(self.peft_model.parameters(), max_norm=1.0)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()

                        current_lr = lr_scheduler.get_last_lr()[0]
                        metrics['train_losses'].append(loss.item())
                        metrics['learning_rates'].append(current_lr)
                        train_loss += loss.item()

                        if self.device == "cuda" and batch_idx % 10 == 0:
                            torch.cuda.empty_cache()

                    except RuntimeError as e:
                        self.logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                        continue

                avg_train_loss = train_loss / len(self.train_loader)

                self.peft_model.eval()
                val_loss = 0

                with torch.no_grad():
                    for inputs, answers in tqdm(self.val_loader,
                                            desc=f"Validation Epoch {epoch + 1}/{epochs}"):
                        try:
                            input_ids = inputs["input_ids"]
                            pixel_values = inputs["pixel_values"]
                            labels = self.processor.tokenizer(
                                text=answers,
                                return_tensors="pt",
                                padding=True,
                                return_token_type_ids=False
                            ).input_ids.to(self.device)

                            outputs = self.peft_model(
                                input_ids=input_ids,
                                pixel_values=pixel_values,
                                labels=labels
                            )
                            val_loss += outputs.loss.item()

                        except RuntimeError as e:
                            self.logger.error(f"Error in validation batch: {str(e)}")
                            continue

                avg_val_loss = val_loss / len(self.val_loader)
                metrics['val_losses'].append(avg_val_loss)

                self._save_training_plots(vis_path, metrics, epoch)

                if avg_val_loss < metrics['best_val_loss']:
                    metrics['best_val_loss'] = avg_val_loss
                    metrics['best_epoch'] = epoch
                    patience_counter = 0

                    best_model_path = os.path.join(save_dir, "best_model")
                    os.makedirs(best_model_path, exist_ok=True)
                    self.peft_model.save_pretrained(best_model_path)
                    self.processor.save_pretrained(best_model_path)
                else:
                    patience_counter += 1

                if (epoch + 1) % 5 == 0:
                    save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}")
                    os.makedirs(save_path, exist_ok=True)

                    self.peft_model.save_pretrained(save_path)
                    self.processor.save_pretrained(save_path)

                    torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': lr_scheduler.state_dict(),
                        'metrics': metrics
                    }, os.path.join(save_path, "training_state.pt"))

                self.logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train Loss: {avg_train_loss:.4f} - "
                    f"Val Loss: {avg_val_loss:.4f} - "
                    f"LR: {current_lr:.2e}"
                )

                if patience_counter >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 5 == 0:
                    random_idx = random.randint(0, len(self.val_dataset) - 1)
                    prefix, suffix, image = self.val_dataset[random_idx]
                    _ = self.inference(image, task="<OD>", visualize=True)

            with open(os.path.join(vis_path, 'final_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)

            return metrics

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            emergency_path = os.path.join(save_dir, "emergency_checkpoint")
            os.makedirs(emergency_path, exist_ok=True)
            self.peft_model.save_pretrained(emergency_path)
            raise e

    def evaluate(self, test_dataset: Dataset, classes: Dict[int, str], device: str = 'cpu'):
        """
        Evaluate the model on a test dataset.

        Args:
            test_dataset: The test dataset.
            classes: Dictionary mapping class IDs to class names.
            device: The device to use for evaluation.

        Returns:
            The mean average precision result.
        """
        self.logger.info("Starting evaluation...")
        images = []
        targets = []
        predictions = []

        with torch.inference_mode():
            for i in tqdm(range(len(test_dataset))):
                image, label = test_dataset[i]
                prefix = "<image>" + label["prefix"]
                suffix = label["suffix"]

                inputs = self.processor(
                    text=prefix,
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

                prefix_length = inputs["input_ids"].shape[-1]

                generation = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
                generation = generation[0][prefix_length:]
                generated_text = self.processor.decode(generation, skip_special_tokens=True)

                w, h = image.size
                prediction = sv.Detections.from_lmm(
                    lmm='paligemma',
                    result=generated_text,
                    resolution_wh=(w, h),
                    classes=classes)

                prediction.class_id = np.array([classes.index(class_name) for class_name in prediction['class_name']])
                prediction.confidence = np.ones(len(prediction))

                target = sv.Detections.from_lmm(
                    lmm='paligemma',
                    result=suffix,
                    resolution_wh=(w, h),
                    classes=classes)

                target.class_id = np.array([classes.index(class_name) for class_name in target['class_name']])

                images.append(image)
                targets.append(target)
                predictions.append(prediction)

        map_metric = MeanAveragePrecision(metric_target=MetricTarget.BOXES)
        map_result = map_metric.update(predictions, targets).compute()

        print(map_result)

        map_result.plot()

        confusion_matrix = sv.ConfusionMatrix.from_detections(
            predictions=predictions,
            targets=targets,
            classes=classes
        )

        _ = confusion_matrix.plot()

        annotated_images = []
        for i in range(25):
            image = images[i]
            detections = predictions[i]

            annotated_image = image.copy()
            annotated_image = sv.BoxAnnotator(thickness=4).annotate(annotated_image, detections)
            annotated_image = sv.LabelAnnotator(text_scale=2, text_thickness=4, smart_position=True).annotate(annotated_image, detections)
            annotated_images.append(annotated_image)

        sv.plot_images_grid(annotated_images, (5, 5))

        self.logger.info("Evaluation complete.")

        return map_result

    @staticmethod
    def convert_yolo_to_jsonl(yolo_dataset_path: str, output_jsonl_path: str, classes: Dict[int, str]):
        """
        Convert a YOLO dataset to JSONL format.

        Args:
            yolo_dataset_path: Path to the YOLO dataset.
            output_jsonl_path: Path to save the output JSONL file.
            classes: Dictionary mapping class IDs to class names.
        """
        annotations = []

        image_files = [f for f in os.listdir(yolo_dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        for image_file in image_files:
            image_path = os.path.join(yolo_dataset_path, image_file)
            label_file = os.path.join(yolo_dataset_path, os.path.splitext(image_file)[0] + '.txt')

            if not os.path.exists(label_file):
                continue

            with open(label_file, 'r') as f:
                lines = f.readlines()

            suffix_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])

                x1 = int((x_center - width / 2) * 1000)
                y1 = int((y_center - height / 2) * 1000)
                x2 = int((x_center + width / 2) * 1000)
                y2 = int((y_center + height / 2) * 1000)

                class_name = classes.get(class_id, f"Unknown Class {class_id}")
                suffix_line = f"{class_name}<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>"
                suffix_lines.append(suffix_line)

            annotations.append({
                "image": image_file,
                "prefix": "<OD>",
                "suffix": "".join(suffix_lines)
            })

        with open(output_jsonl_path, 'w') as f:
            for annotation in annotations:
                f.write(json.dumps(annotation) + '\n')

        print(f"Converted {len(annotations)} annotations to {output_jsonl_path}")

