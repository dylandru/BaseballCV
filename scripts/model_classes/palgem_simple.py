import torch
from torch.utils.data import Dataset
from transformers import (PaliGemmaProcessor, PaliGemmaForConditionalGeneration, 
                          Trainer, TrainingArguments, EarlyStoppingCallback)
import torch.backends
from tqdm import tqdm
import os
from PIL import Image, ImageEnhance
from typing import Dict, Tuple, List, Any
import numpy as np
from datetime import datetime
import supervision as sv
from supervision.metrics import MeanAveragePrecision, MetricTarget
import logging
import shutil
import random
import string
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import json

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
        self.transforms = self.get_augmentation_transforms() if augment else []
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
    
    def get_augmentation_transforms(self):
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

class PaliGemma2:
    def __init__(self, 
                 model_id: str = 'google/paligemma2-3b-pt-224', 
                 model_run_path: str = f'paligemma2_run_{datetime.now().strftime("%Y%m%d")}', 
                 batch_size: int = 8, 
                 torch_dtype: torch.dtype = torch.float32):
        """
        Initialize the PaliGemma2 model.

        Args:
            model_id: The identifier of the model to use.
            model_run_path: The path to save model run information.
            batch_size: The batch size for training and inference.
            torch_dtype: The torch data type to use.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                else "mps" if torch.backends.mps.is_available()
                                else "cpu")
        self.model_id = model_id
        self.batch_size = batch_size
        self.model_run_path = model_run_path
        self.model_name = "PaliGemma2"
        self.torch_dtype = torch_dtype
        self.augment = True

        # Setup logging
        self.logger = self._setup_logger()

        # Initialize model components
        self.quantization_config = self._setup_quantization() if self.device == "cuda" else None
        self.model = None
        self.processor = None
        self.peft_model = None
        self._init_model()

    def _setup_logger(self):
        """Setup logging configuration."""
        logger = logging.getLogger(self.model_name)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        os.makedirs(self.model_run_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.model_run_path, f"{self.model_name}.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger

    def _setup_quantization(self, load_in_4bit: bool = True, bnb_4bit_quant_type: str = "nf4"):
        """Setup quantization configuration for CUDA devices."""
        return BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_quant_storage=torch.bfloat16
        )

    def _init_model(self):
        """Initialize the model and processor."""
        if self.device == "cuda":
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_id, device_map="auto", quantization_config=self.quantization_config)
        else:
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_id)
        
        self.processor = PaliGemmaProcessor.from_pretrained(self.model_id)

    def _augment_suffix(self, suffix: str) -> str:
        """Augment the suffix with a random string."""
        return suffix + "_" + "".join(random.choices(string.ascii_letters + string.digits, k=4))

    def collate_fn(self, batch):
        """Collate function for data loading."""
        images, labels = zip(*batch)
        prefixes = ["<image>" + label["prefix"] for label in labels]
        suffixes = [self._augment_suffix(label["suffix"]) for label in labels]

        inputs = self.processor(
            text=prefixes,
            images=images,
            return_tensors="pt",
            suffix=suffixes,
            padding="longest"
        )

        processed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if key == "pixel_values":
                    processed_inputs[key] = value.to(self.torch_dtype).requires_grad_(True)
                else:
                    processed_inputs[key] = value.long().requires_grad_(False)
            else:
                processed_inputs[key] = value

        processed_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in processed_inputs.items()}

        return processed_inputs

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

        train_file_path = self._convert_annotations(base_path, "train", dict_classes)
        test_file_path = self._convert_annotations(base_path, "test", dict_classes)
        valid_file_path = self._convert_annotations(base_path, "valid", dict_classes)

        return os.path.join(base_path, "train", "images/"), os.path.join(base_path, "valid", "images/"), train_file_path, test_file_path, valid_file_path

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
                
        return output_file

    def create_detection_dataset(self, jsonl_file_path: str, image_directory_path: str,
                               augment: bool = True) -> Dataset:
        """Create a detection dataset from a JSONL file."""
        entries = JSONLDetection.load_jsonl_entries(jsonl_file_path=jsonl_file_path, logger=self.logger)
        return JSONLDetection(entries=entries, 
                            image_directory_path=image_directory_path, 
                            logger=self.logger, 
                            augment=augment)

    def setup_peft(self, r: int = 8, alpha: int = 8, dropout: float = 0.05,
                  create_peft_config: bool = True):
        """
        Set up Parameter-Efficient Fine-Tuning (PEFT).

        Args:
            r: Rank for LoRA.
            alpha: Scaling factor for LoRA.
            dropout: Dropout rate for LoRA.
            create_peft_config: Whether to create a new PEFT configuration.
        """
        if create_peft_config:
            if hasattr(self.model, 'peft_config'):
                self.logger.info("Existing PEFT configuration found. Removing old configuration...")
                peft_core_attrs = ['peft_config', 'base_model_prepare_inputs']
                for attr in peft_core_attrs:
                    if hasattr(self.model, attr):
                        delattr(self.model, attr)

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

        self.model.train()
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
            self.logger.info("Input gradients enabled for base model")
        
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled for base model")

        self.peft_model = get_peft_model(self.model, config)
    
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        
        self.logger.info(
            f"PEFT Model Configuration Complete:\n"
            f"- Total Parameters: {total_params:,}\n"
            f"- Trainable Parameters: {trainable_params:,}\n"
            f"- Trainable Percentage: {100 * trainable_params / total_params:.2f}%\n")
            
        return self.peft_model

    def freeze_vision_encoders(self):
        """Freeze the vision encoders of the model."""
        for param in self.model.vision_tower.parameters():
            param.requires_grad = False
        for param in self.model.multi_modal_projector.parameters():
            param.requires_grad = False
        self.logger.info("Vision encoders frozen.")

    def finetune(self, dataset: str, classes: Dict[int, str],
                 train_test_split: Tuple[int, int, int] = (80, 10, 10), 
                 freeze_vision_encoders: bool = False,
                 epochs: int = 20, lr: float = 4e-6, save_dir: str = "model_checkpoints",
                 num_workers: int = 0, lora_r: int = 8, lora_scaling: int = 8, 
                 patience: int = 10, patience_threshold: float = 0.0, 
                 gradient_accumulation_steps: int = 16, lora_dropout: float = 0.05, 
                 warmup_ratio: float = 0.03, lr_schedule_type: str = "cosine", 
                 create_peft_config: bool = True, random_seed: int = 22, 
                 optimizer: str = "adamw_hf", weight_decay: float = 0.01, 
                 logging_steps: int = 100, save_eval_steps: int = 1000, 
                 save_limit: int = 3, metric_for_best_model: str = "loss"):
        """
        Fine-tune the model on the given dataset.
        """
        self.logger.info(f"Finetuning {self.model_id} on {dataset}")
        save_dir = os.path.join(self.model_run_path, "model_checkpoints")

        if freeze_vision_encoders:
            self.freeze_vision_encoders()

        training_args = TrainingArguments(
                seed=random_seed,
                output_dir=save_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_ratio=warmup_ratio,
                learning_rate=lr,
                weight_decay=weight_decay,
                logging_steps=logging_steps,
                eval_strategy="steps",
                save_strategy="steps", 
                save_steps=save_eval_steps,
                eval_steps=save_eval_steps,
                save_total_limit=save_limit,
                load_best_model_at_end=True,
                metric_for_best_model=metric_for_best_model,
                lr_scheduler_type=lr_schedule_type,
                report_to=["tensorboard", "wandb"],
                dataloader_pin_memory=False,
                dataloader_num_workers=num_workers,
                dataloader_persistent_workers=True if num_workers > 0 else False,
                gradient_checkpointing=False, 
                remove_unused_columns=True
            )

        try:
            train_image_path, valid_image_path, train_jsonl_path, test_jsonl_path, valid_jsonl_path = self.prepare_dataset(
                base_path=dataset,
                dict_classes=classes,
                train_test_split=train_test_split
            )
            self.logger.info(f"Dataset Preparation Complete - Train Path: {train_image_path}, Valid Path: {valid_image_path}")

            self.train_dataset = self.create_detection_dataset(
                jsonl_file_path=train_jsonl_path,
                image_directory_path=train_image_path,
                augment=True
            )
            self.val_dataset = self.create_detection_dataset(
                jsonl_file_path=valid_jsonl_path,
                image_directory_path=valid_image_path,
                augment=False
            )
            self.logger.info("Data Loader Setup Complete")

            self.peft_model = self.setup_peft(lora_r, lora_scaling, lora_dropout, create_peft_config)

            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                data_collator=self.collate_fn,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=patience, 
                                              early_stopping_threshold=patience_threshold)]
            )

            train_result = trainer.train()
            trainer.save_model()
            
            return train_result

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise e

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

        return parsed_answer[task]

    def evaluate(self, test_dataset: Dataset, classes: Dict[int, str]):
        """
        Evaluate the model on a test dataset.

        Args:
            test_dataset: The test dataset.
            classes: Dictionary mapping class IDs to class names.

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
        for i in range(min(25, len(images))):
            image = images[i]
            detections = predictions[i]

            annotated_image = image.copy()
            annotated_image = sv.BoxAnnotator(thickness=4).annotate(annotated_image, detections)
            annotated_image = sv.LabelAnnotator(text_scale=2, text_thickness=4, smart_position=True).annotate(annotated_image, detections)
            annotated_images.append(annotated_image)

        sv.plot_images_grid(annotated_images, (5, 5))

        self.logger.info("Evaluation complete.")

        return map_result