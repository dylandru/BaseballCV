import torch
import logging
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model
from .yolo_to_jsonl import YOLOToJSONLDetection
from typing import Dict

class ModelFunctionUtils:
    def __init__(self, model_name: str, model_run_path: str, batch_size: int, 
                 device: torch.device, processor: None, model: None, 
                 peft_model: None, logger: logging.Logger, yolo_to_jsonl: YOLOToJSONLDetection):
        """
        Initialize the ModelFunctionUtils class.

        Args:
            model_name: Name of the model.
            model_run_path: Path to the model run directory.
            batch_size: Batch size for training and validation.
            device: Device to use for training and validation.
            processor: Processor to use for training and validation.
            model: Model to use for training and validation.
            peft_model: PEFT model to use for training and validation.
            logger: Logger to use for logging.
            yolo_to_jsonl: YOLOToJSONLDetection to use for creating the dataset.
        """
        self.model_name = model_name
        self.model_run_path = model_run_path
        self.batch_size = batch_size
        self.device = device
        self.processor = processor
        self.model = model
        self.peft_model = peft_model
        self.logger = logger
        self.YOLOToJSONLDetection = yolo_to_jsonl

    def collate_fn(self, batch):
        prefixes, suffixes, images = zip(*batch)
        inputs = self.processor(
            text=list(prefixes), 
            images=list(images), 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        return inputs, suffixes
    
    def setup_data_loaders(self, train_path: str, valid_path: str, num_workers: int = 0):
        """
        Set up data loaders for training and validation.

        Args:
            train_path: Path to the training dataset.
            valid_path: Path to the validation dataset.
            num_workers: Number of worker processes for data loading.
        """
        self.train_dataset = self.YOLOToJSONLDetection.create_detection_dataset(
            jsonl_file_path=f"{train_path}train_annotations.json",
            image_directory_path=train_path,
            augment=True
        )
        self.val_dataset = self.YOLOToJSONLDetection.create_detection_dataset(
            jsonl_file_path=f"{valid_path}valid_annotations.json",
            image_directory_path=valid_path,
            augment=False
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            shuffle=True,
            persistent_workers=False if num_workers == 0 else True,
            pin_memory=True if self.device == 'cuda' else False
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            persistent_workers=False if num_workers == 0 else True,
            pin_memory=True if self.device == 'cuda' else False
        )
        return self.train_loader, self.val_loader

    def setup_peft(self, r: int = 8, alpha: int = 8, dropout: float = 0.05, create_peft_config: bool = True):
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
        return trainable_params_info, self.peft_model
    
    def create_detection_dataset(self, jsonl_file_path: str, image_directory_path: str,
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

        return self.YOLOToJSONLDetection(self, entries, image_directory_path, augment)
    
    def return_clean_text_output(self, results: Dict) -> str:
        """
        Return clean text output from the results.

        Args:
            results: Dictionary containing the results.

        Returns:
            Clean text output.
        """
        return next(iter(results.values())).strip()
    
    
    