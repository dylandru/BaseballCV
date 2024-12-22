import logging
import random
import string
from typing import Dict
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import BitsAndBytesConfig
from .yolo_to_jsonl import JSONLDetection


class ModelFunctionUtils:
    def __init__(self, model_name: str, model_run_path: str, batch_size: int, 
                 device: torch.device, processor: None, model: None, 
                 peft_model: None, logger: logging.Logger, torch_dtype: torch.dtype):
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
            yolo_to_jsonl: JSONLDetection to use for creating the dataset.
        """
        self.model_name = model_name
        self.model_run_path = model_run_path
        self.batch_size = batch_size
        self.device = device
        self.processor = processor
        self.model = model
        self.peft_model = peft_model
        self.logger = logger
        self.torch_dtype = torch_dtype

    def augment_suffix(self, suffix: str) -> str:
        """
        Augment the suffix with a random string.
        """
        return suffix + "_" + "".join(random.choices(string.ascii_letters + string.digits, k=4))


    def collate_fn(self, batch):
        images, labels = zip(*batch)

        paths = [label["image"] for label in labels]
        prefixes = ["<image>" + label["prefix"] for label in labels]
        suffixes = [self.augment_suffix(label["suffix"]) for label in labels]

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
                    value = value.to(self.torch_dtype)
                    processed_inputs[key] = value.requires_grad_(True)
                else:
                    processed_inputs[key] = value.long().requires_grad_(False)
            else:
                processed_inputs[key] = value

        return processed_inputs
    
    def setup_data_loaders(self, train_image_path: str, valid_image_path: str, train_jsonl_path: str, valid_jsonl_path: str, num_workers: int):
        """
        Set up data loaders for training and validation.

        Args:
            train_image_path: Path to the training images.
            valid_image_path: Path to the validation images.
            train_jsonl_path: Path to the training JSONL file.
            valid_jsonl_path: Path to the validation JSONL file.
            num_workers: Number of worker processes for data loading.
        """

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

        loader_kwargs = {
            "batch_size": self.batch_size,
            "collate_fn": self.collate_fn,
            "pin_memory": True if self.device == "cuda" else False,
            "num_workers": num_workers,
            "persistent_workers": False
        }

        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            **loader_kwargs
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            shuffle=False,
            **loader_kwargs
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

        self.peft_model = get_peft_model(self.model, config)
    
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        
        self.logger.info(
            f"PEFT Model Configuration Complete:\n"
            f"- Total Parameters: {total_params:,}\n"
            f"- Trainable Parameters: {trainable_params:,}\n"
            f"- Trainable Percentage: {100 * trainable_params / total_params:.2f}%\n")
            
        return self.peft_model
    
    def freeze_vision_encoders(self, model):
        """
        Freeze the vision encoders of the model.
        """
        for param in model.vision_tower.parameters():
            param.requires_grad = False

        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False

        self.logger.info("Vision encoders frozen.")

        return model
    
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
        entries = JSONLDetection.load_jsonl_entries(jsonl_file_path=jsonl_file_path, logger=self.logger)

        return JSONLDetection(entries=entries, 
                                       image_directory_path=image_directory_path, 
                                       logger=self.logger, 
                                       augment=augment)
    
    def return_clean_text_output(self, results: Dict) -> str:
        """
        Return clean text output from the results.

        Args:
            results: Dictionary containing the results.

        Returns:
            Clean text output.
        """
        return next(iter(results.values())).strip()

    def save_checkpoint(self, path, epoch, model, optimizer, scheduler, loss, scaler=None) -> logging.Logger:
        """Save a model checkpoint with all training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'scaler_state_dict': scaler.state_dict() if scaler else None
        }
        torch.save(checkpoint, path)
        self.processor.save_pretrained(os.path.dirname(path))

        return self.logger.info(f"Checkpoint saved to {path}")
    
    
    @staticmethod
    def setup_quantization(load_in_4bit: bool = True, bnb_4bit_quant_type: str = "nf4"):
        """
        Set up a static method for a quantization config. 
        """
        quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_quant_storage=torch.bfloat16
        )
        return quant_config


    
    
    
