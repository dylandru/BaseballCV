import logging
import random
import os
import shutil
import subprocess
import string
from typing import Dict, Tuple, Any, List
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import BitsAndBytesConfig
from baseballcv.datasets import JSONLDetection
from pkg_resources import resource_filename

class ModelFunctionUtils:
    def __init__(self, model_name: str, model_run_path: str, batch_size: int, 
                 device: torch.device, processor: Any, model: Any, 
                 peft_model: Any, logger: logging.Logger, torch_dtype: torch.dtype) -> None:
        """
        Initialize the ModelFunctionUtils class.

        Args:
            model_name (str): Name of the model.
            model_run_path (str): Path to the model run directory.
            batch_size (int): Batch size for training and validation.
            device (torch.device): Device to use for training and validation.
            processor (Any): Processor to use for training and validation.
            model (Any): Model to use for training and validation.
            peft_model (Any): PEFT model to use for training and validation.
            logger (logging.Logger): Logger to use for logging.
            torch_dtype (torch.dtype): The torch dtype to use.
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

        Args:
            suffix (str): The suffix to augment.

        Returns:
            augmented_suffix (str): The augmented suffix.
        """
        return suffix + "_" + "".join(random.choices(string.ascii_letters + string.digits, k=4))


    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """
        Collate function for dataset.

        Args:
            batch: Batch to collate.

        Returns:
            Dict[str, torch.Tensor]: The collated batch.
        """
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
    
    def setup_data_loaders(self, train_image_path: str, valid_image_path: str, train_jsonl_path: str, 
                           valid_jsonl_path: str, num_workers: int) -> Tuple[DataLoader, DataLoader]:
        """
        Set up data loaders for training and validation.

        Args:
            train_image_path (str): Path to the training images.
            valid_image_path (str): Path to the validation images.
            train_jsonl_path (str): Path to the training JSONL file.
            valid_jsonl_path (str): Path to the validation JSONL file.
            num_workers (int): Number of worker processes for data loading.

        Returns:
            train_loader (DataLoader): The training data loader.
            val_loader (DataLoader): The validation data loader.
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


    def setup_peft(self, r: int = 8, alpha: int = 8, dropout: float = 0.05, create_peft_config: bool = True) -> LoraConfig:
        """
        Set up Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

        Args:
            r (int): Rank for LoRA.
            alpha (int): Scaling factor for LoRA.
            dropout (float): Dropout rate for LoRA.
            create_peft_config (bool): Whether to create a new PEFT configuration.

        Returns:
            self.peft_model (LoraConfig): The PEFT model.
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
    
    def freeze_vision_encoders(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Freeze the vision encoders of the model.

        Args:
            model (torch.nn.Module): The model to freeze.

        Returns:
            model (torch.nn.Module): The frozen model.
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
            jsonl_file_path (str): Path to the JSONL file.
            image_directory_path (str): Path to the directory containing images.
            augment (bool): Whether to apply data augmentation.

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

    def save_checkpoint(self, path: str, epoch: int, optimizer: torch.optim.Optimizer, 
                        scheduler: torch.optim.lr_scheduler._LRScheduler, loss: float, 
                        scaler: torch.cuda.amp.GradScaler = None) -> logging.Logger:
        """
        Save checkpoint with all HF required files.

        Args:
            path (str): Path to the checkpoint.
            epoch (int): The epoch number.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
            loss (float): The loss.
            scaler (torch.cuda.amp.GradScaler): The scaler.
        """

        checkpoint_dir = os.path.dirname(path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        if hasattr(self, 'peft_model'):
            self.peft_model.save_pretrained(checkpoint_dir)
            
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': loss,
                'scaler': scaler.state_dict() if scaler else None
            }, os.path.join(checkpoint_dir, 'training_state.pt'))
            
            self.processor.save_pretrained(checkpoint_dir)
            
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.safetensors') and '-of-' in file:
                    parts = file.split('-of-')
                    if len(parts) == 2:
                        correct = f"{parts[0]}-of-{parts[1][:5]}.safetensors"
                        os.rename(os.path.join(checkpoint_dir, file), 
                                os.path.join(checkpoint_dir, correct))
        else:
            raise ValueError("PEFT model not found. Cannot save checkpoint.")
        
    def freeze_layers(self, layers_to_freeze: List[str] = None, show_params: bool = True) -> None:
        """
        Freeze specified layers in PyTorch model
        
        Args:
            layers_to_freeze (List[str]): List of layer names or patterns to freeze.
                                        If None, shows all available layers.
            show_params (bool): Whether to print the trainable status of all parameters.
        
        """
            
        for name, param in self.model.named_parameters():
            if any(pattern in name for pattern in layers_to_freeze):
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        if show_params:
            for name, param in self.model.named_parameters():
                self.logger.info(f"{name} -  Frozen: {not param.requires_grad}")
            frozen_count = sum(1 for p in self.model.parameters() if not p.requires_grad)
            total_count = sum(1 for _ in self.model.parameters())
            self.logger.info(f"\nFrozen parameters: {frozen_count}/{total_count}")

    
    @staticmethod
    def setup_quantization(load_in_4bit: bool = True, bnb_4bit_quant_type: str = "nf4") -> BitsAndBytesConfig:
        """
        Static method for a quantization config. 

        Args:
            load_in_4bit (bool): Whether to load in 4-bit.
            bnb_4bit_quant_type (str): The type of quantization.

        Returns:
            quant_config (BitsAndBytesConfig): The quantization config.
        """
        quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_quant_storage=torch.bfloat16
        )
        return quant_config
    
    @staticmethod
    def setup_yolo_weights(model_file: str, output_dir: str = None) -> str:
        """
        Retrieve YOLO model weights files.

        Args:
            model_file (str): The model file to retrieve.
            output_dir (str): The output directory. If None, uses current directory.

        Returns:
            str: The path to the model weights file.
        """
        if not output_dir:
            output_dir = os.getcwd()
        
        os.makedirs(os.path.join(output_dir, "weights"), exist_ok=True)
        output = os.path.join(output_dir, "weights", os.path.basename(model_file))
        if not os.path.exists(output):
            try:
                script = resource_filename("yolov9", "scripts/get_model_weights.sh")
                subprocess.run(["bash", script, model_file, output_dir], check=True)
            except Exception as e:
                print(f"Error downloading weights: {e}")
                print("Please download weights manually from: https://github.com/WongKinYiu/yolov9/releases")
        else:
            print(f"Model weights file found at {output}.")

        return output
    
    @staticmethod
    def setup_rfdetr_dataset(data_path: str) -> str:
        """
        Setup the expected COCO format for the RF DETR model.

        Args:
            data_path (str): The path to the dataset.
        """
        #Check 1 - Check if Val or Valid Folder / Fileand Change to Val
        if os.path.exists(os.path.join(data_path, "val")):
            os.rename(os.path.join(data_path, "val"), os.path.join(data_path, "valid"))

        if os.path.exists(os.path.join(data_path, "COCO_annotations", "instances_val.json")):
            os.rename(os.path.join(data_path, "COCO_annotations", "instances_val.json"), os.path.join(data_path, "COCO_annotations", "instances_valid.json"))


        #Check 2 - Move the COCO annotations and images to the correct split directories
        if os.path.exists(os.path.join(data_path, "COCO_annotations")):
            coco_annotations_dir = os.path.join(data_path, "COCO_annotations") 

            for split in ["train", "valid", "test"]:
                split_dir = os.path.join(data_path, split)
                images_dir = os.path.join(split_dir, "images")
                
                # Move annotation file if it exists
                src_file = os.path.join(coco_annotations_dir, f"instances_{split}.json")
                if os.path.exists(src_file):
                    shutil.move(src_file, os.path.join(split_dir, "_annotations.coco.json"))
                
                # Move all image files to the split directory IF EXISTS
                if os.path.exists(images_dir):
                    for img in [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]:
                        shutil.move(os.path.join(images_dir, img), os.path.join(split_dir, img))

        return "Dataset organized successfully!"            



    
    
    
