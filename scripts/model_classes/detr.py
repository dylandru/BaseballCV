import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, Trainer, TrainingArguments, EarlyStoppingCallback
import os
import warnings
from PIL import Image
from datetime import datetime
from typing import Dict, List, Tuple
from .utils import CocoDetectionDataset
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from .utils import ModelLogger, ModelFunctionUtils, CocoDetectionDataset, CustomProgressBarCallback

class DETR:
    def __init__(self, 
                 num_labels: int,
                 device: str = None, 
                 model_id: str = "facebook/detr-resnet-50",
                 model_run_path: str = f'detr_run_{datetime.now().strftime("%Y%m%d")}', 
                 batch_size: int = 8,
                 image_size: Tuple[int, int] = (800, 1333)):
        
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                else "mps" if torch.backends.mps.is_available()
                                else "cpu") if device is None else torch.device(device)
        
        self.model_id = model_id
        self.model_run_path = model_run_path
        self.model_name = "DETR"
        self.batch_size = batch_size

        self.logger = ModelLogger(self.model_name, self.model_run_path, 
                                self.model_id, self.batch_size, self.device).orig_logging()
        self.processor = DetrImageProcessor.from_pretrained(self.model_id,
                            size={"shortest_edge": image_size[0], "longest_edge": image_size[1]},
                            do_resize=True,
                            do_pad=True,
                            do_normalize=True)
        self.model = DetrForObjectDetection.from_pretrained(
            self.model_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)

        self.ModelFunctionUtils = ModelFunctionUtils(self.model_name, 
                            self.model_run_path, self.batch_size, 
                            self.device, self.processor, self.model, 
                            None, self.logger, torch.float16)

    
    def finetune(self, dataset_dir: str,
            save_dir: str = "finetuned_detr",
            batch_size: int = 4,
            epochs: int = 10,
            lr: float = 4e-6,
            weight_decay: float = 0.01,
            warmup_ratio: float = 0.1,
            lr_schedule_type: str = "cosine",
            gradient_accumulation_steps: int = 2,
            logging_steps: int = 10,
            save_limit: int = 2,
            patience: int = 3,
            patience_threshold: float = 0.01,
            metric_for_best_model: str = "loss",
            resume_from_checkpoint: str = None,
            num_workers: int = None,
            freeze_layers: List[str] = None,
            custom_freezing: bool = False,
            show_model_params: bool = True) -> Dict:

        try:

            self.logger.info("Setting Up Tensorboard")
            tensorboard_dir = os.path.join(self.model_run_path, save_dir, "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            writer = SummaryWriter(tensorboard_dir)

            model_path = os.path.join(self.model_run_path, save_dir, "model")
            os.makedirs(model_path, exist_ok=True)



            if not freeze_layers and not custom_freezing:
                self.logger.info("Defaulting to Quick Training - Head Layer remains trainable...")
                self.ModelFunctionUtils.freeze_layers([
                'model.backbone',
                'model.encoder',
                'model.decoder',
                'model.input_projection',
                'model.query_position_embeddings',
                'model.layernorm'], show_params=show_model_params)

            elif custom_freezing and not freeze_layers:
                raise ValueError("Custom freezing can only be used if freeze_layers is provided")
            
            elif freeze_layers and not custom_freezing:
                self.logger.info("Freezing Layers specified but custom freezing is not True.")
                self.logger.info("Defaulting to Quick Training - Head Layer remains trainable...")
                self.ModelFunctionUtils.freeze_layers([
                'model.backbone',
                'model.encoder',
                'model.decoder',
                'model.input_projection',
                'model.query_position_embeddings',
                'model.layernorm'], show_params=show_model_params)

            else:
                self.ModelFunctionUtils.freeze_layers(freeze_layers, show_params=show_model_params)


            self.logger.info("Setting Up Data Loaders")
            if not num_workers and self.device != "mps":
                num_workers = min(12, mp.cpu_count() - 1)
                self.logger.info(f"Using Default of {num_workers} workers for Data Loading")
            elif num_workers and num_workers > 0 and self.device == "mps":
                num_workers = 0
                self.logger.info("Using 0 workers for Data Loading on MPS")

            self.logger.info("Loading Training and Validation Datasets")

            train_dataset = CocoDetectionDataset(dataset_dir, "train", self.processor)
            val_dataset = CocoDetectionDataset(dataset_dir, "val", self.processor)

            total_steps = epochs * (len(train_dataset) // batch_size)
            steps_save = (len(train_dataset) // batch_size) // 2

            training_args = TrainingArguments(
                output_dir=model_path,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                learning_rate=lr,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_schedule_type,
                gradient_accumulation_steps=gradient_accumulation_steps,
                logging_steps=logging_steps,
                save_strategy="steps",
                save_steps=steps_save,
                save_total_limit=save_limit,
                eval_strategy="steps",
                eval_steps=steps_save,
                metric_for_best_model=metric_for_best_model,
                load_best_model_at_end=True,
                greater_is_better=False if metric_for_best_model == "loss" else True,
                remove_unused_columns=False,
                dataloader_num_workers=num_workers,
                dataloader_pin_memory=True,
                fp16=True,
                report_to=["tensorboard"],
            )

            if patience and patience_threshold > 0:
                callbacks = [
                    EarlyStoppingCallback(
                        early_stopping_patience=patience,
                        early_stopping_threshold=patience_threshold
                    )
                ]
            else:
                callbacks = []

            self.logger.info("Setting Up Trainer")

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=self._collate_fn,
                callbacks=callbacks,
            )


            print(f"\n=== Starting Training ===\n"
                  f"Model: {self.model_id}\n"
                  f"Total Epochs: {epochs}\n"
                  f"Batch Size: {batch_size}\n"
                  f"Learning Rate: {lr}\n"
                  f"Device: {self.device}\n")

            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            metrics = train_result.metrics

            trainer.save_model(model_path)
            self.processor.save_pretrained(model_path)

            return {
                'best_metric': trainer.state.best_metric,
                'final_train_loss': metrics.get("train_loss"),
                'final_eval_loss': metrics.get("eval_loss"),
                'tensorboard_dir': tensorboard_dir,
                'early_stopped': trainer.state.global_step < total_steps,
                'model_path': os.path.join(self.model_run_path, save_dir)
            }

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            if 'writer' in locals():
                writer.close()
            raise e
    
    def inference(self, image_path: str, conf: float = 0.9) -> List[Dict]:
        self.model.eval()
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        width, height = image.size
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=[(height, width)],
            threshold=conf  
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            xmin, ymin, xmax, ymax = box.tolist()
            detection = {
                'image_id': os.path.basename(image_path),
                'category_id': label.item(),
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],  
                'score': score.item(),
                'area': (xmax - xmin) * (ymax - ymin),
                'iscrowd': 0
            }
            detections.append(detection)
        
        return detections
    
    def _collate_fn(self, batch):
        max_h = max([item['pixel_values'].shape[1] for item in batch])
        max_w = max([item['pixel_values'].shape[2] for item in batch])
        
        pixel_values = []
        pixel_mask = []
        
        for item in batch:
            _, h, w = item['pixel_values'].shape
            
            pad_h = max_h - h
            pad_w = max_w - w

            padded_values = torch.nn.functional.pad(
                item['pixel_values'],
                (0, pad_w, 0, pad_h),
                mode='constant',
                value=0
            )
            pixel_values.append(padded_values)
            
            padded_mask = torch.nn.functional.pad(
                item['pixel_mask'],
                (0, pad_w, 0, pad_h),
                mode='constant',
                value=0
            )
            pixel_mask.append(padded_mask)
        
        pixel_values = torch.stack(pixel_values)
        pixel_mask = torch.stack(pixel_mask)
        labels = [item['labels'] for item in batch]
        
        return {
            'pixel_values': pixel_values,
            'pixel_mask': pixel_mask,
            'labels': labels
        }
