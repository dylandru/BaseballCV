import torch
from tqdm.auto import tqdm
import statistics
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import time
import torch.multiprocessing as mp
from torch.optim import AdamW
from transformers import (PaliGemmaProcessor, PaliGemmaForConditionalGeneration, get_scheduler)
from torch.utils.data import DataLoader
import torch.backends
import os
from PIL import Image
from typing import Dict, Tuple, List
import numpy as np
from peft import PeftModel
from datetime import datetime
import supervision as sv
from baseballcv.model.utils import ModelFunctionUtils, ModelVisualizationTools
from baseballcv.datasets import DataProcessor
from baseballcv.utilities import BaseballCVLogger

"""
To use PaliGemma2 from HuggingFace, the user must accept Google's Usage License and be approved by Google.
"""

class PaliGemma2:
    '''
    A class to initialize and run PaliGemma2 from HuggingFace based in PyTorch.
    '''

    def __init__(self, 
                 device: str = None, 
                 model_id: str = 'google/paligemma2-3b-pt-224', 
                 model_run_path: str = f'paligemma2_run_{datetime.now().strftime("%Y%m%d")}', 
                 batch_size: int = 8, torch_dtype: torch.dtype = torch.float32,
                 use_pretrained_lora: bool = False):
        """
        Initialize the PaliGemma2 model.

        Args:
            model_id: The identifier of the model to use.
            model_run_path: The path to save model run information.
            batch_size: The batch size for training and inference.
        """
    
        self.device = torch.device("cuda" if torch.cuda.is_available()
                                else "mps" if torch.backends.mps.is_available()
                                else "cpu") if device is None else torch.device(device)
                                
        self.model_id = model_id
        self.batch_size = batch_size
        self.model_run_path = model_run_path
        self.model_name = "PaliGemma2"
        self.image_directory_path = "" 
        self.torch_dtype = torch_dtype
        self.augment = True
        self.use_pretrained_lora = use_pretrained_lora

        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__)

        self.quantization_config = None
        self.processor = None
        self.peft_model = None
        self._init_model()

        self.DataProcessor = DataProcessor(self.logger)
        
        self.ModelFunctionUtils = ModelFunctionUtils(
            self.model_name, 
            self.model_run_path,
            self.batch_size, 
            self.device, 
            self.processor, 
            self.model, 
            self.peft_model,
            self.logger,
            self.torch_dtype
        )
        
        self.ModelVisualizationTools = ModelVisualizationTools(
            self.model_name, self.model_run_path, self.logger
        )

    def _init_model(self):
        """Initialize the model and processor."""
        try:
            if self.device == "cuda":
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                    self.model_id, 
                    device_map="auto", 
                    use_flash_attention_2=True, 
                    torch_dtype=self.torch_dtype, 
                    torch_dtype_details={
                        "compute_dtype": self.torch_dtype, 
                        "storage_dtype": self.torch_dtype
                    }
                )
            else:
                self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                    self.model_id
                )

            if self.use_pretrained_lora:
                try:
                    self.model = PeftModel.from_pretrained(self.model, 
                                                       self.model_id,
                                                       is_trainable=True)
                except Exception as e:
                    raise RuntimeError(f"LoRA Model Loading Failed: {str(e)}")
                
            
            if self.model is None:
                raise ValueError("Model failed to load")

            self.logger.info("Model Initialization Successful!")

            self.processor = PaliGemmaProcessor.from_pretrained(self.model_id)

        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def inference(self, image_path: str, text_input: str, task: str = "<TEXT_TO_TEXT>",
                  classes: List[str] = None) -> Tuple[str, str]:
        """
        Perform inference on an image.

        Args:
            image_path (str): Path to the input image.
            text_input (str): Optional text input for the task.
            task (str): The task to perform (e.g., object detection).
            classes (List[str]): Optional list of classes for object detection.
        Returns:
            generated_text (str): The result of the inference.
            image_path (str): The path to the annotated image if task is <TEXT_TO_OD>, else None.
        """

        if hasattr(self.model, 'merge_and_unload'):
            self.logger.info("Merging LoRA weights for Inference")
            self.model = self.model.merge_and_unload()

        image = Image.open(image_path)
        prompt = text_input
        self.model.eval()

        
        if task == "<TEXT_TO_TEXT>":
            inputs = self.processor(
                text=prompt,
                images=image.convert("RGB"),
                return_tensors="pt"
            )

            generation = self.model.generate(
                **inputs,
                max_new_tokens=256,
                early_stopping=True,
                do_sample=True,
                num_beams=2,
            )

            generated_text = (self.processor.decode(
                generation[0], skip_special_tokens=False))[len(prompt):]
            
            print(f"Output: {generated_text}")

            return generated_text
        
        elif task == "<TEXT_TO_OD>":
            if self.device != "cuda":
                self.model.to("cpu")

            inputs = self.processor(
                text="".join(["<image>", text_input if text_input else "detect all objects"]),
                images=image.convert("RGB"),
                return_tensors="pt"
            )
        
            prefix_length = inputs["input_ids"].shape[-1]

            self.logger.info("Conduncting Object Detection Generation...")

            with torch.inference_mode():
                generation = self.model.generate(**inputs, 
                                                 max_new_tokens=256, 
                                                 do_sample=True,
                                                 num_beams=3,
                                                 early_stopping=True)
                
                generation = generation[0][prefix_length:]
                generated_text = self.processor.decode(generation, skip_special_tokens=True)

            w, h = image.size

            self.logger.info("Processing Detections")

            if classes:
                detections = sv.Detections.from_lmm(
                    lmm='paligemma',
                    result=generated_text,
                    resolution_wh=(w, h),
                    classes=classes)
            else:
                detections = sv.Detections.from_lmm(
                    lmm='paligemma',
                    result=generated_text,
                    resolution_wh=(w, h))

            self.logger.info("Annotating Image")

            annotated_image = image.copy()
            annotated_image = sv.BoxAnnotator().annotate(scene=annotated_image, detections=detections)
            annotated_image = sv.LabelAnnotator(smart_position=True).annotate(scene=annotated_image, detections=detections)
            os.makedirs(os.path.join(self.model_run_path, "inference_results"), exist_ok=True)
            image_path = os.path.join(self.model_run_path, "inference_results", "annotated_image.png")
            annotated_image.save(image_path)
            return generated_text, image_path

        else:
            raise ValueError(f"Task {task} not supported")

    
    def finetune(self, dataset: str, classes: Dict[int, str],
                 train_test_split: Tuple[int, int, int] = (80, 10, 10),
                 freeze_vision_encoders: bool = False,
                 epochs: int = 20,
                 lr: float = 4e-6,
                 save_dir: str = "model_checkpoints",
                 num_workers: int = None,
                 lora_r: int = 8,
                 lora_scaling: int = 8,
                 patience: int = 10,
                 patience_threshold: float = 0.0,
                 gradient_accumulation_steps: int = 2,
                 lora_dropout: float = 0.05,
                 warmup_ratio: float = 0.03,
                 lr_schedule_type: str = "cosine",
                 create_peft_config: bool = True,
                 random_seed: int = 22,
                 logging_steps: int = 1000,
                 weight_decay: float = 0.01,
                 save_limit: int = 3,
                 metric_for_best_model: str = "loss",
                 dataset_type: str = "yolo",
                 resume_from_checkpoint: str = None) -> Dict:
        """
        FineTune PaliGemma2 on a custom dataset (currently configured for YOLO format)
        Utilizes PyTorch Training Loop and LoRA with Hugging Face Model for training.
        Has TensorBoard for visualization and metric tracking.
        """

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

        save_dir = os.path.join(self.model_run_path, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        run_name = f"run_{time.strftime('%Y%m%d-%H%M%S')}"
        tensorboard_dir = os.path.join(self.model_run_path, "tensorboard_logs", run_name)
        writer = SummaryWriter(tensorboard_dir)

        try:
            if dataset_type == "yolo":
                train_image_path, valid_image_path, train_jsonl_path, _, valid_jsonl_path = (
                    self.DataProcessor.prepare_dataset(
                        base_path=dataset,
                        dict_classes=classes,
                        train_test_split=train_test_split,
                        dataset_type=dataset_type
                    )
                )

            else:
                train_image_path, valid_image_path, train_jsonl_path, _, valid_jsonl_path = (
                    self.DataProcessor.prepare_dataset(
                        base_path=dataset,
                        dict_classes=classes,
                        dataset_type=dataset_type
                    )
                )

            self.logger.info(f"Dataset Preparation Complete - Train Path: {train_image_path}, Valid Path: {valid_image_path}")

            if self.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True

            if not num_workers and self.device != "mps":
                num_workers = min(12, mp.cpu_count() - 1)
                self.logger.info(f"Using Default of {num_workers} workers for Data Loading")
            elif num_workers and num_workers > 0 and self.device == "mps":
                num_workers = 0
                self.logger.info("Using 0 workers for Data Loading on MPS")

            self.train_loader, self.val_loader = self.ModelFunctionUtils.setup_data_loaders(
                train_image_path=train_image_path,
                valid_image_path=valid_image_path,
                train_jsonl_path=train_jsonl_path,
                valid_jsonl_path=valid_jsonl_path,
                num_workers=num_workers
            )

            if freeze_vision_encoders:
                self.model = self.ModelFunctionUtils.freeze_vision_encoders(self.model)

            self.peft_model = self.ModelFunctionUtils.setup_peft(
                lora_r, lora_scaling, lora_dropout, create_peft_config)
            self.model = self.peft_model
            self.model.to(self.device)

            optimizer_group_params = [
                {
                    'params': [p for n, p in self.model.named_parameters()
                            if p.requires_grad and not any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
                    'weight_decay': weight_decay
                },
                {
                    'params': [p for n, p in self.model.named_parameters()
                            if p.requires_grad and any(nd in n for nd in ['bias', 'LayerNorm.weight'])],
                    'weight_decay': 0.0
                }
            ]
            optimizer = AdamW(optimizer_group_params, lr=lr) #hardcoded AdamW for now...

            num_training_steps = epochs * len(self.train_loader)
            num_warmup_steps = int(warmup_ratio * num_training_steps)
            scheduler = get_scheduler(
                name=lr_schedule_type,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )

            # Load checkpoint if resuming
            if resume_from_checkpoint:
                self.ModelFunctionUtils.load_checkpoint(resume_from_checkpoint)

            best_metric = float('inf') if metric_for_best_model == "loss" else float('-inf')
            patience_counter = 0
            saved_checkpoints = []
            global_step = 0
            scaler = torch.amp.GradScaler() if self.device == "cuda" else None

            print(f"\n=== Starting Training ===\n"
                f"Model: {self.model_id}\n"
                f"Total Epochs: {epochs}\n"
                f"Batch Size: {self.batch_size}\n"
                f"Learning Rate: {lr}\n"
                f"Device: {self.device}\n")

            for epoch in range(epochs):

                epoch_dir = os.path.join(save_dir, f"epoch_{epoch + 1}")
                os.makedirs(epoch_dir, exist_ok=True)
                self.model.train()
                train_loss = 0
                epoch_losses = []

                train_progress = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}/{epochs} [Train]",
                    bar_format="{desc}\n{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]\n{postfix}",
                    postfix=dict(loss="0.0000", lr="0.0"),
                    dynamic_ncols=True,
                    initial=0
                )

                for step, batch in enumerate(train_progress):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(**batch)
                            loss = outputs.loss / gradient_accumulation_steps
                        scaler.scale(loss).backward()
                    else:
                        outputs = self.model(**batch)
                        loss = outputs.loss / gradient_accumulation_steps
                        loss.backward()

                    current_loss = loss.item()
                    epoch_losses.append(current_loss)
                    train_loss += current_loss

                    train_progress.set_postfix({
                        'loss': f'{statistics.mean(epoch_losses[-100:]):.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })

                    if (step + 1) % gradient_accumulation_steps == 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            optimizer.step()

                        scheduler.step()
                        optimizer.zero_grad()

                        if global_step % logging_steps == 0:
                            writer.add_scalar('Training/Loss', current_loss, global_step)
                            writer.add_scalar('Training/LearningRate', scheduler.get_last_lr()[0], global_step)

                        global_step += 1

                avg_train_loss = train_loss / len(self.train_loader)

                self.model.eval()
                val_loss = 0
                val_losses = []

                val_progress = tqdm(
                    self.val_loader,
                    desc=f"Epoch {epoch + 1}/{epochs} [Valid]",
                    bar_format="{desc}\n{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]\n{postfix}",
                    postfix=dict(loss="0.0000"),
                    dynamic_ncols=True,
                    initial=0
                )

                with torch.no_grad():
                    for batch in val_progress:
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                for k, v in batch.items()}
                        outputs = self.model(**batch)
                        current_val_loss = outputs.loss.item()
                        val_losses.append(current_val_loss)
                        val_loss += current_val_loss

                        val_progress.set_postfix({
                            'loss': f'{statistics.mean(val_losses[-100:]):.4f}'
                        })

                val_loss = val_loss / len(self.val_loader)
                current_metric = val_loss if metric_for_best_model == "loss" else -val_loss
                is_better = current_metric < best_metric

                epoch += 1

                writer.add_scalars('Loss', {
                    'train': avg_train_loss,
                    'validation': val_loss
                }, epoch)

                print(f"\nEpoch {epoch}/{epochs} Summary:")
                print(f"Training Loss: {avg_train_loss:.4f}")
                print(f"Validation Loss: {val_loss:.4f}")
                print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

                if is_better:
                    print(f"New Best Model! Previous: {best_metric:.4f} | New: {current_metric:.4f}")
                    print("Saving checkpoint for Model...")
                    best_metric = current_metric
                    patience_counter = 0
                    checkpoint_path = os.path.join(epoch_dir, f"checkpoint_epoch_{epoch}.pt")
                    self.ModelFunctionUtils.save_checkpoint(
                        path=checkpoint_path,
                        epoch=epoch,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss=val_loss,
                        scaler=scaler
                    )
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epochs. Best: {best_metric:.4f}")

                print("-" * 50)

                if (epoch) % 3 == 0:
                    print("Saving checkpoint for Model...")
                    checkpoint_path = os.path.join(epoch_dir, f"checkpoint_epoch_{epoch}.pt")
                    self.ModelFunctionUtils.save_checkpoint(
                        path=checkpoint_path,
                        epoch=epoch,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss=val_loss,
                        scaler=scaler
                    )
                    saved_checkpoints.append(checkpoint_path)

                    if len(saved_checkpoints) > save_limit:
                        os.remove(saved_checkpoints.pop(0))

                if patience_counter >= patience and abs(current_metric - best_metric) > patience_threshold:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            return {
                'best_metric': best_metric,
                'final_train_loss': avg_train_loss,
                'final_val_loss': val_loss,
                'tensorboard_dir': tensorboard_dir,
                'early_stopped': patience_counter >= patience,
                'model_path': os.path.join(epoch_dir, "best_model.pt")
            }

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            writer.close()
            raise e

    def evaluate(self, base_path: str, classes: Dict[int, str], num_workers: int = 5, dataset: Dataset = None, dataset_type: str = "yolo"):
        """
        Evaluate the model on a dataset. Defaults to valid split if creating new dataset.

        Args:
            base_path: The path where the entire dataset is stored (or will be stored if non-existent).
            classes: Dictionary mapping class IDs to class names.
            num_workers: The number of workers to use for data loading.
            dataset: The dataset to evaluate on, if one is provided. If None, the valid split of the full dataset is used.
            dataset_type: The type of dataset (yolo or paligemma). Default is yolo.

        Returns:
            The mean average precision result.
        """
        self.logger.info("Starting evaluation...")
        self.model.eval().to(self.device)

        if hasattr(self.model, 'merge_and_unload'):
            self.logger.info("Merging LoRA weights for evaluation")
            self.model = self.model.merge_and_unload()


        if dataset:
            split_dataset = dataset
        else:
            if dataset_type == "yolo":
                _, valid_image_path, _, _, valid_jsonl_path = self.DataProcessor.prepare_dataset(
                    base_path=base_path,
                    dict_classes=classes,
                    train_test_split=(80, 10, 10),
                    dataset_type=dataset_type
                )
            else:
                _, valid_image_path, _, _, valid_jsonl_path = self.DataProcessor.prepare_dataset(
                    base_path=base_path,
                    dict_classes=classes,
                    dataset_type=dataset_type
                )
            
            split_dataset = self.ModelFunctionUtils.create_detection_dataset(
                jsonl_file_path=valid_jsonl_path,
                image_directory_path=valid_image_path,
                augment=False
            )
        
        eval_loader = DataLoader(
            split_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.ModelFunctionUtils.collate_fn,
            pin_memory=True if self.device == "cuda" else False,
            num_workers=num_workers,
            persistent_workers=False
        )

        images = []
        targets = []
        predictions = []

        with torch.inference_mode():
          for batch_idx, batch in enumerate(tqdm(
                    eval_loader,
                    desc=f"Evaluating Model",
                    bar_format="{desc}\n{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    dynamic_ncols=True,
                    initial=0
                )):

            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = self.model.generate(
                **batch,
                max_new_tokens=256, 
                do_sample=False
            )

            prefix_length = batch["input_ids"].shape[-1]
            
            start_idx = batch_idx * self.batch_size
            end_idx = start_idx + len(outputs)
            
            orig_samples = [eval_loader.dataset[i] for i in range(start_idx, end_idx)]
            orig_images = [sample[0] for sample in orig_samples]
            batch_suffixes = [sample[1]["suffix"] for sample in orig_samples]

            for idx, (image, generation, suffix) in enumerate(zip(orig_images, outputs, batch_suffixes)):
                generated_text = self.processor.decode(generation[prefix_length:], skip_special_tokens=True)
        
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

