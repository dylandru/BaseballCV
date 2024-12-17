import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, get_scheduler
import torch.backends
from tqdm import tqdm
import os
import json
from PIL import Image
import random
from typing import Dict, Tuple
import torch.multiprocessing as mp
import numpy as np
from datetime import datetime
import supervision as sv
from supervision.metrics import MeanAveragePrecision, MetricTarget
from utils import YOLOToJSONLDetection, ModelFunctionUtils, ModelVisualizationTools, ModelLogger

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
        self.model_run_path = model_run_path
        self.model_name = "PaliGemma2"
        self.logger = ModelLogger(self.model_name, self.model_run_path).orig_logging()
        self.YOLOToJSONLDetection = YOLOToJSONLDetection(self, self.entries, self.image_directory_path, self.logger, self.augment)
        self.ModelFunctionUtils = ModelFunctionUtils(self.model_name, self.model_run_path, 
                                      self.batch_size, self.device, self.processor, self.model, self.peft_model, self.logger, self.YOLOToJSONLDetection)
        self.ModelVisualizationTools = ModelVisualizationTools(self.model_name, self.model_run_path, self.logger)
        self._init_model()

    def _init_model(self):
        """Initialize the model and processor."""
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, trust_remote_code=True).to(self.device)
        self.processor = PaliGemmaProcessor.from_pretrained(
            self.model_id, trust_remote_code=True)

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
            self.ModelVisualizationTools.visualize_results(image, text_output)

        if task == "CAPTION_TO_PHASE_GROUNDING" or task == "<OPEN_VOCABULARY_DETECTION>":
            if text_input is not None:
                if task == "CAPTION_TO_PHASE_GROUNDING":
                    self.ModelVisualizationTools.visualize_results(image, text_output)
                else:
                    boxes = text_output.get('bboxes', [])
                    labels = text_output.get('bboxes_labels', [])
                    results = {
                        'bboxes': boxes,
                        'labels': labels
                    }
                    self.ModelVisualizationTools.visualize_results(image, results)
            else:
                raise ValueError("Text input is needed for this type of task")

        if task == "<CAPTION>" or task == "<DETAILED_CAPTION>" or task == "<MORE_DETAILED_CAPTION>":
            print(self.ModelFunctionUtils.return_clean_text_output(text_output))

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

            train_path, valid_path = self.YOLOToJSONLDetection.prepare_dataset(
                base_path=dataset,
                dict_classes=classes,
                train_test_split=train_test_split
            )

            self.logger.info(f"Dataset Preparation Complete - Train Path: {train_path}, Valid Path: {valid_path}")

            self.train_loader, self.val_loader = self.ModelFunctionUtils.setup_data_loaders(train_path, valid_path, num_workers=num_workers)

            self.logger.info(f"Data Loader Setup Complete")

            trainable_params_info, self.peft_model = self.ModelFunctionUtils.setup_peft(r=lora_r, alpha=lora_scaling, dropout=lora_dropout, create_peft_config=create_peft_config)

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

