import torch
from torch.utils.data import Dataset
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Trainer, TrainingArguments, EarlyStoppingCallback
import torch.backends
from tqdm import tqdm
import os
from PIL import Image
from typing import Dict, Tuple
import numpy as np
from datetime import datetime
import supervision as sv
from supervision.metrics import MeanAveragePrecision, MetricTarget
from .utils import YOLOToJSONLDetection, ModelFunctionUtils, ModelVisualizationTools, ModelLogger

"""
To use PaliGemma2 from HuggingFace, the user must accept Google's Usage License and be approved by Google.
"""

class PaliGemma2:
    def __init__(self, 
                 model_id: str = 'google/paligemma2-3b-pt-224', 
                 model_run_path: str = f'paligemma2_run_{datetime.now().strftime("%Y%m%d")}', 
                 batch_size: int = 8):
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
        self.model_run_path = model_run_path
        self.model_name = "PaliGemma2"
        self.entries = [] 
        self.image_directory_path = "" 
        self.augment = True 
        self._init_model()
        self.logger = ModelLogger(self.model_name, self.model_run_path, self.model_id, self.batch_size, self.device).orig_logging()
        self.YOLOToJSONLDetection = YOLOToJSONLDetection(self, self.entries, self.image_directory_path, self.logger, self.augment)
        self.ModelFunctionUtils = ModelFunctionUtils(self.model_name, self.model_run_path, 
                                      self.batch_size, self.device, self.processor, self.model, self.peft_model, self.logger, self.YOLOToJSONLDetection)
        self.ModelVisualizationTools = ModelVisualizationTools(self.model_name, self.model_run_path, self.logger)

    def _init_model(self):
        """Initialize the model and processor."""
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id).to(self.device)
        self.processor = PaliGemmaProcessor.from_pretrained(
            self.model_id)

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
                 num_workers: int = 6, lora_r: int = 8, lora_scaling: int = 8, patience: int = 10,
                 patience_threshold: float = 0.0, gradient_accumulation_steps: int = 4,
                 lora_dropout: float = 0.05, warmup_ratio: float = 0.1, lr_schedule_type: str = "cosine", 
                 create_peft_config: bool = True, random_seed: int = 22, use_fp16: bool = False,
                 optimizer: str = "adamw_torch_fused", weight_decay: float = 0.01, logging_steps: int = 100,
                 save_eval_steps: int = 1000, save_limit: int = 3, metric_for_best_model: str = "loss"):
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
            lora_dropout: Dropout rate for LoRA.
            patience: Number of epochs to wait before early stopping.
            patience_threshold: Threshold to beat for early stopping.
            gradient_accumulation_steps: Number of gradient accumulation steps.
            warmup_ratio: Ratio of warmup steps to total steps.
            lr_schedule_type: Learning rate schedule type.
            create_peft_config: Whether to create a new PEFT configuration.
            random_seed: Random seed for reproducibility.
            use_fp16: Whether to use FP16 for training.
            optimizer: Optimizer to use.
            weight_decay: Weight decay for optimizer.
            logging_steps: Number of steps to log training metrics.
            save_eval_steps: Number of steps to save and evaluate model checkpoints.
            save_limit: Maximum number of model checkpoints to save (from the last saved checkpoint)
            metric_for_best_model: Metric to use for saving the best model.

        Returns:
            Dictionary containing training metrics.
        """
        self.logger.info(f"Finetuning {self.model_id} on {dataset}")

        save_dir = os.path.join(self.model_run_path, "model_checkpoints")
        training_args = TrainingArguments(
                seed=random_seed,
                output_dir=save_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_ratio = warmup_ratio,
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
                greater_is_better=False,
                fp16=use_fp16,
                optim=optimizer, 
                lr_scheduler_type=lr_schedule_type,
                report_to=["tensorboard", "wandb"],
                dataloader_pin_memory=True,
                dataloader_num_workers=num_workers,
                dataloader_persistent_workers=True,
                gradient_checkpointing=True, 
                ddp_find_unused_parameters=False,
                remove_unused_columns=True,
            )

        vis_path = os.path.join(
            self.model_run_path,
            'training_visualizations',
            self.model_id.replace('/', '_'),
            'finetuning_info',
            os.path.basename(dataset.rstrip('/'))
        )
        os.makedirs(vis_path, exist_ok=True)

        try:
            train_path, valid_path = self.YOLOToJSONLDetection.prepare_dataset(
                base_path=dataset,
                dict_classes=classes,
                train_test_split=train_test_split
            )
            self.logger.info(f"Dataset Preparation Complete - Train Path: {train_path}, Valid Path: {valid_path}")

            self.train_loader, self.val_loader = self.ModelFunctionUtils.setup_data_loaders(
                train_path, valid_path, num_workers=num_workers
            )
            self.logger.info("Data Loader Setup Complete")

            trainable_params_info, self.peft_model = self.ModelFunctionUtils.setup_peft(lora_r, lora_scaling, lora_dropout, create_peft_config)
            self.logger.info(trainable_params_info)

            self.logger.info(
                f"Training Configuration:\n"
                f"- Model ID: {self.model_id}\n"
                f"- Model Run Path: {self.model_run_path}\n"
                f"- Dataset: {dataset}\n"
                f"- Train Test Split: {train_test_split}\n"
                f"- Epochs: {epochs}\n"
                f"- Learning Rate: {lr}\n"
                f"- Optimizer: {optimizer}\n" 
                f"- Scheduler: {lr_schedule_type}\n"
                f"- Batch Size: {self.batch_size}\n"
                f"- Gradient Accumulation: {gradient_accumulation_steps}\n"
                f"- Warmup Ratio: {warmup_ratio}\n"
                f"- Early Stopping Patience: {patience}\n"
                f"- Patience Threshold: {patience_threshold}\n"
                f"- Mixed Precision: {'fp16' if use_fp16 else 'none'}\n"
                f"- Number of Workers: {num_workers}\n"
                f"- Save & Eval Steps: {save_eval_steps}\n"
                f"- Save Limit: {save_limit}\n"
                f"- Metric for Best Model: {metric_for_best_model}\n"

            )

            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=self.train_loader.dataset,
                eval_dataset=self.val_loader.dataset,
                data_collator=self.ModelFunctionUtils.collate_fn,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=patience_threshold)]
            )

            train_result = trainer.train()
            trainer.save_model()
            
            return train_result

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise e




    def evaluate(self, test_dataset: Dataset, classes: Dict[int, str]):
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

