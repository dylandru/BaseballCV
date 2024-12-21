import torch
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from transformers import (PaliGemmaProcessor, PaliGemmaForConditionalGeneration, 
                          Trainer, TrainingArguments, EarlyStoppingCallback, DefaultFlowCallback, ProgressCallback)
import torch.backends
from tqdm import tqdm
import os
from PIL import Image
from typing import Dict, Tuple, List
import numpy as np
from datetime import datetime
import supervision as sv
from supervision.metrics import MeanAveragePrecision, MetricTarget
from .utils import ModelFunctionUtils, ModelVisualizationTools, ModelLogger, DataProcessor

"""
To use PaliGemma2 from HuggingFace, the user must accept Google's Usage License and be approved by Google.
"""

class PaliGemma2:
    def __init__(self, 
                 model_id: str = 'google/paligemma2-3b-pt-224', 
                 model_run_path: str = f'paligemma2_run_{datetime.now().strftime("%Y%m%d")}', 
                 batch_size: int = 8, torch_dtype: torch.dtype = torch.float32):
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
        self.model_run_path = model_run_path
        self.model_name = "PaliGemma2"
        self.image_directory_path = "" 
        self.torch_dtype = torch_dtype
        self.augment = True 
        self.mp_method = "spawn" if self.device == "cuda" else "fork"

        self.logger = ModelLogger(self.model_name, self.model_run_path, 
                                self.model_id, self.batch_size, self.device).orig_logging()

        self.quantization_config = None
        if self.device == "cuda":
            self.quantization_config = ModelFunctionUtils.setup_quantization()
        self.model = None
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
        if self.device == "cuda":
            mp.set_start_method('spawn', force=True)
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_id, devic_map="auto", quantization_config=self.quantization_config)
        else:
            mp.set_start_method('fork', force=True)
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.model_id)
        
        self.processor = PaliGemmaProcessor.from_pretrained(
            self.model_id)

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
                 train_test_split: Tuple[int, int, int] = (80, 10, 10), freeze_vision_encoders: bool = False,
                 epochs: int = 20, lr: float = 4e-6, save_dir: str = "model_checkpoints",
                 num_workers: int = 0, lora_r: int = 8, lora_scaling: int = 8, patience: int = 10,
                 patience_threshold: float = 0.0, gradient_accumulation_steps: int = 16,
                 lora_dropout: float = 0.05, warmup_ratio: float = 0.03, lr_schedule_type: str = "cosine", 
                 create_peft_config: bool = True, random_seed: int = 22, 
                 optimizer: str = "adamw_hf", weight_decay: float = 0.01, logging_steps: int = 100,
                 save_eval_steps: int = 1000, save_limit: int = 3, metric_for_best_model: str = "loss") -> Dict:
        """
        Fine-tune the model on the given dataset.

        Args:
            dataset (str): Path to the dataset.
            classes (Dict[int, str]): Dictionary mapping class IDs to class names.
            train_test_split (Tuple[int, int, int]): Tuple specifying the train, test, and validation split ratios.
            epochs (int): Number of epochs to train.
            lr (float): Learning rate.
            save_dir (str): Directory to save the model checkpoints.
            num_workers (int): Number of worker processes for data loading.
            lora_r (int): Rank for LoRA.
            lora_scaling (int): Scaling factor for LoRA.
            lora_dropout (float): Dropout rate for LoRA.
            patience (int): Number of epochs to wait before early stopping.
            patience_threshold (float): Threshold to beat for early stopping.
            gradient_accumulation_steps (int): Number of gradient accumulation steps.
            warmup_ratio (float): Ratio of warmup steps to total steps.
            lr_schedule_type (str): Learning rate schedule type.
            create_peft_config (bool): Whether to create a new PEFT configuration.
            random_seed (int): Random seed for reproducibility.
            optimizer (str): Optimizer to use.
            weight_decay (float): Weight decay for optimizer.
            logging_steps (int): Number of steps to log training metrics.
            save_eval_steps (int): Number of steps to save and evaluate model checkpoints.
            save_limit (int): Maximum number of model checkpoints to save (from the last saved checkpoint)
            metric_for_best_model (str): Metric to use for saving the best model.

        Returns:
            train_result (Dict): Dictionary containing training metrics.
        """
        self.logger.info(f"Finetuning {self.model_id} on {dataset}")
        save_dir = os.path.join(self.model_run_path, "model_checkpoints")

        if freeze_vision_encoders:
            self.model = self.ModelFunctionUtils.freeze_vision_encoders(self.model)


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
                lr_scheduler_type=lr_schedule_type,
                report_to=["tensorboard", "wandb"],
                dataloader_pin_memory=False,
                dataloader_num_workers=num_workers,
                dataloader_persistent_workers=True if num_workers > 0 else False,
                gradient_checkpointing=True, 
                remove_unused_columns=True
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
            train_image_path, valid_image_path, train_jsonl_path, test_jsonl_path, valid_jsonl_path = self.DataProcessor.prepare_dataset(
                base_path=dataset,
                dict_classes=classes,
                train_test_split=train_test_split
            )
            self.logger.info(f"Dataset Preparation Complete - Train Path: {train_image_path}, Valid Path: {valid_image_path}")

            self.train_dataset, self.val_dataset = self.ModelFunctionUtils.setup_data_loaders(
                train_image_path, valid_image_path, train_jsonl_path, valid_jsonl_path, num_workers=num_workers
            )
            self.logger.info("Data Loader Setup Complete")

            self.peft_model = self.ModelFunctionUtils.setup_peft(lora_r, lora_scaling, lora_dropout, create_peft_config)

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
                f"- Number of Workers: {num_workers}\n"
                f"- Save & Eval Steps: {save_eval_steps}\n"
                f"- Save Limit: {save_limit}\n"
                f"- Metric for Best Model: {metric_for_best_model}\n"

            )

            trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.val_dataset,
                data_collator=self.ModelFunctionUtils.collate_fn,
                callbacks=[
                    EarlyStoppingCallback(
                        early_stopping_patience=patience,
                        early_stopping_threshold=patience_threshold
                    ),
                    DefaultFlowCallback(),
                    ProgressCallback()
                ]
            )
            
            if self.device == "cuda":
                train_result = mp.spawn(trainer.train, nprocs=1, args=(self.device,))
            else:
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

