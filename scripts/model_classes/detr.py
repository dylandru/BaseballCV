import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection, Trainer, TrainingArguments, EarlyStoppingCallback
import os
import warnings
from PIL import Image
from datetime import datetime
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from .utils import ModelLogger, ModelFunctionUtils, CocoDetectionDataset

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
            logging_steps: int = None,
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

            training_args = TrainingArguments(
                output_dir=os.path.join(model_path, 'checkpoints'),
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=epochs,
                learning_rate=lr,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_schedule_type,
                gradient_accumulation_steps=gradient_accumulation_steps,
                logging_steps=logging_steps,
                save_strategy="epoch",
                save_total_limit=save_limit,
                eval_strategy="epoch",
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
                'early_stopped': trainer.state.global_step < total_steps,
                'model_path': os.path.join(self.model_run_path, save_dir)
            }

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise e
    
    def inference(self, file_path: str,
                  conf: float = 0.9, 
                  save: bool = False, 
                  save_viz_dir: str = 'visualizations',
                  show_video: bool = False) -> List[Dict]:
        
        self.model.eval()

        if file_path.endswith('.png', '.jpg', '.jpeg'):
            image = Image.open(file_path).convert("RGB")
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
                    'image_id': image.filename,
                    'category_id': label.item(),
                    'bbox': [xmin, ymin, xmax, ymax],  
                    'score': score.item(),
                }
                detections.append(detection)
            
            ModelVisualizationTools(self.model_name, self.model_run_path, self.logger).visualize_detection_results(image, detections, save=save, save_viz_dir=os.path.join(self.model_run_path, save_viz_dir))

        elif file_path.endswith('.mp4'):
            cap = cv2.VideoCapture(file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if save:
                save_path = os.path.join(self.model_run_path, save_viz_dir, f'{file_path.split("/")[-1]}.mp4')
                writer = cv2.VideoWriter(save_path, 
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               fps, (width, height))
                
            all_detections = []
            frame_count = 0

            progress_bar = tqdm(total=total_frames, desc=f'Predicting Video: Frame {frame_count} / {total_frames}',
                                bar_format="{desc}\n{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]\n{postfix}",
                                dynamic_ncols=True,
                                initial=0)

            with progress_bar as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    image = Image.fromarray(frame_rgb)
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)

                    results = self.processor.post_process_object_detection(
                            outputs,
                            target_sizes=[(height, width)],
                            threshold=conf  
                        )[0]
                    
                    frame_detections = []
                    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                        xmin, ymin, xmax, ymax = box.tolist()
                        detection = {
                            'frame': frame_count,
                            'category_id': label.item(),
                            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                            'score': score.item(),
                        }
                        frame_detections.append(detection)
                        ModelVisualizationTools(self.model_name, self.model_run_path, self.logger).visualize_detection_results(image, frame_detections, save=False)
                    
                    all_detections.extend(frame_detections)

                    if save:
                        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

                    if show_video:
                        cv2.imshow('frame', frame_rgb)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    frame_count += 1
                    pbar.update(1)
            
   
        return detections if file_path.endswith('.png', '.jpg', '.jpeg') else all_detections
    
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


    def evaluate(self, dataset_dir: str, num_workers: int = 4, confidence_threshold: float = 0.5) -> Dict:
        """
        Evaluate the DETR model on a dataset.
        
        Args:
            dataset_dir (str): Directory containing the dataset with COCO format annotations
            num_workers (int): Number of workers for data loading
            confidence_threshold (float): Confidence threshold for detections
            
        Returns:
            Dict: Dictionary containing evaluation metrics
        """
        self.logger.info("Starting evaluation...")
        self.model.eval()

        # Create validation dataset and dataloader
        val_dataset = CocoDetectionDataset(dataset_dir, "val", self.processor)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )

        all_predictions = []
        all_targets = []
        images = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
                try:
                    # Move inputs to device
                    pixel_values = batch['pixel_values'].to(self.device)
                    pixel_mask = batch['pixel_mask'].to(self.device)

                    # Forward pass
                    outputs = self.model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask
                    )

                    # Process each item in the batch
                    for b in range(len(pixel_values)):
                        # Get logits and boxes for this item
                        logits = outputs.logits[b]  # Shape: [num_queries, num_classes + 1]
                        boxes = outputs.pred_boxes[b]  # Shape: [num_queries, 4]

                        # Get probabilities and apply threshold
                        probs = logits.softmax(-1)  # Shape: [num_queries, num_classes + 1]
                        scores, labels = probs[:, :-1].max(-1)  # Exclude no-object class
                        keep_mask = scores > confidence_threshold
                        
                        # Filter predictions
                        filtered_boxes = boxes[keep_mask]
                        filtered_scores = scores[keep_mask]
                        filtered_labels = labels[keep_mask]

                        # Convert boxes to image coordinates
                        img_h, img_w = pixel_values[b].shape[1:]
                        scaled_boxes = self.processor.post_process_box_predictions(
                            filtered_boxes,
                            target_sizes=[(img_h, img_w)]
                        )[0]

                        predictions = {
                            'boxes': scaled_boxes.cpu().numpy(),
                            'scores': filtered_scores.cpu().numpy(),
                            'labels': filtered_labels.cpu().numpy(),
                        }
                        all_predictions.append(predictions)

                        # Get corresponding ground truth
                        target = batch['labels'][b]
                        if isinstance(target, dict):
                            target_boxes = target['boxes'].cpu().numpy()
                            target_labels = target['labels'].cpu().numpy()
                        else:
                            target_boxes = target.get('boxes', []).cpu().numpy()
                            target_labels = target.get('labels', []).cpu().numpy()
                        
                        all_targets.append({
                            'boxes': target_boxes,
                            'labels': target_labels
                        })

                        # Store image for visualization (first 25 only)
                        if len(images) < 25:
                            images.append(pixel_values[b])

                except Exception as e:
                    self.logger.warning(f"Error processing batch {batch_idx}: {str(e)}")
                    continue

            # Calculate metrics
            metrics = self._calculate_metrics(all_predictions, all_targets)

            # Visualize results if we have images
            if images:
                self._visualize_results(
                    images[:25], 
                    all_predictions[:25], 
                    all_targets[:25], 
                    metrics
                )

            self.logger.info("Evaluation complete.")
            self.logger.info(f"mAP: {metrics['mAP']:.4f}")
            
            return metrics

    def _calculate_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """
        Calculate evaluation metrics including mAP.
        
        Args:
            predictions (List[Dict]): List of prediction dictionaries
            targets (List[Dict]): List of target dictionaries
            
        Returns:
            Dict: Dictionary containing calculated metrics
        """
        import supervision as sv
        from supervision.metrics import MeanAveragePrecision, MetricTarget
        
        # Convert to supervision Detections format
        sv_predictions = []
        sv_targets = []
        
        for pred, target in zip(predictions, targets):
            # Handle empty predictions/targets
            if len(pred['boxes']) == 0 or len(target['boxes']) == 0:
                continue

            sv_pred = sv.Detections(
                xyxy=pred['boxes'],
                confidence=pred['scores'],
                class_id=pred['labels']
            )
            sv_predictions.append(sv_pred)
            
            sv_target = sv.Detections(
                xyxy=target['boxes'],
                class_id=target['labels']
            )
            sv_targets.append(sv_target)

        # Calculate metrics
        map_metric = MeanAveragePrecision(metric_target=MetricTarget.BOXES)
        metrics = map_metric.update(sv_predictions, sv_targets).compute()

        return {
            'mAP': metrics.map,
            'mAP_50': metrics.map_50,
            'mAP_75': metrics.map_75,
            'mAR': metrics.mar,
        }

    def _visualize_results(self, images: List[torch.Tensor], predictions: List[Dict], 
                        targets: List[Dict], metrics: Dict) -> None:
        """
        Visualize evaluation results including example detections and metrics.
        
        Args:
            images (List[torch.Tensor]): List of images
            predictions (List[Dict]): List of predictions
            targets (List[Dict]): List of targets
            metrics (Dict): Dictionary of calculated metrics
        """
        import supervision as sv
        import matplotlib.pyplot as plt
        import os
        
        # Create visualization directory
        vis_dir = os.path.join(self.model_run_path, "evaluation_visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        # Plot metrics summary
        plt.figure(figsize=(10, 6))
        plt.bar(metrics.keys(), metrics.values())
        plt.title("Evaluation Metrics")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "metrics_summary.png"))
        plt.close()

        # Visualize example detections
        annotated_images = []
        for img, pred, target in zip(images[:25], predictions[:25], targets[:25]):
            # Convert image tensor to PIL Image
            img_array = (img.cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0)
            img_pil = Image.fromarray(img_array)
            
            # Create annotated image with predictions and ground truth
            annotated_img = img_pil.copy()
            
            # Draw predictions in red
            box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.default())
            if len(pred['boxes']) > 0:  # Only draw if we have predictions
                pred_detections = sv.Detections(
                    xyxy=pred['boxes'],
                    confidence=pred['scores'],
                    class_id=pred['labels']
                )
                annotated_img = box_annotator.annotate(scene=annotated_img, detections=pred_detections)
            
            # Draw ground truth in green
            box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.default())
            if len(target['boxes']) > 0:  # Only draw if we have ground truth
                target_detections = sv.Detections(
                    xyxy=target['boxes'],
                    class_id=target['labels']
                )
                annotated_img = box_annotator.annotate(scene=annotated_img, detections=target_detections)
            
            annotated_images.append(annotated_img)

        # Create grid visualization
        sv.plot_images_grid(
            images=annotated_images,
            grid_size=(5, 5),
            output_path=os.path.join(vis_dir, "detection_examples.png")
        )

        self.logger.info(f"Visualizations saved to {vis_dir}")