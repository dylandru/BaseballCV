import cv2
import torch
from coco_eval import CocoEvaluator
from tqdm import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection, Trainer, TrainingArguments, EarlyStoppingCallback
import os
import warnings
from PIL import Image
from datetime import datetime
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import multiprocessing as mp
import supervision as sv
import matplotlib.pyplot as plt
from .utils import ModelLogger, ModelFunctionUtils, CocoDetectionDataset, ModelVisualizationTools

class DETR:
    def __init__(self, 
                 num_labels: int,
                 device: str = None, 
                 model_id: str = "facebook/detr-resnet-101",
                 model_run_path: str = f'detr_run_{datetime.now().strftime("%Y%m%d")}', 
                 batch_size: int = 8,
                 image_size: Tuple[int, int] = (800, 1333),
                 inference_mode: bool = False):
        
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
        if inference_mode:
            self.processor = DetrImageProcessor.from_pretrained(self.model_id, revision="no_timm",
                                size={"shortest_edge": image_size[0], "longest_edge": image_size[1]},
                                do_resize=True,
                                do_pad=True,
                                do_normalize=True)
            
            self.model = DetrForObjectDetection.from_pretrained(
                self.model_id,
                num_labels=num_labels,
                ignore_mismatched_sizes=True, 
                revision="no_timm"
            ).to(self.device)

        else:
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
            head_freeze: bool = False,
            show_model_params: bool = True) -> Dict:

        try:
            model_path = os.path.join(self.model_run_path, save_dir, "model")
            os.makedirs(model_path, exist_ok=True)

            if head_freeze: 
                self.logger.info("Freezing Head Layer...")
                self.ModelFunctionUtils.freeze_layers([
                    'model.backbone',
                    'model.encoder',
                    'model.decoder', 
                    'model.input_projection',
                    'model.query_position_embeddings',
                    'model.layernorm'
                ], show_params=show_model_params)

            elif freeze_layers is None and not head_freeze: 
                self.logger.info("No Freezing Layers Specified - Training Entire Model")

            elif freeze_layers is not None:
                self.ModelFunctionUtils.freeze_layers(freeze_layers, show_params=show_model_params)

            else:
                raise ValueError("Invalid Freezing Layers Specified")


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
            self.logger.info(f"Model and processor saved to {model_path}")

            self.logger.info("Conducting Evaluation...")
            self.evaluate(dataset_dir)



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
        
        box_annotator = sv.BoxAnnotator()

        if file_path.endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)

            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                
                target_sizes = [(image.shape[:2])]
                results = self.processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=conf  
                )[0]
        
            detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=conf)
            labels = [
                f"{self.model.config.id2label[class_id]} {confidence:0.2f}" 
                for _, confidence, class_id, _ 
                in detections
            ]

            annotated_frame = ModelVisualizationTools(self.model_name, self.model_run_path, self.logger).visualize_detection_results(file_path=file_path, detections=detections, labels=labels, save=save, save_viz_dir=os.path.join(self.model_run_path, save_viz_dir))

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

                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    target_sizes = [(image.shape[:2])]
                    results = self.processor.post_process_object_detection(
                        outputs,
                        target_sizes=target_sizes,
                        threshold=conf  
                    )[0]
                    
                    detections = sv.Detections.from_transformers(transformers_results=results).with_nms(threshold=conf)
                    labels = [
                        f"{self.model.config.id2label[class_id]} {confidence:0.2f}" 
                        for _, confidence, class_id, _ 
                        in detections
                    ]
                    
                    frame_detections = []
                    for score, class_id, box in zip(detections.confidence, detections.class_id, detections.xyxy):
                        xmin, ymin, xmax, ymax = box.tolist()
                        detection = {
                            'video_id': file_path,
                            'frame': frame_count,
                            'labels': self.model.config.id2label[class_id],
                            'boxes': [xmin, ymin, xmax, ymax],  
                            'scores': score.item(),
                        }
                        frame_detections.append(detection)
                        annotated_frame = ModelVisualizationTools(self.model_name, self.model_run_path, self.logger).visualize_detection_results(file_path=file_path, detections=detections, labels=labels, save=False)
                    
                    all_detections.extend(frame_detections)

                    if save:
                        writer.write(annotated_frame)

                    if show_video:
                        cv2.imshow('frame', annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    frame_count += 1
                    pbar.update(1)
            
   
        return detections if file_path.endswith(('.png', '.jpg', '.jpeg')) else all_detections


    def evaluate(self, dataset_dir: str) -> Dict:
        try:
            self.logger.info("Starting evaluation...")
            self.model.eval()

            test_dataset = CocoDetectionDataset(dataset_dir, "test", self.processor)
            evaluator = CocoEvaluator(coco_gt=test_dataset.coco, iou_types=["bbox"])
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self._collate_fn
            )

            def convert_to_xywh(boxes):
                xmin, ymin, xmax, ymax = boxes.unbind(1)
                return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

            def prepare_for_coco_detection(predictions):
                coco_results = []
                for original_id, prediction in predictions.items():
                    if len(prediction) == 0:
                        continue

                    boxes = prediction["boxes"]
                    boxes = convert_to_xywh(boxes).tolist()
                    scores = prediction["scores"].tolist()
                    labels = prediction["labels"].tolist()

                    coco_results.extend(
                        [
                            {
                                "image_id": original_id,
                                "category_id": labels[k],
                                "bbox": box,
                                "score": scores[k],
                            }
                            for k, box in enumerate(boxes)
                        ]
                    )
                return coco_results

            all_coco_results = []
            for batch in tqdm(test_loader, desc="Evaluating"):
                pixel_values = batch['pixel_values'].to(self.device)
                pixel_mask = batch['pixel_mask'].to(self.device)
                labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch['labels']]

                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

                    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
                    results = self.processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes)

                    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
                    coco_results = prepare_for_coco_detection(predictions)
                    all_coco_results.extend(coco_results)

            if not all_coco_results:
                self.logger.warning("No predictions to evaluate.")
                return {}

            evaluator.update(all_coco_results)
            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            metrics = evaluator.summarize()

            return metrics

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise e

        
    def _collate_fn(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }

    def _calculate_metrics(self, predictions: List[Dict], targets: List[Dict]) -> Dict:
        """
        Calculate evaluation metrics including mAP.
        
        Args:
            predictions (List[Dict]): List of prediction dictionaries
            targets (List[Dict]): List of target dictionaries
            
        Returns:
            Dict: Dictionary containing calculated metrics
        """
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
            'mAP': metrics.map50_95,
            'mAP_50': metrics.map50,
            'mAP_75': metrics.map75
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
            box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.red())
            if len(pred['boxes']) > 0:  # Only draw if we have predictions
                pred_detections = sv.Detections(
                    xyxy=pred['boxes'],
                    confidence=pred['scores'],
                    class_id=pred['labels']
                )
                annotated_img = box_annotator.annotate(scene=annotated_img, detections=pred_detections)
            
            # Draw ground truth in green
            box_annotator = sv.BoxAnnotator(color=sv.ColorPalette.green())
            if len(target['boxes']) > 0:  # Only draw if we have ground truth
                target_detections = sv.Detections(
                    xyxy=target['boxes'],
                    class_id=target['labels']
                )
                annotated_img = box_annotator.annotate(scene=annotated_img, detections=target_detections)
            
            annotated_images.append(annotated_img)

        # Create grid visualization
        figure, axs = plt.subplots(5, 5, figsize=(20, 20))
        for idx, img in enumerate(annotated_images):
            if idx < 25:  # Only show up to 25 images
                row = idx // 5
                col = idx % 5
                axs[row, col].imshow(img)
                axs[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "detection_examples.png"))
        plt.close()

        self.logger.info(f"Visualizations saved to {vis_dir}")