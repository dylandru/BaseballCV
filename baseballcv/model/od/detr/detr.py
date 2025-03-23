import logging
import cv2
import torch
from coco_eval import CocoEvaluator
from tqdm import tqdm
from transformers import DetrImageProcessor, DetrForObjectDetection
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import multiprocessing as mp
import supervision as sv
from baseballcv.model.utils import ModelFunctionUtils, ModelVisualizationTools
from baseballcv.datasets import CocoDetectionDataset
import pytorch_lightning as pl
from baseballcv.utilities import BaseballCVLogger

"""

DETR Class Implementation is based on the following tutorial:
- https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb

"""

class DETR(pl.LightningModule):
    """
    DETR (DEtection TRansformer) model class for object detection tasks using PyTorch Lightning and HuggingFace Transformers.
    Current class only supports COCO detection datasets.
    
    This class provides functionality for:
    - Model initialization and configuration
    - Fine-tuning on custom datasets
    - Inference on images and videos
    - Model evaluation
    
    Attributes:
        detr_device (torch.device): Device to run model on (cuda/mps/cpu)
        model_id (str): HuggingFace model identifier
        model_run_path (str): Directory path for model outputs
        model_name (str): Name of the model
        batch_size (int): Batch size for training/inference
        processor (DetrImageProcessor): Image processor for DETR
        model (DetrForObjectDetection): DETR model instance
        detr_logger (ModelLogger): Logger instance
        ModelFunctionUtils (ModelFunctionUtils): Utility functions for model operations
    """
    def __init__(self, 
                 num_labels: int,
                 device: str = None, 
                 model_id: str = "facebook/detr-resnet-50",
                 model_run_path: str = f'detr_run_{datetime.now().strftime("%Y%m%d")}',
                 batch_size: int = 8,
                 image_size: Tuple[int, int] = (800, 800)):
        """
        Initialize DETR model.

        Args:
            num_labels (int): Number of object classes to detect
            device (str, optional): Device to run model on. Defaults to None (auto-detect).
            model_id (str, optional): HuggingFace model ID. Defaults to "facebook/detr-resnet-50".
            model_run_path (str, optional): Output directory path. Defaults to timestamped directory.
            batch_size (int, optional): Batch size. Defaults to 8.
            image_size (Tuple[int, int], optional): Input image dimensions. Defaults to (800, 800).
        """
        super().__init__()
        self.save_hyperparameters(ignore=['device'])
        
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        # Device and Logger are both PL Lightning attributes - overriding with DETR specific attributes

        self.detr_device = "cuda" if torch.cuda.is_available() else "cpu" if device is None else device
        self.model_id = model_id
        self.model_run_path = model_run_path
        self.model_name = "DETR"
        self.batch_size = batch_size

        self.detr_logger = BaseballCVLogger.get_logger(self.__class__.__name__)

        self.processor = DetrImageProcessor.from_pretrained(
            self.model_id,
            size={"shortest_edge": image_size[0], "longest_edge": image_size[1]},
            do_resize=True,
            do_pad=True,
            do_normalize=True
        )
        
        self.model = DetrForObjectDetection.from_pretrained(
            self.model_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.detr_device)

        self.ModelFunctionUtils = ModelFunctionUtils(self.model_name, 
                            self.model_run_path, self.batch_size, 
                            self.detr_device, self.processor, self.model, 
                            None, self.detr_logger, torch.float16)
        
    def finetune(self, dataset_dir: str,
            classes: dict,
            save_dir: str = "finetuned_detr",
            batch_size: int = 4,
            epochs: int = 10,
            lr: float = 1e-4,
            lr_backbone: float = 1e-5,
            weight_decay: float = 0.01,
            gradient_accumulation_steps: int = 2,
            logging_steps: int = 100,
            save_limit: int = 2,
            patience: int = 3,
            patience_threshold: float = 0.0001,
            precision: str = '16-mixed',
            num_workers: int = None,
            freeze_layers: List[str] = None,
            freeze_head: bool = False,
            show_model_params: bool = True,
            push_to_hub_with_path: str = None,
            conf: float = 0.2) -> Dict:
        """
        Fine-tune the DETR model on a custom dataset.

        Args:
            dataset_dir (str): Path to dataset directory
            classes (dict): Dictionary mapping class IDs to class names
            save_dir (str, optional): Directory to save model. Defaults to "finetuned_detr".
            batch_size (int, optional): Training batch size. Defaults to 4.
            epochs (int, optional): Number of training epochs. Defaults to 10.
            lr (float, optional): Learning rate. Defaults to 1e-4.
            lr_backbone (float, optional): Backbone learning rate. Defaults to 1e-5.
            weight_decay (float, optional): Weight decay. Defaults to 0.01.
            gradient_accumulation_steps (int, optional): Gradient accumulation steps. Defaults to 2.
            logging_steps (int, optional): Steps between logging. Defaults to 100.
            save_limit (int, optional): Maximum checkpoints to save. Defaults to 2.
            patience (int, optional): Early stopping patience. Defaults to 3.
            patience_threshold (float, optional): Early stopping threshold. Defaults to 0.01.
            num_workers (int, optional): DataLoader workers. Defaults to None.
            freeze_layers (List[str], optional): Layers to freeze. Defaults to None.
            freeze_head (bool, optional): Whether to freeze head. Defaults to False.
            show_model_params (bool, optional): Show model parameters. Defaults to True.
            push_to_hub_with_path (str, optional): HuggingFace Hub path. Defaults to None.
            conf (float, optional): Confidence threshold. Defaults to 0.2.

        Returns:
            Dict: Training results...
                - best_model_path: Path to best checkpoint
                - best_model_score: Best validation score
                - early_stopped: Whether training stopped early
                - model_path: Path to saved model
        """
        try:
            self.model.config.id2label = classes
            self.lr = lr
            self.lr_backbone = lr_backbone
            self.weight_decay = weight_decay
            self.push_to_hub_with_path = push_to_hub_with_path

            model_path = os.path.join(self.model_run_path, save_dir, "model")
            os.makedirs(model_path, exist_ok=True)

            if freeze_head: 
                self.detr_logger.info("Freezing Head Layer...")
                self.ModelFunctionUtils.freeze_layers([
                    'model.backbone',
                    'model.encoder',
                    'model.decoder', 
                    'model.input_projection',
                    'model.query_position_embeddings',
                    'model.layernorm'
                ], show_params=show_model_params)

            elif freeze_layers is not None:
                self.ModelFunctionUtils.freeze_layers(freeze_layers, show_params=show_model_params)
            else:
                self.detr_logger.info("No Freezing Layers Specified - Training Entire Model")

            if not num_workers and self.detr_device != "mps":
                num_workers = min(12, mp.cpu_count() - 1)
                self.detr_logger.info(f"Using Default of {num_workers} workers")
            elif num_workers and num_workers > 0 and self.detr_device == "mps":
                num_workers = 0
                self.detr_logger.info("Using 0 workers for MPS")

            self.detr_logger.info("Loading Datasets...")
            train_dataset = CocoDetectionDataset(dataset_dir, "train", self.processor)
            val_dataset = CocoDetectionDataset(dataset_dir, "val", self.processor)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=self.collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=self.collate_fn
            )

            callbacks = [
                pl.callbacks.ModelCheckpoint(
                    dirpath=os.path.join(model_path, 'checkpoints'),
                    filename='detr-{epoch:02d}-{validation_loss:.2f}',
                    monitor='validation_loss',
                    mode='min',
                    save_top_k=save_limit
                )
            ]

            if patience:
                callbacks.append(
                    pl.callbacks.EarlyStopping(
                        monitor='validation_loss',
                        patience=patience,
                        min_delta=patience_threshold,
                        mode='min'
                    )
                )

            trainer = pl.Trainer(
                max_epochs=epochs,
                accelerator='auto',
                devices=1,
                precision=precision,
                accumulate_grad_batches=gradient_accumulation_steps,
                callbacks=callbacks,
                logger=pl.loggers.TensorBoardLogger(
                    save_dir=model_path,
                    name=f'pl_tensorboard_logs_{self.model_name}'
                ),
                log_every_n_steps=logging_steps
            )

            self.detr_logger.info("Starting training...")
            trainer.fit(
                model=self,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader
            )

            self.model.save_pretrained(model_path)
            self.processor.save_pretrained(model_path)
            self.detr_logger.info("Model and processor saved locally.")

            if self.push_to_hub_with_path is not None:
                self.detr_logger.info("Pushing to HuggingFace Hub...")

                try:
                    self.model.push_to_hub(self.push_to_hub_with_path)
                    self.processor.push_to_hub(self.push_to_hub_with_path)
                    self.detr_logger.info("Model and processor pushed to HuggingFace Hub successfully")
                except Exception as e:
                    self.detr_logger.error(f"Failed to push to HuggingFace Hub: {str(e)}")
            
            self.detr_logger.info("Training complete...")
            
            try:
                self.evaluate(dataset_dir=dataset_dir, conf=conf)
            except Exception as e:
                self.detr_logger.warning(f"Evaluation failed: {str(e)}")

            return {
                'best_model_path': trainer.checkpoint_callback.best_model_path,
                'best_model_score': trainer.checkpoint_callback.best_model_score,
                'early_stopped': trainer.early_stopping_callback.stopped_epoch > 0 if patience else False,
                'model_path': model_path
            }

        except Exception as e:
            self.detr_logger.error(f"Training failed: {str(e)}")
            raise e
        
    def inference(self, file_path: str,
                  classes: dict = None,
                  conf: float = 0.2, 
                  save: bool = False, 
                  save_viz_dir: str = 'visualizations',
                  show_video: bool = False) -> List[Dict]:
        """
        Run inference on image or video file.

        Args:
            file_path (str): Path to image or video file
            classes (dict, optional): Class ID to name mapping. Defaults to None.
            conf (float, optional): Confidence threshold. Defaults to 0.9.
            save (bool, optional): Whether to save visualizations. Defaults to False.
            save_viz_dir (str, optional): Directory to save visualizations. Defaults to 'visualizations'.
            show_video (bool, optional): Whether to display video during inference. Defaults to False.

        Returns:
            List[Dict]: List of detections, each containing:
                For images:
                    - boxes: Bounding box coordinates
                    - scores: Confidence scores
                    - labels: Predicted class labels
                For videos:
                    - video_id: Video file path
                    - frame: Frame number
                    - labels: Predicted class labels
                    - boxes: Bounding box coordinates
                    - scores: Confidence scores
        """

        if classes:
            self.model.config.id2label = classes

        self.model.eval()
        self.model.to(self.detr_device)
        if file_path.endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(file_path)

            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.detr_device)
                outputs = self.model(**inputs)
                
                target_sizes = [(image.shape[:2])]
                results = self.processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=conf  
                )[0]
        
            detections = sv.Detections.from_transformers(transformers_results=results)
            labels = [
                f"{self.model.config.id2label[class_id]} {confidence:0.2f}" 
                for _, confidence, class_id, _ 
                in detections
            ]

            annotated_frame = ModelVisualizationTools(
                self.model_name, 
                self.model_run_path, 
                self.detr_logger
            ).visualize_detection_results(
                file_path=file_path, 
                detections=detections, 
                labels=labels, 
                save=save, 
                save_viz_dir=os.path.join(self.model_run_path, save_viz_dir)
            )

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
                    inputs = {k: v.to(self.detr_device) for k, v in inputs.items()}
                    
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
                        annotated_frame = ModelVisualizationTools(
                            self.model_name, 
                            self.model_run_path, 
                            self.detr_logger
                        ).visualize_detection_results(
                            file_path=file_path, 
                            detections=detections, 
                            labels=labels, 
                            save=False, 
                            save_viz_dir=os.path.join(self.model_run_path, save_viz_dir)
                        )
                    
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


    def evaluate(self, dataset_dir: str,
                 conf: float = 0.2) -> Dict:
        """
        Evaluate model performance on test dataset.

        Args:
            dataset_dir (str): Path to dataset directory
            conf (float, optional): Confidence threshold. Defaults to 0.2.

        Returns:
            Dict: Evaluation metrics including mAP and other COCO metrics
        """
        try:
            self.detr_logger.info("Starting evaluation...")
            self.model.eval()
            self.model = self.model.to(self.detr_device)

            test_dataset = CocoDetectionDataset(dataset_dir, "test", self.processor)
            evaluator = CocoEvaluator(coco_gt=test_dataset.coco, iou_types=["bbox"])
            print(f"evaluator: {evaluator}")
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn
            )

            def convert_to_xywh(boxes):
                xmin, ymin, xmax, ymax = boxes.unbind(1)
                return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

            def prepare_for_coco_detection(predictions, conf):
                coco_results = []
                for original_id, prediction in predictions.items():
                    if len(prediction["boxes"]) == 0:
                        continue

                    boxes = convert_to_xywh(prediction["boxes"]).tolist()
                    scores = prediction["scores"].tolist()
                    labels = prediction["labels"].tolist()

                    for k, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                        if score > conf:
                            coco_results.append({
                                "image_id": original_id,
                                "category_id": label,
                                "bbox": box,
                                "score": score,
                            })

                return coco_results

            for idx, batch in tqdm(enumerate(test_loader), desc="Evaluating", total=len(test_loader)):
                pixel_values = batch['pixel_values'].to(self.detr_device)
                pixel_mask = batch['pixel_mask'].to(self.detr_device)
                labels = [{k: v.to(self.detr_device) for k, v in t.items()} for t in batch['labels']]
                with torch.no_grad():
                    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

                    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
                    results = self.processor.post_process_object_detection(
                        outputs, 
                        target_sizes=orig_target_sizes, 
                        threshold=conf)
                    predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
                    coco_results = prepare_for_coco_detection(predictions, conf)
                    if coco_results:
                        evaluator.update(coco_results)

            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            metrics = evaluator.summarize()

            return metrics

        except Exception as e:
            self.detr_logger.error(f"Evaluation failed: {str(e)}")
            raise e

    def forward(self, pixel_values, pixel_mask):
        """
        Forward pass of the model.

        Args:
            pixel_values: Input image pixel values
            pixel_mask: Input image pixel mask

        Returns:
            Model outputs
        """
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

    def common_step(self, batch, batch_idx):
        """
        Common step for training and validation.

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            Tuple[torch.Tensor, Dict]: Loss and loss dictionary
        """
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.detr_device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            torch.Tensor: Training loss
        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Input batch
            batch_idx: Batch index

        Returns:
            torch.Tensor: Validation loss
        """
        loss, loss_dict = self.common_step(batch, batch_idx)
        
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        """
        Configure optimizers for training.

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        if not hasattr(self, 'lr') or not hasattr(self, 'lr_backbone') or not hasattr(self, 'weight_decay'):
            raise ValueError("Learning rates and weight decay must be set via finetune() before training")
        
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() 
                          if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.named_parameters() 
                          if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        return optimizer

    def collate_fn(self, batch):
        """
        Collate function for data loading.

        Args:
            batch: Input batch

        Returns:
            Dict: Collated batch with pixel values, mask and labels
        """
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        return {
            'pixel_values': encoding['pixel_values'],
            'pixel_mask': encoding['pixel_mask'],
            'labels': labels
        }