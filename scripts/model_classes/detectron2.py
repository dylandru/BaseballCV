from torch.utils.data import Dataset, DataLoader
import torch.backends
import torch.multiprocessing as mp
from datetime import datetime
import logging
import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any
import supervision as sv
from supervision.metrics import MeanAveragePrecision, MetricTarget

from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

from .utils import ModelFunctionUtils, ModelVisualizationTools, ModelLogger, DataProcessor

class Detectron2Trainer(DefaultTrainer):
    """Extension of Detectron2's DefaultTrainer with custom evaluation."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return evaluators[0]

class Detectron2:
    '''
    A class to initialize and run Detectron2 models for object detection tasks in baseball.
    '''
    def __init__(self, 
                 device: str = None,
                 model_id: str = 'faster_rcnn_R_50_FPN_3x',
                 model_run_path: str = f'detectron2_run_{datetime.now().strftime("%Y%m%d")}',
                 batch_size: int = 8,
                 torch_dtype: torch.dtype = torch.float32):
        """
        Initialize the Detectron2 model.

        Args:
            device: Device to run the model on ('cuda', 'cpu')
            model_id: Model architecture identifier
            model_run_path: Path to save model artifacts
            batch_size: Batch size for training/inference
            torch_dtype: Torch data type for model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "cpu") if device is None else torch.device(device)
        
        self.model_id = model_id
        self.batch_size = batch_size
        self.model_run_path = model_run_path
        self.model_name = "Detectron2"
        self.torch_dtype = torch_dtype
        self.cfg = None
        self.predictor = None

        self.logger = ModelLogger(self.model_name, self.model_run_path,
                                self.model_id, self.batch_size, self.device).orig_logging()

        self._init_model()

        self.DataProcessor = DataProcessor(self.logger)
        self.ModelFunctionUtils = ModelFunctionUtils(
            self.model_name,
            self.model_run_path,
            self.batch_size,
            self.device,
            None,
            None,
            None,
            self.logger,
            self.torch_dtype
        )
        self.ModelVisualizationTools = ModelVisualizationTools(
            self.model_name, self.model_run_path, self.logger
        )

    def _init_model(self):
        """Initialize Detectron2 configuration and model."""
        try:
            self.cfg = get_cfg()
            
            # Load base config from model zoo
            config_path = model_zoo.get_config_file(f"COCO-Detection/{self.model_id}.yaml")
            self.cfg.merge_from_file(config_path)
            
            # Set basic parameters
            self.cfg.MODEL.DEVICE = str(self.device)
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{self.model_id}.yaml")
            
            # Set inference threshold
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            
            self.logger.info("Model Configuration Initialized Successfully!")

        except Exception as e:
            self.logger.error(f"Failed to initialize model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _register_dataset(self, dataset_name: str, images_path: str, annotations: List[Dict]):
        """Register dataset with Detectron2."""
        
        def _get_data_dicts():
            data_dicts = []
            for annotation in annotations:
                record = {}
                
                # Get image info
                img_path = os.path.join(images_path, annotation['file_name'])
                height, width = cv2.imread(img_path).shape[:2]
                
                record["file_name"] = img_path
                record["image_id"] = annotation['image_id']
                record["height"] = height
                record["width"] = width
                
                objs = []
                for obj in annotation['annotations']:
                    obj_dict = {
                        "bbox": obj['bbox'],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "category_id": obj['category_id'],
                        "iscrowd": 0
                    }
                    objs.append(obj_dict)
                record["annotations"] = objs
                data_dicts.append(record)
            
            return data_dicts
        
        if dataset_name in DatasetCatalog:
            DatasetCatalog.remove(dataset_name)
        
        DatasetCatalog.register(dataset_name, _get_data_dicts)
        MetadataCatalog.get(dataset_name).set(thing_classes=self.class_names)

    def inference(self, image_path: str, conf_threshold: float = 0.5) -> Dict:
        """
        Perform inference on an image.

        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections

        Returns:
            Dictionary containing detection results
        """
        try:
            # Update confidence threshold
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
            
            if not self.predictor:
                self.predictor = DefaultPredictor(self.cfg)
            
            # Read image and run inference
            image = cv2.imread(image_path)
            outputs = self.predictor(image)
            
            # Process predictions
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            
            # Prepare results
            results = {
                'boxes': boxes,
                'scores': scores,
                'classes': classes
            }
            
            # Visualize results
            v = Visualizer(image[:, :, ::-1], 
                         metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                         scale=1.0)
            vis_output = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            
            # Save visualization
            vis_path = os.path.join(self.model_run_path, "inference_results")
            os.makedirs(vis_path, exist_ok=True)
            cv2.imwrite(os.path.join(vis_path, "result.jpg"), 
                       vis_output.get_image()[:, :, ::-1])
            
            return results
            
        except Exception as e:
            self.logger.error(f"Inference failed: {str(e)}")
            raise

    def finetune(self, dataset: str, classes: Dict[int, str],
                 train_test_split: Tuple[int, int, int] = (80, 10, 10),
                 epochs: int = 20,
                 lr: float = 0.00025,
                 save_dir: str = "model_checkpoints",
                 num_workers: int = 4,
                 warmup_iters: int = 1000) -> Dict:
        """
        Fine-tune Detectron2 model on a custom dataset.

        Args:
            dataset: Path to dataset
            classes: Dictionary mapping class IDs to names
            train_test_split: Train/test/val split ratios
            epochs: Number of training epochs
            lr: Learning rate
            save_dir: Directory to save checkpoints
            num_workers: Number of worker processes
            warmup_iters: Number of warmup iterations

        Returns:
            Dictionary containing training metrics
        """
        try:
            # Prepare dataset
            train_path, valid_path = self.DataProcessor.prepare_dataset(
                base_path=dataset,
                dict_classes=classes,
                train_test_split=train_test_split
            )
            
            # Store class names
            self.class_names = list(classes.values())
            
            # Update config for training
            self.cfg.DATASETS.TRAIN = ("baseball_train",)
            self.cfg.DATASETS.TEST = ("baseball_val",)
            
            # Set training parameters
            self.cfg.DATALOADER.NUM_WORKERS = num_workers
            self.cfg.SOLVER.IMS_PER_BATCH = self.batch_size
            self.cfg.SOLVER.BASE_LR = lr
            self.cfg.SOLVER.WARMUP_ITERS = warmup_iters
            self.cfg.SOLVER.MAX_ITER = epochs * len(DatasetCatalog.get("baseball_train")) // self.batch_size
            self.cfg.SOLVER.STEPS = []  # No learning rate decay
            
            # Set number of classes
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)
            
            # Set output directory
            self.cfg.OUTPUT_DIR = os.path.join(self.model_run_path, save_dir)
            os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
            
            # Create trainer and train
            trainer = Detectron2Trainer(self.cfg)
            trainer.resume_or_load(resume=False)
            trainer.train()
            
            # Load best model
            checkpoints = [f for f in os.listdir(self.cfg.OUTPUT_DIR) 
                         if f.endswith('.pth')]
            latest_checkpoint = sorted(checkpoints)[-1]
            self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, latest_checkpoint)
            
            return {
                'output_dir': self.cfg.OUTPUT_DIR,
                'final_model': self.cfg.MODEL.WEIGHTS
            }

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def evaluate(self, base_path: str, classes: Dict[int, str]) -> Dict:
        """
        Evaluate model performance on a dataset.

        Args:
            base_path: Path to evaluation dataset
            classes: Dictionary mapping class IDs to names

        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Prepare dataset
            _, valid_path = self.DataProcessor.prepare_dataset(
                base_path=base_path,
                dict_classes=classes
            )
            
            # Create evaluator
            evaluator = COCOEvaluator(
                "baseball_val",
                self.cfg,
                False,
                output_dir=os.path.join(self.cfg.OUTPUT_DIR, "evaluation")
            )
            
            # Run evaluation
            val_loader = build_detection_test_loader(
                self.cfg,
                "baseball_val",
                mapper=DatasetMapper(self.cfg, is_train=False)
            )
            
            results = inference_on_dataset(self.predictor.model, val_loader, evaluator)
            
            return results

        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise