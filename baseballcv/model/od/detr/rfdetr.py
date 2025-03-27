import multiprocessing as mp

from scipy.fft import set_workers
mp.set_start_method('spawn', force=True)

import os
from typing import Tuple, List
from rfdetr import RFDETRBase, RFDETRLarge
from baseballcv.model.utils import ModelVisualizationTools, ModelFunctionUtils
from baseballcv.utilities import BaseballCVLogger
from tqdm import tqdm
import supervision as sv
import cv2

"""

RF DETR Class Implementation is based on the following tutorial:
- https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-rf-detr-on-detection-dataset.ipynb#scrollTo=gcjxmZeqqAdv

"""

class RFDETR:
    """
    RF DETR Class Implementation
    """

    def __init__(self, device: str = "cpu", model_path: str = None, imgsz: int = 560, model_type: str = "base", labels: List[str] = None, project_path: str = "rfdetr_runs"):
        self.device = device
        self.imgsz = imgsz
        self.model = RFDETRBase(device=device, pretrained_weights=model_path if model_path else None) if model_type == "base" else RFDETRLarge(device=device, pretrained_weights=model_path if model_path else None)
        self.model_name = "rfdetr"

        self.model_run_path = os.path.join(os.getcwd(), project_path)
        os.makedirs(self.model_run_path, exist_ok=True)

        self.model.resolution = self.imgsz
        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__)
        self.ModelVisualizationTools = ModelVisualizationTools(self.model_name, self.model_run_path, self.logger)
        self.logger.info("Initializing RF DETR model...")
        self.labels = labels

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Enable MPS fallback as implementation buggy


    def inference(self, source_path: str, conf: float = 0.2, save_viz: bool = True) -> Tuple[List[sv.Detections], str]:
        """
        Inference method for the RF DETR model.

        Args:
            source_path (str): The path to the source file (image or video).
            conf (float): The confidence threshold for the detections.
            save_viz (bool): Whether to save the visualization of the detections.

        Returns:
            detections (list): A list of detections.
            output_path (str): The path to the output file (image or video).
        """
        
        is_video = os.path.isfile(source_path) and os.path.splitext(source_path)[1].lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        class_mapping = {str(i): label for i, label in enumerate(self.labels)}
        if is_video:
            
            frame_generator = sv.get_video_frames_generator(source_path)
            video_info = sv.VideoInfo.from_video_path(source_path)

            all_detections = []
            output_path = os.path.join(self.model_run_path, os.path.basename(source_path))

            with sv.VideoSink(target_path=output_path, video_info=video_info) as sink:
                for frame in tqdm(frame_generator, total=video_info.total_frames, desc="Processing frames"):
                    detections = self.model.predict(frame, threshold=conf)
                    all_detections.append(detections)
                    if save_viz:
                        labels = []
                        for class_id in detections.class_id:
                            class_id_str = str(class_id)
                            class_name = class_mapping.get(class_id_str)
                            labels.append(class_name)


                        annotated_image = sv.BoxAnnotator().annotate(scene=frame.copy(), detections=detections)
                        annotated_image = sv.LabelAnnotator(text_thickness=1).annotate(
                            scene=annotated_image, 
                            detections=detections,
                            labels=labels
                        )
                        sink.write_frame(annotated_image)

                self.logger.info(f"Inference completed." + ("Saved to " + output_path if save_viz else ""))

            return all_detections, output_path if save_viz else all_detections
        
        else:
            self.logger.info("Starting image inference...")
            detections = self.model.predict(source_path, threshold=conf)

            if save_viz:
                image = cv2.imread(source_path)

                labels = []
                for class_id in detections.class_id:
                    class_id_str = str(class_id)
                    class_name = class_mapping.get(class_id_str)
                    labels.append(class_name)

                annotated_image = sv.BoxAnnotator().annotate(scene=image.copy(), detections=detections)
                annotated_image = sv.LabelAnnotator(text_thickness=2).annotate(
                    scene=annotated_image, 
                    detections=detections,
                    labels=labels
                )
                output_path = os.path.join(self.model_run_path, os.path.basename(source_path))
                cv2.imwrite(output_path, annotated_image)
                self.logger.info(f"Inference completed. Detections saved to {output_path}")
            else:
                self.logger.info("Inference completed.")

            return detections, output_path if save_viz else detections
            

    def finetune(self, data_path: str, epochs: int = 50, batch_size: int = 4, 
                 lr: float = 0.0001, lr_encoder: float = 0.00015, weight_decay: float = 0.0001,
                 lr_drop: int = 100, clip_max_norm: float = 0.1, lr_vit_layer_decay: float = 0.8,
                 lr_component_decay: float = 0.7, grad_accum_steps: int = 4, amp: bool = True,
                 dropout: float = 0, drop_path: float = 0.0,
                 checkpoint_interval: int = 10, use_ema: bool = True, ema_decay: float = 0.993,
                 ema_tau: int = 100, num_workers: int = 2, warmup_epochs: int = 0) -> RFDETRBase | RFDETRLarge:
        """
        Finetune the RF DETR model.
        
        Args:
            data_path (str): Path to the dataset directory
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            lr (float): Learning rate
            lr_encoder (float): Learning rate for encoder
            weight_decay (float): Weight decay for optimizer
            lr_drop (int): Learning rate drop epoch
            clip_max_norm (float): Gradient clipping max norm
            lr_vit_layer_decay (float): Learning rate decay for ViT layers
            lr_component_decay (float): Learning rate decay for components
            grad_accum_steps (int): Gradient accumulation steps
            amp (bool): Whether to use automatic mixed precision
            dropout (float): Dropout rate
            drop_path (float): Drop path rate
            resolution (int): Input image resolution
            checkpoint_interval (int): Interval to save checkpoints
            use_ema (bool): Whether to use EMA
            ema_decay (float): EMA decay rate
            ema_tau (int): EMA tau parameter
            num_workers (int): Number of workers for data loading
            warmup_epochs (int): Number of warmup epochs
        """

        # Check if the dataset is already organized
        msg = ModelFunctionUtils.setup_rfdetr_dataset(data_path)
        self.logger.info(msg)

        # Multiprocessing on CPU addressed. 
        if self.device == "cpu" and num_workers > 0:
            self.logger.warning(f"Running on CPU with num_workers={num_workers}. This might cause issues. Consider setting num_workers=0 or wrapping in if __name__ == '__main__'.")
        
        self.model.train(
            dataset_dir=str(data_path),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            lr_encoder=lr_encoder,
            weight_decay=weight_decay,
            lr_drop=lr_drop,
            clip_max_norm=clip_max_norm,
            lr_vit_layer_decay=lr_vit_layer_decay,
            lr_component_decay=lr_component_decay,
            grad_accum_steps=grad_accum_steps,
            amp=amp,
            dropout=dropout,
            drop_path=drop_path,
            checkpoint_interval=checkpoint_interval,
            use_ema=use_ema,
            ema_decay=ema_decay,
            ema_tau=ema_tau,
            num_workers=num_workers,
            warmup_epochs=warmup_epochs
        )

        self.logger.info("Finetuning completed.")
        return self.model