from baseballcv.functions import LoadTools
from ultralytics import YOLO
from datetime import datetime
import cv2
import streamlit as st
import torch
import numpy as np

__all__ = ['ModelManager']

class ModelManager:
    """
    A class to manage model loading and prediction.

    Args:
        conf (float): Confidence threshold for predictions. Defaults to 0.8.
    """
    def __init__(self, conf: float = 0.8):
        self.load_tools = LoadTools()
        self.conf = conf
        self._model_cache = {}
        torch.set_num_threads(8)
        
    def load_model(self, model_alias: str) -> YOLO:
        """
        Load model with caching to avoid repeated loading of the same model.

        Args:
            model_alias (str): Alias of the model to load.

        Returns:
            YOLO: The loaded YOLO model.
        """
        if model_alias not in self._model_cache:
            model_path = self.load_tools.load_model(model_alias=model_alias)
            model = YOLO(model_path)
            model.model.eval()
            model.model.float()
            self._model_cache[model_alias] = model
        return self._model_cache[model_alias]
        
    def predict_image(self, image_path: str, model: YOLO) -> list[dict]:
        """
        Predict on image and return annotations.

        Args:
            image_path (str): Path to image to predict on.
            model (YOLO): The YOLO model to use for prediction.

        Returns:
            list[dict]: List of annotations.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        height, width = img.shape[:2]

        results = next(model.predict(
            img,
            conf=self.conf,
            verbose=True,
            imgsz=(480, 800),
            stream=True,
            device='cpu',
            max_det=5,
            agnostic_nms=True
        ))
        
        annotations = []
        if len(results.boxes) > 0:
            boxes = results.boxes.xywhn.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            
            boxes[:, 0] *= width
            boxes[:, 1] *= height
            boxes[:, 2] *= width
            boxes[:, 3] *= height
            
            for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
                annotations.append({
                    "id": i + 1,
                    "image_id": 1,
                    "category_id": int(cls) + 1,
                    "bbox": box.tolist(),
                    "date_created": datetime.now().isoformat(),
                    "user_id": "model",
                    "auto_generated": True,
                    "score": float(score)
                })
        
        return annotations

    @staticmethod
    def get_model_instance(model_alias: str) -> YOLO:
        """
        Get the model using Streamlit's session state for caching.

        Args:
            model_alias (str): Alias of the model to get.

        Returns:
            YOLO: The loaded YOLO model.
        """
        if 'model_instances' not in st.session_state:
            st.session_state.model_instances = {}
            
        if model_alias not in st.session_state.model_instances:
            manager = ModelManager()
            st.session_state.model_instances[model_alias] = manager.load_model(model_alias)
            
        return st.session_state.model_instances[model_alias]