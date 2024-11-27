from load_tools import LoadTools
from ultralytics import YOLO
from datetime import datetime
import cv2
import streamlit as st

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
            self._model_cache[model_alias] = YOLO(model_path)
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
        #TODO: Add support for keypoint predictions

        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        results = model.predict(image_path, conf=self.conf, verbose=True, imgsz=(480, 800))[0] 
        
        annotations = []
        for i, (box, score, cls) in enumerate(zip(results.boxes.xywhn, 
                                                 results.boxes.conf, 
                                                 results.boxes.cls)):
            x, y, w, h = box.tolist()

            x = x * width
            y = y * height
            w = w * width
            h = h * height

            annotations.append({
                "id": i + 1,
                "image_id": 1,
                "category_id": int(cls.item()) + 1,
                "bbox": [x, y, w, h],
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