from load_tools import LoadTools
from ultralytics import YOLO
from datetime import datetime
import cv2
import streamlit as st

__all__ = ['ModelManager']

class ModelManager:
    def __init__(self, conf: float = 0.8):
        self.load_tools = LoadTools()
        self.conf = conf
        
    @st.cache_resource(show_spinner="Loading model...")
    def load_model(_self, model_alias: str) -> YOLO: #TODO: Adjust based on More Model Types
        model_path = _self.load_tools.load_model(model_alias=model_alias)
        model = YOLO(model_path)
        return model
        
    def predict_image(self, image_path: str, model: YOLO) -> list[dict]:
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