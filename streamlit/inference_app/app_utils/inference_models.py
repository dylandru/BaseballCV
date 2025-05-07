from ultralytics import YOLO
from baseballcv.functions import LoadTools
import numpy as np
from PIL import Image
from typing import List, Any
import streamlit as st
import cv2
from abc import ABC, abstractmethod

class InferenceModel(ABC):
    def __init__(self) -> None:
        self.tools = LoadTools()
    
    @abstractmethod
    def infer_image(self) -> Any:
        pass

    @abstractmethod
    def infer_video(self) -> None:
        pass

class YOLOInference(InferenceModel):
    def __init__(self, model_alias, confidence, dir):
        super().__init__()
        self.confidence = confidence
        st.warning("Loading Model. This may take a moment.")
        self.model = YOLO(self.tools.load_model(model_alias, 'YOLO', output_dir=dir))
    
    def infer_image(self, file):
        img = np.array(Image.open(file))
        pred = self.model(img, conf=self.confidence)[0].plot()
        return pred
    
    def infer_video(self, out: cv2.VideoWriter, cap: cv2.VideoCapture, length: int):
        progress_bar = st.progress(0)
        for i in range(length):
            read, frame = cap.read()

            if read:
                results = self.model(frame, stream=True, device='mps', conf=self.confidence) # Does streamlit support GPU?

                for result in results:
                    out.write(result.plot())
                
            progress_bar.progress(int((i + 1) / length * 100), text=f"Processing frames {i+1}/{length}")
                
        cap.release()
        out.release()
        progress_bar.empty()
