from ultralytics import YOLO
from baseballcv.functions import LoadTools
import numpy as np
from pathlib import Path
import os
import tempfile
from PIL import Image
from typing import List, Any
import streamlit as st
import cv2
from abc import ABC, abstractmethod

class InferenceModel(ABC):
    def __init__(self) -> None:
        working_dir = Path().cwd()
        if 'BaseballCV' in str(working_dir): # Not sure the current working directory streamlit will have when published
            working_dir = working_dir / 'streamlit' / 'inference_app'

        self.dir =  working_dir / 'models'
        self.tools = LoadTools()
    
    @abstractmethod
    def infer_image(self) -> Any:
        pass

    @abstractmethod
    def infer_video(self) -> str:
        pass

    @staticmethod
    def create_temp_video_file(file): # Static method so it's not called when class is instantiated
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(file.read())
            tmp_file_path = tmp_file.name
        
        return tmp_file_path


class YOLOInference(InferenceModel):
    def __init__(self, model_alias, confidence):
        super().__init__()
        self.confidence = confidence
        self.model = YOLO(self.tools.load_model(model_alias, 'YOLO', output_dir=self.dir))
    
    def infer_image(self, files):
        preds = []
        for file in files:
            img = np.array(Image.open(file))
            preds.append(self.model(img, conf=self.confidence)[0].plot())

        return preds
    
    def infer_video(self, files):
        # Will probably move this somewhere, depending on if the other models have the same strucutre
        # Right now, the only way to get a video to work is to write it to a temp mp4 file, display and remove it
        tmp_file_path = self.create_temp_video_file(files[0])

        cap = cv2.VideoCapture(tmp_file_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

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

        os.remove(tmp_file_path)
        
        return out_path
