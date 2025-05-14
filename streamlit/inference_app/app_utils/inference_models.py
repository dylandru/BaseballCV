from ultralytics import YOLO
from baseballcv.functions import LoadTools
from baseballcv.model import RFDETR, PaliGemma2, YOLOv9, Florence2
import supervision as sv
import numpy as np
from PIL import Image
from typing import Any
import streamlit as st
import cv2
from abc import ABC, abstractmethod
import os
from .file_manager import File

class InferenceModel(ABC):
    def __init__(self, dir) -> None:
        self.tools = LoadTools()
        self.dir = dir
        self.file_manager = File()
        st.warning("Loading Model. This may take a moment.")

    def _handle_uploaded_file(self, file):
        """Helper method to handle Streamlit UploadedFile objects"""
        if hasattr(file, 'getvalue'):  # Streamlit UploadedFile
            file_path = self.dir / 'working_files' / 'img' / file.name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())
            return str(file_path), True
        return str(file), False

    @abstractmethod
    def infer_image(self, file) -> Any:
        pass
    
    @abstractmethod
    def infer_video(self, *args, **kwargs) -> None:
        pass

class YOLOInference(InferenceModel):
    def __init__(self, input_alias, dir, **kwargs):
        super().__init__(dir)
        self.confidence = kwargs.get('confidence', 0.5)
        self.model_path = self.tools.load_model(input_alias, 'YOLO', dir)
        self.model = YOLO(self.model_path)

    def infer_image(self, file):
        input_path, is_temp = self._handle_uploaded_file(file)
        try:
            img = np.array(Image.open(input_path))
            pred = self.model(img, conf=self.confidence)[0].plot()
            return pred
        finally:
            if is_temp and os.path.exists(input_path):
                os.unlink(input_path)
    
    def infer_video(self, out: cv2.VideoWriter, cap: cv2.VideoCapture, length: int, *args):
        for arg in args:
            if callable(arg):
                styling = arg
                styling()
                break

        progress_bar = st.progress(0)
        for i in range(length):
            read, frame = cap.read()

            if read:
                results = self.model(frame, stream=True, device='mps', conf=self.confidence)
                for result in results:
                    out.write(result.plot())
                
            progress_bar.progress(int((i + 1) / length * 100), text=f"Processing Frames: {i+1}/{length}")
                
        cap.release()
        out.release()
        progress_bar.empty()

class YOLOv9Inference(InferenceModel):
    def __init__(self, input_alias, dir, **kwargs):
        super().__init__(dir)
        self.confidence = kwargs.get('confidence', 0.5)
        model_path = self.tools.load_model(input_alias, 'YOLO', output_dir=str(self.dir))
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        self.model = YOLOv9(
            device='cpu',
            model_path=model_path,
            name=model_name,
            custom_weights=True
        )
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def infer_image(self, file):
        input_path, is_temp = self._handle_uploaded_file(file)
        try:
            results = self.model.inference(
                input_path, 
                conf_thres=self.confidence
            )

            img = cv2.imread(input_path) # Load image in BGR format

            predictions = results.get("predictions", [])
            if not predictions:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            xyxys = np.array([
                [int(p["x"]), int(p["y"]), int(p["x"]) + int(p["width"]), int(p["y"]) + int(p["height"])]
                for p in predictions
            ])
            confidences = np.array([p["confidence"] for p in predictions])
            class_ids = np.array([p["class_id"] for p in predictions]) 
            class_names_list = [p["class"] for p in predictions]

            detections = sv.Detections(
                xyxy=xyxys,
                confidence=confidences,
                class_id=class_ids,
                data={'class_name': class_names_list} # Pass list of strings, supervision handles it
            )

            annotated_image = self.box_annotator.annotate(scene=img.copy(), detections=detections)
            annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections)
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            return annotated_image_rgb
            
        finally:
            if is_temp and os.path.exists(input_path):
                os.unlink(input_path)
    
    def infer_video(self, out: cv2.VideoWriter, cap: cv2.VideoCapture, length: int, *args):
        for arg in args:
            if callable(arg):
                styling = arg
                styling()
                break

        try:
            progress_bar = st.progress(0)
            for i in range(length):
                read, frame = cap.read()

                if read:
                    results = self.model.inference(
                        frame, 
                        conf_thres=self.confidence,
                        nosave=True
                    )
                    
                    detections = []
                    for pred in results["predictions"]:
                        x, y = pred["x"], pred["y"]
                        w, h = pred["width"], pred["height"]
                        conf = pred["confidence"]
                        cls_id = pred["class_id"]
                        
                        detections.append(sv.Detection(
                            xyxy=np.array([x, y, x + w, y + h]),
                            confidence=conf,
                            class_id=cls_id,
                            tracker_id=None
                        ))
                    
                    if detections:
                        detections = sv.Detections.from_yolo(detections)
                        frame = self.box_annotator.annotate(scene=frame, detections=detections)
                        frame = self.label_annotator.annotate(scene=frame, detections=detections)
                    
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                progress_bar.progress(int((i + 1) / length * 100), text=f"Processing Frames: {i+1}/{length}")
                    
            cap.release()
            out.release()
            progress_bar.empty()
        except Exception as e:
            st.error(f"Error during video inference: {e}")
            raise e

class RFDETRInference(InferenceModel):
    def __init__(self, input_alias, dir, **kwargs):
        super().__init__(dir)
        self.confidence = kwargs.get('confidence', 0.5)
        model_path = self.tools.load_model(input_alias, 'RFDETR', use_bdl_api=True, output_dir=str(dir))
        self.model = RFDETR(model_path=model_path, 
                          labels=['Glove'],
                          project_path=str(dir),
                          imgsz=640)

    def infer_image(self, file):
        input_path, is_temp = self._handle_uploaded_file(file)
        try:
            pred = self.model.inference(input_path, conf=self.confidence, save_viz=False)[0]
            img = cv2.imread(input_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = sv.BoxAnnotator().annotate(scene=img, detections=pred)
            img = sv.LabelAnnotator().annotate(scene=img, detections=pred)

            return img
        finally:
            if is_temp and os.path.exists(input_path):
                os.unlink(input_path)
    
    def infer_video(self, *args, **kwargs):
        st.error("Video inference not supported for RFDETR yet")

class PaligemmaInference(InferenceModel):
    def __init__(self, input_alias, dir, **kwargs):
        super().__init__(dir)
        model_path = self.tools.load_model(input_alias, 'PALIGEMMA2')
        self.model = PaliGemma2(model_id=model_path,
                              model_run_path=str(dir))

    def infer_image(self, file):
        input_path, is_temp = self._handle_uploaded_file(file)
        try:
            return self.model.inference(input_path)
        finally:
            if is_temp and os.path.exists(input_path):
                os.unlink(input_path)
    
    def infer_video(self, *args, **kwargs):
        st.error("Video inference not supported for PaliGemma2 yet")

class FlorenceInference(InferenceModel):
    def __init__(self, input_alias, dir, **kwargs):
        super().__init__(dir)
        model_path = self.tools.load_model(input_alias, 'FLORENCE2')
        self.model = Florence2(model_id=model_path,
                             model_run_path=str(dir))

    def infer_image(self, file):
        input_path, is_temp = self._handle_uploaded_file(file)
        try:
            return self.model.inference(input_path)
        finally:
            if is_temp and os.path.exists(input_path):
                os.unlink(input_path)
    
    def infer_video(self, *args, **kwargs):
        st.error("Video inference not supported for Florence2 yet")