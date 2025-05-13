from ultralytics import YOLO
from baseballcv.functions import LoadTools
from baseballcv.model import RFDETR, PaliGemma2, YOLOv9, Florence2
import numpy as np
from PIL import Image
from typing import List, Any
import streamlit as st
import cv2
from abc import ABC, abstractmethod

class InferenceModel(ABC):
    def __init__(self) -> None:
        self.tools = LoadTools()

        st.warning("Loading Model. This may take a moment.")

    @abstractmethod
    def infer_image(self, file) -> Any:
        pass
    
    @abstractmethod
    def infer_video(self, *args, **kwargs) -> None:
        pass

class YOLOInference(InferenceModel):
    def __init__(self, input_alias, dir, **kwargs):
        super().__init__()

        self.confidence = kwargs.get('confidence') # this will always have a value

        self.model = YOLO(self.tools.load_model(input_alias, 'YOLO', output_dir=dir))

    def infer_image(self, file):
        img = np.array(Image.open(file))
        pred = self.model(img, conf=self.confidence)[0].plot()
        return pred
    
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
                results = self.model(frame, stream=True, device='mps', conf=self.confidence) # Does streamlit support GPU?

                for result in results:
                    out.write(result.plot())
                
            progress_bar.progress(int((i + 1) / length * 100), text=f"Processing frames {i+1}/{length}")
                
        cap.release()
        out.release()
        progress_bar.empty()

class YOLOv9Inference(InferenceModel):
    """
    Don't think I did this properly.
    """
    def __init__(self, input_alias, dir, **kwargs):
        super().__init__()
        self.dir = dir

        self.confidence = kwargs.get('confidence')

        path = self.tools.load_model(input_alias, 'YOLO', output_dir=self.dir / 'weights')

        self.model = YOLOv9(device = 'cpu', name=input_alias, model_path=dir)

    def infer_image(self, file):
        pred = self.model.inference(file, conf_thres=self.confidence, project=self.dir / 'predict') # This file will need to be removed
        img = cv2.imread(file)

        print(pred)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(pred.xyxy)):

            x1, y1, x2, y2 = map(int, pred.xyxy[i])
            conf = float(pred.confidence[i])
            cls_id = int(pred.class_id[i])

            if cls_id != 0:
                continue

            label = f"{cls_id} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(173, 216, 230), thickness=10)

            cv2.putText(img, label, (x1, max(y1 - 10, 0)), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                        color=(173, 216, 230), thickness=2)

        return img
    
    def infer_video(self):
        st.error("Doesn't support this yet")

class RFDETRInference(InferenceModel):
    """
    This model is causing issues to skipping for now
    """
    def __init__(self, input_alias, dir, **kwargs):
        super().__init__()

        self.confidence = kwargs.get('confidence') # this will always have a value


        self.model = RFDETR(model_path=self.tools.load_model(input_alias, 'RFDETR', output_dir=dir), 
                            labels=['Glove'],
                            project_path=dir)

    def infer_image(self, file):
        pred = self.model.inference(file, conf=self.confidence, save_viz=False)[0] # It's a still image, so just duplicated

        img = cv2.imread(file)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(pred.xyxy)):

            x1, y1, x2, y2 = map(int, pred.xyxy[i])
            conf = float(pred.confidence[i])
            cls_id = int(pred.class_id[i])

            if cls_id != 0:
                continue

            label = f"{cls_id} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(173, 216, 230), thickness=10)

            cv2.putText(img, label, (x1, max(y1 - 10, 0)), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                        color=(173, 216, 230), thickness=2)

        return img
    
    def infer_video(self):
        st.error("Not supporting video inferences for now.")

class PaligemmaInference(InferenceModel):
    """
    Skipping this model for now due to load tools issues
    """
    def __init__(self, input_alias, dir, **kwargs):
        super().__init__()

        self.model = PaliGemma2(model_id=self.tools.load_model(input_alias, 'PALIGEMMA2', output_dir=dir),
                                 model_run_path=dir)

    def infer_image(self, file):
        pred = self.model.inference(file)

        return pred
        
    
    def infer_video(self, *args, **kwargs):
        st.error("Not supporting video inferences for now.")

class FlorenceInference(InferenceModel):
    def __init__(self, input_alias, dir):
        super().__init__()

        self.model = PaliGemma2(model_id=self.tools.load_model(input_alias, 'FLORENCE2', output_dir=dir),
                                 model_run_path=dir)

    def infer_image(self, file):
        return super().infer_image(file)
    
    def infer_video(self, *args, **kwargs):
        return super().infer_video(*args, **kwargs)