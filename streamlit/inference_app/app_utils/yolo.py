from ultralytics import YOLO
from baseballcv.functions import LoadTools
import numpy as np

class YOLOInference:
    def __init__(self, model_alias: str = 'phc_detector'):
        self.tools = LoadTools()

        # What I can do is make a new parameter to load tools to specify the path to load this model so it doesn't
        # conflict with the models folder.
        self.model = YOLO(self.tools.load_model(model_alias=model_alias, model_type='YOLO'))

    def inference(self, img: np.array):
        return self.model(img)