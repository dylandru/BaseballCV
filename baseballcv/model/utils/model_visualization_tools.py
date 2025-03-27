import os
import cv2
import matplotlib.pyplot as plt
from typing import List
import logging
import supervision as sv


class ModelVisualizationTools:
    '''
    A class to visualize the results of a model.
    '''
    def __init__(self, model_name: str, model_run_path: str, logger: logging.Logger) -> None:
        """
        Initialize the ModelVisualizationTools class.

        Args:
            model_name (str): The name of the model.
            model_run_path (str): The path to the model run directory.
            logger (logging.Logger): The logger instance for logging.
        """
        self.model_name = model_name
        self.model_run_path = model_run_path
        self.logger = logger

    def visualize_detection_results(self, file_path: str, detections: sv.Detections, labels: List[str], save: bool = True, save_viz_dir: str = 'visualizations') -> logging.Logger:
        """
        Visualize the results.

        Args:
            file_path (str): The path to the image file.
            detections (sv.Detections): Detections object containing boxes, class IDs and confidence scores
            labels (List[str]): List of formatted label strings
            save_viz_dir (str): Directory to save the visualizations.

        Returns:
            logger_message (logging.Logger): The logger message for logging the completed visualization saving.
        """
        if isinstance(file_path, str):
            image = cv2.imread(file_path)
        else:
            image = file_path

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        ax.axis('off')

        if save:
            os.makedirs(save_viz_dir, exist_ok=True)
            
            if isinstance(file_path, str):
                base_name = os.path.basename(file_path)
                save_name = os.path.splitext(base_name)[0]
            else:
                save_name = 'frame'
                
            save_path = os.path.join(save_viz_dir, f'detection_{save_name}.jpg')
            
            cv2.imwrite(save_path, annotated_frame)
            self.logger.info(f"Visualizations saved to {save_path}")
        
        plt.show()
        plt.close()

        return (save_path, annotated_frame) if save else annotated_frame


        

    