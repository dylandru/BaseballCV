import os
import cv2
import torch
import matplotlib.pyplot as plt
from typing import Dict, List
from tqdm import tqdm
from PIL import Image
import logging


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

    def visualize_detection_results(self, file_path: str, results: Dict, save: bool = True, save_viz_dir: str = 'visualizations') -> logging.Logger:
        """
        Visualize the results.

        Args:
            file_path (str): The path to the image file.
            results (Dict): Dictionary containing the results.
            save_viz_dir (str): Directory to save the visualizations.

        Returns:
            logger_message (logging.Logger): The logger message for logging the completed visualization saving.
        """
        image = plt.imread(file_path)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)

        for detection in results:
            bbox = detection['boxes']
            label = detection['labels']
            score = detection['scores']
            xmin, ymin, xmax, ymax = bbox
            
            rect = plt.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor='red',
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(
                xmin,
                ymin - 2,
                f"{label} ({score:.2f})",
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=12,
                color='white',
                fontweight='bold'
            )

        ax.axis('off')

        if save:
            os.makedirs(save_viz_dir, exist_ok=True)
            
            if isinstance(file_path, str):
                base_name = os.path.basename(file_path)
                save_name = os.path.splitext(base_name)[0]
            else:
                save_name = 'frame'
                
            save_path = os.path.join(save_viz_dir, f'detection_{save_name}.jpg')
            
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300, facecolor='auto', edgecolor='auto')
            self.logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
        plt.close()

        return save_path if save else None


        

    