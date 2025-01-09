import os
import matplotlib.pyplot as plt
from typing import Dict
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

    def visualize_results(self, image: Image.Image, results: Dict, save_viz_dir: str = 'visualizations') -> logging.Logger:
        """
        Visualize the results.

        Args:
            image (PIL.Image.Image): The input image.
            results (Dict): Dictionary containing the results.
            save_viz_dir (str): Directory to save the visualizations.

        Returns:
            logger_message (logging.Logger): The logger message for logging the completed visualization saving.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        ax = plt.gca()

        for bbox, label in zip(results['bboxes'], results['labels']):
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
                label,
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=12,
                color='white'
            )

        plt.axis('off')
        plt.show()
        os.makedirs(save_viz_dir, exist_ok=True)
        plt.savefig(os.path.join(self.model_run_path, save_viz_dir, 'result.png'))
        plt.close()
        return self.logger.info(f"Visualization saved to {os.path.join(self.model_run_path, save_viz_dir, 'result.png')}")

    