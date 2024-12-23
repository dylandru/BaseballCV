import os
import logging
from datetime import datetime
import sys

class ModelLogger:
    def __init__(self, model_name: str, model_run_path: str, model_id: str, batch_size: int, device: str) -> None:
        self.model_name = model_name
        self.model_run_path = model_run_path
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device

    def orig_logging(self) -> logging.Logger:
        """
        Set up logging for a given model.

        Returns:
            logging.Logger: The logger.
        """
        log_dir = os.path.join(self.model_run_path, "logs")
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{self.model_name}_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[  
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(f"{self.model_name}_({self.model_id})")
        self.logger.info(f"Initializing {self.model_name} model with Batch Size: {self.batch_size}")
        self.logger.info(f"Device: {self.device}")
        return self.logger
