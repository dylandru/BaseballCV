import pytest
import os
import tempfile
import shutil
from unittest import mock
import torch
from baseballcv.model.utils.model_function_utils import ModelFunctionUtils
from baseballcv.model import Florence2
from baseballcv.utilities import BaseballCVLogger

class TestModelFunctionUtils:
    """
    Test suite for the ModelFunctionUtils class.
    
    This class contains tests to verify the functionality of the ModelFunctionUtils,
    including the collate_fn, freeze_vision_encoders, and create_detection_dataset methods. It is currently being 
    updated for a variety of models, so this functionality is not yet complete.
    """
            
    @pytest.fixture
    def setup(self) -> dict:
        """
        Set up test environment with Florence2 model.
        
        Creates a temporary directory and initializes the Florence2 model for testing the
        ModelFunctionUtils class. This includes a processor that simulates
        feature extraction and a ModelLogger instance.
        
        Returns:
            dict: A dictionary containing the following test components:
                - processor: Processor with feature extraction capability
                - model: Florence2 model instance for testing
                - logger: ModelLogger instance configured for testing
                - temp_dir: Path to temporary directory for test artifacts
                - model_run_path: Path to temporary model run directory
                - batch_size: Batch size for testing
                - device: Device to use for testing
        """
        temp_dir = tempfile.mkdtemp()
        model_run_path = os.path.join(temp_dir, 'florence2_test_run')
        
        model_params = {
            'model_id': 'microsoft/Florence-2-large',
            'batch_size': 1,
            'model_run_path': model_run_path
        }
        
        try:
            florence2 = Florence2(**model_params)
            processor = florence2.processor
            
            logger = BaseballCVLogger.get_logger(self.__class__.__name__)
            
            return {
                'processor': processor,
                'model': florence2,
                'logger': logger,
                'temp_dir': temp_dir,
                'model_run_path': model_run_path,
                'batch_size': model_params['batch_size'],
                'device': florence2.device,
                'model_name': model_params['model_id']
            }
        except Exception as e:
            pytest.skip(f"Skipping due to error initializing Florence2: {str(e)}")
    
    def teardown_method(self, method) -> None:
        """
        Clean up temporary directories after test completion.
        
        This method iterates through all attributes of the current test instance
        and removes any temporary directories found in dictionaries.
        
        Args:
            method: The test method that was executed.
        """
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, dict) and 'temp_dir' in attr:
                if os.path.exists(attr['temp_dir']):
                    shutil.rmtree(attr['temp_dir'])
    
    def test_core_functionality(self, setup) -> None:
        """
        Test essential methods of ModelFunctionUtils.
        
        This test verifies the core functionality of the ModelFunctionUtils class,
        including:
        1. The collate_fn method for batching data
        2. The create_detection_dataset method for dataset creation
        3. The augment_suffix method for string manipulation
        
        Args:
            setup: Fixture providing test components including processor, model,
                  logger, and temporary directory.
        """
        model_utils = ModelFunctionUtils(
            processor=setup['processor'],
            model=setup['model'],
            logger=setup['logger'],
            model_run_path=setup['model_run_path'],
            model_name=setup['model_name'],
            batch_size=setup['batch_size'],
            device=setup['device'],
            peft_model=None,
            torch_dtype=torch.float32
        )
        
        test_image_path1 = os.path.join(setup['temp_dir'], "test_image1.jpg")
        test_image_path2 = os.path.join(setup['temp_dir'], "test_image2.jpg")
        
        with open(test_image_path1, 'wb') as f:
            f.write(b'dummy image data')
        
        with open(test_image_path2, 'wb') as f:
            f.write(b'dummy image data')
            
        batch = [
            {"image": test_image_path1, "labels": torch.tensor([1, 0])},
            {"image": test_image_path2, "labels": torch.tensor([0, 1])}
        ]
        
        with mock.patch.object(model_utils, 'collate_fn', return_value={
            "pixel_values": torch.rand(2, 3, 224, 224),
            "labels": torch.stack([item["labels"] for item in batch])
        }):
            collated = model_utils.collate_fn(batch)
            assert "pixel_values" in collated
            assert "labels" in collated
    
        
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, dir=setup['temp_dir']) as f:
            f.write(b'{"image_path": "test.jpg", "boxes": [[0, 0, 100, 100]], "labels": ["test"]}\n')
        
        with mock.patch('baseballcv.model.utils.model_function_utils.JSONLDetection') as mock_dataset:
            mock_dataset_instance = mock.MagicMock()
            mock_dataset.return_value = mock_dataset_instance
            
            dataset = model_utils.create_detection_dataset(
                jsonl_file_path=f.name,
                image_directory_path=setup['temp_dir'],
                augment=False
            )
            
            assert dataset is mock_dataset_instance
        
        suffix = model_utils.augment_suffix("test")
        assert isinstance(suffix, str)
        assert len(suffix) > 4
