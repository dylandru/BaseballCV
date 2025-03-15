import pytest
import torch
import os
from unittest import mock
from baseballcv.model import YOLOv9

class TestYOLOv9:
    """Minimal test cases for YOLOv9 model."""

    @pytest.fixture
    def setup(self):
        """Set up minimal test environment."""
        # Create a test input tensor
        test_input = torch.rand(1, 3, 640, 640)
        
        # Basic model parameters
        model_params = {
            'name': 'yolov9-c',
            'device': 'cpu',
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45
        }
        
        return {
            'test_input': test_input,
            'model_params': model_params
        }

    def test_model_initialization(self, setup, monkeypatch):
        """Test model initialization and device selection."""
        # Mock the YOLO module to avoid actual model loading
        mock_yolo = mock.MagicMock()
        monkeypatch.setattr('ultralytics.YOLO', lambda model_path: mock_yolo)
        
        # Test 1: Initialize with specified device
        model = YOLOv9(
            device=setup['model_params']['device'],
            name=setup['model_params']['name']
        )
        
        assert model is not None, "YOLOv9 model should initialize"
        assert model.device == setup['model_params']['device'], "Device should be set correctly"
        
        # Test 2: Test CUDA device selection
        with monkeypatch.context() as m:
            # Mock torch.cuda.is_available to return True
            m.setattr(torch.cuda, 'is_available', lambda: True)
            
            # Initialize with 'auto' device to test auto-detection
            model = YOLOv9(name=setup['model_params']['name'])
            
            assert model.device == 'cuda:0', "Should select CUDA when available"
            
        # Test 3: Test CPU fallback
        with monkeypatch.context() as m:
            # Mock torch.cuda.is_available to return False
            m.setattr(torch.cuda, 'is_available', lambda: False)
            
            # Initialize with 'auto' device to test fallback
            model = YOLOv9(name=setup['model_params']['name'])
            
            assert model.device == 'cpu', "Should fall back to CPU when CUDA is not available"

    def test_inference(self, setup, monkeypatch):
        """Test basic inference functionality."""
        # Create mock YOLO model that returns expected format for inference
        mock_yolo = mock.MagicMock()
        mock_results = mock.MagicMock()
        mock_results.boxes.xyxy = torch.tensor([[10, 20, 110, 120]])
        mock_results.boxes.conf = torch.tensor([0.95])
        mock_results.boxes.cls = torch.tensor([0])
        mock_yolo.predict.return_value = [mock_results]
        
        monkeypatch.setattr('ultralytics.YOLO', lambda model_path: mock_yolo)
        
        model = YOLOv9(
            device=setup['model_params']['device'],
            name=setup['model_params']['name'],
            confidence_threshold=setup['model_params']['confidence_threshold'],
            iou_threshold=setup['model_params']['iou_threshold']
        )
        
        temp_path = "temp_test_image.jpg"
        with open(temp_path, 'w') as f:
            f.write('dummy image')
        
        try:
            with mock.patch('PIL.Image.open'):
                result = model.inference(
                    image_path=temp_path,
                    save_dir=None
                )
                
                assert result is not None, "Inference should return results"
                mock_yolo.predict.assert_called_once()
                
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
