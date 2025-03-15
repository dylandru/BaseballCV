import pytest
import torch
import os
from unittest import mock
from PIL import Image
from baseballcv.model import PaliGemma2

class TestPaliGemma2:
    """Minimal test cases for PaliGemma2 model."""
    
    @pytest.fixture
    def setup(self):
        """Set up minimal test environment."""
        test_image = Image.new('RGB', (224, 224), color='white')
        
        model_params = {
            'model_id': 'google/paligemma2-3b-pt-224',
            'batch_size': 1,
            'torch_dtype': torch.float32,
            'device': 'cpu'
        }
        
        test_text = "What is in this image?"
        
        return {
            'test_image': test_image,
            'test_text': test_text,
            'model_params': model_params
        }

    def test_model_initialization(self, setup, monkeypatch):
        """Test model initialization and device selection."""
        def mock_init_model(self):
            self.processor = mock.MagicMock()
            self.model = mock.MagicMock()
            if hasattr(self, 'logger'):
                self.logger.info("Model Initialization Successful!")
            
        monkeypatch.setattr(PaliGemma2, '_init_model', mock_init_model)
        
        model = PaliGemma2(
            device=setup['model_params']['device'],
            model_id=setup['model_params']['model_id'],
            batch_size=setup['model_params']['batch_size'],
            torch_dtype=setup['model_params']['torch_dtype']
        )
        
        assert model is not None, "PaliGemma2 model should initialize"
        assert hasattr(model, 'model'), "Model should have model attribute"
        assert hasattr(model, 'processor'), "Model should have processor attribute"
        
        with monkeypatch.context() as m:
            m.setattr(torch.cuda, 'is_available', lambda: True)
            m.setattr(torch.backends.mps, 'is_available', lambda: False)
            
            model = PaliGemma2(
                model_id=setup['model_params']['model_id'],
                batch_size=setup['model_params']['batch_size'],
                torch_dtype=setup['model_params']['torch_dtype']
            )
            
            assert str(model.device) == "cuda", "Should select CUDA when available"

    def test_inference(self, setup, monkeypatch):
        """Test basic inference functionality."""
        def mock_init_model(self):
            self.processor = mock.MagicMock()
            self.model = mock.MagicMock()
            self.model.eval = mock.MagicMock(return_value=self.model)
            self.model.generate = mock.MagicMock(return_value=[torch.tensor([1, 2, 3, 4])])
            self.processor.decode = mock.MagicMock(return_value="Generated text response")
            if hasattr(self, 'logger'):
                self.logger.info("Model Initialization Successful!")
            
        monkeypatch.setattr(PaliGemma2, '_init_model', mock_init_model)
        
        model = PaliGemma2(
            device=setup['model_params']['device'],
            model_id=setup['model_params']['model_id'],
            batch_size=setup['model_params']['batch_size'],
            torch_dtype=setup['model_params']['torch_dtype']
        )
        
        temp_path = "temp_test_image.jpg"
        setup['test_image'].save(temp_path)
        
        try:
            with mock.patch('PIL.Image.open', return_value=setup['test_image']):
                result = model.inference(
                    image_path=temp_path,
                    text_input=setup['test_text'],
                    task="<TEXT_TO_TEXT>"
                )
                
                assert result is not None, "Inference should return a result"
                
            with mock.patch('PIL.Image.open', return_value=setup['test_image']), \
                 mock.patch('supervision.Detections.from_lmm'), \
                 mock.patch('supervision.BoxAnnotator'), \
                 mock.patch('supervision.LabelAnnotator'), \
                 mock.patch('os.makedirs'):
                
                result, image_path = model.inference(
                    image_path=temp_path,
                    text_input="detect all objects",
                    task="<TEXT_TO_OD>",
                    classes=["baseball", "glove"]
                )
                
                assert result is not None, "Inference should return a result"
                assert image_path is not None, "Inference should return an image path"
                
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
