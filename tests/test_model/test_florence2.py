import pytest
import torch
import os
from unittest import mock
from PIL import Image
from baseballcv.model.vlm.florence2.florence2 import Florence2

class TestFlorence2:
    """Minimal test cases for Florence2 model."""

    @pytest.fixture
    def setup(self):
        """Set up minimal test environment."""
        test_image = Image.new('RGB', (224, 224), color='white')
        
        model_params = {
            'model_id': 'microsoft/Florence-2-large',
            'batch_size': 1,
            'model_run_path': 'florence2_test_run'
        }
        
        return {
            'test_image': test_image,
            'model_params': model_params
        }

    def test_model_initialization(self, setup, monkeypatch):
        """Test model initialization and device selection."""
        def mock_init_model(self):
            from transformers import AutoModelForCausalLM, AutoProcessor
            self.model = mock.MagicMock(spec=AutoModelForCausalLM)
            self.processor = mock.MagicMock(spec=AutoProcessor)
            
        monkeypatch.setattr(Florence2, '_init_model', mock_init_model)
        monkeypatch.setattr(Florence2, '_orig_logging', lambda self: None)
        
        model = Florence2(
            model_id=setup['model_params']['model_id'],
            batch_size=setup['model_params']['batch_size'],
            model_run_path=setup['model_params']['model_run_path']
        )
        
        assert model is not None, "Florence2 model should initialize"
        assert hasattr(model, 'device'), "Model should have device attribute"
        
        with monkeypatch.context() as m:
            m.setattr(torch.cuda, 'is_available', lambda: True)
            m.setattr(torch.backends.mps, 'is_available', lambda: False)
            
            model = Florence2(
                model_id=setup['model_params']['model_id'],
                batch_size=setup['model_params']['batch_size'],
                model_run_path=setup['model_params']['model_run_path']
            )
            
            assert str(model.device) == "cuda", "Should select CUDA when available"
            
        with monkeypatch.context() as m:
            # Mock device availability
            m.setattr(torch.cuda, 'is_available', lambda: False)
            m.setattr(torch.backends.mps, 'is_available', lambda: True)
            
            model = Florence2(
                model_id=setup['model_params']['model_id'],
                batch_size=setup['model_params']['batch_size'],
                model_run_path=setup['model_params']['model_run_path']
            )
            
            assert str(model.device) == "mps", "Should select MPS when available and CUDA is not"

    def test_inference(self, setup, monkeypatch):
        """Test basic inference functionality."""
        def mock_init_model(self):
            self.model = mock.MagicMock()
            self.processor = mock.MagicMock()
            self.model.eval = mock.MagicMock(return_value=None)
            self.model.generate = mock.MagicMock(return_value=torch.tensor([[1, 2, 3, 4]]))
            self.processor.batch_decode = mock.MagicMock(return_value=["Generated text"])
            self.processor.post_process_generation = mock.MagicMock(
                return_value={"<OD>": {"bboxes": [], "labels": []}}
            )
            
        monkeypatch.setattr(Florence2, '_init_model', mock_init_model)
        monkeypatch.setattr(Florence2, '_orig_logging', lambda self: None)
        monkeypatch.setattr(Florence2, '_visualize_results', lambda self, image, results: None)
        monkeypatch.setattr(Florence2, '_return_clean_text_output', lambda self, results: "Clean text output")
        
        model = Florence2(
            model_id=setup['model_params']['model_id'],
            batch_size=setup['model_params']['batch_size'],
            model_run_path=setup['model_params']['model_run_path']
        )
        
        temp_path = "temp_test_image.jpg"
        setup['test_image'].save(temp_path)
        
        try:
            with mock.patch('PIL.Image.open', return_value=setup['test_image']):
                result = model.inference(
                    image_path=temp_path,
                    task="<OD>"
                )
                
                assert result is not None, "Inference should return a result"
                
            with mock.patch('PIL.Image.open', return_value=setup['test_image']):
                with mock.patch('builtins.print') as mock_print:  # Mock print to avoid console output
                    result = model.inference(
                        image_path=temp_path,
                        task="<CAPTION>"
                    )
                    
                    assert result is not None, "Inference should return a result"
                    mock_print.assert_called_once_with("Clean text output")
                
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
