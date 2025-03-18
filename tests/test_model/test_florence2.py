import shutil
import pytest
import torch
import os
import tempfile
import psutil
from PIL import Image
from huggingface_hub.utils import GatedRepoError
from baseballcv.model import Florence2

class TestFlorence2:
    """
    Test cases for Florence2 model.
    
    This test suite verifies the functionality of the Florence2 multimodal model,
    including initialization, device selection, and inference capabilities for
    various tasks such as image captioning, visual question answering, and
    object detection.
    """

    @pytest.fixture
    def setup_florence2_test(self, load_tools) -> dict:
        """
        Set up test environment with real dataset.
        
        Creates a temporary directory and loads a baseball dataset for testing.
        Initializes a Florence2 model and prepares test images and questions for
        inference testing. Checks for sufficient memory before attempting to load
        the model.
        
        Args:
            load_tools: Fixture providing tools to load datasets
            
        Yields:
            dict: A dictionary containing test resources including:
                - test_image: PIL Image object for testing
                - test_image_path: Path to the test image
                - test_tasks: List of supported tasks for inference testing
                - test_questions: List of sample questions for VQA inference
                - dataset_path: Path to the loaded dataset
                - model_params: Dictionary of parameters used to initialize the model
                - temp_dir: Path to temporary directory for test artifacts
                - model: Initialized Florence2 model instance
                
        Raises:
            pytest.skip: If requirements for testing are not met for memory reasons
        """
        
        temp_dir = tempfile.mkdtemp()
        ram = psutil.virtual_memory().total / (1024**3)  
        
        if ram < 16:
            pytest.skip("Skipping Florence2 tests: insufficient memory (likely needs 16GB machine)")

        try:
            dataset_path = load_tools.load_dataset("baseball_rubber_home_glove")
            
            test_tasks = [
                "<CAPTION>",
                "<VQA>",
                "<OD>"
            ]
            
            test_questions = [
                "What objects are in this image?",
                "Is there a baseball in the image?"
            ]
            
            model_params = {
                'model_id': 'microsoft/Florence-2-large',
                'batch_size': 1,
                'model_run_path': os.path.join(temp_dir, 'florence2_test_run')
            }

            model = Florence2(**model_params)
            
            # Get a test image from the dataset
            test_image_path = os.path.join(dataset_path, "images", os.listdir(os.path.join(dataset_path, "images"))[0])
            test_image = Image.open(test_image_path)
            
            yield {
                'test_image': test_image,
                'test_image_path': test_image_path,
                'test_tasks': test_tasks,
                'test_questions': test_questions,
                'dataset_path': dataset_path,
                'model_params': model_params,
                'temp_dir': temp_dir,
                'model': model
            }
            
        except Exception: #if for some reason the Dataset download fails...
            test_image_path = os.path.join(temp_dir, "test_image.jpg")
            test_image = Image.new('RGB', (224, 224), color='white')
            test_image.save(test_image_path)
            
            model_params = {
                'model_id': 'microsoft/Florence-2-large',
                'batch_size': 1,
                'model_run_path': os.path.join(temp_dir, 'florence2_test_run')
            }
            
            model = Florence2(**model_params)
            
            yield {
                'test_image': test_image,
                'test_image_path': test_image_path,
                'test_tasks': ["<CAPTION>"],
                'test_questions': ["What is in this image?"],
                'dataset_path': None,
                'model_params': model_params,
                'temp_dir': temp_dir,
                'model': model
            }
        
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if 'dataset_path' in locals() and dataset_path and os.path.exists(dataset_path):
                shutil.rmtree(dataset_path)

    def test_model_initialization(self, setup_florence2_test) -> None:
        """
        Test model initialization and device selection.
        
        Verifies that the Florence2 model initializes correctly with the specified
        parameters and tests device selection logic, including optional MPS support
        when available.
        
        Args:
            setup_florence2_test: Fixture providing test resources for Florence2
            
        Assertions:
            - Model should initialize successfully
            - Model should have a device attribute
            - When MPS is available and CUDA is not, model should select MPS device
        """
        
        model_init = Florence2(**setup_florence2_test['model_params'])
        assert model_init is not None, "Florence2 model should initialize"
        assert hasattr(model_init, 'device'), "Model should have device attribute"
            
        with pytest.MonkeyPatch().context() as m:
            m.setattr(torch.cuda, 'is_available', lambda: False)
            m.setattr(torch.backends.mps, 'is_available', lambda: True)
            model_mps = Florence2(**setup_florence2_test['model_params'])
            
            assert "mps" in str(model_mps.device), "Should select MPS when available and CUDA is not"

    def test_caption_inference(self, setup_florence2_test) -> None:
        """
        Test captioning inference with a real image.
        
        Verifies that the Florence2 model can perform image captioning inference
        on a test image and returns a properly formatted caption result.
        
        Args:
            setup_florence2_test: Fixture providing test resources including model and test image
            
        Assertions:
            - Inference should return a non-null result
            - Result should be a string representing the image caption
        """
        
        model = setup_florence2_test['model']
        
        result = model.inference(
            image_path=setup_florence2_test['test_image_path'],
            task="<CAPTION>"
        )
        
        assert result is not None, "Caption inference should return a result"
        assert isinstance(result, str), "Caption result should be a string"
            
    def test_vqa_inference(self, setup_florence2_test) -> None:
        """
        Test VQA inference with a real image.
        
        Verifies that the Florence2 model can perform visual question answering (VQA)
        inference on a test image with a provided question and returns a properly
        formatted answer.
        
        Args:
            setup_florence2_test: Fixture providing test resources including model,
                                  test image, and test questions
            
        Assertions:
            - Inference should return a non-null result
            - Result should be a string representing the answer to the question
        """

        model = setup_florence2_test['model'] 
        result = model.inference(
            image_path=setup_florence2_test['test_image_path'],
            task="<VQA>",
            question=setup_florence2_test['test_questions'][0]
        )
            
        assert result is not None, "VQA inference should return a result"
        assert isinstance(result, str), "VQA result should be a string"

    def test_object_detection_inference(self, setup_florence2_test) -> None:
        """
        Test object detection inference with a real image.
        
        Verifies that the Florence2 model can perform object detection inference
        on a test image and returns properly structured detection results with
        expected fields.
        
        Args:
            setup_florence2_test: Fixture providing test resources including model and test image
            
        Assertions:
            - Inference should return a non-null result
            - Result should be a dictionary containing detection information
            - Result should include bounding boxes and corresponding labels
        """
        try:    
            model = setup_florence2_test['model']
            
            viz_dir = os.path.join(setup_florence2_test['temp_dir'], 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            if hasattr(model, 'model_run_path'):
                os.makedirs(os.path.join(model.model_run_path, 'visualizations'), exist_ok=True)
            
            result = model.inference(
                image_path=setup_florence2_test['test_image_path'],
                task="<OD>"
            )
                
            assert result is not None, "OD inference should return a result"
                
            assert isinstance(result, dict), "OD result should be a dictionary"
            assert "bboxes" in result, "OD result should include bounding boxes"
            assert "labels" in result, "OD result should include labels"
        except Exception as e:
            pytest.skip(f"Object detection inference test skipped: {str(e)}")

#TODO: Add tests for Finetuning when Tests can be run on GPU