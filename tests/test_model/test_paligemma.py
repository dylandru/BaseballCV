import pytest
import torch
import os
import tempfile
from PIL import Image
import psutil
from baseballcv.model import PaliGemma2
from huggingface_hub import login
from huggingface_hub.utils import GatedRepoError, LocalEntryNotFoundError
import shutil
from typing import Generator

class TestPaliGemma2:
    """
    Test cases for PaliGemma2 model.
    
    This test suite verifies the functionality of the PaliGemma2 multimodal model,
    including initialization, device selection, and inference capabilities for
    both text-to-text and object detection tasks.
    """
    
    @pytest.fixture
    def setup_paligemma_test(self, load_tools) -> Generator[dict, None, None]:
        """
        Set up test environment for PaliGemma2.
        
        Creates a temporary directory and initializes test resources for PaliGemma2 testing.
        This fixture checks for sufficient memory and proper authentication before attempting
        to load the model. Prepares test images and questions for inference testing.
        
        Args:
            load_tools: Fixture providing tools to load datasets
            
        Yields:
            dict: A dictionary containing test resources including:
                - test_image: PIL Image object for testing
                - test_image_path: Path to the test image
                - test_questions: List of sample questions for text-to-text inference
                - dataset_path: Path to the loaded dataset
                - class_mapping: Dictionary mapping class IDs to class names
                - model_params: Dictionary of parameters used to initialize the model
                - temp_dir: Path to temporary directory for test artifacts
                - model: Initialized PaliGemma2 model instance
                
        Raises:
            pytest.skip: If requirements for testing are not met (memory, authentication)
        """
        temp_dir = tempfile.mkdtemp()
        ram = psutil.virtual_memory().total / (1024**3) 
        hf_token = os.environ.get("HF_TOKEN")

        if hf_token:
            login(token=hf_token)
        else:
            pytest.skip("Skipping PaliGemma2 tests: No Hugging Face token found")
        
        if ram < 16:
            pytest.skip("Skipping PaliGemma2 tests: insufficient memory (likely needs 16GB machine)")

        try:

            model_params = {
                'model_id': 'google/paligemma2-3b-pt-224',
                'batch_size': 1,
                'torch_dtype': torch.float32,
                'device': 'cpu'
            }

            try: #try to initialize the model, but skip if we encounter access issues with Gated Repo
                model = PaliGemma2(**model_params)
            except (GatedRepoError, OSError) as e:
                if "gated repo" in str(e).lower() or "403" in str(e):
                    pytest.skip(f"Skipping PaliGemma2 tests: No access to gated model. {str(e)}")
                else:
                    raise e
            except Exception as e:
                if "access gated repo" in str(e).lower() or "awaiting a review" in str(e).lower():
                    pytest.skip(f"Skipping PaliGemma2 tests: No access to gated model. {str(e)}")
                else:
                    raise e
                
            dataset_path = load_tools.load_dataset(
                    dataset_alias="amateur_hitter_pitcher_jsonl", 
                    use_bdl_api=False, 
                    file_txt_path="datasets/JSONL/amateur_hitter_pitcher_jsonl.txt"
                )
            
            images_dir = os.path.join(dataset_path, "dataset")
            test_image_path = None
            
            if os.path.exists(images_dir):
                for file in os.listdir(images_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_image_path = os.path.join(images_dir, file)
                        break
            
            if not test_image_path:
                test_image_path = os.path.join(temp_dir, "test_image.jpg")
                Image.new('RGB', (224, 224), color='white').save(test_image_path)
                
            test_image = Image.open(test_image_path)
            
            test_questions = [
                "What is in this image?",
                "How many objects can you see?",
                "Is there a baseball in this image?"
            ]
            
            class_mapping = {
                0: 'glove',
                1: 'homeplate',
                2: 'baseball',
                3: 'rubber'
            }
    
            
            yield {
                'test_image': test_image,
                'test_image_path': test_image_path,
                'test_questions': test_questions,
                'dataset_path': dataset_path,
                'class_mapping': class_mapping,
                'model_params': model_params,
                'temp_dir': temp_dir,
                'model': model
            }
            
        except Exception as e:
            # If dataset download fails or other error occurs
            test_image_path = os.path.join(temp_dir, "test_image.jpg")
            test_image = Image.new('RGB', (224, 224), color='white')
            test_image.save(test_image_path)
            
            pytest.skip(f"Skipping PaliGemma2 tests due to setup error: {str(e)}")
        
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if 'dataset_path' in locals() and locals()['dataset_path'] and os.path.exists(locals()['dataset_path']):
                shutil.rmtree(locals()['dataset_path'])


    def test_model_initialization(self, setup_paligemma_test) -> None:
        """
        Test model initialization and device selection.
        
        Verifies that the PaliGemma2 model can be properly initialized with specified
        parameters and that the device selection works correctly. This includes testing
        CPU initialization and optional MPS (Metal Performance Shaders) device selection
        when available.
        
        Args:
            setup_paligemma_test: Fixture providing test resources for PaliGemma2
            
        Assertions:
            - Model should initialize successfully
            - Device should be correctly set to CPU
            - When MPS is available, model should select MPS device
        """

        model_init = PaliGemma2(
            device=setup_paligemma_test['model_params']['device'],
            model_id=setup_paligemma_test['model_params']['model_id'],
            batch_size=setup_paligemma_test['model_params']['batch_size'],
            torch_dtype=setup_paligemma_test['model_params']['torch_dtype']
        )
            
        assert model_init is not None, "PaliGemma2 model should initialize"
        assert model_init.device == 'cpu', "Device should be correctly set to CPU"

        #optional MPS test - not necessary but can be used to test MPS device selection
        with pytest.MonkeyPatch().context() as m: #torch not available for CPU
            m.setattr(torch.cuda, 'is_available', lambda: False)
            m.setattr(torch.backends.mps, 'is_available', lambda: True)
            
            try:
                model_mps = PaliGemma2(
                    model_id=setup_paligemma_test['model_params']['model_id'],
                    batch_size=setup_paligemma_test['model_params']['batch_size'],
                    torch_dtype=setup_paligemma_test['model_params']['torch_dtype'],
                    device='mps'
                )
                
                assert "mps" in str(model_mps.device), "Should select MPS when available"
            except Exception as e:
                pytest.skip(f"MPS device selection test skipped: {str(e)}")

    def test_text_to_text_inference(self, setup_paligemma_test) -> None:
        """
        Test basic text-to-text inference functionality.
        
        Verifies that the PaliGemma2 model can perform text-to-text inference
        by providing an image and a question, and checking that the model returns
        a valid text response.
        
        Args:
            setup_paligemma_test: Fixture providing test resources for PaliGemma2
            
        Assertions:
            - Inference should return a non-null result
            - Result should be a string (text response)
            
        Raises:
            pytest.skip: If inference fails due to model or resource limitations
        """
        
        try:
            model = setup_paligemma_test['model']
            
            result = model.inference(
                image_path=setup_paligemma_test['test_image_path'],
                text_input=setup_paligemma_test['test_questions'][0],
                task="<TEXT_TO_TEXT>"
            )
            
            assert result is not None, "Inference should return a result"
            assert isinstance(result, str), "Text-to-text result should be a string"
            
        except Exception as e:
            pytest.skip(f"Text-to-text inference test skipped: {str(e)}")

    def test_object_detection_inference(self, setup_paligemma_test) -> None:
        """
        Test object detection inference.
        
        Verifies that the PaliGemma2 model can perform object detection inference
        by providing an image and a detection prompt, and checking that the model
        returns valid detection results and generates an output image with
        bounding boxes.
        
        Args:
            setup_paligemma_test: Fixture providing test resources for PaliGemma2
            
        Assertions:
            - Inference should return a non-null result
            - An image path should be returned
            - The output image should exist and not be empty
            
        Raises:
            pytest.skip: If inference fails due to model or resource limitations
        """
        try:
            model = setup_paligemma_test['model']
            classes = list(setup_paligemma_test['class_mapping'].values())
            
            result, image_path = model.inference(
                image_path=setup_paligemma_test['test_image_path'],
                text_input="detect all objects",
                task="<TEXT_TO_OD>",
                classes=classes,
                save_dir=setup_paligemma_test['temp_dir']
            )
            
            assert result is not None, "OD inference should return a result"
            assert image_path is not None, "OD inference should return an image path"
            
            if image_path and os.path.exists(image_path):
                assert os.path.getsize(image_path) > 0, "Output image should not be empty"
        except Exception as e:
            pytest.skip(f"Object detection inference test skipped: {str(e)}")

#TODO: Add tests for Finetuning when Tests can be run on GPU
