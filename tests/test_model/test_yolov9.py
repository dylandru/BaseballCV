import pytest
import torch
import shutil
import os
import tempfile
from PIL import Image
from baseballcv.model import YOLOv9

class TestYOLOv9:
    """
    Test cases for YOLOv9 model.
    
    This test suite verifies the functionality of the YOLOv9 object detection model,
    including initialization, device selection, inference capabilities, and parameter
    sensitivity.
    """

    @pytest.fixture
    def setup_yolo_test(self, load_tools) -> dict:
        """
        Set up test environment with BaseballCV dataset.
        
        Creates a temporary directory and loads a baseball dataset for testing.
        Initializes a YOLOv9 model and prepares test images either from the dataset
        or creates a synthetic test image if needed.
        
        Args:
            load_tools: Fixture providing tools to load datasets
            
        Yields:
            dict: A dictionary containing test resources including:
                - test_image_path: Path to an image for testing inference
                - dataset_path: Path to the loaded dataset
                - class_mapping: Dictionary mapping class IDs to class names
                - temp_dir: Path to temporary directory for test artifacts
                - model: Initialized YOLOv9 model instance
        """
        
        try:
            temp_dir = tempfile.mkdtemp()
            dataset_path = load_tools.load_dataset("baseball_rubber_home_glove")

            model = YOLOv9(
                device='cpu'
            ) #using default model yolov9-c
            
            train_images_dir = os.path.join(dataset_path, "train", "images")
            test_image_path = None
            
            if os.path.exists(train_images_dir) and os.listdir(train_images_dir):
                test_image_path = os.path.join(train_images_dir, [f for f in os.listdir(train_images_dir) 
                                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))][0])
            
            if not test_image_path:
                test_image_path = os.path.join(temp_dir, "test_image.jpg")
                Image.new('RGB', (640, 640), color='green').save(test_image_path)
            
            class_mapping = {
                0: 'glove',
                1: 'homeplate',
                2: 'baseball',
                3: 'rubber'
            }
            
            yield {
                'test_image_path': test_image_path,
                'dataset_path': dataset_path,
                'class_mapping': class_mapping,
                'temp_dir': temp_dir,
                'model': model
            }
            
        except Exception as e:
            print(f"Error setting up YOLOv9 test: {e}")
            test_image_path = os.path.join(temp_dir, "test_image.jpg")
            img = Image.new('RGB', (640, 640), color='red')
            img.save(test_image_path)
            
            yield {
                'test_image_path': test_image_path,
                'dataset_path': None,
                'class_mapping': {0: 'test_class'},
                'temp_dir': temp_dir,
                'model': model
            }
        
        finally:
            shutil.rmtree(temp_dir)
            shutil.rmtree(dataset_path)

    def test_model_initialization(self, setup_yolo_test) -> None:
        """
        Test model initialization and device selection.
        
        Verifies that the YOLOv9 model initializes correctly with the specified
        device and tests device selection logic, including optional MPS support
        when available.
        
        Args:
            setup_yolo_test: Fixture providing test resources
        """

        model = YOLOv9(
            device='cpu'
        )
        
        assert model is not None, "YOLOv9 model should initialize"
        assert model.device == 'cpu', "Device should be set correctly"
        
        with pytest.MonkeyPatch().context() as m: #optional MPS test - not necessary but can be used to test MPS device selection
            m.setattr(torch.cuda, 'is_available', lambda: False)
            m.setattr(torch.backends.mps, 'is_available', lambda: True)
            
            try:
                model = YOLOv9(device='mps')
                assert model.device == 'mps', "Should select MPS when available"
            except Exception as e:
                pytest.skip(f"MPS model initialization test skipped: {str(e)}")

    def test_inference(self, setup_yolo_test) -> None:
        """
        Test basic inference functionality.
        
        Verifies that the YOLOv9 model can perform inference on a test image
        and returns properly structured detection results with expected fields.
        
        Args:
            setup_yolo_test: Fixture providing test resources including model and test image
        """
        
        model = setup_yolo_test['model']
        
        result = model.inference(
            source=setup_yolo_test['test_image_path'],
            project=setup_yolo_test['temp_dir']
        )
        
        assert result is not None, "Inference should return results"
        
        if isinstance(result, list) and len(result) > 0:
            assert 'box' in result[0], "Detection should include bounding box"
            assert 'confidence' in result[0], "Detection should include confidence"
            assert 'class_id' in result[0], "Detection should include class_id"
            
            assert len(result[0]['box']) == 4, "Box should have 4 coordinates"
                
    
    def test_threshold_parameters(self, setup_yolo_test) -> None:
        """
        Test confidence and IoU threshold parameters.
        
        Verifies that the confidence threshold parameter correctly filters
        detection results, with lower thresholds yielding more detections
        than higher thresholds.
        
        Args:
            setup_yolo_test: Fixture providing test resources including model and test image
        """
        
        model = setup_yolo_test['model']
        
        high_conf_results = model.inference(
            source=setup_yolo_test['test_image_path'],
            conf_thres=0.9
        )
        
        low_conf_results = model.inference(
            source=setup_yolo_test['test_image_path'],
            conf_thres=0.05
        )
        
        if high_conf_results and low_conf_results:
            assert len(high_conf_results) <= len(low_conf_results), \
                "Lower confidence threshold should yield more or equal detections"

#TODO: Add tests for Finetuning when Tests can be run on GPU
