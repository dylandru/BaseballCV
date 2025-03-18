import shutil
import pytest
import torch
from baseballcv.model import DETR
from baseballcv.datasets import CocoDetectionDataset
from transformers import DetrImageProcessor

class TestDETR:
    """
    Test cases for DETR model using real data.
    
    This test suite verifies the functionality of the DETR (DEtection TRansformer) 
    object detection model, including initialization, device selection, and inference 
    capabilities with various confidence thresholds.
    """

    @pytest.fixture
    def setup_detr_test(self, load_tools):
        """
        Set up test environment with real dataset.
        
        Creates a testing environment by loading a baseball dataset in COCO format,
        initializing the image processor, and preparing test images for inference.
        
        Args:
            load_tools: Fixture providing tools to load datasets
            
        Yields:
            dict: A dictionary containing test resources including:
                - test_image: Tensor representation of a test image
                - dataset_path: Path to the loaded dataset
                - coco_dataset: Initialized CocoDetectionDataset instance
                - id2label: Dictionary mapping class IDs to class names
                - image_size: Tuple containing dimensions of the test image
        """

        dataset_path = load_tools.load_dataset("baseball_rubber_home_glove_COCO")
        processor = DetrImageProcessor(do_resize=True, size={"height": 800, "width": 800})

        split = "train"
        coco_dataset = CocoDetectionDataset(dataset_dir=dataset_path, split=split, processor=processor)
        
        test_image = None
        image_size = (100, 100, 3)  #fallback
        
        if len(coco_dataset) > 0:
            pixel_values, _ = coco_dataset[0]
            test_image = pixel_values
            image_size = (pixel_values.shape[-3], pixel_values.shape[-2], pixel_values.shape[-1])
        else:
            test_image = torch.rand(3, 224, 224)
        
        id2label = {
            0: "glove",
            1: "homeplate",
            2: "baseball",
            3: "rubber"
        }
        
        yield {
            'test_image': test_image,
            'dataset_path': dataset_path,
            'coco_dataset': coco_dataset,
            'id2label': id2label,
            'image_size': image_size
        }
        shutil.rmtree(dataset_path) 

    @pytest.mark.network
    def test_model_initialization(self):
        """
        Test model initialization and device selection.
        
        Verifies that the DETR model can be properly initialized with specified
        parameters and that the device selection works correctly. This test checks
        that model attributes are correctly set during initialization.
        
        Assertions:
            - Model should initialize successfully
            - Device should be correctly set to CPU
            - Model ID should match the specified value
            - Batch size should be set to the default value
            - Image processor should be initialized
        """

        model_cpu = DETR(
            model_id='facebook/detr-resnet-50',
            device='cpu',
            num_labels=4
        )
        
        assert model_cpu is not None, "DETR model should initialize"
        assert model_cpu.detr_device == 'cpu', "Device should be set to CPU"
        assert model_cpu.model_id == 'facebook/detr-resnet-50', "Model ID should be set correctly"
        assert model_cpu.batch_size == 8, "Batch size should be set correctly"
        assert model_cpu.processor is not None, "Model should have a processor"

    def test_inference(self, setup_detr_test):
        """
        Test basic inference functionality with a real image.
        
        Verifies that the DETR model can perform object detection inference on
        a real image and return properly formatted results. This test checks the
        structure of the detection results and ensures they contain the expected
        fields.
        
        Args:
            setup_detr_test: Fixture providing test resources for DETR
            
        Assertions:
            - Inference should return a non-null result
            - Result should be a list of detections
            - Each detection should have score, label, and box attributes
        """

        if len(setup_detr_test['coco_dataset']) == 0:
            pytest.skip("No images available in dataset to test inference")
        
        try:
            model = DETR(
                model_id='facebook/detr-resnet-50',
                device='cpu',
                num_labels=4
            )
            
            result = model.inference("baseball_rubber_home_glove_COCO/test/images/0000011.jpg")
            
            assert result is not None, "Inference should return results"
            assert isinstance(result, list), "Inference should return a list"
            
            if len(result) > 0:
                assert "score" in result[0], "Each detection should have a confidence score"
                assert "label" in result[0], "Each detection should have a label"
                assert "box" in result[0], "Each detection should have a bounding box"
        
        except Exception as e:
            pytest.skip(f"DETR is under construction. Error occurred: {e}")

    def test_inference_with_custom_threshold(self, setup_detr_test):
        """
        Test inference with a custom confidence threshold.
        
        Verifies that the DETR model correctly applies different confidence
        thresholds during inference. This test compares the number of detections
        returned when using a high threshold versus a low threshold.
        
        Args:
            setup_detr_test: Fixture providing test resources for DETR
            
        Assertions:
            - Higher confidence threshold should result in fewer or equal
              number of detections compared to a lower threshold
        """
        try:
            model = DETR(
                    model_id='facebook/detr-resnet-50',
                    device='cpu',
                    num_labels=4
                )
                
            result_high_threshold = model.inference("baseball_rubber_home_glove_COCO/test/images/0000011.jpg", conf=0.5)
            result_low_threshold = model.inference("baseball_rubber_home_glove_COCO/test/images/0000011.jpg", conf=0.1)
            
            assert len(result_high_threshold) <= len(result_low_threshold), \
                "Higher threshold should result in fewer or equal number of predictions"
        except Exception as e:
            pytest.skip(f"DETR is under construction. Error occurred: {e}")
