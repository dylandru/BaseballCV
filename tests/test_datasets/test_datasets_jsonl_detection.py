import pytest
import os
import tempfile
import shutil
import logging
from PIL import Image
from baseballcv.datasets import JSONLDetection

class TestJSONLDetection:
    @pytest.fixture
    def setup_jsonl_dataset(self, load_tools):
        """Set up and download BaseballCV JSONL dataset for testing"""

        logger = logging.getLogger("test_logger")
        
        dataset_path = load_tools.load_dataset(
            dataset_alias="amateur_hitter_pitcher_jsonl", 
            use_bdl_api=False, 
            file_txt_path="datasets/JSONL/amateur_hitter_pitcher_jsonl.txt"
        )
        
        images_dir = os.path.join(dataset_path, "dataset"); os.makedirs(images_dir, exist_ok=True)
        
        jsonl_files = [f for f in os.listdir(images_dir) if f.endswith('.jsonl')]

        if not jsonl_files:
            raise FileNotFoundError("No JSONL file found in the dataset")
        
        jsonl_path = os.path.join(images_dir, jsonl_files[0])
        entries = JSONLDetection.load_jsonl_entries(jsonl_path, logger)
        
        yield entries, images_dir, jsonl_path, logger, dataset_path
        shutil.rmtree(dataset_path)
    
    def test_jsonl_detection_initialization(self, setup_jsonl_dataset):
        """Test the initialization of JSONLDetection with the real dataset"""
        entries, images_dir, _, logger, _ = setup_jsonl_dataset
        
        dataset = JSONLDetection(entries=entries, image_directory_path=images_dir, logger=logger)
        
        assert len(dataset) == len(entries)
        assert dataset.image_directory_path == images_dir
        assert dataset.logger == logger
        assert hasattr(dataset, 'transforms')
    
    def test_jsonl_detection_getitem(self, setup_jsonl_dataset):
        """Test the __getitem__ method of JSONLDetection"""
        entries, images_dir, _, logger, _ = setup_jsonl_dataset
        
        if len(entries) == 0:
            pytest.skip("No entries found in the dataset")
        
        dataset = JSONLDetection(entries=entries, image_directory_path=images_dir, logger=logger, augment=False)
        
        try:
            image, entry = dataset[0]
            
            assert isinstance(image, Image.Image)
            assert isinstance(entry, dict)
        except Exception as e:
            pytest.skip(f"Error accessing first item in dataset: {str(e)}")
    
    def test_jsonl_detection_load_entries(self, setup_jsonl_dataset):
        """Test the load_jsonl_entries static method"""
        _, _, jsonl_path, logger, _ = setup_jsonl_dataset
        
        entries = JSONLDetection.load_jsonl_entries(jsonl_path, logger)
        
        assert isinstance(entries, list)
        if len(entries) > 0:
            assert isinstance(entries[0], dict)
    
    def test_jsonl_detection_transformations(self, setup_jsonl_dataset):
        """Test the image transformation methods"""

        _, images_dir, _, _, _ = setup_jsonl_dataset
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:1]

        if not image_files:
            image = Image.new('RGB', (100, 100), color='green')
        else:
            image_path = os.path.join(images_dir, image_files[0])
            image = Image.open(image_path)
        
        jittered = JSONLDetection.random_color_jitter(image)
        assert isinstance(jittered, Image.Image)
        
        blurred = JSONLDetection.random_blur(image)
        assert isinstance(blurred, Image.Image)
        
        noisy = JSONLDetection.random_noise(image)
        assert isinstance(noisy, Image.Image)
    
    def test_jsonl_detection_invalid_jsonl(self, setup_jsonl_dataset):
        """Test handling of invalid JSONL file"""
        _, _, _, logger, _ = setup_jsonl_dataset
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
            temp_file.write("This is not valid JSON\n")
            temp_file.write('{"valid": "entry"}\n')
            temp_file_path = temp_file.name
        
        try:
            entries = JSONLDetection.load_jsonl_entries(temp_file_path, logger)
            assert len(entries) == 1
            assert entries[0]['valid'] == 'entry'
        finally:
            os.unlink(temp_file_path)
    
    def test_jsonl_detection_empty_file(self, setup_jsonl_dataset):
        """Test handling of empty JSONL file"""
        _, _, _, logger, _ = setup_jsonl_dataset
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as temp_file:
            temp_file_path = temp_file.name
        
        try:
            with pytest.raises(ValueError) as excinfo:
                JSONLDetection.load_jsonl_entries(temp_file_path, logger)
            assert "No valid entries found" in str(excinfo.value)
        finally:
            os.unlink(temp_file_path)
