import pytest
import os
import tempfile
import shutil
import json
import logging
from baseballcv.datasets import DataProcessor

class TestDataProcessor:
    @pytest.fixture
    def setup_data_processor_and_dataset(self, load_tools):
        """Set up a DataProcessor instance and temporary directories"""
        logger = logging.getLogger("test_logger")
        
        processor = DataProcessor(logger)
        dict_classes = {0: 'glove', 1: 'homeplate', 2: 'baseball', 3: 'rubber'}

        processed_dataset_path = load_tools.load_dataset(dataset_alias="baseball")
        
        yield processor, processed_dataset_path, dict_classes
        shutil.rmtree(processed_dataset_path)

    def test_prepare_dataset_with_existing_split(self, setup_data_processor_and_dataset):
        """Test prepare_dataset method with an existing train/test/valid split"""
        processor, processed_dataset_path, dict_classes = setup_data_processor_and_dataset
        
        result = processor.prepare_dataset(processed_dataset_path, dict_classes)
        
        assert len(result) == 5, "Should return 5 paths"
        
        for split in ["train", "test", "valid"]:
            assert os.path.exists(os.path.join(processed_dataset_path, split, "images"))
            assert os.path.exists(os.path.join(processed_dataset_path, split, "labels"))
    
    def test_convert_annotations(self, setup_data_processor_and_dataset):
        """Test convert_annotations method"""
        processor, processed_dataset_path, dict_classes = setup_data_processor_and_dataset
        split = "train"
        
        output_file = processor.convert_annotations(processed_dataset_path, split, dict_classes)
        
        assert os.path.exists(output_file)
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
            first_annotation = json.loads(lines[0])
            assert "image" in first_annotation, "image key should be present"
            assert "prefix" in first_annotation, "prefix key should be present"
            assert "suffix" in first_annotation, "suffix key should be present"
            assert first_annotation["prefix"] == "<OD>", "prefix should be <OD>"
            assert "baseball<loc_" in first_annotation["suffix"], "suffix should contain baseball class"
