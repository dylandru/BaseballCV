import pytest
import torch
import tempfile
import shutil
from transformers import DetrImageProcessor
from baseballcv.datasets import CocoDetectionDataset

class TestCocoDetectionDataset:
    @pytest.fixture
    def setup_coco_dataset(self, load_tools):
        """Download and set up the actual COCO dataset provided by BaseballCV for testing"""

        dataset_path = load_tools.load_dataset("baseball_rubber_home_glove_COCO")
        processor = DetrImageProcessor(do_resize=True, size={"height": 800, "width": 800})
        
        yield dataset_path, processor
        shutil.rmtree(dataset_path)
    
    def test_coco_dataset_initialization(self, setup_coco_dataset):
        """Test the initialization of CocoDetectionDataset with the real dataset"""
        dataset_path, processor = setup_coco_dataset
        
        split = "train"
        dataset = CocoDetectionDataset(dataset_dir=dataset_path, split=split, processor=processor)
        
        assert dataset.img_dir is not None
        assert dataset.ann_file is not None
        assert len(dataset) > 0
        assert isinstance(dataset.processor, DetrImageProcessor)
    
    def test_coco_dataset_getitem(self, setup_coco_dataset):
        """Test the __getitem__ method of CocoDetectionDataset with the real dataset"""
        dataset_path, processor = setup_coco_dataset
        
        split = "train"
        dataset = CocoDetectionDataset(dataset_dir=dataset_path, split=split, processor=processor)
        
        if len(dataset) > 0:
            pixel_values, target = dataset[0]
            
            assert isinstance(pixel_values, torch.Tensor)
            assert isinstance(target, dict)
            assert 'labels' in target
            assert 'boxes' in target