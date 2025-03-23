import logging
import os
from transformers import DetrImageProcessor
import torchvision
from pycocotools.coco import COCO

class CocoDetectionDataset(torchvision.datasets.CocoDetection):
    """
    Dataset class for COCO-format detection datasets that supports both:
    1. Hierarchical structure with split/images and split/labels
    2. Flat structure with all images and instances_{split}.json files in root dir
    """
    def __init__(self, dataset_dir: str, split: str, logger: logging.Logger, processor: DetrImageProcessor):
        """
        Initialize COCO detection dataset.
        
        Args:
            dataset_dir (str): Root directory containing the dataset
            split (str): Dataset split ('train', 'test', or 'val')
            logger (logging.Logger): Logger instance for logging
            processor (DetrImageProcessor): DETR image processor for preprocessing
        """
        self.logger = logger
        
        # Check for hierarchical structure first
        hierarchical_img_dir = os.path.join(dataset_dir, split)
        hierarchical_ann_file = os.path.join(dataset_dir, split, "_coco.json")
        
        # Check for flat structure
        flat_img_dir = os.path.join(dataset_dir, split, "images")
        flat_ann_file = os.path.join(dataset_dir, "COCO_annotations", f"instances_{split}.json")
        
        # Determine which structure to use
        if os.path.exists(hierarchical_img_dir) and os.path.exists(hierarchical_ann_file):
            self.img_dir = hierarchical_img_dir
            self.ann_file = hierarchical_ann_file
            self.logger.info(f"Using hierarchical COCO dataset structure: {hierarchical_img_dir} and {hierarchical_ann_file}")
        elif os.path.exists(flat_img_dir) and os.path.exists(flat_ann_file):
            self.img_dir = flat_img_dir
            self.ann_file = flat_ann_file
            self.logger.info(f"Using flat COCO dataset structure: {flat_img_dir} and {flat_ann_file}")
        else:
            self.logger.error(f"Could not find valid COCO dataset structure in {dataset_dir}. "
                f"Expected either:\n"
                f"1. Hierarchical: {hierarchical_img_dir} and {hierarchical_ann_file}\n"
                f"2. Flat: {flat_img_dir} and {flat_ann_file}")
            
        super(CocoDetectionDataset, self).__init__(self.img_dir, self.ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetectionDataset, self).__getitem__(idx)        
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
