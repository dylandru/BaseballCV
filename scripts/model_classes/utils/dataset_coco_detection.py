import os
from torch.utils.data import Dataset
from transformers import DetrImageProcessor
from torchvision.datasets import CocoDetection

class CocoDetectionDataset(Dataset):
    """
    Dataset class for COCO-format detection datasets that supports both:
    1. Hierarchical structure with split/images and split/labels
    2. Flat structure with all images and instances_{split}.json files in root dir
    """
    def __init__(self, dataset_dir: str, split: str, processor: DetrImageProcessor):
        """
        Initialize COCO detection dataset.
        
        Args:
            dataset_dir (str): Root directory containing the dataset
            split (str): Dataset split ('train', 'test', or 'val')
            processor (DetrImageProcessor): DETR image processor for preprocessing
        """
        self.processor = processor
        
        # Check for hierarchical structure first
        hierarchical_img_dir = os.path.join(dataset_dir, split)
        hierarchical_ann_file = os.path.join(dataset_dir, split, "_coco.json")
        
        # Check for flat structure
        flat_img_dir = dataset_dir
        flat_ann_file = os.path.join(dataset_dir, f"instances_{split}.json")
        
        # Determine which structure to use
        if os.path.exists(hierarchical_img_dir) and os.path.exists(hierarchical_ann_file):
            self.img_dir = hierarchical_img_dir
            self.ann_file = hierarchical_ann_file
        elif os.path.exists(flat_img_dir) and os.path.exists(flat_ann_file):
            self.img_dir = flat_img_dir
            self.ann_file = flat_ann_file
        else:
            raise ValueError(
                f"Could not find valid COCO dataset structure in {dataset_dir}. "
                f"Expected either:\n"
                f"1. Hierarchical: {hierarchical_img_dir} and {hierarchical_ann_file}\n"
                f"2. Flat: {flat_img_dir} and {flat_ann_file}"
            )
            
        self.dataset = CocoDetection(self.img_dir, self.ann_file)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            dict: Dictionary containing:
                - pixel_values: Preprocessed image tensor
                - pixel_mask: Attention mask tensor
                - labels: Target labels
        """
        image, annots = self.dataset[idx]
        image_id = self.dataset.ids[idx]
        
        target = {'image_id': image_id, 'annotations': annots}
        encoding = self.processor(images=image, 
                                annotations=target, 
                                return_tensors="pt",
                                size=self.processor.size)
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'pixel_mask': encoding['pixel_mask'].squeeze(),
            'labels': encoding['labels'][0]
        }