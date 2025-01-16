import os
from torch.utils.data import Dataset
from transformers import DetrImageProcessor
from torchvision.datasets import CocoDetection

class CocoDetectionDataset(Dataset):
    def __init__(self, dataset_dir: str, split: str, processor: DetrImageProcessor):
        img_dir = os.path.join(dataset_dir, split)
        ann_file = os.path.join(dataset_dir, split, "_coco.json")
        self.dataset = CocoDetection(img_dir, ann_file)
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
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