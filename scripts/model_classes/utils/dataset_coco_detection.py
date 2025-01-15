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

        for ann in annots:
            bbox = ann['bbox']
            x0, y0 = bbox[0], bbox[1]
            x1, y1 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            
            ann['bbox'] = [
                min(x0, x1),
                min(y0, y1),
                max(x0, x1),
                max(y0, y1)
            ]
        
        target = {'image_id': image_id, 'annotations': annots}
        encoding = self.processor(images=image, annotations=target, return_tensors="pt")
        
        return {
            'pixel_values': encoding['pixel_values'].squeeze(),
            'pixel_mask': encoding['pixel_mask'].squeeze(),
            'labels': encoding['labels'][0]
        }