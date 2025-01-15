import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, Trainer, TrainingArguments
import os
from PIL import Image
from typing import Dict, List
from .utils import CocoDetectionDataset

class DETR:
    def __init__(self, num_labels: int, model_name: str = "facebook/detr-resnet-50"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)
    
    def _collate_fn(self, batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        pixel_mask = torch.stack([item['pixel_mask'] for item in batch])
        labels = [item['labels'] for item in batch]
        return {
            'pixel_values': pixel_values,
            'pixel_mask': pixel_mask,
            'labels': labels
        }

    def finetune(
        self,
        dataset_dir: str,
        output_dir: str = "baseballcv-detr",
        batch_size: int = 4,
        num_epochs: int = 10,
        learning_rate: float = 1e-4
    ):

        train_dataset = CocoDetectionDataset(dataset_dir, self.processor)
        val_dataset = CocoDetectionDataset(dataset_dir, self.processor)

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            fp16=True,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=learning_rate,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            gradient_checkpointing=True,
            gradient_accumulation_steps=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self._collate_fn,
        )

        # Train and save
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)

    def inference(self, image_path: str, confidence_threshold: float = 0.9) -> List[Dict]:
        self.model.eval()
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        width, height = image.size
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=[(height, width)],
            threshold=confidence_threshold
        )[0]
        
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            xmin, ymin, xmax, ymax = box.tolist()
            detection = {
                'image_id': os.path.basename(image_path),
                'category_id': label.item(),
                'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],  
                'score': score.item(),
                'area': (xmax - xmin) * (ymax - ymin),
                'iscrowd': 0
            }
            detections.append(detection)
        
        return detections