---
layout: default
title: Training Pipeline
parent: Using the Repository
nav_order: 3
---

# Building a Training Pipeline with DETR

This guide walks through creating a complete training pipeline for baseball object detection using BaseballCV's DETR implementation. We'll cover data preparation, training configuration, and model evaluation.

## Setting Up the Training Environment

First, let's set up a comprehensive training pipeline that handles data preparation, model training, and evaluation:

```python
from baseballcv.functions import LoadTools, DataTools
from baseballcv.model import DETR
import torch
import os

class BaseballTrainingPipeline:
    def __init__(self, base_data_dir: str):
        """Initialize training pipeline"""
        self.load_tools = LoadTools()
        self.data_tools = DataTools()
        self.base_data_dir = base_data_dir
        
        # Define classes for baseball detection
        self.classes = {
            0: "baseball",
            1: "glove",
            2: "homeplate",
            3: "rubber"
        }
        
        # Initialize DETR model
        self.model = DETR(
            num_labels=len(self.classes),
            model_id="facebook/detr-resnet-50",
            batch_size=8
        )

    def prepare_training_data(self):
        """
        Prepare training dataset using BaseballCV's utilities
        """
        # Generate dataset from baseball footage
        self.data_tools.generate_photo_dataset(
            output_frames_folder="raw_dataset",
            max_plays=1000,  # Adjust based on needs
            max_num_frames=6000,
            start_date="2024-05-01",
            end_date="2024-06-01"
        )
        
        # Automatically annotate using existing model
        self.data_tools.automated_annotation(
            model_alias="ball_tracking",
            image_dir="raw_dataset",
            output_dir="annotated_dataset",
            conf=0.8
        )

    def train_model(self):
        """
        Configure and run model training
        """
        training_args = {
            'dataset_dir': "annotated_dataset",
            'classes': self.classes,
            'save_dir': "baseball_detector",
            'batch_size': 4,
            'epochs': 50,
            'lr': 1e-4,
            'lr_backbone': 1e-5,
            'weight_decay': 0.01,
            'gradient_accumulation_steps': 2,
            'patience': 5,
            'patience_threshold': 0.0001,
            'precision': '16-mixed',
            'freeze_backbone': True
        }
        
        # Start training with progress monitoring
        metrics = self.model.finetune(**training_args)
        
        return metrics

    def evaluate_model(self):
        """
        Evaluate model performance
        """
        return self.model.evaluate(
            dataset_dir="annotated_dataset",
            conf=0.25  # Adjust based on application needs
        )

# Usage example
pipeline = BaseballTrainingPipeline(base_data_dir="training_data")

# Execute complete training workflow
pipeline.prepare_training_data()
training_metrics = pipeline.train_model()
evaluation_metrics = pipeline.evaluate_model()
```

## Data Preparation Details

The pipeline uses BaseballCV's data tools to create a high-quality training dataset:

```python
def prepare_coco_dataset(self):
    """
    Convert annotated dataset to COCO format for DETR training
    """
    from baseballcv.utils import CocoDetectionDataset
    
    # Setup directory structure
    os.makedirs("coco_dataset/train/images", exist_ok=True)
    os.makedirs("coco_dataset/valid/images", exist_ok=True)
    os.makedirs("coco_dataset/test/images", exist_ok=True)
    
    # Process annotations
    for split in ['train', 'valid', 'test']:
        annotations = []
        image_dir = f"annotated_dataset/{split}/images"
        
        for image_file in os.listdir(image_dir):
            if image_file.endswith(('.jpg', '.png')):
                image_id = len(annotations)
                image_path = os.path.join(image_dir, image_file)
                
                # Process corresponding annotation
                ann_file = os.path.join(
                    f"annotated_dataset/{split}/labels",
                    os.path.splitext(image_file)[0] + '.txt'
                )
                
                if os.path.exists(ann_file):
                    with open(ann_file, 'r') as f:
                        ann_data = f.read().splitlines()
                        
                    # Convert YOLO to COCO format
                    coco_annotations = self._convert_to_coco(
                        ann_data, 
                        image_id
                    )
                    
                    annotations.append({
                        'image': image_path,
                        'annotations': coco_annotations
                    })
        
        # Save split-specific annotations
        self._save_coco_annotations(
            annotations,
            f"coco_dataset/{split}/_annotations.json"
        )
```

## Training Configuration

DETR training can be customized through various parameters:

{: .note }
Training configuration should be adjusted based on your available computational resources and dataset size.

```python
def configure_training(self):
    """
    Configure detailed training parameters
    """
    training_config = {
        # Data parameters
        'batch_size': 4,
        'num_workers': min(8, os.cpu_count() - 1),
        
        # Optimization parameters
        'lr': 1e-4,
        'lr_backbone': 1e-5,
        'weight_decay': 0.01,
        'gradient_accumulation_steps': 2,
        
        # Training schedule
        'epochs': 50,
        'warmup_epochs': 3,
        'patience': 5,
        
        # Model parameters
        'freeze_backbone': True,
        'precision': '16-mixed',
        
        # Hardware utilization
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_amp': True  # Automatic Mixed Precision
    }
    
    return training_config
```

## Model Training Workflow

The complete training workflow includes several key components:

1. **Data Loading**: Efficient data loading with proper batching and augmentation
2. **Model Configuration**: Setup of model architecture and training parameters
3. **Training Loop**: Robust training with progress monitoring and checkpointing
4. **Validation**: Regular validation to track model performance
5. **Checkpointing**: Save model states for resuming training

Here's how to execute the complete workflow:

```python
def execute_training_workflow(self):
    """
    Execute complete training workflow with monitoring
    """
    # Initial setup
    config = self.configure_training()
    
    # Prepare datasets
    self.prepare_coco_dataset()
    
    # Initialize training state
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop with monitoring
    for epoch in range(config['epochs']):
        train_metrics = self.model.training_step()
        val_metrics = self.model.validation_step()
        
        # Log progress
        self.log_training_progress(
            epoch,
            train_metrics,
            val_metrics
        )
        
        # Save checkpoints
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            self.save_checkpoint(
                epoch,
                val_metrics,
                is_best=True
            )
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping check
        if patience_counter >= config['patience']:
            print("Early stopping triggered")
            break
```

## Evaluation and Analysis

After training, comprehensive evaluation helps understand model performance:

```python
def analyze_model_performance(self):
    """
    Analyze trained model performance
    """
    # Compute COCO metrics
    metrics = self.model.evaluate(
        dataset_dir="coco_dataset/test",
        conf=0.25
    )
    
    # Visualize results
    self.visualize_detections(
        num_samples=10,
        save_dir="evaluation_results"
    )
    
    return metrics
```

## Common Issues and Solutions

When training DETR models, consider these common challenges:

1. **Memory Usage**
   - Monitor GPU memory usage
   - Adjust batch size and gradient accumulation steps
   - Use mixed precision training

2. **Training Stability**
   - Start with a smaller learning rate
   - Use proper learning rate scheduling
   - Implement gradient clipping

3. **Dataset Quality**
   - Ensure proper annotation format
   - Validate dataset splits
   - Check class balance

This training pipeline provides a robust foundation for training DETR models on baseball data. Adjust parameters based on your specific needs and computational resources.
