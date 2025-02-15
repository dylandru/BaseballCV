---
layout: default
title: Training Florence2
parent: Using the Repository
nav_order: 4
---

# Training Florence2 for Ball Detection

This guide demonstrates how to train a Florence2 model specifically for baseball detection using BaseballCV's ball dataset. Florence2's vision-language capabilities make it particularly suitable for precise ball detection and tracking.

## Setting Up the Training Pipeline

First, let's create a comprehensive training pipeline that leverages BaseballCV's utilities:

```python
from baseballcv.functions import LoadTools
from baseballcv.model import Florence2
import os
import torch

class BallDetectionPipeline:
    def __init__(self):
        """Initialize pipeline components"""
        self.load_tools = LoadTools()
        
        # Initialize Florence2 with appropriate batch size
        self.model = Florence2(
            model_id='microsoft/Florence-2-large',
            batch_size=4  # Adjust based on GPU memory
        )
        
        # Define class mapping
        self.classes = {
            0: "baseball"  # Single class for ball detection
        }
        
        # Set up output directories
        self.output_dir = "ball_detector"
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_dataset(self):
        """
        Load and prepare the ball detection dataset
        """
        # Load the baseball-only dataset
        dataset_path = self.load_tools.load_dataset("baseball")
        
        return dataset_path

    def configure_training(self):
        """
        Configure Florence2 training parameters
        """
        training_config = {
            # Dataset parameters
            'train_test_split': (80, 10, 10),  # Training/Test/Validation split
            
            # Training hyperparameters
            'epochs': 20,
            'lr': 4e-6,
            'batch_size': 4,
            
            # LoRA parameters for efficient fine-tuning
            'lora_r': 8,
            'lora_scaling': 8,
            'lora_dropout': 0.05,
            
            # Training optimizations
            'warmup_epochs': 1,
            'gradient_accumulation_steps': 2,
            'lr_schedule': "cosine",
            
            # Early stopping settings
            'patience': 5,
            'patience_threshold': 0.01,
            
            # Model saving
            'save_dir': self.output_dir,
            
            # Hardware utilization
            'num_workers': 4  # Adjust based on CPU cores
        }
        
        return training_config

    def train(self):
        """
        Execute the training pipeline
        """
        dataset_path = self.prepare_dataset()
        config = self.configure_training()
        
        metrics = self.model.finetune(
            dataset=dataset_path,
            classes=self.classes,
            **config
        )
        
        return metrics

# Initialize and run the pipeline
pipeline = BallDetectionPipeline()
training_metrics = pipeline.train()
```

## Dataset Structure and Preparation

The baseball dataset in BaseballCV is already optimized for training, but understanding its structure is important:

```python
def inspect_dataset(dataset_path):
    """
    Analyze the ball detection dataset structure
    """
    # Dataset statistics
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "valid")
    test_path = os.path.join(dataset_path, "test")
    
    print(f"Training images: {len(os.listdir(os.path.join(train_path, 'images')))}")
    print(f"Validation images: {len(os.listdir(os.path.join(val_path, 'images')))}")
    print(f"Test images: {len(os.listdir(os.path.join(test_path, 'images')))}")
```

## Training Configuration Details

Florence2 training can be customized through various parameters. Here's a detailed configuration:

```python
def detailed_training_config(self):
    """
    Detailed training configuration with explanations
    """
    return {
        # Model configuration
        'dataset': self.dataset_path,
        'classes': self.classes,
        'epochs': 20,
        'batch_size': 4,
        
        # Optimization parameters
        'lr': 4e-6,  # Learning rate for fine-tuning
        'weight_decay': 0.01,
        'warmup_epochs': 1,
        
        # LoRA specific parameters
        'lora_r': 8,        # LoRA attention dimension
        'lora_scaling': 8,  # LoRA alpha scaling factor
        'lora_dropout': 0.05,
        
        # Training stability
        'gradient_accumulation_steps': 2,
        'patience': 5,      # Early stopping patience
        'patience_threshold': 0.01,
        
        # Hardware utilization
        'num_workers': 4,   # DataLoader workers
        
        # Model saving
        'save_dir': "ball_detector",
        'create_peft_config': True  # Enable LoRA configuration
    }
```

## Monitoring Training Progress

Monitoring training progress helps identify issues early:

```python
def setup_training_monitoring(self):
    """
    Configure training monitoring and visualization
    """
    import matplotlib.pyplot as plt
    
    class TrainingMonitor:
        def __init__(self):
            self.train_losses = []
            self.val_losses = []
            self.learning_rates = []
        
        def update(self, train_loss, val_loss, lr):
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(lr)
        
        def plot_metrics(self, save_path):
            # Plot loss curves
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, label='Train Loss')
            plt.plot(self.val_losses, label='Val Loss')
            plt.title('Training Progress')
            plt.legend()
            
            # Plot learning rate
            plt.subplot(1, 2, 2)
            plt.plot(self.learning_rates)
            plt.title('Learning Rate Schedule')
            
            plt.savefig(save_path)
            plt.close()
    
    return TrainingMonitor()
```

## Model Evaluation

After training, evaluate the model's performance:

```python
def evaluate_model(self):
    """
    Evaluate trained model performance
    """
    # Load test dataset
    test_dataset = self.load_tools.load_dataset(
        "baseball",
        subset="test"
    )
    
    # Run evaluation
    metrics = self.model.evaluate(
        base_path=test_dataset,
        classes=self.classes,
        num_workers=4,
        dataset_type="yolo"
    )
    
    return metrics
```

## Practical Tips for Florence2 Training

When training Florence2 for ball detection, consider these important factors:

### 1. Memory Management

Florence2 is a large model. Optimize memory usage by:
- Using gradient accumulation
- Implementing LoRA for efficient fine-tuning
- Adjusting batch size based on available GPU memory
- Utilizing mixed precision training


This training pipeline provides a robust foundation for fine-tuning Florence2 specifically for baseball detection. The configuration can be adjusted based on your specific needs and computational resources.
