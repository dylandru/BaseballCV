---
layout: default
title: Troubleshooting
parent: Getting Started
nav_order: 2
---

# Troubleshooting BaseballCV Installation and Setup

When working with a sophisticated framework like BaseballCV, you might encounter various challenges during setup and usage. This guide will help you understand and resolve common issues, whether you're working in a local environment or Google Colab. We'll explore the underlying causes and provide comprehensive solutions for each problem.

## Environment-Related Issues

Understanding your working environment is crucial for successful troubleshooting. Let's explore common issues in both local and cloud environments.

### Local Environment Challenges

When working locally, GPU configuration often presents the first challenge. Let's verify your setup with a diagnostic script:

```python
import torch
import sys
import platform

def diagnose_environment():
    """
    Perform a comprehensive environment check and provide detailed feedback
    """
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"Operating System: {platform.platform()}")
    
    if torch.cuda.is_available():
        print("\nGPU Information:")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name()}")
        print(f"Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\nNo GPU detected. Consider:")
        print("1. Verifying CUDA installation")
        print("2. Checking PyTorch CUDA compatibility")
        print("3. Examining system PATH variables")
```

### Google Colab Specific Issues

When using Colab, we need to consider its unique environment. Here's a diagnostic approach:

```python
def check_colab_environment():
    """
    Verify Colab-specific settings and connections
    """
    import psutil
    
    try:
        # Check GPU allocation
        !nvidia-smi
        
        # Verify Drive mount
        from google.colab import drive
        drive_path = "/content/drive"
        if not os.path.exists(drive_path):
            print("Drive not mounted. Mounting now...")
            drive.mount(drive_path)
            
        # Check available resources
        ram = psutil.virtual_memory()
        print(f"\nAvailable RAM: {ram.available / 1e9:.2f} GB")
        
        return True
    except Exception as e:
        print(f"Colab setup issue: {str(e)}")
        return False
```

## Model Loading and Execution

Model-related issues often stem from resource constraints or incorrect configurations. Let's examine both scenarios.

### Memory Management

Understanding memory usage helps prevent common crashes:

```python
def monitor_memory_usage():
    """
    Track memory usage during model operations
    """
    import numpy as np
    
    def get_memory_stats():
        if torch.cuda.is_available():
            # GPU memory tracking
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return {
                'allocated_gb': allocated / 1e9,
                'reserved_gb': reserved / 1e9
            }
        return None
    
    # Create a memory checkpoint
    initial_mem = get_memory_stats()
    
    return initial_mem
```

## Training Stability

Training interruptions can occur for various reasons. Let's implement robust recovery mechanisms.

### Checkpoint Management

Creating reliable checkpointing systems:

```python
def implement_checkpointing(model, save_dir, frequency=5):
    """
    Set up a robust checkpointing system
    """
    import os
    from datetime import datetime
    
    os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(epoch, model, optimizer):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}_{timestamp}.pt')
        
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }
        
        torch.save(checkpoint, path)
        return path

    return save_checkpoint
```

### Recovery Procedures

When training interruptions occur, having proper recovery procedures is essential:

```python
def recover_training_state(checkpoint_path, model, optimizer):
    """
    Recover training state from a checkpoint
    """
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        
        print(f"Successfully restored training from epoch {start_epoch}")
        return start_epoch
        
    except Exception as e:
        print(f"Recovery failed: {str(e)}")
        return 0
```

## Specific Error Solutions

Let's examine solutions for common specific errors you might encounter.

### CUDA Memory Issues

When encountering CUDA out-of-memory errors:

```python
def handle_cuda_memory_error(model, input_size, batch_size):
    """
    Adjust model configuration to handle memory errors
    """
    suggested_batch = batch_size // 2
    
    print("Memory optimization suggestions:")
    print(f"1. Reduce batch size from {batch_size} to {suggested_batch}")
    print("2. Enable gradient checkpointing")
    print("3. Use mixed precision training")
    
    return suggested_batch
```

### Dataset Format Issues

When dataset loading fails:

```python
def validate_dataset_structure(dataset_path):
    """
    Verify dataset structure and format
    """
    expected_structure = {
        'images': ['.jpg', '.png'],
        'labels': ['.txt']
    }
    
    for directory, extensions in expected_structure.items():
        dir_path = os.path.join(dataset_path, directory)
        if not os.path.exists(dir_path):
            print(f"Missing directory: {directory}")
            continue
            
        files = os.listdir(dir_path)
        valid_files = [f for f in files if any(f.endswith(ext) for ext in extensions)]
        
        if not valid_files:
            print(f"No valid files found in {directory}")