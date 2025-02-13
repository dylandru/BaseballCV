---
layout: default
title: Getting Started
nav_order: 5
has_children: true
permalink: /getting-started
---

## Using Google Colab for Training

While local installation works well for many use cases, training deep learning models often requires substantial computational resources. Google Colab provides free access to GPU resources, making it an excellent platform for training BaseballCV models. Let's explore how to effectively use BaseballCV in Google Colab.

### Setting Up Colab Environment

First, we need to configure our Colab environment properly. This setup process needs to be performed at the start of each Colab session, as Colab environments reset with each new connection. Here's a complete setup script that handles all necessary configuration:

```python
# Install BaseballCV and its dependencies
!pip install git+https://github.com/dylandru/BaseballCV.git
!pip install supervision==0.3.0 ultralytics>=8.2.90 transformers==4.48.0

# Connect to Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Verify GPU availability and specifications
!nvidia-smi

# Import essential modules
from baseballcv.functions import LoadTools, DataTools
from baseballcv.models import Florence2, DETR
```

### Managing Training Data in Colab

Training deep learning models requires efficient data management. When working with Colab, we need to consider both accessibility and persistence of our datasets. Here's how to organize your training data effectively:

```python
# Create persistent directories in Google Drive
!mkdir -p /content/drive/MyDrive/BaseballCV/datasets
!mkdir -p /content/drive/MyDrive/BaseballCV/models

# Initialize data tools with Drive paths
data_tools = DataTools()

# Download and prepare a dataset
dataset_path = data_tools.load_dataset(
    "baseball_rubber_home_glove",
    output_dir="/content/drive/MyDrive/BaseballCV/datasets"
)
```

### Optimizing Training Workflows

Training in Colab requires careful consideration of session time limits and potential disconnections. Here's how to implement robust training workflows:

```python
def setup_training_session(model_type="DETR", dataset_name=None):
    """
    Set up a complete training session with checkpointing and optimization
    """
    # Configure model with optimizations for Colab
    if model_type == "DETR":
        model = DETR(
            num_labels=4,
            batch_size=8,  # Adjust based on available memory
            device='cuda'  # Use GPU when available
        )
    elif model_type == "Florence2":
        model = Florence2(
            batch_size=8,
            torch_dtype=torch.float16  # Use mixed precision
        )
    
    # Set up checkpoint directory in Drive
    checkpoint_dir = "/content/drive/MyDrive/BaseballCV/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Configure training parameters
    training_config = {
        'dataset': dataset_path,
        'epochs': 100,
        'save_dir': checkpoint_dir,
        'gradient_accumulation_steps': 4,  # Help with memory management
        'save_frequency': 5  # Save every 5 epochs
    }
    
    return model, training_config
```

### Best Practices for Colab Training

When training models in Colab, follow these essential practices for optimal results:

1. Start with a small subset of your data to verify the setup and workflow.
2. Use Google Drive for persistent storage of both models and datasets.
3. Implement regular checkpointing to handle potential disconnections.
4. Monitor GPU memory usage and adjust batch sizes accordingly.
5. Consider using mixed precision training to improve performance.

Here's an example of implementing these practices:

```python
def train_with_monitoring():
    """
    Training workflow with monitoring and safety measures
    """
    # Initialize monitoring
    import psutil
    import torch
    
    def check_resources():
        gpu_memory = torch.cuda.memory_allocated()
        ram_usage = psutil.virtual_memory().percent
        return gpu_memory, ram_usage
    
    # Training loop with monitoring
    try:
        while training:
            gpu_mem, ram = check_resources()
            if gpu_mem > 0.95 * torch.cuda.get_device_properties(0).total_memory:
                print("Warning: GPU memory near capacity")
                # Implement memory optimization strategies
                
            # Continue training
            model.train_step()
            
    except RuntimeError as e:
        print(f"Training interrupted: {str(e)}")
        # Save emergency checkpoint
        model.save_checkpoint("emergency_save")
```

## Next Steps

Now that you have BaseballCV installed and configured, whether locally or on Colab, you're ready to start analyzing baseball footage! Consider exploring:

- Our [tutorial notebooks](https://github.com/dylandru/BaseballCV/tree/main/notebooks) for practical examples
- The model documentation to understand available capabilities
- Our dataset tools to prepare your own analysis projects

Remember, BaseballCV is a powerful framework with many capabilities. Take time to experiment with different features and configurations to find what works best for your specific needs.