# BaseballCV Model Classes

This directory contains the model class implementations for BaseballCV. Each model class provides a standardized interface for loading and using different types of computer vision / VLM models for baseball analysis.

## Available Models

### 1. Florence2
An open-source Microsoft Florence 2 VLM model for baseball analysis.

#### Using Inference

```python
from scripts.model_classes import Florence2

# Initialize model
model = Florence2()

# Run object detection
detection_results = model.inference(
    image_path='baseball_game.jpg', 
    task='<OD>' 
)

# Run detailed captioning
caption_results = model.inference(
    image_path='baseball_game.jpg',
    task='<DETAILED_CAPTION>'  
)

# Run open vocabulary detection
specific_objects = model.inference(
    image_path='baseball_game.jpg',
    task='<OPEN_VOCABULARY_DETECTION>',
    text_input='Find the baseball, pitcher, and catcher' 
```

#### Using Fine-Tuning

```python
from scripts.model_classes import Florence2

# Initialize model
model = Florence2()

# Fine-tune model on dataset
metrics = model.finetune(
    dataset='baseball_dataset',
    classes={
        0: 'baseball',
        1: 'glove',
        2: 'bat',
        3: 'player'
    },
    epochs=20,
    lr=4e-6,
    save_dir='baseball_model_checkpoints',
    num_workers=4,
    patience=5,
    warmup_epochs=1
)
```

#### Features
- VLM
- Batch processing capability
- CPU, MPS, and CUDA GPU support
- Configurable confidence threshold
- Memory-efficient processing


## Adding New Models

When adding new model classes, follow these guidelines:

1. Create a new Python file for your model class
2. Implement the standard interface:
   ```python
   class NewModel:
       def __init__(self, **kwargs):
           # Initialize model and parameters
           pass
           
       def inference(self, image_path: str, task: str, **kwargs) -> list:
           # Run inference on image
           pass
           
       def fine_tune(self, dataset: str, classes: list, **kwargs) -> list:
           # Fine-tune model on dataset
           pass
   ```

3. Add model to `__init__.py`
4. Update this README with model documentation

## Common Requirements

All model classes should:
- Provide consistent output format
- Include proper error handling
- Support batch processing (when possible)
- Be memory efficient
- Include docstrings and type hints
- Support GPU, MPS, and CPU (the more the merrier)

## Future Models
Future model classes will be added for:
- PaliGemma 2
- Phi 3.5
- Other object detection frameworks

## Contributing

When contributing new model classes:
1. Follow the standard interface
2. Include comprehensive documentation
3. Add tests in `tests/test_modelname.py`
4. Update requirements.txt if needed
5. Provide example usage
