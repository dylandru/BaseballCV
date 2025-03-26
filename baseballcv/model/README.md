# BaseballCV Model Classes

This directory contains the model class implementations for BaseballCV. Each model class provides a standardized interface for loading and using different types of computer vision / VLM models for baseball analysis.

## Available Models

### 1. Florence2
An open-source Microsoft Florence 2 VLM model for baseball analysis.

#### Using Inference

```python
from baseballcv.models import Florence2

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
)

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

### 2. DETR (Detection Transformer)
A Hugging Face implementation of Facebook's DETR model for object detection, optimized for baseball player detection.

#### Using Inference
```python
from baseballcv.models import DETR

# Initialize model
classes = {1: "hitter", 2: "pitcher"}
detr = DETR(
    num_labels=len(classes),
    device="cuda",  # or "cpu", "mps"
    model_id="facebook/detr-resnet-50" #can also be path to local model or other HF DETR model
)

# Run detection on single image
detections = detr.inference(
    file_path="baseball_game.jpg",
    classes=classes,
    conf=0.2
)

```

#### Using Fine-Tuning
```python
# Fine-tune on custom dataset
detr.finetune(
    dataset_dir="baseball_dataset",
    classes=classes,
    save_dir="finetuned_detr",
    batch_size=4,
    epochs=10,
    patience=3
)

# Evaluate model
metrics = detr.evaluate(
    dataset_dir="test_dataset",
    conf=0.2
)
```

#### Features
- PyTorch Lightning integration
- COCO format dataset support
- Automatic mixed precision training
- Early stopping and model checkpointing
- Visualization tools for evaluation
- Multi-GPU support
- Configurable backbone freezing

### 3. PaliGemma 2
A fine-tuned version of the PaliGemma 2B model optimized for baseball play analysis and description.

#### Using Inference
```python
from baseballcv.models import PaliGemma2

# Initialize model
model = PaliGemma2(
    device="cuda",
    model_id="google/paligemma2-3b-pt-224"
)

# Generate play description
description = model.inference(
    image_path="baseball_play.jpg",
    text_input="Describe this baseball play:",
    task="<TEXT_TO_TEXT>"
)

# Answer specific questions for object detection
answer  = model.inference(
    image_path="pitch.jpg",
    text_input="pitcher.",
    task="<TEXT_TO_OD>"
)

```

#### Features
- Vision-language capabilities
- Optimized for baseball terminology
- Memory-efficient processing
- Support for various image formats

### 4. YOLOv9 (GPL-3.0 license)
The `YOLOv9` class provides a similar functionality to the Ultralytics YOLO class with a more permissive license. It acts as a wrapper for this forked repo: [https://github.com/dylandru/yolov9](https://github.com/dylandru/yolov9)

```python
from baseballcv.model.od import YOLOv9

# Initialize model
model = YOLOv9(
    device="cuda",  # or 'cpu', 'mps'
    name="yolov9-c",  # optional, defaults to latest YOLOv9
)

# Run inference on single image
detections = model.inference(
    source="baseball_play.jpg",
    conf=0.25  # confidence threshold
)

# Run inference on video
video_results = model.inference(
    source="game_clip.mp4",
    conf=0.25
)

# Fine-tune on custom dataset
metrics = model.finetune(
    data_path="baseball_dataset/data.yaml",
    epochs=50,
    batch_size=16,
    save_dir="checkpoints/"
)
```

The class automatically handles:
- Model initialization
- Batch processing for efficient inference
- Standardized output format for easy post-processing
- Automatic YOLOv9 installation and weight downloads if needed

### 5. RF DETR (Recurrent Feature DETR)
A better version of DETR, available to be used for inference and finetuning. The `RFDETR` class is a wrapper for the `rfdetr` package, available to be used for inference and finetuning.

```python
from baseballcv.model.od import RFDETR

# Initialize model
model = RFDETR(
    device="cpu",
    model_type="large",
    model_path="models/od/RFDETR/glove_tracking/model_weights/rfdetr_glove_tracking.txt"
)

# Run inference on single image
detections = model.inference(
    source="baseball_play.jpg",
    conf=0.25
)
```

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
           
       def finetune(self, dataset: str, classes: list, **kwargs) -> list:
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
- Qwen 2.5
- Phi 3.5
- Other object detection frameworks

## Contributing

When contributing new model classes:
1. Follow the standard interface
2. Include comprehensive documentation
3. Add tests in `tests/test_modelname.py`
4. Update requirements.txt if needed
5. Provide example usage
