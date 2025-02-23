# Using YOLOv9

The BaseballCV package provides a streamlined interface for using YOLOv9 models for object detection in baseball contexts.

## Quick Start

```python
from baseballcv.model import YOLOv9

# Initialize the model
model = YOLOv9(
    device="cuda",  # Use GPU if available
    name="yolov9-c"  # Model configuration name
)

# Run inference on an image or video
results = model.inference(
    source="baseball_game.mp4",
    conf_thres=0.25,  # Confidence threshold
    iou_thres=0.45   # NMS IoU threshold
)
```

## Model Initialization

The YOLOv9 class can be initialized with several parameters:

```python
model = YOLOv9(
    device="cuda",           # Device to run on ("cuda" or "cpu")
    model_path='',          # Path to custom weights
    cfg_path='models/detect/yolov9-c.yaml',  # Model configuration
    name='yolov9-c'        # Model name
)
```

## Running Inference

The `inference()` method supports both images and videos:

```python
results = model.inference(
    source="path/to/image_or_video",  # File path or list of paths
    imgsz=(640, 640),      # Input image size
    conf_thres=0.25,       # Confidence threshold
    iou_thres=0.45,        # NMS IoU threshold
    max_det=1000,          # Maximum detections per image
    view_img=False,        # Show results
    save_txt=False,        # Save results to *.txt
    save_conf=False,       # Save confidences in --save-txt labels
    save_crop=False,       # Save cropped prediction boxes
    hide_labels=False,     # Hide labels
    hide_conf=False,       # Hide confidences
    vid_stride=1           # Video frame-rate stride
)
```

## Model Evaluation

To evaluate model performance on a dataset:

```python
metrics = model.evaluate(
    data_path="data.yaml",  # Dataset configuration file
    batch_size=32,                  # Batch size
    imgsz=640,                      # Image size
    conf_thres=0.001,              # Confidence threshold
    iou_thres=0.7,                 # IoU threshold
    max_det=300                    # Maximum detections per image
)
```

## Fine-tuning

To fine-tune the model on your own dataset:

```python
results = model.finetune(
    data_path="data.yaml",  # Dataset configuration
    epochs=100,                     # Number of epochs
    imgsz=640,                      # Image size
    batch_size=16,                  # Batch size
    patience=100,                   # Early stopping patience
    optimizer='SGD'                 # Optimizer (SGD, Adam, AdamW, LION)
)
```

## Dataset Format

The data configuration file (data.yaml) should follow this format:

```yaml
path: dataset  # Dataset root directory
train: images/train    # Train images (relative to 'path')
val: images/val        # Validation images (relative to 'path')

# Classes
names:
  0: baseball
  1: glove
  2: bat
```

## Advanced Usage

### Custom Training Configuration

```python
model.finetune(
    data_path="data.yaml",
    epochs=100,
    imgsz=640,
    batch_size=16,
    # Training optimizations
    multi_scale=True,      # Vary img-size ±50%
    cos_lr=True,          # Cosine LR scheduler
    label_smoothing=0.1,  # Label smoothing epsilon
    # Early stopping
    patience=50,          # Epochs to wait for improvement
    # Save settings
    save_period=10,       # Save checkpoint every X epochs
    project="baseball_detection"  # Project name for saving
)
```

### Processing Results

The inference method returns a list of dictionaries containing detection results:

```python
results = model.inference("baseball_game.mp4")
for detection in results:
    # Access bounding boxes, classes, and confidences
    boxes = detection['boxes']      # [x1, y1, x2, y2]
    classes = detection['classes']  # Class IDs
    scores = detection['scores']    # Confidence scores
```

## Performance Tips

1. Use GPU when available by setting `device="cuda" / 0` or `device="cpu"`
2. Adjust batch size based on available memory
3. Use appropriate image size (`imgsz`) for your use case
4. Tune confidence and IoU thresholds for optimal results
5. Enable half-precision (`half=True`) for faster inference 