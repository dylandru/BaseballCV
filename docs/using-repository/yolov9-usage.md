---
layout: default
title: Using YOLOv9
parent: Using the Repository
nav_order: 5
---

# Using YOLOv9

BaseballCV provides a streamlined interface for using YOLOv9 models for object detection in baseball contexts. This guide will help you get started with using YOLOv9 models for efficient and accurate detection tasks.

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
    device="cuda",           # Device to run on ("cuda", "cpu", or GPU index)
    model_path='',          # Path to custom weights
    cfg_path='models/detect/yolov9-c.yaml',  # Model configuration
    name='yolov9-c'        # Model name
)
```

Available model names include:
- `yolov9-c` - Compact model with balanced speed and accuracy
- `yolov9-e` - Enhanced model with higher accuracy
- `yolov9-s` - Small model for speed-critical applications

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

The method returns a list of dictionaries containing detection results that you can process in your application.

## Practical Example: Tracking the Baseball

Here's a practical example of using YOLOv9 for baseball tracking:

```python
from baseballcv.model import YOLOv9
from baseballcv.functions import LoadTools
import cv2
import numpy as np

# Load ball tracking model
load_tools = LoadTools()
model = YOLOv9(device="cuda", name="ball_tracking")

# Process a baseball video
video_path = "baseball_pitch.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = "tracked_pitch.mp4"

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Track ball through video
ball_trajectory = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference
    results = model.inference(
        source=frame,
        conf_thres=0.35,
        iou_thres=0.45
    )
    
    # Process results
    for detection in results:
        boxes = detection.get('boxes', [])
        scores = detection.get('scores', [])
        labels = detection.get('labels', [])
        
        for box, score, label in zip(boxes, scores, labels):
            if model.model.names[int(label)].lower() == 'baseball':
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Add to trajectory
                ball_trajectory.append((frame_idx, center_x, center_y))
                
                # Draw box and center
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    
    # Draw trajectory
    if len(ball_trajectory) > 1:
        for i in range(1, len(ball_trajectory)):
            if ball_trajectory[i][0] - ball_trajectory[i-1][0] <= 3:  # Only connect nearby frames
                pt1 = (ball_trajectory[i-1][1], ball_trajectory[i-1][2])
                pt2 = (ball_trajectory[i][1], ball_trajectory[i][2])
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
    
    # Write frame
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
```

## Model Evaluation

To evaluate model performance on a dataset:

```python
metrics = model.evaluate(
    data_path="data.yaml",  # Dataset configuration file
    batch_size=32,          # Batch size
    imgsz=640,              # Image size
    conf_thres=0.001,       # Confidence threshold
    iou_thres=0.7,          # IoU threshold
    max_det=300             # Maximum detections per image
)

print(f"mAP@0.5: {metrics[0]}")
print(f"mAP@0.5:0.95: {metrics[1]}")
```

## Fine-tuning

To fine-tune the model on your own dataset:

```python
results = model.finetune(
    data_path="data.yaml",  # Dataset configuration
    epochs=100,             # Number of epochs
    imgsz=640,              # Image size
    batch_size=16,          # Batch size
    patience=100,           # Early stopping patience
    optimizer='SGD'         # Optimizer (SGD, Adam, AdamW)
)
```

### Dataset Format

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
    boxes = detection.get('boxes', [])     # [x1, y1, x2, y2]
    classes = detection.get('classes', []) # Class IDs
    scores = detection.get('scores', [])   # Confidence scores
    
    for box, class_id, score in zip(boxes, classes, scores):
        class_name = model.model.names[int(class_id)]
        print(f"Detected {class_name} with confidence {score:.2f} at {box}")
```

## Performance Tips

1. Use GPU when available by setting `device="cuda"` or `device="0"` (specific GPU index)
2. Adjust batch size based on available memory
3. Use appropriate image size (`imgsz`) for your use case
4. Tune confidence and IoU thresholds for optimal results
5. Use `vid_stride` to skip frames when processing long videos
6. Enable half-precision with `half=True` for faster inference
