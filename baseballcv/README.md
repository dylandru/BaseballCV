# BaseballCV

**A Computer Vision Toolkit for Baseball Analytics**

[![PyPI version](https://badge.fury.io/py/baseballcv.svg)](https://pypi.org/project/baseballcv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Installation

```bash
pip install baseballcv
```

## Package Structure

![alt text](/assets/print.png)




## Key Features

### 1. Dataset Handling

```python

from baseballcv.datasets import CocoDetectionDataset, JSONLDetection

# Load COCO dataset
coco_data = CocoDetectionDataset("path/to/dataset", "train", processor)

# Convert YOLO to JSONL format
entries = JSONLDetection.load_jsonl_entries("annotations.jsonl", logger)
dataset = JSONLDetection(entries, "images/", logger, augment=True)
```

### 2. Model Implementations

Florence-2 Fine-tuning:

```python
from baseballcv.model import Florence2

# Initialize and fine-tune
model = Florence2(model_id="microsoft/Florence-2-large")
model.finetune(dataset="baseball_data/", classes={0: "ball", 1: "bat"})

# Inference
results = model.inference("pitch_sequence.mp4", task="<OPEN_VOCABULARY_DETECTION>")
```

DETR Object Detection:

```python
from baseballcv.model import DETR

# Initialize and train
detr = DETR(num_labels=5)
detr.finetune(dataset_dir="coco_data/", classes={0: "pitcher", 1: "batter"})

# Detect objects
detections = detr.inference("game_frame.jpg", conf=0.5)
```

### 3. Data Processing Pipeline

```python
from baseballcv import DataTools

# Generate dataset from videos
dt = DataTools()
dt.generate_photo_dataset(
    output_frames_folder="dataset/",
    start_date="2024-05-01",
    end_date="2024-05-30",
    max_plays=500
)

# Auto-annotate with YOLO
dt.automated_annotation(
    model_alias="yolov8_baseball",
    image_dir="dataset/",
    output_dir="annotated_data/"
)
```

### 4. Baseball Savant Integration

```python
from baseballcv.functions import BaseballSavVideoScraper

# Scrape game videos
scraper = BaseballSavVideoScraper('2024-04-10', '2024-05-10', download_folder='videos')
scraper.run_executor()
```

## Core Components

### Model Utilities

Model Utilities 

Visualization - model_visualization_tools.py

```python
from baseballcv.model import ModelVisualizationTools

viz = ModelVisualizationTools("yolov8", "runs/", logger)
viz.visualize_detection_results("image.jpg", detections, labels)
```

Logging - model_logger.py

```python
from baseballcv.model import ModelLogger

logger = ModelLogger(
    model_name="florence2",
    model_run_path="runs/",
    model_id="baseball-analysis",
    batch_size=8,
    device="cuda"
).orig_logging()
```

### Dataset Tools

COCO ↔ JSONL Conversion:

```python
from baseballcv.datasets.processing import DataProcessor

processor = DataProcessor(logger)
train_path, valid_path = processor.prepare_dataset(
    base_path="dataset/",
    dict_classes={0: "ball", 1: "bat"},
    train_test_split=(80, 10, 10)
)
```

### Advanced Usage

Training Configuration

```python
from baseballcv.model.model_function_utils import ModelFunctionUtils

utils = ModelFunctionUtils(
    model_name="florence2",
    model_run_path="runs/",
    batch_size=8,
    device="cuda",
    processor=processor,
    model=model,
    logger=logger
)

# Configure PEFT
peft_config = utils.setup_peft(r=8, alpha=8, dropout=0.1)

# Setup data loaders
train_loader, val_loader = utils.setup_data_loaders(
    train_image_path="train/images/",
    valid_image_path="valid/images/",
    train_jsonl_path="train_annotations.jsonl",
    valid_jsonl_path="valid_annotations.jsonl"
)
```

### Dependencies

| Package | Version |
|---------|---------|
| Python | ≥3.8 |
| PyTorch | ≥2.0 |
| Transformers | Latest |
| Ultralytics YOLO | Latest |
| Supervision | Latest |
| Pandas | Latest |
| OpenCV | Latest |

For a complete list of dependencies and version requirements, see [requirements.txt](requirements.txt).

### Contributing

1 Clone repository:

```python
git clone https://github.com/dylandru/BaseballCV.git
cd BaseballCV
```

2 Install development dependencies:

```python
pip install -e .[dev]
```

3 Run tests:

```python
pytest tests/
```

### License

This project is licensed under the MIT License - please see model folders for more information on individual licenses.