---
layout: default
title: Datasets
nav_order: 7
has_children: false
permalink: /datasets
---

# BaseballCV Datasets

BaseballCV provides several carefully curated datasets for computer vision tasks in baseball. These datasets are designed to support various analysis needs, from basic object detection to complex tracking tasks.

## Available Datasets

### YOLO Format Datasets

{: .note }
YOLO format datasets are primarily used for object detection tasks and are compatible with YOLO models.

1. **baseball_rubber_home_glove**
   - Content: Baseball, rubber, homeplate, and catcher's glove annotations
   - Distribution:
     - Train: 3,925 images with annotations
     - Test: 88 images with annotations
     - Valid: 98 images with annotations
   - Class Distribution:
     ```python
     classes = {
         0: 'glove',
         1: 'homeplate',
         2: 'baseball',
         3: 'rubber'
     }
     ```

2. **baseball_rubber_home**
   - Content: Baseball, rubber, and homeplate annotations
   - Distribution:
     - Train: 3,905 images with annotations
     - Test: 87 images with annotations
     - Valid: 98 images with annotations
   - Class Distribution:
     ```python
     classes = {
         15: 'homeplate',
         16: 'baseball',
         17: 'rubber'
     }
     ```

3. **baseball**
   - Content: Baseball-only annotations, optimized for ball tracking
   - Distribution:
     - Train: 4,534 images
     - Test: 426 images
     - Valid: 375 images

4. **OKD_NOKD**
   - Content: Catcher stance classification (One Knee Down vs No One Knee Down)
   - Distribution:
     - OKD: 1,408 images
     - NOKD: 1,408 images

5. **amateur_pitcher_hitter**
   - Content: Pitchers and hitters in amateur baseball games
   - Distribution:
     - Train: NA
     - Test: NA
     - Valid: NA

### COCO Format Datasets

{: .note }
COCO format datasets are ideal for training DETR and other transformer-based models.

1. **baseball_rubber_home_COCO**
   - Content: COCO format annotations for baseball, rubber, and homeplate
   - Distribution:
     - Train: 3,994 images
     - Test: 774 images
     - Valid: 400 images

2. **baseball_rubber_home_glove_COCO**
   - Content: COCO format annotations including glove tracking
   - Distribution:
     - Train: 3,475 images
     - Test: 408 images
     - Valid: 206 images

### Raw Photo Datasets

1. **broadcast_10k_frames**
   - Content: 10,000 unannotated MLB broadcast frames
   - Resolution: 1280 x 720 pixels
   - Usage: Custom annotation and model training

2. **broadcast_15k_frames**
   - Content: 15,000 unannotated MLB broadcast frames
   - Resolution: 1280 x 720 pixels
   - Usage: Custom annotation and model training

## Using the Datasets

### Loading Datasets

BaseballCV provides a simple interface for loading datasets:

```python
from baseballcv.functions import LoadTools

# Initialize LoadTools
load_tools = LoadTools()

# Load a pre-annotated dataset
dataset_path = load_tools.load_dataset("baseball_rubber_home_glove")

# Load raw photos for custom annotation
raw_dataset = load_tools.load_dataset("broadcast_10k_frames")
```

### Working with Different Formats

#### YOLO Format
```python
def process_yolo_dataset(dataset_path):
    """Process a YOLO format dataset"""
    # Label format: class_id x_center y_center width height
    labels_dir = os.path.join(dataset_path, "labels")
    images_dir = os.path.join(dataset_path, "images")
    
    for label_file in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            annotations = f.readlines()
            # Each line: class_id x_center y_center width height
```

#### COCO Format
```python
def process_coco_dataset(dataset_path):
    """Process a COCO format dataset"""
    from pycocotools.coco import COCO
    
    annotation_file = os.path.join(dataset_path, "_annotations.json")
    coco = COCO(annotation_file)
    
    # Access images and annotations
    image_ids = coco.getImgIds()
    categories = coco.loadCats(coco.getCatIds())
```

### Creating Custom Datasets

BaseballCV supports creating custom datasets from video footage:

```python
from baseballcv.functions import DataTools

data_tools = DataTools()

# Generate dataset from videos
data_tools.generate_photo_dataset(
    output_frames_folder="custom_dataset",
    max_plays=100,
    max_num_frames=1000,
    start_date="2024-05-01",
    end_date="2024-05-31"
)

# Auto-annotate using existing model
data_tools.automated_annotation(
    model_alias="ball_tracking",
    image_dir="custom_dataset",
    output_dir="annotated_dataset",
    conf=0.8
)
```

## Dataset Quality Considerations

When working with BaseballCV datasets, consider:

1. **Resolution and Quality**
   - All datasets are derived from MLB broadcast footage
   - Standard resolution: 1280x720 pixels
   - Various lighting conditions and camera angles

2. **Annotation Consistency**
   - Ball annotations are centered on the baseball
   - Glove annotations include the pocket area
   - Plate and rubber annotations cover the entire visible area

3. **Class Balance**
   - Baseball class is present in most frames
   - Other objects may have varying frequencies
   - Consider class weights during training

4. **Data Splits**
   - Train/test/validation splits are consistent across formats
   - Splits maintain temporal separation to prevent data leakage

## Best Practices

1. **Dataset Selection**
   - Choose format based on model architecture (YOLO vs COCO)
   - Consider using combined datasets for multi-object detection
   - Start with smaller datasets for prototyping

2. **Data Augmentation**
   - Apply appropriate augmentations for baseball scenarios
   - Consider motion blur and lighting variations
   - Maintain object visibility during augmentation

3. **Validation**
   - Regularly validate annotations during training
   - Use visualization tools to check detection quality
   - Monitor class distribution in custom datasets