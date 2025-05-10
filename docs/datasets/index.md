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

4. **phc**
   - Content: Pitcher, hitter, and catcher annotations
   - Distribution:
     - Train, Test, Valid splits
   - Primary use: Player tracking and positioning analysis

5. **amateur_pitcher_hitter**
   - Content: Pitchers and hitters in amateur baseball games
   - Distribution:
     - Train, Test, Valid splits
   - Primary use: Amateur baseball analysis

6. **OKD_NOKD**
   - Content: Catcher stance classification (One Knee Down vs No One Knee Down)
   - Distribution:
     - OKD: 1,408 images
     - NOKD: 1,408 images

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

### JSONL Format Datasets

{: .note }
JSONL format datasets are optimized for vision-language models like Florence2 and PaliGemma2.

1. **amateur_hitter_pitcher_jsonl**
   - Content: JSONL format annotations for amateur baseball players
   - Distribution:
     - Train, Test, Valid splits
   - Primary use: Vision-language model training for amateur baseball analysis

### Raw Photo Datasets

1. **broadcast_10k_frames**
   - Content: 10,000 unannotated MLB broadcast frames
   - Resolution: 1280 x 720 pixels
   - Usage: Custom annotation and model training

2. **broadcast_15k_frames**
   - Content: 15,000 unannotated MLB broadcast frames
   - Resolution: 1280 x 720 pixels
   - Usage: Custom annotation and model training

### HuggingFace Hosted CV Datasets

1. **international_amateur_baseball_pitcher_photo**
   - Content: 10,000 unannotated international amateur baseball pitcher photos
   - Usage: Custom annotation and model training for amateur baseball pitcher detection

2. **international_amateur_baseball_photos**
   - Content: 100,000 unannotated international amateur baseball photos
   - Usage: Custom annotation and model training for amateur baseball detection

3. **international_amateur_baseball_catcher_photos**
   - Content: 15,000 unannotated international amateur baseball catcher photos
   - Usage: Custom annotation and model training for amateur baseball catcher detection

4. **international_amateur_baseball_catcher_video**
   - Content: Video clips of amateur baseball catchers
   - Usage: Catcher motion analysis and detection model training

5. **international_amateur_baseball_game_video**
   - Content: Complete amateur baseball game footage
   - Usage: Full-game analysis and player tracking

6. **international_amateur_baseball_bp_video**
   - Content: Batting practice footage from amateur baseball
   - Usage: Swing analysis and training

### HuggingFace Hosted Numerical Datasets

1. **mlb_glove_tracking_april_2024**
   - Content: Over 100,000 plays worth of data for Glove Tracking throughout Pitch
   - Usage: Command Estimation, Catcher Training

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

# Load a HuggingFace hosted dataset
hf_dataset = load_tools.load_dataset("international_amateur_baseball_photos")
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

#### JSONL Format
```python
def process_jsonl_dataset(dataset_path):
    """Process a JSONL format dataset for vision-language models"""
    import json
    
    jsonl_path = os.path.join(dataset_path, "train_annotations.json")
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            image_path = os.path.join(dataset_path, entry['image'])
            prefix = entry['prefix']  # Usually a task token like "<OD>"
            suffix = entry['suffix']  # Annotation text with location tags
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

# Auto-annotate using existing model with YOLO format
data_tools.automated_annotation(
    model_alias="ball_tracking",
    image_dir="custom_dataset",
    output_dir="annotated_dataset",
    conf=0.8,
    mode="legacy"
)

# Auto-annotate using Autodistill and natural language
ontology = { 
    "a mitt worn by a baseball player for catching a baseball": "glove",
    "a baseball in flight": "baseball",
    "the white pentagon-shaped home plate on a baseball field": "homeplate"
}

data_tools.automated_annotation(
    model_type="detection",
    image_dir="custom_dataset",
    output_dir="annotated_dataset",
    conf=0.8,
    mode="autodistill",
    ontology=ontology
)
```

## Dataset Quality Considerations

When working with BaseballCV datasets, consider:

1. **Resolution and Quality**
   - Professional datasets are derived from MLB broadcast footage
   - Standard resolution: 1280x720 pixels
   - Amateur datasets vary in quality and camera angles

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
   - Choose format based on model architecture (YOLO vs COCO vs JSONL)
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

4. **Domain Adaptation**
   - When moving between professional and amateur footage, consider retraining
   - Use transfer learning from MLB models to amateur scenarios
   - Validate performance across different camera angles and lighting conditions