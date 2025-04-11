---
layout: default
title: Using the Repository
nav_order: 6
has_children: true
permalink: /using-repository
---

# Using BaseballCV: A Practical Guide

Welcome to the practical guide for BaseballCV. We'll explore how to use the framework effectively, starting with basic operations and progressing to more sophisticated analyses. Throughout this guide, we'll use real-world examples to demonstrate BaseballCV's capabilities in analyzing baseball footage and extracting meaningful insights.

## Basic Object Detection

Let's begin with fundamental object detection, which forms the basis of most baseball analysis tasks. We'll start by detecting basic elements like baseballs, players, and equipment in single images.

```python
from baseballcv.functions import LoadTools
from ultralytics import YOLO

# Initialize our tools
load_tools = LoadTools()

# Load a pre-trained model for basic detection
model_path = load_tools.load_model("ball_tracking")
model = YOLO(model_path)

# Perform detection on an image
results = model.predict(
    source="game_footage.jpg",
    conf=0.25,  # Confidence threshold
    show=True   # Display results
)

# Access the detections
for r in results:
    for box in r.boxes:
        # Extract coordinates and confidence
        x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
        confidence = box.conf[0]       # Detection confidence
        class_id = box.cls[0]         # Class ID of the detection
        class_name = model.names[int(class_id)]
        
        print(f"Found {class_name} with {confidence:.2f} confidence")
```

In this example, we're using one of BaseballCV's pre-trained models to detect objects in a baseball scene. The model has been specifically trained to recognize baseball-related objects, making it more accurate than general-purpose detectors for this specific domain.

## Advanced Object Detection with YOLOv9

For improved performance and accuracy in object detection, BaseballCV provides access to the latest YOLOv9 models:

```python
from baseballcv.model import YOLOv9

# Initialize the YOLOv9 model
model = YOLOv9(
    device="cuda",  # Use GPU for faster processing
    name="yolov9-c"  # Use the compact model variant
)

# Run inference on an image or video
results = model.inference(
    source="baseball_game.mp4",
    conf_thres=0.25,  # Confidence threshold
    iou_thres=0.45   # NMS IoU threshold
)

# Process the results
for detection in results:
    boxes = detection['boxes']      # Bounding box coordinates
    classes = detection['classes']  # Class IDs
    scores = detection['scores']    # Confidence scores
    
    for box, cls, score in zip(boxes, classes, scores):
        print(f"Detected {model.model.names[int(cls)]} with confidence {score:.2f}")
```

## Using Vision Language Models

BaseballCV provides powerful vision language models like Florence2 and PaliGemma2 that allow natural language interaction with baseball imagery:

```python
from baseballcv.model import Florence2

# Initialize Florence2 model
model = Florence2()

# Analyze an image with a natural language prompt
results = model.inference(
    image_path="pitcher_windup.jpg",
    task="<DETAILED_CAPTION>"
)

print(f"Image analysis: {results}")

# Ask a specific question about the image
answer = model.inference(
    image_path="pitcher_windup.jpg",
    task="<VQA>",
    text_input="What pitch grip is the pitcher using?"
)

print(f"Answer: {answer}")

# Detect specific objects with open vocabulary detection
detections = model.inference(
    image_path="baseball_field.jpg",
    task="<OPEN_VOCABULARY_DETECTION>",
    text_input="Find the baseball, pitcher's mound, and catcher"
)
```

## Ball Trajectory Analysis

Understanding ball movement is crucial in baseball analytics. Here's how to track and analyze pitch trajectories:

```python
from baseballcv.functions import BaseballTools

# Initialize the baseball tools
baseball_tools = BaseballTools(device="cuda")

# Analyze pitch distance to strike zone
results = baseball_tools.distance_to_zone(
    start_date="2024-05-01",
    end_date="2024-05-02",
    team_abbr="NYY",         # Optional team filter
    pitch_type="FF",        # Optional pitch type filter (e.g., "FF" for fastball)
    max_videos=5,           # Limit number of videos
    create_video=True       # Generate annotated videos
)

# Process the results
for result in results:
    print(f"Play ID: {result['play_id']}")
    print(f"Distance to zone: {result['distance_to_zone']:.2f} inches")
    print(f"Position: {result['position']}")
    print(f"In strike zone: {result['in_zone']}")
```

## Glove Tracking and Analysis

BaseballCV includes advanced catcher's glove tracking capabilities for analyzing positioning, framing, and receiving techniques:

```python
from baseballcv.functions import BaseballTools

# Initialize the baseball tools
baseball_tools = BaseballTools(device="cuda")

# Track catcher's glove in a single video
results = baseball_tools.track_gloves(
    mode="regular",
    video_path="catcher_video.mp4",
    confidence_threshold=0.5,
    enable_filtering=True,           # Filter out physically impossible movements
    generate_heatmap=True,          # Create position heatmap
    show_plot=True                  # Show real-time tracking plot
)

print(f"Video analyzed: {results['output_video']}")
print(f"Tracking data saved to: {results['tracking_data']}")
print(f"Heatmap saved to: {results['heatmap']}")

# Movement statistics
stats = results['movement_stats']
print(f"Total glove movement: {stats['total_distance_inches']:.2f} inches")
print(f"Range of motion (X): {stats['x_range_inches']:.2f} inches")
print(f"Range of motion (Y): {stats['y_range_inches']:.2f} inches")

# Batch process multiple videos
batch_results = baseball_tools.track_gloves(
    mode="batch",
    input_folder="catcher_videos/",
    max_workers=4,               # Use parallel processing
    generate_batch_info=True    # Generate combined statistics
)

# Scrape and analyze videos from Baseball Savant
savant_results = baseball_tools.track_gloves(
    mode="scrape",
    start_date="2024-05-01",
    end_date="2024-05-05", 
    team_abbr="NYY",
    max_videos=10
)
```

## Custom Dataset Generation

Creating custom datasets for model training is straightforward with BaseballCV:

```python
from baseballcv.functions import DataTools

# Initialize DataTools
data_tools = DataTools()

# Generate dataset from videos
dataset_path = data_tools.generate_photo_dataset(
    output_frames_folder="custom_dataset",
    max_plays=100,
    max_num_frames=5000,
    start_date="2024-05-01",
    end_date="2024-05-31",
    delete_savant_videos=True  # Clean up temporary files
)

# Auto-annotate with YOLO model
annotated_path = data_tools.automated_annotation(
    model_alias="ball_tracking",
    image_dir=dataset_path,
    output_dir="annotated_dataset",
    conf=0.8,
    mode="legacy"
)

# Auto-annotate with Autodistill using natural language
ontology = {
    "a baseball in flight": "baseball",
    "a catcher's mitt or glove": "glove",
    "the white home plate on a baseball field": "homeplate"
}

data_tools.automated_annotation(
    image_dir=dataset_path,
    output_dir="annotated_natural_language",
    mode="autodistill",
    ontology=ontology
)
```

## Training Custom Models

To train your own custom models on baseball data:

```python
from baseballcv.model import DETR

# Initialize DETR model for training
model = DETR(
    num_labels=4,  # Number of object classes to detect
    device="cuda",
    batch_size=8
)

# Define the classes
classes = {
    0: "baseball",
    1: "glove",
    2: "homeplate",
    3: "pitcher"
}

# Fine-tune the model
training_results = model.finetune(
    dataset_dir="annotated_dataset",
    classes=classes,
    epochs=50,
    lr=1e-4,
    lr_backbone=1e-5,
    patience=5  # Early stopping
)

# Evaluate the trained model
metrics = model.evaluate(
    dataset_dir="annotated_dataset",
    conf=0.25
)

print(f"Model performance: {metrics}")
```

## Building Complete Analysis Pipelines

BaseballCV's true power emerges when combining its components into comprehensive analysis pipelines:

```python
from baseballcv.functions import LoadTools, DataTools, BaseballTools
from baseballcv.model import Florence2
import os

class PitchAnalysisPipeline:
    """Complete pipeline for pitch analysis"""
    
    def __init__(self, output_dir="analysis_results"):
        self.load_tools = LoadTools()
        self.data_tools = DataTools()
        self.baseball_tools = BaseballTools(device="cuda")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load our models
        self.ball_model = YOLO(self.load_tools.load_model("ball_trackingv4"))
        self.context_model = Florence2()
    
    def analyze_game(self, team, date, pitch_type=None, max_videos=10):
        """Analyze pitches from a specific game"""
        # Step 1: Get pitch videos
        self.baseball_tools.distance_to_zone(
            start_date=date,
            end_date=date,
            team_abbr=team,
            pitch_type=pitch_type,
            max_videos=max_videos,
            create_video=True,
            csv_path=os.path.join(self.output_dir, f"{team}_{date}_pitch_analysis.csv")
        )
        
        # Step 2: Track glove positioning
        self.baseball_tools.track_gloves(
            mode="scrape",
            start_date=date,
            end_date=date,
            team_abbr=team,
            max_videos=max_videos
        )
        
        # Step 3: Extract frames for further analysis
        frames = self.data_tools.generate_photo_dataset(
            output_frames_folder=os.path.join(self.output_dir, "frames"),
            start_date=date,
            end_date=date,
            max_plays=max_videos
        )
        
        # Step 4: Run contextual analysis on key frames
        for frame_file in os.listdir(frames)[:5]:  # Analyze first 5 frames
            frame_path = os.path.join(frames, frame_file)
            context = self.context_model.inference(
                frame_path,
                task="<DETAILED_CAPTION>"
            )
            print(f"Frame analysis: {context}")
        
        print(f"Analysis complete. Results saved to {self.output_dir}")

# Run the pipeline
pipeline = PitchAnalysisPipeline()
pipeline.analyze_game(team="NYY", date="2024-05-01", pitch_type="FF")
```

Through these examples, we've explored several key capabilities of BaseballCV. The framework's modular design allows you to combine these techniques in various ways to create comprehensive analysis pipelines suited to your specific needs.
