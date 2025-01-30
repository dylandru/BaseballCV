---
layout: default
title: Using the Repository
nav_order: 6
has_children: false
permalink: /using-repository
---

# Using BaseballCV: A Practical Guide

Welcome to the practical guide for BaseballCV. We'll explore how to use the framework effectively, starting with basic operations and progressing to more sophisticated analyses. Throughout this guide, we'll use real-world examples to demonstrate BaseballCV's capabilities in analyzing baseball footage and extracting meaningful insights.

## Basic Object Detection

Let's begin with fundamental object detection, which forms the basis of most baseball analysis tasks. We'll start by detecting basic elements like baseballs, players, and equipment in single images.

```python
from baseballcv import LoadTools
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
        
        print(f"Found object of class {class_id} with {confidence:.2f} confidence")
```

In this example, we're using one of BaseballCV's pre-trained models to detect objects in a baseball scene. The model has been specifically trained to recognize baseball-related objects, making it more accurate than general-purpose detectors for this specific domain.

## Working with Video Sequences

Baseball analysis typically involves working with video footage. Let's explore how to process video sequences effectively:

```python
from baseballcv.model_classes import Florence2
import cv2

def analyze_pitch_sequence(video_path):
    """
    Analyze a pitch sequence using Florence 2's multimodal capabilities
    """
    # Initialize Florence 2 model for detailed analysis
    model = Florence2()
    
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Analyze each frame
        results = model.inference(
            frame,
            task="<OD>",  # Object detection task
            text_input="Find the baseball, pitcher, and catcher"
        )
        
        frames.append(results)
    
    cap.release()
    return frames

# Process the sequence
sequence_results = analyze_pitch_sequence("pitch_video.mp4")
```

This code demonstrates how to process video sequences frame by frame, maintaining context across the sequence. Florence 2's multimodal capabilities allow us to specify exactly what we're looking for in natural language.

## Advanced Player Analysis

One of BaseballCV's most powerful features is its ability to analyze player mechanics in detail. Here's how to perform detailed player analysis:

```python
from baseballcv.model_classes import DETR

def analyze_pitcher_mechanics(video_path):
    """
    Perform detailed analysis of pitcher mechanics
    """
    # Initialize DETR model with specific focus on player detection
    model = DETR(num_labels=4)  # Players, ball, glove, plate
    
    results = model.inference(
        video_path,
        task="<TEXT_TO_OD>",
        text_input="Track the pitcher's delivery sequence"
    )
    
    # Extract key points from the delivery
    key_frames = {
        'setup': results[0],      # Initial stance
        'leg_lift': results[30],  # Maximum leg lift
        'drive': results[45],     # Drive toward home
        'release': results[60],   # Ball release
        'follow': results[75]     # Follow-through
    }
    
    return key_frames
```

This example shows how we can use DETR's sophisticated detection capabilities to analyze specific aspects of player mechanics. The model understands the sequence of movements and can identify key points in the delivery.

## Ball Trajectory Analysis

Understanding ball movement is crucial in baseball analytics. Here's how to track and analyze pitch trajectories:

```python
from baseballcv import DataTools
import numpy as np

def analyze_pitch_trajectory(video_path):
    """
    Track and analyze pitch trajectory from release to plate
    """
    data_tools = DataTools()
    
    # Generate frame sequence
    frames = data_tools.generate_photo_dataset(
        video_path,
        max_num_frames=30,  # Capture key points in trajectory
        output_frames_folder="trajectory_analysis"
    )
    
    # Track ball positions
    ball_positions = []
    for frame in frames:
        position = model.inference(
            frame,
            task="<OPEN_VOCABULARY_DETECTION>",
            text_input="Locate the baseball"
        )
        ball_positions.append(position)
    
    # Calculate trajectory metrics
    trajectory = np.array(ball_positions)
    
    # Estimate pitch metrics
    initial_velocity = calculate_velocity(trajectory[:5])
    break_amount = calculate_break(trajectory)
    spin_rate = estimate_spin(trajectory)
    
    return {
        'velocity': initial_velocity,
        'break': break_amount,
        'spin_rate': spin_rate,
        'trajectory': trajectory
    }
```

This code demonstrates how to track a baseball through space and calculate important metrics about its movement. The analysis combines multiple frames to build a complete understanding of the pitch's characteristics.

## Catcher Positioning Analysis

Let's examine how to analyze catcher positioning and pitch framing:

```python
def analyze_catcher_framing(video_path):
    """
    Analyze catcher's receiving position and framing technique
    """
    # Load specialized model for glove tracking
    model_path = load_tools.load_model("glove_tracking")
    model = YOLO(model_path)
    
    frames = extract_frames(video_path)
    
    glove_positions = []
    for frame in frames:
        # Detect glove position
        results = model.predict(frame)
        
        # Extract glove coordinates
        glove_box = [box for box in results[0].boxes if box.cls == 0][0]
        glove_positions.append(glove_box.xyxy[0])
    
    # Analyze framing movement
    setup_position = glove_positions[0]
    receive_position = glove_positions[-1]
    
    framing_distance = calculate_distance(setup_position, receive_position)
    framing_time = len(frames) / 30  # Assuming 30fps
    
    return {
        'setup_position': setup_position,
        'receive_position': receive_position,
        'framing_distance': framing_distance,
        'framing_time': framing_time
    }
```

This analysis focuses on the subtle movements of the catcher's glove, helping evaluate receiving technique and framing ability.

Through these examples, we've explored several key capabilities of BaseballCV. Each analysis builds upon the basic object detection capabilities to provide deeper insights into specific aspects of the game. The framework's modular design allows you to combine these techniques in various ways to create comprehensive analysis pipelines suited to your specific needs.