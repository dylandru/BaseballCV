# docs/pages/theory/cv_fundamentals.md
---
layout: default
title: Computer Vision Fundamentals
parent: Theory
nav_order: 1
---

# Fundamentals of Computer Vision in Baseball

Computer vision in baseball analytics involves several key concepts that form the foundation of automated visual analysis. This section explores these fundamental concepts and their specific applications in baseball.

## Object Detection

Object detection in baseball involves identifying and localizing key elements in images or video frames:

```python
import cv2
import torch
from ultralytics import YOLO

# Basic example of baseball object detection
def detect_baseball_objects(image_path):
    # Load a pre-trained model
    model = YOLO('yolov8n.pt')
    
    # Read image
    image = cv2.imread(image_path)
    
    # Perform detection
    results = model(image)
    
    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            # Get class and confidence
            cls = int(box.cls)
            conf = float(box.conf)
            
            if conf > 0.5:  # Confidence threshold
                # Draw bounding box
                cv2.rectangle(image, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
    
    return image