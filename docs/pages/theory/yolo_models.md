# docs/pages/theory/yolo_models.md
---
layout: default
title: YOLO Models in Baseball
parent: Theory
nav_order: 2
---

# YOLO Object Detection Models in Baseball Analytics

This section covers the evolution and application of YOLO (You Only Look Once) models in baseball analytics, from basic implementation to advanced use cases.

## Basic YOLO Implementation for Baseball

Before diving into BaseballCV's specialized tools, let's understand how to implement basic YOLO detection for baseball scenarios:

```python
from ultralytics import YOLO
import cv2
import numpy as np

def baseball_detection_pipeline():
    """
    Basic pipeline for baseball detection using YOLO
    """
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')
    
    # Custom training configuration for baseball
    training_config = {
        'epochs': 100,
        'batch_size': 16,
        'img_size': 640,
        'device': 'cuda:0'
    }
    
    # Baseball-specific class mapping
    baseball_classes = {
        0: 'baseball',
        1: 'player',
        2: 'glove',
        3: 'bat'
    }
    
    return model, training_config, baseball_classes

def train_baseball_detector(model, dataset_path, config):
    """
    Train YOLO model on baseball dataset
    """
    results = model.train(
        data=dataset_path,
        epochs=config['epochs'],
        batch=config['batch_size'],
        imgsz=config['img_size']
    )
    return results

def detect_baseball_objects(model, image_path, conf_threshold=0.25):
    """
    Detect baseball objects in image
    """
    results = model(image_path, conf=conf_threshold)
    return results

    