---
layout: default
title: Repository Structure
nav_order: 4
has_children: false
permalink: /repository-structure
---

# Repository Structure and Components

BaseballCV is organized into a modular structure that promotes maintainability and ease of use. Understanding this organization helps developers effectively utilize and extend the framework's capabilities.

## Core Directory Structure

```
BaseballCV/
├── baseballcv/
│   ├── model/
│   │   ├── od/
│   │   │   ├── detr/
│   │   │   ├── yolo/
│   │   │   └── rfdetr/
│   │   ├── vlm/
│   │   │   ├── florence2/
│   │   │   └── paligemma2/
│   │   └── utils/
│   │       ├── model_function_utils.py
│   │       └── model_visualization_tools.py
│   ├── datasets/
│   │   ├── formats/
│   │   │   ├── datasets_coco_detection.py
│   │   │   └── datasets_jsonl_detection.py
│   │   └── processing/
│   │       └── datasets_processor.py
│   ├── functions/
│   │   ├── dataset_tools.py
│   │   ├── load_tools.py
│   │   ├── savant_scraper.py
│   │   ├── baseball_tools.py
│   │   └── utils/
│   │       ├── baseball_utils/
│   │       │   ├── distance_to_zone.py
│   │       │   └── glove_tracker.py
│   │       └── savant_utils/
│   │           ├── crawler.py
│   │           └── gameday.py
│   └── utilities/
│       ├── logger/
│       │   ├── baseballcv_logger.py
│       │   └── baseballcv_prog_bar.py
│       └── dependencies/
│           └── git_dependency_installer.py
├── datasets/
│   ├── yolo/
│   ├── COCO/
│   └── raw_photos/
├── models/
│   ├── od/
│   │   ├── YOLO/
│   │   ├── DETR/
│   │   └── RFDETR/
│   └── vlm/
│       ├── Florence2/
│       └── PaliGemma2/
├── notebooks/
├── tests/
├── docs/
├── README.md
└── LICENSE
```

## Key Components

The repository is built around several core components that work together to provide comprehensive baseball analysis capabilities. Let's explore each major component in detail.

### Models (model/)

BaseballCV implements several state-of-the-art computer vision models, organized by type:

**Object Detection (od/)**
- **DETR** implementation for precise player, ball and equipment detection
- **YOLOv9** for real-time object detection with improved performance
- **RFDETR** (Receptive Field DETR) for enhanced detection capabilities
- All optimized for baseball-specific scenarios
- Support for both training and inference pipelines

**Vision Language Models (vlm/)**
- **Florence2** for multi-modal understanding and queries
- **PaliGemma2** for enhanced contextual analysis
- Support for natural language queries about baseball scenes

**Model Utilities (utils/)**
- Function utilities for model operations
- Visualization tools for detection results
- Common model operations and helper functions

### Functions (functions/)

Core utility functions that power BaseballCV's capabilities:

**BaseballTools**
- Distance to zone calculation for pitch analysis
- Glove tracking and movement analysis
- Comprehensive analysis of catcher positioning

**DataTools**
- Dataset generation from videos
- Automated annotation with pre-trained models
- Dataset conversion between formats

**LoadTools**
- Model loading and management
- Dataset downloading and preparation
- Resource handling

**BaseballSavVideoScraper**
- Video acquisition from Baseball Savant
- Pitch-level metadata retrieval
- Filtering by team, pitcher, and pitch type

### Utilities (utilities/)

Shared utility components:

**Logger**
- Comprehensive logging system
- Progress tracking via custom progress bars
- Structured output for debugging and monitoring

**Dependencies**
- Automatic dependency management
- Git-based package installation
- Compatibility verification

### Datasets (datasets/)

Dataset handling and processing:

**Formats**
- COCO format support for object detection
- JSONL format for vision-language model training
- YOLO format for compatibility with YOLO models

**Processing**
- Data processing and transformation
- Format conversion utilities
- Augmentation strategies

## Component Interaction

BaseballCV's components are designed to work together seamlessly. A typical workflow might involve:
1. Using `BaseballSavVideoScraper` to obtain game footage
2. Processing videos with `BaseballTools` for analysis
3. Generating datasets with `DataTools` for model training
4. Training models using the appropriate model implementation
5. Visualizing results with the visualization tools

This modular design allows users to easily customize and extend functionality while maintaining robust integration between components.
