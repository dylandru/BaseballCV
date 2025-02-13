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
│   ├── models/
│   │   ├── od/
│   │   │   └── detr/
│   │   ├── vlm/
│   │   │   ├── florence2/
│   │   │   └── paligemma2/
│   │   └── utils/
│   │       ├── model_function_utils.py
│   │       ├── model_logger.py
│   │       └── model_visualization_tools.py
│   ├── datasets/
│   │   └── formats/
│   │       └── datasets_coco_detection.py
│   │       └── datasets_jsonl_detection.py
│   │   └── processing/
│   │       └── datasets_processor.py
│   └── functions/
│       ├── dataset_tools.py
│       └── load_tools.py
├── datasets/
│   ├── yolo/
│   ├── COCO/
│   └── raw_photos/
├── models/
│   ├── YOLO/
│   └── vlm/
├── notebooks/
├── tests/
├── docs/
├── README.md
└── LICENSE

```

## Key Components

The repository is built around several core components that work together to provide comprehensive baseball analysis capabilities. Let's explore each major component in detail.

### Models (models/)

BaseballCV implements several state-of-the-art computer vision models, organized by type:

**Object Detection (od/)**
- DETR implementation for precise player and equipment detection
- Optimized for baseball-specific scenarios
- Supports both training and inference pipelines

**Vision Language Models (vlm/)**
- Florence2 for multi-modal understanding and queries
- PaliGemma2 for enhanced contextual analysis
- Support for natural language queries about baseball scenes

**Model Utilities (utils/)**
- Function utilities for model operations
- Visualization tools for detection results
- Logging and monitoring capabilities
- Common model operations and helper functions

### Datasets (datasets/)

The datasets module provides standardized dataset handling:
- COCO format support for object detection
- Custom dataset format conversions
- Efficient data loading pipelines
- Support for multiple dataset structures

### Functions (functions/)

Core utility functions that power BaseballCV's capabilities:
- Dataset tools for automated annotation and processing
- Model loading and management utilities
- Data preprocessing and augmentation tools
- Resource optimization functions

### Testing Framework (tests/)

Comprehensive testing infrastructure:
- Unit tests for all major components
- Integration tests for model interactions
- Performance benchmarks
- Dataset validation tools

## Component Interaction

BaseballCV's components are designed to work together seamlessly. A typical workflow might involve:
1. Using dataset_tools for data preparation
2. Loading models through appropriate model classes
3. Processing data using model-specific pipelines
4. Visualizing results with the visualization tools

This modular design allows users to easily customize and extend functionality while maintaining robust integration between components.