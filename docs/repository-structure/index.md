---
layout: default
title: Repository Structure
nav_order: 4
has_children: true
permalink: /repository-structure
---

# Repository Structure and Components

BaseballCV is organized into a modular structure that promotes maintainability and ease of use. Understanding this organization helps developers effectively utilize and extend the framework's capabilities.

## Core Directory Structure

```
BaseballCV/
├── scripts/
│   ├── model_classes/
│   ├── function_utils/
│   ├── dataset_tools.py
│   ├── load_tools.py
│   └── savant_scraper.py
├── datasets/
│   ├── yolo/
│   ├── COCO/
│   └── raw_photos/
├── models/
│   ├── YOLO/
│   └── vlm/
├── notebooks/
├── tests/
└── docs/
```

## Key Components

The repository is built around several core components that work together to provide comprehensive baseball analysis capabilities. Let's explore each major component in detail.

### Utility Scripts (scripts/)

The scripts directory contains the essential tools that form the backbone of BaseballCV's functionality. These utilities handle everything from data management to model operations.

**dataset_tools.py**
This module provides robust tools for dataset creation and management. It includes sophisticated functions for:
- Automated photo dataset generation from baseball footage
- Intelligent frame extraction with configurable parameters
- Semi-supervised annotation tools
- Dataset format conversion and validation

**load_tools.py**
The loading utility serves as a central hub for managing models and datasets. It handles:
- Efficient model weight management and caching
- Dynamic dataset loading and organization
- Automatic format detection and conversion
- Resource optimization for different hardware configurations

**savant_scraper.py**
This specialized tool interfaces with Baseball Savant to provide comprehensive data collection capabilities. It includes:
- Intelligent video scraping with customizable filters
- Automated metadata extraction
- Play-by-play synchronization
- Concurrent download management

### Model Classes (scripts/model_classes/)

BaseballCV implements several state-of-the-art computer vision models, each chosen for specific strengths in baseball analysis:

**DETR (Detection Transformer)**
Our DETR implementation excels at understanding complex spatial relationships in baseball scenes. It provides:
- Global scene understanding through transformer architecture
- Precise player and equipment localization
- Robust handling of occlusions and overlapping objects
- Baseball-specific optimizations for improved performance

**Florence 2**
The Florence 2 implementation brings powerful vision-language capabilities to baseball analysis:
- Natural language query processing for baseball scenarios
- Multi-modal understanding of game situations
- Context-aware analysis of player actions
- Integration with baseball-specific terminology

**PaliGemma 2**
Our PaliGemma 2 implementation enhances contextual understanding:
- Advanced sequence modeling for game situations
- Integrated caption generation for automated analysis
- Fine-grained action recognition
- Robust performance on broadcast footage

**YOLO Models**
The YOLO implementation provides real-time detection capabilities:
- Custom-trained models for baseball-specific objects
- Optimized for broadcast footage analysis
- Multiple specialized variants for different tasks
- High-performance inference pipelines

### Datasets

The datasets directory contains carefully curated collections organized by format:
- YOLO format datasets for object detection
- COCO format annotations for advanced analysis
- Raw photo collections for custom training
- Specialized datasets for specific tasks

### Function Utilities (scripts/function_utils/)

These utilities provide essential support functions:
- Video frame extraction and processing
- Data format conversion tools
- Visualization helpers
- Performance optimization utilities

### Testing Framework

The tests directory contains comprehensive testing suites:
- Unit tests for all major components
- Integration tests for model interactions
- Performance benchmarks
- Dataset validation tools

## Component Interaction

BaseballCV's components are designed to work together seamlessly. For example, a typical workflow might involve:
1. Using savant_scraper to collect game footage
2. Processing with dataset_tools to create training data
3. Loading models through load_tools
4. Applying multiple models for comprehensive analysis

This modular design allows users to easily customize and extend functionality while maintaining robust integration between components.