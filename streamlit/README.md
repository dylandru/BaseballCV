# BaseballCV Streamlit App

This directory contains various Streamlit applications for BaseballCV. Each app serves a specific purpose in helping the sabermetric community perform analysis on baseball data utilizing our available computer vision models, datasets, and tools.

## Apps

### 1. Annotation App (`/annotation_app`)

A Streamlit-based app for managing and annotating baseball image and video using BaseballCV's computer vision models.

    - Open-source project designed to crowdsource baseball annotations for training of models
    - Upload personal baseball videos or photos to annotate
    - Annotation interface for labeling objects and keypoints
    - Integration with AWS S3 for data / photo storage
    - Built-in BaseballCV models for predicted annotations to quicken annotation process
    - See detailed instructions in `/annotation_app/README.md`

### 2. Gemini Annotation App (`/gemini_annotating_app`)

A Streamlit-bassed app for implementing automated annotation using Google AI Studio and Gemini.

    - Open-source implementation of a use-case for model distillation using large LLM API calls to create datasets for supervised CV models.
    - Use own photos for annotation
    - Allows for multiple CV annotation formats (including YOLO and COCO)
    - Generates complete dataset with uploaded photos into a downloadable ZIP file.
    - Previews results for user with the annotations shown on image and actual model output
    - See detailed instructions in `/gemini_annotating_app/README.md`


### 3. Inference App (`/inference_app`)

A Streamlit-based app for implementing inference models on baseball video or image feed without using code.

    - Designed to elimate using code to make inferences on images and videos
    - Ability to download the annotated inferences to show off on friends, social media, etc.
    - See detailed instructions in `/inference_app/README.md`

## Common Features

All Streamlit applications in this directory (will) share:
- Consistent UIs
- Integration with BaseballCV library
- AWS compatibility
- Docker deployment options

## Directory Structure

```
streamlit/
├── annotation_app/       # Main annotation application
│   ├── app.py            # Application entry point
│   ├── Dockerfile        # Container config
│   ├── docker-compose.yml # Docker services setup
│   ├── app_utils/        # Utility modules
```

## Development Guidelines

When creating new Streamlit applications:
1. Follow the existing directory structure pattern
2. Include a detailed README.md
3. Use shared utilities from BaseballCV core
4. Attempt to create a similar UI for app
5. Implement Docker support

## Getting Started

1. Choose the appropriate app for your needs
2. Follow the README in the specific app directory
3. Set up required credentials and env
4. Use Docker (currently recommended)

## Requirements

- Python 3.11+
- Streamlit
- Docker (recommended)
- AWS credentials (if needed)
- BaseballCV core libraries
