# Notebooks

The notebooks directory contains Jupyter notebooks demonstrating various functionalities and use-cases of the BaseballCV project.

## Available Notebooks

- `ball_inference_YOLOv9.ipynb`: Demonstrates ball tracking using YOLOv9
- `Glove_tracking.ipynb`: Shows how to use the glove tracking model and extract predictions from the video
- `YOLO_PHC_detection_extraction.ipynb`: Demonstrates pitcher, hitter, catcher detection and prediction extraction (future use-cases will be integrated)
- `glove_framing_tracking.ipynb`: Demonstrates how to extract the glove tracking model coordinates and transpose them to a 2D plane and plots the result.
- `model_fine_tuning.ipynb`: Demonstrates how to auto fine tune our models, using the glove tracking model as an example.
- `okd_nkd`: Creates a model through a classifier to detect One Knee Down postion, or others.
- `autolabeling_semisupervised.ipynb`: Demonstrates how to use DataTools class for auto annotating images using the BaseballCV pipeline, in a semisupervised manner.

## Usage

To use these notebooks:

1. Install all required dependencies (see main README.md)
2. Launch Jupyter Notebook, JupyterLab, or Google Colab
3. Navigate to this directory and open the desired notebook

Each notebook contains step-by-step instructions and explanations for using the models and the built-in repo functions.

## Notebook Details

### Ball Inference Notebook

The `ball_inference_YOLOv9.ipynb` notebook demonstrates how to:
- Load the ball_inference model
- Perform inference on video frames
- Visualize the results of ball detection

### Glove Tracking Notebook

The `Glove_tracking.ipynb` notebook shows how to:
- Load the glove tracking detection model
- Perform inference on video
- Extract predictions from inference for use
- Visualize the results of glove tracking

### Pitcher Hitter Catcher Detection and Extraction Notebook

The `YOLO_PHC_detection_extraction.ipynb` notebook demonstrates how to:
- Load the PHC detection model
- Perform inference on video frames
- Visualize the results of PHC detection

### Glove Tracking 2D Transposing Notebook

The `glove_framing_tracking.ipynb` notebook shows how to:
- Load the glove tracking detection model
- Perform inference on video
- Extract predictions from inference for use
- Transpose the glove predictions to a 2D reference plane.
- Visualize the results of glove tracking

### OKD / NOKD Classifier Notebook

The `okd_nokd_classifier.ipynb` notebook shows how to:
- Load the OKD / NOKD dataset
- Extract pose data from the dataset
- Train a classifier model to predict if a catcher is in a one-knee down position
- Visualize the results of the classification

### Semi-supervised Auto-Labeling Notebook

The `autolabeling_semisupervised.ipynb` notebook shows how to:
- Create a base dataset
- Auto annotate it
- View the results for manual filtering

These notebooks should guide the user in using the BaseballCV tools in their own projects.
