# Models

This directory contains pre-trained models (only in YOLO, as of now)for various baseball-related object detection tasks.

## Available Models

- `bat_tracking.pt`: Utilizes YOLOv8 Detection model to detect the bat in broadcast feeds (model in PyTorch).
- `ball_tracking.pt`: Utilizes YOLOv9 Detection Model to track the baseball from pitcher's hand to home plate (model in PyTorch).
- `pitcher_hitter_catcher_detector.pt`: Utilizes YOLOv8 Detection model to detect the pitcher, hitter, and catcher in broadcast feeds (model in PyTorch).
- `glove_tracking.pt`: Utilizes YOLOv9 Detection model to detect the catcher's glove, the ball, the rubber, and home plate (model in PyTorch).

## Model Weights

The model weights can be downloaded using the following links:

- Pitcher Hitter Catcher Detector: [pitcher_hitter_catcher_detector_v3.pt](https://data.balldatalab.com/index.php/s/SciCLNYR5QGkjfK/download/pitcher_hitter_catcher_detector_v3.pt)
- Glove Tracking: [glove_tracking_v1.pt](https://data.balldatalab.com/index.php/s/QHmGwgYnwwbXybx/download/glove_tracking_v1.pt)
- Ball Tracking: [ball_tracking_v2-YOLOv9.pt](https://data.balldatalab.com/index.php/s/dczeCqJyTaQX7aq/download/ball_tracking_v2-YOLOv9.pt)
- Bat Tracking: [bat_tracking.pt](https://data.balldatalab.com/index.php/s/SqMzsxKkCrzojSF/download/bat_tracking.pt)

## Usage

If you prefer to load the model directly using built-in functions versus downloading the model weights, you can use the `load_model` function from `scripts.load_tools`:

```python
from scripts.load_tools import load_model

#Download from .txt file
model = YOLO(load_model("models/pitcher_hitter_catcher_detector/model_weights/pitcher_hitter_catcher_detector_v3.txt"))

#Download from alias
model = YOLO(load_model("pitcher_hitter_catcher_detector"))
```
