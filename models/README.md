# Models

This directory contains pre-trained models (only in YOLO, as of now)for various baseball-related object detection tasks.

## Available YOLO Models

- `bat_tracking`: Utilizes YOLOv8 Detection model to detect the bat in broadcast feeds (model in PyTorch).
- `ball_tracking`: Utilizes YOLOv9 Detection Model to track the baseball and other objects from pitcher's hand to home plate (model in PyTorch).
- `pitcher_hitter_catcher_detector`: Utilizes YOLOv8 Detection model to detect the pitcher, hitter, and catcher in broadcast feeds (model in PyTorch).
- `glove_tracking`: Utilizes YOLOv9 Detection model to detect the catcher's glove, the ball, the rubber, and home plate (model in PyTorch).
- `ball_trackingv4`: Utilizes YOLOv11 Detection model to track ONLY the baseball from pitcher's hand to home plate (model in PyTorch).

## Available Florence2 Models

- `florence_ball_tracking`: Utilizes a fine-tuned version of Florence2 to track the baseball from pitcher's hand to home plate among other VLM tasks.

## Available PaliGemma2 Models

- `paligemma2_ball_tracking`: Utilizes a fine-tuned version of PaliGemma2 to track the baseball from pitcher's hand to home plate among other VLM tasks.

## Available YOLOv9 Models

- `amateur_pitcher_hitter`: Utilizes YOLOv9 Detection model to detect pitchers and hitters in amateur baseball games (model in PyTorch).
- `homeplate_tracking`: Utilizes YOLOv9 Detection model to detect the home plate during broadcast feeds (model in PyTorch).

## Model Weights

The model weights can be downloaded directly from the links below:

### YOLO Models

- Pitcher Hitter Catcher Detector: [pitcher_hitter_catcher_detector_v3.pt](https://data.balldatalab.com/index.php/s/SciCLNYR5QGkjfK/download/pitcher_hitter_catcher_detector_v3.pt)
- Glove Tracking: [glove_tracking_v1.pt](https://data.balldatalab.com/index.php/s/QHmGwgYnwwbXybx/download/glove_tracking_v1.pt)
- Ball Tracking: [ball_tracking_v3-YOLOv11.pt](https://data.balldatalab.com/index.php/s/xdaTeMtmnpx8NbE/download/ball_tracking_v3-YOLOv11.pt)
- Bat Tracking: [bat_tracking.pt](https://data.balldatalab.com/index.php/s/SqMzsxKkCrzojSF/download/bat_tracking.pt)
- Ball Tracking v4: [ball_tracking_v4-YOLOv11.pt](https://data.balldatalab.com/index.php/s/cfoxjLS5BiZAgKs/download/ball_tracking_v4-YOLOv11.pt)
- Amateur Pitcher Hitter: [amateur_pitcher_hitter_v1.pt](https://data.balldatalab.com/index.php/s/bqQZG5Gz7qsMq9X/download/amateur_pitcher_hitter_v1.pt)
- Homeplate Tracking: [homeplate_tracking.pt](https://data.balldatalab.com/index.php/s/fgTr8w9ASc6XJpS/download/homeplate_tracking.pt)

### Florence2 Models

- Ball Tracking: [florence_ball_tracking](https://data.balldatalab.com/index.php/s/eFmKZXp3A3ko3jF/download/)

### PaliGemma2 Models

- Ball Tracking: [paligemma2_ball_tracking](https://data.balldatalab.com/index.php/s/CqDQFRoGR7wp6Ls/download/paligemma2_ball_tracking.zip)


## Usage

If you prefer to load the model directly using built-in functions versus downloading the model weights, you can use the `load_model` function from `scripts.load_tools`:

```python
from baseballcv.functions import LoadTools
from ultralytics import YOLO

# Initialize LoadTools class
load_tools = LoadTools()

#Download from .txt file
model_path = load_tools.load_model("models/pitcher_hitter_catcher_detector/model_weights/pitcher_hitter_catcher_detector_v4.txt")

#Download from alias
model_path = load_tools.load_model("pitcher_hitter_catcher_detector")

# Initialize model with YOLO
model = YOLO(model_path)
```
