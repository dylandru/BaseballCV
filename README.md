# BaseballCV
![BaseballCV Logo](assets/baseballcvlogo.png)

 *** A collection of tools and models designed to aid in the use of Computer Vision in baseball. ***

## Goals

Our goal is to provide access to high-quality datasets and computer vision models for baseball. By sharing our annotated data and model weights, we hope to foster innovation in sports analytics and make it easier for developers to create tools that analyze baseball games.

## Available Assets

### 1. Datasets
We provide open-source datasets containing images and corresponding annotations from baseball broadcasts and other video sources. These datasets are curated and annotated with bounding boxes and labels for:

- Pitchers
- Hitters
- Catchers
- Baseball
- Pitcher's Glove
- Other objects of interest

Datasets are available in common formats like YOLO, allowing for easy integration with computer vision pipelines. 

**Download datasets**:

- [YOLO-format dataset](datasets/yolo)

If you are interested in creating your own dataset, we have a variety of images and tools you can use to get started.

**Download raw photos**:

- [Raw photos dataset](datasets/raw_photos)

### 2. Pre-trained Models
We offer pre-trained YOLO models for object detection. The models are trained to detect the aforementioned objects with high accuracy.

**Available YOLO Models**:

- `bat_tracking.pt`: Trained to detect bat movement.
- `ball_tracking.pt`: Trained to detect the baseball from the pitcher's hand to home.
- `pitcher_hitter_catcher.pt`: Trained to detect the pitcher, hitter and catcher.

**Download model weights**:


## Examples

Below are some examples showcasing our models in action. These include both image and video examples where the models detect various objects within a baseball broadcast.

### Image Example

![Example Detection](assets/phc_example_prediction.jpg)

The above image demonstrates our YOLO model detecting a pitcher, hitter, and catcher during a game broadcast.

### Video Example
https://github.com/user-attachments/assets/7f56df7e-2bdb-4057-a1d7-d4d50d81708e



This video showcases our model's ability to track multiple objects, including the ball and mound, in real-time.

## Example Code

### Inference Example

Here's an example of how to use our pre-trained YOLO models to run inference on an image or video.

```python
from ultralytics import YOLO
model = YOLO("models/ball_tracking/model_weights/ball_tracking.pt")
model.predict("example_baseball_broadcast.jpg")
```


## Installation and Setup

To get started with our datasets and models, follow these steps:

### 1. Clone the Repository

```python
git clone https://github.com/your_username/BaseballCV.git
cd BaseballCV
```

### 2. Install Dependencies

```python
pip install -r requirements.txt
```

## Contributing

We welcome contributions from the community! Whether you're looking to improve our datasets, train better models, or build new tools on top of our work, feel free to open a pull request or start a discussion.

### How to Contribute

 - Fork the repository
 - Create a new branch (git checkout -b feature/YourFeature)
 - Commit your changes (git commit -m 'Add YourFeature')
 - Push to the branch (git push origin feature/YourFeature)
 - Open a pull request

## License

This project is licensed under the XXXXX License - see the ![MIT LICENSE](LICENSE) file for details.

## Contact

If you have any questions or would like to discuss the project, please reach out to us at dylandrummey22@gmail.com or c.marcano@balldatalab.com. You can also follow us on Twitter/X to stay updated:

- Dylan Drummey: [@drummeydylan](https://x.com/DrummeyDylan)
- Carlos Marcano: [@camarcano](https://x.com/camarcano)

