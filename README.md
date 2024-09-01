# BaseballCV
 A collection of tools and models designed to aid in the use of Computer Vision in baseball

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

- [YOLO-format dataset](link_to_dataset)

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

![Example Detection](link_to_image_example)

The above image demonstrates our YOLO model detecting a pitcher, hitter, and catcher during a live broadcast.

### Video Example
[Watch Video Detection](link_to_video_example)

This video showcases our model's ability to track multiple objects, including the ball and pitcher's glove, in real-time.

## Example Code

Here's an example of how to use our pre-trained YOLO models to run inference on an image or video.

```python
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

This project is licensed under the XXXXX License - see the ![LICENSE](link_to_license)  file for details.

## Contact

If you have any questions or would like to discuss the project, please reach out to us at xxxx@example.com.