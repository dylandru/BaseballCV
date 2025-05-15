# Inference App

## Overview
The Inference App is a Streamlit web application designed to use pretrained models built in the package to make inferences on uploaded video. This prevents the tedious process of having to use code to make inferences on videos as this UI makes it possible for users to simply upload a file and select model configuration options to generate inferences on the file, in which the user can use to show off to friends, social media, etc. 

*Note: Based on computational limits, these inference models may be a little slow. If you are running an inference on a video, there are 400+ frames to make predictions on so it could take a couple of minutes depending on the device.*

## Features
The following are current supported features for this inferface
* **YOLO**: 
    * Pitcher Hitter Catcher Detector
    * Bat Tracking
    * Ball Tracking
    * Ball Tracking v4
    * Glove Tracking
* **YOLOv9**: 
    * Homeplate Tracking

## Setup
*   Python 3.8+
*   pip
*   poetry==2.1.2 (optional)

### Installation
1.  **Clone Repo & Install Dependencies**:
    Clone this very repository, change your current directory to BaseballCV, and create a virtual environment:

    With poetry (recommended)
    ```bash
    poetry install
    source .venv/bin/activate
    ```
    **Note**: You will need to install `poetry` with pip. See the main [`README.md`](https://github.com/dylandru/BaseballCV/blob/main/README.md) for more details.

    With python's virtual environment
    ```bash
    python -m venv venv
    source venv/bin/activate 
    ```
    Install the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

### Project Structure

*   `app.py`: The main Streamlit application script.
*   `app_utils/`: Contains helper modules:
    *   `dataset_creation.py`: Generates the video files if the user does not have any to upload. For now, the videos generated are at random. It also generates random frames from a video in case the user has a video, but doesn't want to inference every frame.
    *   `file_manager.py`: Manages the directory structure for each streamlit session, ensuring proper files are going to the correct directory. 
    *   `inference_models.py`: Stores every implemented model with their inferences implemented.
*   `README.md`: This file.

## Running the App

Once the setup is complete, you can run the Streamlit application from the BaseballCV repository w/ `streamlit/inference_app/app.py`:

```bash
cd BaseballCV
streamlit run streamlit/inference_app/app.py
```

This will typically open the application in your default web browser.

## How to Use
1. Upload a video **or** image(s). 1 video and up to 10 images are supported. This ensures time and computation are not mishandled. **Optional**: If you don't have a video to use, select the `Download Random Video` button and one will be generated for you. You may have to do it a couple of times if an error pops up. 
2. Select the type of model you want to use for inferencing then select it's respective alias. 
3. Set the confidence of the inference on the respective detected object. Not required, but can be used as a learning tool.
4. Select the option of whether you want to generate random frames from a video. This is in place in case the user doesn't have any images to upload and doesn't want to wait a few minutes for the model to make an inference on every frame of a video. If  `Yes` is selected, 3 random frames with annotations are outputted. 
5. Easiest step: Select `Run Model` to make the inferences
6. **Optional**: You can download the inference image(s) or video to show off to collegues. 

## Troubleshooting
Errors that occur are most likely outside of the user's control, but will be printed on the screen. If the user finds an issue, please create a [issue](https://github.com/dylandru/BaseballCV/issues)

## Future Implentation
This is the first iteration of this UI and we would like to add in more tools and functionality to make this app more interactive. Some things we are actively working on are improving and adding to the model classes for inferences as well as the accuracy for the pretrained models. Some ideas we have that we would like to see in future iterations are:
* Have a specific way to download videos of interest instead of random
* Incorporate the functions in `BaseballTools` to generate a more descriptive output
* Add more parameters for user input aside from model confidence
* Add more models to give a broader landscape of how these models are used and what their outputs look like
* Potentially a way to handle video input in place instead of writing it and accessing it internally through the streamlit session