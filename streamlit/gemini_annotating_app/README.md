# Gemini Baseball Annotation App

## Overview

The Gemini Baseball Annotation App is a Streamlit web application designed for automated annotation of baseball images using Google's Gemini generative AI models. Users can upload images, configure annotation parameters (such as the Gemini model, output format, temperature, and custom prompts), process the images in parallel, view the annotated results with visualizations, and download the annotations in various formats (JSON, YOLO, COCO). This tool aims to streamline the process of creating datasets for computer vision tasks in baseball analytics.

**At this time, the App ONLY supports Detection Annotations.**

## Features

*   **Google Gemini Integration**: Leverages powerful Gemini vision models for object detection and annotation.
*   **Flexible Model Selection**: Supports multiple Gemini models (e.g., `gemini-1.5-flash`, `gemini-1.5-pro`).
*   **Custom Prompts**: Users can define specific prompts to guide the annotation process (e.g., "Identify all baseball players," "Identify the pitcher's glove").
*   **Multiple Output Formats**: Annotations can be generated and downloaded in:
    *   **JSON**: Raw structured data from the model.
    *   **YOLO**: Format suitable for training YOLO object detection models.
    *   **COCO**: Common format for object detection datasets.
*   **Adjustable Temperature**: Control the randomness/creativity of the model's output.
*   **Image Uploader**: Supports uploading multiple JPG, JPEG, and PNG images.
*   **Interactive Visualization**: Displays original images, generated annotations (as text/JSON), and images with bounding boxes drawn.
*   **Downloadable Results**: Provides a ZIP file containing the original images and their corresponding annotation files.

## Setup

### Prerequisites

*   Python 3.8+
*   pip

### Installation

1.  **Clone Repo & Install Dependencies**:
    Clone this very repository, change your current directory to BaseballCV, and create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate 
    ```
    Install the required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r streamlit/gemini_annotating_app/requirements.txt
    ```


2.  **Google AI Studio API Key**:
    *   You need a Google AI Studio API key to use this application. You can obtain one from the [Google AI Studio](aistudio.google.com).
    *   The application will prompt you to enter this API key in the sidebar.

### Project Structure

*   `app.py`: The main Streamlit application script.
*   `app_utils/`: Contains helper modules:
    *   `detection_processor.py`: Handles communication with the Gemini API, parses responses, converts annotation formats, and generates visualizations.
    *   `data_downloader.py`: Manages the creation of ZIP files for downloading annotation results.
*   `app_assets/`: Stores static assets like the repo logo (`baseballcvlogo.png`).
*   `README.md`: This file.

## Running the App

Once the setup is complete, you can run the Streamlit application from the BaseballCV repository w/ `streamlit/gemini_annotating_app/app.py`:

```bash
cd BaseballCV
streamlit run streamlit/gemini_annotating_app/app.py
```

This will typically open the application in your default web browser.

## How to Use

1.  **Navigate to the App**: Open the URL provided by Streamlit (usually `http://localhost:8501`).
2.  **Enter API Key**: In the sidebar, input your Google Gemini API Key.
3.  **Select Model**: Choose a Gemini model from the dropdown list (`gemini-2.0-flash` is reccomended for API Free-Tier Users).
4.  **Choose Format**: Select your desired output format for annotations (JSON, YOLO, or COCO).
5.  **Set Temperature**: Adjust the temperature slider. Lower values make the output more deterministic, while higher values make it more random.
6.  **Write Prompt**: In the text area, describe what objects you want the model to identify in the images. Be specific for better results.
7.  **Upload Images**: Drag and drop or browse to upload one or more image files (`.jpg`, `.jpeg`, `.png`).
8.  **Process Images**: Click the "Process Images" button in the sidebar. A progress bar will show the status.
9.  **View Results**:
    *   Once processing is complete, the results will appear below the uploader.
    *   You'll see the original image, the raw annotation data (e.g., JSON), and (if applicable) the image with bounding boxes visualized.
    *   If errors occur during processing for specific images, they will be displayed.
10. **Download Results**:
    *   A "Download Annotations" button will appear.
    *   Clicking this button will download a ZIP file containing the uploaded images and their corresponding annotation files in the format you selected.
11. **Clear Results**: Use the "Clear Results" button to remove all processed images and their annotations from the current session.

### Example Prompts

*   "Identify all baseball players. There should be a maximum of 3 players in the image."
*   "Identify a pitcher, hitter, and catcher."
*   "Identify the catcher's glove."
*   "Identify only the pitcher in the image."

*Note: The quality and clarity of the input images, as well as the quality of the prompt, significantly impact the annotation results.*

## Troubleshooting

*   **API Key Issues**: Ensure your Gemini API key is valid, active, and has the necessary permissions.
*   **API Call Error (Quota Exceeded)**: Google AI Studio Free-Tier Users are severely rate-limited, and API Calls may fail if submitting too many requests. 
*   **Model Not Available**: If a selected model fails to initialize, it might not be available in your region or for your account. Try a different model.
*   **No Annotations**: If no annotations are generated, check your prompt for clarity. The model might not have understood the request, or the objects might not be clearly visible.
*   **Incorrect Bounding Boxes**: The accuracy of bounding boxes depends on the model, prompt, and image quality. Experiment with different prompts and temperatures.
