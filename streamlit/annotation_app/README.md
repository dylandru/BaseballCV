# BaseballCV Annotation App - An App Solution Designed to support Open-Source Baseball Annotations

A Streamlit-based app for managing and annotating baseball image and video using BaseballCV's computer vision models.

Run the App on your browser: [Annotation App](https://balldatalab.com/streamlit/baseballcv_annotation_app/)

## Features of Project

- Open-source project designed to crowdsource baseball annotations for training of models
- Upload personal baseball videos or photos to annotate
- Annotation interface for labeling objects and keypoints
- Integration with AWS S3 for data / photo storage
- Built-in BaseballCV models for predicted annotations to quicken annotation process

## Prerequisites

- Docker and Docker Compose
- AWS Account with S3 access
- Python 3.11+ (cannot guarantee compatibility with versions below 3.11)

## Starting with Docker

1. Clone the repository and navigate to the BaseballCV `annotation_app` directory:

```bash
git clone https://github.com/dylandru/BaseballCV.git
cd BaseballCV/streamlit/annotation_app
```

2. Create a `.env` file in `streamlit/annotation_app/` with your AWS credentials that have access to an S3 bucket (currently configured for bucket `baseballcv-annotations`):

```env
AWS_BASEBALLCV_ACCESS_KEY=access_key
AWS_BASEBALLCV_SECRET_KEY=secret_key
AWS_BASEBALLCV_REGION=region
```

3. Build and run the Docker container:

```bash
# Given the user is in the BaseballCV/streamlit/annotation_app directory
docker-compose up -d
```

4. Access the App and Annotations

- App: `http://localhost:8505`
- Annotations: `https://aws.amazon.com/s3/buckets/<your_bucket_name>?region=<your_region>&bucketType=general&prefix=<project_type>/<project_name>/completed/<user>/`

## Starting Manually (without Docker)

1. Install dependencies (would recommend using a virtual environment):

```bash
pip install -r requirements.txt
```

2. Set environment variables:

```bash
export AWS_BASEBALLCV_ACCESS_KEY=access_key
export AWS_BASEBALLCV_SECRET_KEY=secret_key
export AWS_BASEBALLCV_REGION=region
export PYTHONPATH=/path/to/BaseballCV
```

3. Run the app:

```bash
cd BaseballCV
streamlit run streamlit/annotation_app/app.py
```

## Usage Guide

### Creating a New Project

1. Launch the app and click "Create New Project"
2. Enter a project name and select the project type
3. Add any project-specific settings (will be adding the ability to add custom models soon)

### Adding Images to Annotate

1. Select your project from the dashboard
2. Click "Upload Media"
3. Click "Upload Photos" or "Upload Videos" or "Download Photos"
4. Either download photos from S3 or choose your own photos / videos stored locally
5. Wait for processing to complete

### Annotation Process

1. Open your project and click "Open Annotator"
2. If a prediction model is added, the predicted annotations will be shown on the image
3. Select your object you want to annotate from the dropdown menu and mark the elements you want to annotate
4. Click "Save Annotations"

### Viewing Results

- Access project statistics from the dashboard
- View and export annotations
- Track progress of annotation tasks

## Project Structure

```
annotation_app/
├── app.py                 # Main Streamlit app
├── app_utils/
│   ├── app_pages.py       # Page layouts and UI components
│   ├── task_manager.py    # Task and project management
│   ├── model_manager.py   # BaseballCV model integration
│   ├── s3.py              # AWS S3 interactions (uses CLI)
│   ├── file_tools.py      # File handling utilities
│   ├── default_tools.py   # Default annotation tools
│   └── image_manager.py   # Image processing utilities
├── Dockerfile             # Container config
└── docker-compose.yml     # Docker services setup
```

## Troubleshooting

### Common Issues

1. **S3 Access Denied / AWS Credentials Error**
   - Verify AWS credentials in .env file
   - Check S3 bucket permissions
   - ensure AWS user has access to the bucket

2. **Docker Port Conflicts**
   - Change the port mapping in docker-compose.yml
   - Check for other services using the given port (currently 8505)

3. **Missing Model Weights**
   - Ensure model files are downloaded
   - Check model paths configuration
   - Ensure model weights are in the correct directory

### Getting Help

If you are still having trouble hosting the app, there are three primary options:

- Open an issue in the repository
- Contact the maintainers directly
- Review the logs / error messages and debug.
