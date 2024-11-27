import os
import json
import cv2
from typing import Any, List
import requests
from PIL import Image
import io

__all__ = ['FileTools']

class FileTools:
    """A class to manage file operations."""
    def __init__(self):
        pass
        
    def save_json(self, data: Any, filepath: str) -> Any:
        """Save data to a JSON file.

        Args:
            data (Any): Data to save.
            filepath (str): Path to the file to save the data to.

        Returns:
            Any: The saved data.
        """
        with open(filepath, 'w') as f:
            return json.dump(data, f, indent=4)

    def load_json(self, filepath: str) -> Any:
        """Load data from a JSON file.

        Args:
            filepath (str): Path to the file to load the data from.

        Returns:
            Any: The loaded data.
        """
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_image_from_endpoint(self, api_endpoint: str) -> Image.Image:
        """Load an image from an API endpoint.

        Args:
            api_endpoint (str): The API endpoint to load the image from.

        Returns:
            Image.Image: The loaded image.
        """
        response = requests.get(api_endpoint)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        else:
            raise Exception(f"Failed to load image from {api_endpoint}")

    def extract_frames(self, video_file: Any, output_dir: str, frame_interval: int = 1) -> List[str]:
        """Extract frames from a video file and save them as images.

        Args:
            video_file (Any): The video file for the frames to be extracted from.
            output_dir (str): The directory to save the extracted frames.
            frame_interval (int): The interval between frames to extract. Defaults to 1.

        Returns:
            List[str]: List of paths to the extracted frames.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        video_name = os.path.splitext(video_file.name)[0]
            
        temp_path = os.path.join(output_dir, "temp_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(video_file.read())
        
        cap = cv2.VideoCapture(temp_path)
        frame_count = 0
        saved_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"{video_name}_frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_path)
                
            frame_count += 1
        
        cap.release()
        os.remove(temp_path)
        return saved_frames
