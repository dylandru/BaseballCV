import os
import json
import cv2
from typing import Any, List
import tempfile
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
        
        os.makedirs(output_dir, exist_ok=True)
        frames = []
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_file.read())
                tmp_file_path = tmp_file.name
            
            video = cv2.VideoCapture(tmp_file_path)
            if not video.isOpened():
                raise Exception("Failed to open video file")
                
            frame_count = 0
            while True:
                success, frame = video.read()
                if not success:
                    break
                    
                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    success = cv2.imwrite(frame_path, frame)
                    if not success:
                        print(f"Failed to write frame {frame_count}")
                    else:
                        frames.append(frame_path)
                        
                frame_count += 1
                
            video.release()
            
            os.unlink(tmp_file_path)
            
            if not frames:
                raise Exception(f"No frames were extracted from the video")
                
            return frames
            
        except Exception as e:
            raise Exception(f"Error extracting frames: {str(e)}")
