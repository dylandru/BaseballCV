import os
import json
import cv2
from PIL import Image

__all__ = ['FileTools']

class FileTools:
    @staticmethod
    def save_json(data, filepath):
        """Save data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def load_json(filepath):
        """Load data from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def extract_frames(video_file, output_dir, frame_interval=1):
        """Extract frames from video file"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save video to temporary file
        temp_path = os.path.join(output_dir, "temp_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(video_file.read())
        
        # Open video file
        cap = cv2.VideoCapture(temp_path)
        frame_count = 0
        saved_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_path)
                
            frame_count += 1
        
        cap.release()
        os.remove(temp_path)
        return saved_frames