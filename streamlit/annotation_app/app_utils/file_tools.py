import os
import json
import cv2
from typing import Any, List

__all__ = ['FileTools']

class FileTools:
    def __init__(self):
        pass
        
    def save_json(self, data: Any, filepath: str) -> None:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    def load_json(self, filepath: str) -> Any:
        with open(filepath, 'r') as f:
            return json.load(f)

    def extract_frames(self, video_file: Any, output_dir: str, frame_interval: int = 1) -> List[str]:
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
