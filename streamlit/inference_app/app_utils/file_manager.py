from pathlib import Path
import shutil
import cv2
from typing import Tuple


# Dir structure
# streamlit/ ----
#   inference_app/ ------
#       app.py ---------
#       models/ --------
#       working_files/ --------
#           img/ ----------
#           videos/ ---------

class File:
    def __init__(self):
        working_dir = Path.cwd()

        if 'BaseballCV' in str(working_dir):
            working_dir = working_dir / 'streamlit' / 'inference_app'

        
        self.models_dir = working_dir / 'models'
        self.imgs_dir = working_dir / 'working_files' / 'img'
        self.videos_dir = working_dir / 'working_files' / 'videos'

        self.imgs_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    @property
    def annot_video_path(self):
        # This is for the output path of the annotated video
        return self.out_path
    
    @property
    def models_dir_path(self):
        # This is for load tools
        return self.models_dir

    
    def write_video(self, file, create_annotation_output = True) -> Tuple[cv2.VideoWriter, cv2.VideoCapture, int]:
        tmp_path = self._create_temp_video(file)

        cap = cv2.VideoCapture(tmp_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        self.out_path = self.videos_dir / f"annot_{file.name}"

        if create_annotation_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.out_path, fourcc, fps, (width, height))

        return out, cap, length


    def _create_temp_video(self, file):
        video_path = self.videos_dir / file.name

        with open(video_path, 'wb') as video:
            video.write(file.read())
        
        return video_path

    def clear(self):
        if self.imgs_dir.exists():
            shutil.rmtree(self.imgs_dir)
        if self.videos_dir.exists():
            shutil.rmtree(self.videos_dir)

