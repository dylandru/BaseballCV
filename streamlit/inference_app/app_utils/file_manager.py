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
    """
    Class that handles the file structure of the project during each session.
    These files act as caching for the models to prevent continuous loading in one session.
    """
    def __init__(self) -> None:
        """
        Initializes the `File` class with a `models` and `working_files` directories to
        add images, videos, and loaded models. 
        """
        working_dir = Path.cwd()

        if 'BaseballCV' in str(working_dir):
            working_dir = working_dir / 'streamlit' / 'inference_app'

        
        self.models_dir_path = working_dir / 'models'
        self.imgs_dir = working_dir / 'working_files' / 'img'
        self.videos_dir = working_dir / 'working_files' / 'videos'

        self.imgs_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    
    def write_video(self, file: Path, create_annotation_output = True) -> Tuple[cv2.VideoWriter, cv2.VideoCapture, int]:
        """
        Writes a empty video and establishes a capture used to make annotations of the predictions for each frame.

        Args:
            file (Path): The input file of the uploaded video.
            create_annotation_output (bool): Creates a annotation file if the desired method is to make annotations on the video.
            Defaults to True for this purpose, but can be False if the user just wants the video characteristics.

        Returns:
            Tuple: The written output of the file, captured frames, and the number of frames in the video.
        """
        tmp_path = self._create_temp_video(file)

        cap = cv2.VideoCapture(tmp_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = None
        self.annot_video_path = self.videos_dir / f"annot_{file.name}"

        if create_annotation_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.annot_video_path, fourcc, fps, (width, height))

        return out, cap, length


    def _create_temp_video(self, file: Path) -> Path:
        """
        Creates the temporary video file that can be used for annotations.

        Args:
            file (Path): The input file of the uploaded video.
        
        Returns:
            Path: the video path of the written file.
        """
        video_path = self.videos_dir / file.name

        with open(video_path, 'wb') as video:
            video.write(file.read())
        
        return video_path

    def clear(self) -> None:
        """
        Clears the files once the user is satisfied with the uploads to prevent
        session issues with too many files uploaded. 
        """
        if self.imgs_dir.exists():
            shutil.rmtree(self.imgs_dir)
        if self.videos_dir.exists():
            shutil.rmtree(self.videos_dir)