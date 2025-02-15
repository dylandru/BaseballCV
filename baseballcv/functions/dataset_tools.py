import os
import random
from collections import defaultdict
import concurrent.futures
from .savant_scraper import BaseballSavVideoScraper
from .utils import extract_frames_from_video
import shutil
from .load_tools import LoadTools
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime

class DataTools:
    '''
    A class for generating and processing datasets for computer vision tasks in baseball.

    This class provides methods for generating photo datasets from baseball videos,
    and automating the annotation process using pre-trained models.

    Attributes:
        scraper (BaseballSavVideoScraper): An instance of the BaseballSavVideoScraper class.
        process_pool (concurrent.futures.ProcessPoolExecutor): A process pool for parallel execution.
        LoadTools (LoadTools): An instance of the LoadTools class for loading models.
        output_folder (str): The output folder for generated datasets.
    '''

    def __init__(self, max_workers: int = 10):
        self.scraper = BaseballSavVideoScraper()
        self.max_workers = max_workers
        self.LoadTools = LoadTools()
        self.output_folder = ''

    def generate_photo_dataset(self,
                           output_frames_folder: str = "cv_dataset", 
                           video_download_folder: str = "raw_videos",
                           max_plays: int = 10, 
                           max_num_frames: int = 6000,
                           max_videos_per_game: int = 10,
                           start_date: str = "2024-05-22",
                           end_date: str = "2024-07-25",
                           delete_savant_videos: bool = True) -> None:
        """
        Extracts random frames from scraped Baseball Savant broadcast videos to create a photo dataset for a 
        Computer Vision model.
        
        Args:
            output_frames_folder (str): Name of folder where photos will be saved. Default is "cv_dataset".
            video_download_folder (str): Name of folder containing videos. Default is "raw_videos".
            max_plays (int): Maximum number of plays for scraper to download videos. Default is 10.
            max_num_frames (int): Maximum number of frames to extract across all videos. Default is 6000.
            max_videos_per_game (int): Max number of videos to pull for single game to increase variety. Defaults to 10.
            start_date (str): Start date for video scraping in "YYYY-MM-DD" format. Default is "2024-05-22".
            end_date (str): End date for video scraping in "YYYY-MM-DD" format. Default is "2024-05-25".
            max_workers (int): Number of worker processes to use for frame extraction. Default is 10.
            delete_savant_videos (bool): Whether or not to delete scraped savant videos after frames are extracted. Default is True.

        Returns:
            None: Creates a folder of photos from the video frames to use.
        """

        self.output_folder = output_frames_folder
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.scraper.run_statcast_pull_scraper(start_date=start_date, end_date=end_date, 
                                download_folder=video_download_folder, max_videos=max_plays, max_videos_per_game=max_videos_per_game, max_workers=1)
                
        os.makedirs(self.output_folder, exist_ok=True)
        video_files = [f for f in os.listdir(video_download_folder) if f.endswith(('.mp4', '.mov'))]
        
        if not video_files:
            print("No video files found in the specified folder.") 
            return
        
        games = defaultdict(list) #group videos by given game for increased variety
        for video_file in video_files:
            game_id = video_file[:6] 
            games[game_id].append(video_file)
        
        frames_per_game = max_num_frames // len(games)
        remaining_frames = max_num_frames % len(games) #distribute frames evenly across games
        
        extraction_tasks = []
        for game_id, game_videos in games.items():
            frames_for_game = frames_per_game + (1 if remaining_frames > 0 else 0)
            remaining_frames = max(0, remaining_frames - 1)
            
            frames_per_video = frames_for_game // len(game_videos)
            extra_frames = frames_for_game % len(game_videos)
            
            for i, video_file in enumerate(game_videos):
                frames_to_extract = frames_per_video + (1 if i < extra_frames else 0)
                video_path = f"{video_download_folder}/{video_file}"
                extraction_tasks.append((video_path, game_id, self.output_folder, frames_to_extract))
        
        extracted_frames = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            try:
                future_videos = {executor.submit(extract_frames_from_video, *task): task 
                               for task in extraction_tasks}
                for future in concurrent.futures.as_completed(future_videos):
                    video_path, game_id, _, _ = future_videos[future]
                    try:
                        result = future.result()
                        extracted_frames.extend(result)
                    except Exception as e:
                        print(f"Error with {video_path}: {str(e)}")
            except KeyboardInterrupt:
                print("\nShutting down gracefully...")
                executor.shutdown(wait=False)
                raise
            finally:
                for task in extraction_tasks:
                    video_path = task[0]
                    if os.path.exists(video_path):
                        try:
                            os.close(os.open(video_path, os.O_RDONLY))
                        except:
                            pass

        random.shuffle(extracted_frames)
        
        for i, frame_path in enumerate(extracted_frames):
            frame_name = f"{i+1:06d}{os.path.splitext(frame_path)[1]}"
            new_path = os.path.join(self.output_folder, frame_name)
            shutil.move(frame_path, new_path) 

        existing_files = set(os.listdir(self.output_folder))
        extracted_file_names = set(f"{i+1:06d}{os.path.splitext(frame)[1]}" for i, frame in enumerate(extracted_frames))
        files_to_remove = existing_files - extracted_file_names
        for file in files_to_remove:
            os.remove(os.path.join(self.output_folder, file))
        
        print(f"Extracted {len(extracted_frames)} frames from {len(video_files)} videos over {len(games)} games.")
        
        if delete_savant_videos:
            self.scraper.cleanup_savant_videos(video_download_folder)
    
    def automated_annotation(self, 
                             model_alias: str,
                             model_type: str = 'detection',
                             image_dir: str = "cv_dataset",
                             output_dir: str = "labeled_dataset", 
                             conf: float = .80, 
                             device: str = 'cpu') -> None:
        """
        Automatically annotates images using pre-trained YOLO model from BaseballCV repo. The annotated output
        consists of image files in the output directory, and label files in the subfolder "annotations" to 
        work with annotation tools.

        Note: The current implementation only supports YOLO detection models. 

        Args:
            model_alias (str): Alias of model to utilize for annotation.
            model_type (str): Type of CV model to utilize for annotation. Default is 'detection'.
            image_dir (str): Directory with images to annotate. Default is "cv_dataset".
            output_dir (str): Directory to save annotated images / labels. Default is "labeled_dataset".
            conf (float): Minimum confidence threshold for detections. Default is 0.80.
            device (str): Device to run model on ('cpu', 'mps', 'cuda'). Default is 'cpu'.

        Returns:
            None: Saves annotated images and labels to the output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        annotations_dir = os.path.join(output_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)

        model = YOLO(self.LoadTools.load_model(model_alias))
        print(f"Model loaded: {model}")

        annotation_tasks = [image_file for image_file in os.listdir(image_dir)]

        for image_file in tqdm(annotation_tasks, desc="Annotating images"):
            image_path = os.path.join(image_dir, image_file)
            annotations = []

            results = model.predict(source=image_path, save=False, conf=conf, device=device, verbose=False)
            
            if model_type == 'detection':
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls)
                        xywhn = box.xywhn[0].tolist()
                        if len(xywhn) == 4:
                            x_center, y_center, width, height = xywhn
                            annotations.append(f"{cls} {x_center} {y_center} {width} {height}")
                        else:
                            print(f"Invalid bounding box for {image_file}: {xywhn}")

            #TODO: Add annotation format for YOLO Keypoint, Segmentation, and Classification models

            if annotations:
                shutil.copy(image_path, output_dir)
                output_file = os.path.join(annotations_dir, os.path.splitext(image_file)[0] + '.txt')
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w') as f:
                    f.write('\n'.join(annotations))

        print("Annotation process complete.")

        return None

