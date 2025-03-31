import os
import random
import supervision as sv
from collections import defaultdict
import concurrent.futures
from .savant_scraper import BaseballSavVideoScraper
from .utils import extract_frames_from_video
import shutil
from .load_tools import LoadTools
from ultralytics import YOLO
from tqdm import tqdm
from datetime import datetime
from baseballcv.utilities import BaseballCVLogger
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM

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
        logger (BaseballCVLogger): A BaseballCV Logger instance for logging.
    '''

    def __init__(self):
        self.LoadTools = LoadTools()
        self.output_folder = ''
        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__)

    def generate_photo_dataset(self,
                           output_frames_folder: str = "cv_dataset", 
                           video_download_folder: str = "raw_videos",
                           max_plays: int = 10, 
                           max_num_frames: int = 6000,
                           max_videos_per_game: int = 10,
                           start_date: str = "2024-05-22",
                           end_date: str = "2024-07-25",
                           delete_savant_videos: bool = True,
                           use_savant_scraper: bool = True,
                           input_video_folder: str = None,
                           use_supervision: bool = False,
                           frame_stride: int = 30) -> (str | None):
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
            delete_savant_videos (bool): Whether or not to delete scraped savant videos after frames are extracted. Default is True.
            use_savant_scraper (bool): Whether to use the savant scraper to download videos. Default is True.
            input_video_folder (str): Path to folder containing videos if not using savant scraper. Default is None.
            use_supervision (bool): Whether to use supervision library for frame extraction. Default is False.
            frame_stride (int): Number of frames to skip when using supervision. Default is 30.

        Returns:
            output_folder (str): Creates a folder of photos from the video frames to use. Returns the directory where the photos or stored. 
            If there are no video files found in the specific folder, None is returned.
        """
        self.output_folder = output_frames_folder
        
        if use_savant_scraper:
            self.scraper = BaseballSavVideoScraper(start_date, end_date, download_folder=video_download_folder,
                                    max_return_videos=max_plays, max_videos_per_game=max_videos_per_game)
            
            self.scraper.run_executor()
            video_folder = video_download_folder
        else:
            if input_video_folder is None:
                raise ValueError("input_video_folder must be provided when use_savant_scraper is False")
            video_folder = input_video_folder
                
        os.makedirs(self.output_folder, exist_ok=True)
        video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.mov', '.mts'))]
        
        if not video_files:
            self.logger.warning("No video files found in the specified folder.")
            return None
        
        games = defaultdict(list)
        for video_file in video_files:
            if use_savant_scraper:
                game_id = video_file[:6] 
            else:
                game_id = os.path.splitext(video_file)[0][:6]
            games[game_id].append(video_file)
        
        frames_per_game = max_num_frames // len(games)
        remaining_frames = max_num_frames % len(games)
        
        extracted_frames = []
        
        extraction_tasks = []
        for game_id, game_videos in games.items():
            frames_for_game = frames_per_game + (1 if remaining_frames > 0 else 0)
            remaining_frames = max(0, remaining_frames - 1)
            
            frames_per_video = frames_for_game // len(game_videos)
            extra_frames = frames_for_game % len(game_videos)
            
            for i, video_file in enumerate(game_videos):
                frames_to_extract = frames_per_video + (1 if i < extra_frames else 0)
                video_path = os.path.join(video_folder, video_file)
                
                if use_supervision:
                    video_name = os.path.splitext(video_file)[0]
                    image_name_pattern = f"{game_id}_{video_name}-{{:05d}}.png"
                    
                    frame_count = 0
                    with sv.ImageSink(target_dir_path=self.output_folder, image_name_pattern=image_name_pattern) as sink:
                        for image in sv.get_video_frames_generator(source_path=str(video_path), stride=frame_stride):
                            sink.save_image(image=image)
                            frame_count += 1
                            extracted_frames.append(os.path.join(self.output_folder, image_name_pattern.format(frame_count)))
                            
                            if frame_count >= frames_to_extract:
                                break
                else:
                    extraction_tasks.append((video_path, game_id, self.output_folder, frames_to_extract))

        if not use_supervision and extraction_tasks:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                try:
                    future_videos = {executor.submit(extract_frames_from_video, *task): task 
                                   for task in extraction_tasks}
                    for future in concurrent.futures.as_completed(future_videos):
                        video_path, game_id, _, _ = future_videos[future]
                        try:
                            result = future.result()
                            extracted_frames.extend(result)
                        except Exception as e:
                            self.logger.error(f"Error with {video_path}: {str(e)}")
                except KeyboardInterrupt:
                    self.logger.info("\nShutting down gracefully...")
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
            if os.path.exists(frame_path):
                frame_name = f"{i+1:06d}{os.path.splitext(frame_path)[1]}"
                new_path = os.path.join(self.output_folder, frame_name)
                shutil.move(frame_path, new_path) 

        existing_files = set(os.listdir(self.output_folder))
        extracted_file_names = set(f"{i+1:06d}{os.path.splitext(os.path.basename(frame))[1]}" 
                                  for i, frame in enumerate(extracted_frames))
        files_to_remove = existing_files - extracted_file_names
        for file in files_to_remove:
            os.remove(os.path.join(self.output_folder, file))
        
        self.logger.info(f"Extracted {len(extracted_frames)} frames from {len(video_files)} videos over {len(games)} games.")
        
        if delete_savant_videos and use_savant_scraper:
            self.scraper.cleanup_savant_videos()

        return self.output_folder
    
    def automated_annotation(self, 
                             model_alias: str = None,
                             model_type: str = 'detection',
                             image_dir: str = "cv_dataset",
                             output_dir: str = "labeled_dataset", 
                             conf: float = .80, 
                             device: str = 'cpu',
                             mode: str = 'autodistill',
                             ontology: dict = None,
                             extension: str = '.jpg') -> str:
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
            device (str): Device to run model on ('cpu', 'mps', 'cuda'). Default is 'cpu'. MPS is not supported for AutoDistill.
            mode (str): Mode to use for annotation. Default is 'autodistill'.
            ontology (dict): Ontology to use for annotation. Default is None.
            extension (str): Extension of images to annotate. Default is '.jpg'.

        Returns:
            None: Saves annotated images and labels to the output directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        annotations_dir = os.path.join(output_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)

        if mode == 'autodistill': #use RF Autodistill library
            if ontology is not None:
                self.logger.info(f"Using Autodistill mode with ontology: {ontology}")
                self.logger.info(f"This may take a while...")
                auto_model= GroundedSAM(ontology=CaptionOntology(ontology))
                auto_model.label(
                    input_folder=str(image_dir),
                    output_folder=str(output_dir),
                    extension=extension
                )
                self.logger.info("Annotation process complete.")
                return output_dir
            else:
                raise ValueError("ontology must be provided when using autodistill mode")
        
        else: #Legacy Version for using models from YOLO repo
            if model_alias is not None:
                model = YOLO(self.LoadTools.load_model(model_alias))
                self.logger.info(f"Model loaded: {model}")
            else:
                raise ValueError("model_alias must be provided when using legacy mode")

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
                                self.logger.warning(f"Invalid bounding box for {image_file}: {xywhn}")

                #TODO: Add annotation format for YOLO Keypoint, Segmentation, and Classification models

                if annotations:
                    shutil.copy(image_path, output_dir)
                    output_file = os.path.join(annotations_dir, os.path.splitext(image_file)[0] + '.txt')
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, 'w') as f:
                        f.write('\n'.join(annotations))

            self.logger.info("Annotation process complete.")
            return output_dir

