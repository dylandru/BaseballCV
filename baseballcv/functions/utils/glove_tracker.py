# baseballcv/functions/utils/glove_tracker.py
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from typing import Dict, List, Tuple, Optional, Union, Callable
import glob
from tqdm import tqdm
import shutil
import time
from ultralytics import YOLO
import supervision as sv
from baseballcv.utilities import BaseballCVLogger, ProgressBar
from baseballcv.functions.load_tools import LoadTools


class GloveTracker:
    """
    Class for tracking the catcher's glove, home plate, and baseball in videos.
    
    This class supports multiple operational modes:
    - Regular mode: Process individual videos passed to it
    - Batch mode: Process multiple videos in a folder
    - Scrape mode: Download videos using the Savant scraper and process them
    
    The tracker uses YOLO models to detect and track objects, and provides
    visualization and data export capabilities for analysis.
    """
    
    def __init__(
        self, 
        model_alias: str = 'glove_tracking',
        results_dir: str = "glove_tracking_results",
        device: str = None,
        confidence_threshold: float = 0.5,
        enable_filtering: bool = True,
        max_velocity_inches_per_sec: float = 120.0,
        mode: str = "regular",
        logger: Optional[BaseballCVLogger] = None
    ):
        """
        Initialize the GloveTracker.
        
        Args:
            model_alias (str): The alias of the model to use for detection
            results_dir (str): Directory to save results
            device (str): Device to run the model on (cpu, cuda, mps)
            confidence_threshold (float): Confidence threshold for detections
            enable_filtering (bool): Whether to enable outlier filtering for glove detections
            max_velocity_inches_per_sec (float): Maximum plausible velocity for glove movement (for filtering)
            mode (str): Operating mode - "regular", "batch", or "scrape"
            logger (BaseballCVLogger): Logger instance for logging
        """
        self.load_tools = LoadTools()
        self.logger = logger if logger else BaseballCVLogger.get_logger(self.__class__.__name__)
        self.model_path = self.load_tools.load_model(model_alias)
        self.model = YOLO(self.model_path)
        self.device = device if device else 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        self.confidence_threshold = confidence_threshold
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Enable filtering and set velocity threshold
        self.enable_filtering = enable_filtering
        self.max_velocity_inches_per_sec = max_velocity_inches_per_sec
        
        # Set operating mode
        self.mode = mode
        if mode not in ["regular", "batch", "scrape"]:
            self.logger.warning(f"Invalid mode '{mode}'. Defaulting to 'regular'")
            self.mode = "regular"
        
        # Get class names from the model
        self.class_names = self.model.names
        self.logger.info(f"Model loaded with classes: {self.class_names}")
        
        # Define important class indices
        self.glove_class_id = next((id for id, name in self.class_names.items() if 'glove' in name.lower()), None)
        self.homeplate_class_id = next((id for id, name in self.class_names.items() if 'homeplate' in name.lower() or 'home_plate' in name.lower() or 'home plate' in name.lower()), None)
        self.baseball_class_id = next((id for id, name in self.class_names.items() if 'baseball' in name.lower() or 'ball' in name.lower()), None)
        
        if not all([self.glove_class_id is not None, self.homeplate_class_id is not None, self.baseball_class_id is not None]):
            missing_classes = []
            if self.glove_class_id is None:
                missing_classes.append("glove")
            if self.homeplate_class_id is None:
                missing_classes.append("homeplate")
            if self.baseball_class_id is None:
                missing_classes.append("baseball")
            self.logger.warning(f"Some required classes not found in model: {missing_classes}")
        
        self.logger.info(f"Class IDs - Glove: {self.glove_class_id}, Home Plate: {self.homeplate_class_id}, Baseball: {self.baseball_class_id}")
        
        # For storing tracking data
        self.tracking_data = []
        
        # Home plate reference (to be set during tracking)
        self.home_plate_reference = None
        self.pixels_per_inch = None

    def track_video(self, video_path: str, output_path: Optional[str] = None,
                show_plot: bool = True, create_video: bool = True) -> str:
        """
        Track objects in a video and generate visualization with 2D tracking plot.

        Args:
            video_path (str): Path to input video
            output_path (str): Path for output video (if None, auto-generated in results_dir)
            show_plot (bool): Whether to show the 2D plot in the output video

        Returns:
            str: Path to the output video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

        # Create output paths if not provided
        if output_path is None:
            video_filename = os.path.basename(video_path)
            output_path = os.path.join(self.results_dir, f"tracked_{video_filename}")

        # Construct the CSV path correctly
        csv_filename = os.path.splitext(os.path.basename(output_path))[0] + "_tracking.csv"
        csv_path = os.path.join(self.results_dir, csv_filename)

        # Reset tracking data for this video
        self.tracking_data = []
        self.home_plate_reference = None
        self.pixels_per_inch = None

        # Open video
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = total_frames  # Store total frames as an instance attribute

        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if create_video else None

        # If show_plot is True, we'll create a wider output to accommodate the plot
        out_width = width * 2 if show_plot else width
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height)) if create_video else None

        # Create plot for glove tracking
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Process frames
        progress_bar = ProgressBar(total=total_frames, desc=f"Processing video: {os.path.basename(video_path)}")

        frame_idx = 0

        with progress_bar as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLO detection on the frame
                results = self.model.predict(frame, conf=self.confidence_threshold, device=self.device, verbose=False)

                # Process and extract detections
                detections = self._process_detections(results, frame, frame_idx, fps)

                # Draw annotations on the frame
                annotated_frame = self._annotate_frame(frame.copy(), detections, frame_idx)

                # Create the 2D tracking plot if needed
                if show_plot:
                    self._update_tracking_plot(ax, fig)

                    # Convert matplotlib figure to image
                    fig.canvas.draw()
                    buf = fig.canvas.buffer_rgba()
                    plot_img = np.asarray(buf)[:, :, :3]  # Convert RGBA to RGB
                    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    # Resize plot to match frame height
                    plot_img = cv2.resize(plot_img, (width, height))

                    # Create combined frame with original and plot side by side
                    combined_frame = np.hstack((annotated_frame, plot_img))
                    if create_video:
                        out.write(combined_frame) 
                else:
                    if create_video:
                        out.write(annotated_frame) 

                frame_idx += 1
                pbar.update(1)

        # Clean up
        cap.release()
        if create_video:
            out.release()
        plt.close(fig)

        # Save tracking data to CSV
        self._save_tracking_data(csv_path, video_path)

        self.logger.info(f"Tracking completed. Output video saved to {output_path}")
        self.logger.info(f"Tracking data saved to {csv_path}")

        return output_path

    def batch_process(self, 
                    input_folder: str, 
                    delete_after_processing: bool = False, 
                    skip_confirmation: bool = False,
                    show_plot: bool = True,
                    create_video: bool = True,
                    extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv'],
                    max_workers: int = 1) -> str:
        """
        Process all videos in a folder and generate a combined CSV with tracking data.
        
        Args:
            input_folder (str): Path to folder containing videos
            delete_after_processing (bool): Whether to delete videos after processing
            skip_confirmation (bool): Skip confirmation for deletion
            show_plot (bool): Whether to show plot in output videos
            extensions (List[str]): List of video file extensions to process
            max_workers (int): Maximum number of parallel workers (1 = sequential)
            
        Returns:
            str: Path to the combined CSV file
        """
        if not os.path.exists(input_folder):
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        # Find all video files in the folder
        video_files = []
        for ext in extensions:
            video_files.extend(glob.glob(os.path.join(input_folder, f"*{ext}")))
        
        if not video_files:
            self.logger.warning(f"No video files found in {input_folder}")
            return None
        
        self.logger.info(f"Found {len(video_files)} videos to process in {input_folder}")
        
        # Warn about deletion if enabled
        if delete_after_processing and not skip_confirmation:
            confirm = input(f"WARNING: {len(video_files)} videos will be DELETED after processing. Continue? (y/n): ")
            if confirm.lower() != 'y':
                self.logger.info("Batch processing cancelled")
                return None
        
        # Create a dataframe for all videos
        combined_df = pd.DataFrame()

        def _process_wrapper(video_path):
            return self._process_single_video(video_path, show_plot, create_video)

        # Process sequentially or in parallel based on max_workers
        if max_workers > 1:
            self.logger.info(f"Processing videos in parallel with {max_workers} workers")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self._process_single_video, video_path, show_plot): video_path for video_path in video_files}
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    video_path = futures[future]
                    try:
                        video_df, output_path = future.result()
                        if video_df is not None:
                            combined_df = pd.concat([combined_df, video_df], ignore_index=True)
                            
                            # Delete if enabled
                            if delete_after_processing:
                                os.remove(video_path)
                                self.logger.info(f"Deleted video: {os.path.basename(video_path)}")
                    except Exception as e:
                        self.logger.error(f"Error processing {os.path.basename(video_path)}: {str(e)}")
        else:
            self.logger.info("Processing videos sequentially")
            for video_path in video_files:
                try:
                    video_df, output_path = _process_wrapper(video_path)
                    if video_df is not None:
                        combined_df = pd.concat([combined_df, video_df], ignore_index=True)
                        
                        # Delete if enabled
                        if delete_after_processing:
                            os.remove(video_path)
                            self.logger.info(f"Deleted video: {os.path.basename(video_path)}")
                except Exception as e:
                    self.logger.error(f"Error processing {os.path.basename(video_path)}: {str(e)}")
        
        # Save combined results
        if not combined_df.empty:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            combined_csv_path = os.path.join(self.results_dir, f"batch_tracking_results_{timestamp}.csv")
            combined_df.to_csv(combined_csv_path, index=False)
            self.logger.info(f"Combined tracking results saved to {combined_csv_path}")
            
            # Generate summary statistics
            self._generate_batch_summary(combined_df, os.path.join(self.results_dir, f"batch_summary_{timestamp}.csv"))
            
            return combined_csv_path
        else:
            self.logger.warning("No valid tracking data was collected")
            return None

    def scrape_and_process(self, 
                        start_date: str, 
                        end_date: str = None,
                        team_abbr: str = None,
                        player: int = None,
                        pitch_type: str = None,
                        max_videos: int = 10,
                        max_videos_per_game: int = None,
                        delete_after_processing: bool = True,
                        skip_confirmation: bool = False,
                        show_plot: bool = True,
                        create_video: bool = True) -> str:
        """
        Scrape videos from Baseball Savant and process them for glove tracking.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format (optional)
            team_abbr (str): Team abbreviation (optional)
            player (int): Player ID (optional)
            pitch_type (str): Pitch type (optional)
            max_videos (int): Maximum number of videos to process
            max_videos_per_game (int): Maximum videos per game
            delete_after_processing (bool): Whether to delete videos after processing
            skip_confirmation (bool): Skip confirmation for deletion
            show_plot (bool): Whether to show plot in output videos
            
        Returns:
            str: Path to the combined CSV file
        """
        # Import here to avoid circular import
        from baseballcv.functions.savant_scraper import BaseballSavVideoScraper

        # Create a unique download folder within results directory
        download_folder = os.path.join(self.results_dir, f"savant_videos_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(download_folder, exist_ok=True)
        
        # Initialize the scraper
        scraper = BaseballSavVideoScraper(
            start_dt=start_date,  # Changed from start_date
            end_dt=end_date,      # Changed from end_date
            team_abbr=team_abbr,
            player=player,
            pitch_type=pitch_type,
            download_folder=download_folder,
            max_return_videos=max_videos,
            max_videos_per_game=max_videos_per_game
        )
        
        self.logger.info(f"Scraping videos from Baseball Savant...")
        scraper.run_executor()
        
        # Get the play data
        play_ids_df = scraper.get_play_ids_df()
        num_videos = len(play_ids_df)
        
        if num_videos == 0:
            self.logger.warning("No videos were downloaded from Baseball Savant")
            return None
        
        self.logger.info(f"Successfully scraped {num_videos} videos")

        # Process the downloaded videos with create_video parameter
        result = self.batch_process(
            input_folder=download_folder,
            delete_after_processing=delete_after_processing,
            skip_confirmation=skip_confirmation,
            show_plot=show_plot,
            create_video=create_video  # Pass through create_video parameter
        )
        
        # If we didn't delete videos during batch processing, check if we should now
        if not delete_after_processing and os.path.exists(download_folder):
            if not skip_confirmation:
                confirm = input(f"Delete downloaded videos? (y/n): ")
                if confirm.lower() == 'y':
                    scraper.cleanup_savant_videos()
                    self.logger.info(f"Deleted downloaded videos")
            elif delete_after_processing:
                scraper.cleanup_savant_videos()
                self.logger.info(f"Deleted downloaded videos")
        
        return result

    def _process_single_video(self, video_path: str, show_plot: bool, create_video: bool = True) -> Tuple[pd.DataFrame, str]:
        """
        Process a single video file and return its tracking dataframe.
        
        Args:
            video_path (str): Path to the video file
            show_plot (bool): Whether to show plot in output video
            create_video (bool): Whether to create output video
            
        Returns:
            Tuple[pd.DataFrame, str]: (Tracking dataframe, output video path or CSV path)
        """
        try:
            # Process video
            output_path = self.track_video(video_path, show_plot=show_plot, create_video=create_video)
            
            # Determine CSV path based on whether a video was created
            if create_video:
                csv_filename = os.path.splitext(os.path.basename(output_path))[0] + "_tracking.csv"
            else:
                # output_path is already the CSV path in this case
                csv_filename = os.path.basename(output_path)
                
            csv_path = os.path.join(self.results_dir, csv_filename)
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                
                # Add video filename column
                df['video_filename'] = os.path.basename(video_path)
                
                return df, output_path
            else:
                self.logger.warning(f"No tracking data generated for {os.path.basename(video_path)}")
                return None, output_path
        except Exception as e:
            self.logger.error(f"Error processing {os.path.basename(video_path)}: {str(e)}")
            raise

    def _generate_batch_summary(self, combined_df: pd.DataFrame, summary_path: str) -> None:
        """
        Generate summary statistics from batch processing.
        
        Args:
            combined_df (pd.DataFrame): Combined tracking data
            summary_path (str): Path to save summary CSV
        """
        if combined_df.empty:
            return
        
        # Group by video filename
        video_groups = combined_df.groupby('video_filename')
        
        summary_data = []
        for video_name, group in video_groups:
            # Filter to glove positions
            glove_data = group[group['glove_real_x'].notna() & group['glove_real_y'].notna()]
            
            if not glove_data.empty:
                # Calculate movement
                dx = glove_data['glove_real_x'].diff().dropna()
                dy = glove_data['glove_real_y'].diff().dropna()
                distances = np.sqrt(dx**2 + dy**2)
                
                summary = {
                    'video_filename': video_name,
                    'frame_count': len(group),
                    'frames_with_glove': len(glove_data),
                    'frames_with_baseball': group['baseball_real_x'].notna().sum(),
                    'frames_with_homeplate': group['homeplate_center_x'].notna().sum(),
                    'total_distance_inches': distances.sum(),
                    'max_glove_movement_inches': distances.max() if not distances.empty else None,
                    'avg_glove_movement_inches': distances.mean() if not distances.empty else None,
                    'glove_x_range_inches': glove_data['glove_real_x'].max() - glove_data['glove_real_x'].min() if len(glove_data) > 1 else None,
                    'glove_y_range_inches': glove_data['glove_real_y'].max() - glove_data['glove_real_y'].min() if len(glove_data) > 1 else None,
                    'avg_glove_x_position': glove_data['glove_real_x'].mean() if not glove_data.empty else None,
                    'avg_glove_y_position': glove_data['glove_real_y'].mean() if not glove_data.empty else None
                }
                summary_data.append(summary)
        
        # Create and save summary dataframe
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_path, index=False)
            self.logger.info(f"Batch summary saved to {summary_path}")

    def _process_detections(self, results, frame, frame_idx, fps=30.0):
        """
        Process YOLO detection results for the current frame with outlier filtering.
        
        Args:
            results: YOLO detection results
            frame: Current video frame
            frame_idx: Frame index
            fps: Frames per second of the video
                
        Returns:
            dict: Processed detections with glove, homeplate, and baseball info
        """
        detections = {
            'frame_idx': frame_idx,
            'glove': None,
            'homeplate': None,
            'baseball': None,
            'real_world_coords': {}
        }
        
        frame_height, frame_width = frame.shape[:2]
        
        # Extract detections from results
        for result in results:
            for box_idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()  # Convert to numpy for easier handling
                
                x1, y1, x2, y2 = map(int, xyxy)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Store detection based on class
                if cls_id == self.glove_class_id:
                    if detections['glove'] is None or conf > detections['glove']['confidence']:
                        detections['glove'] = {
                            'box': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'confidence': conf
                        }
                
                elif cls_id == self.homeplate_class_id:
                    if detections['homeplate'] is None or conf > detections['homeplate']['confidence']:
                        detections['homeplate'] = {
                            'box': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'confidence': conf,
                            'width': x2 - x1  # Home plate width in pixels
                        }
                        
                        # Update home plate reference if needed
                        if self.home_plate_reference is None:
                            self.home_plate_reference = detections['homeplate']
                            # Calculate pixels per inch (home plate is 17 inches wide)
                            self.pixels_per_inch = detections['homeplate']['width'] / 17.0
                            self.logger.info(f"Home plate reference set: {self.pixels_per_inch:.2f} pixels per inch")
                
                elif cls_id == self.baseball_class_id:
                    if detections['baseball'] is None or conf > detections['baseball']['confidence']:
                        detections['baseball'] = {
                            'box': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'confidence': conf
                        }
        
        # Convert to real-world coordinates if home plate reference is available
        if self.home_plate_reference is not None and detections['glove'] is not None:
            # Use home plate center as origin
            origin_x, origin_y = self.home_plate_reference['center']
            
            # Calculate real-world coordinates (in inches) relative to home plate
            if detections['glove'] is not None:
                glove_x, glove_y = detections['glove']['center']
                real_x = (glove_x - origin_x) / self.pixels_per_inch
                real_y = (origin_y - glove_y) / self.pixels_per_inch  # Invert Y to match real-world coordinates
                detections['real_world_coords']['glove'] = (real_x, real_y)
            
            if detections['baseball'] is not None:
                ball_x, ball_y = detections['baseball']['center']
                real_x = (ball_x - origin_x) / self.pixels_per_inch
                real_y = (origin_y - ball_y) / self.pixels_per_inch
                detections['real_world_coords']['baseball'] = (real_x, real_y)
        
        # Apply filtering to glove detections
        if hasattr(self, 'enable_filtering') and self.enable_filtering and len(self.tracking_data) > 0:
            prev_detection = self.tracking_data[-1]
            if not self._filter_glove_detection(prev_detection, detections, fps, self.max_velocity_inches_per_sec):
                # If detection is an outlier, don't include glove data
                if 'glove' in detections['real_world_coords']:
                    del detections['real_world_coords']['glove']
                detections['glove'] = None
        
        # Add to tracking data if there are relevant detections
        if detections['glove'] is not None or detections['homeplate'] is not None or detections['baseball'] is not None:
            self.tracking_data.append(detections)
        
        return detections
    
    def _fill_missing_detections(self):
        """
        Fill in missing glove detection coordinates by repeating the last known position.
        This prevents gaps and jumps in the tracking visualization.
        
        Returns:
            List[Tuple[float, float]]: Processed glove coordinates for all frames
        """
        if not self.tracking_data:
            return [], []
        
        # Extract frame indices and glove coordinates
        frame_indices = []
        glove_coords = []
        
        for detection in self.tracking_data:
            frame_indices.append(detection['frame_idx'])
            
            if 'glove' in detection['real_world_coords']:
                glove_coords.append(detection['real_world_coords']['glove'])
            else:
                glove_coords.append(None)
        
        # Fill in missing glove coordinates
        filled_x = []
        filled_y = []
        last_valid_x = None
        last_valid_y = None
        
        for i, coords in enumerate(glove_coords):
            if coords is not None:
                # Valid detection
                x, y = coords
                filled_x.append(x)
                filled_y.append(y)
                last_valid_x, last_valid_y = x, y
            elif last_valid_x is not None and last_valid_y is not None:
                # Missing detection, but we have previous valid coordinates
                filled_x.append(last_valid_x)
                filled_y.append(last_valid_y)
            else:
                # No valid previous coordinates to use
                # We'll skip this frame entirely
                pass
        
        return filled_x, filled_y

    def _annotate_frame(self, frame, detections, frame_idx):
        """
        Annotate the frame with detections and tracking information.
        
        Args:
            frame: Current video frame
            detections: Processed detections
            frame_idx: Frame index
            
        Returns:
            annotated_frame: Frame with annotations
        """
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw home plate
        if detections['homeplate'] is not None:
            x1, y1, x2, y2 = detections['homeplate']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"Home Plate", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw center of home plate
            center_x, center_y = detections['homeplate']['center']
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 255, 0), -1)
            
            # Add home plate dimensions if pixels_per_inch is available
            if self.pixels_per_inch is not None:
                cv2.putText(frame, f"Home Plate Width: 17.0 in", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, f"Scale: {self.pixels_per_inch:.2f} px/in", (x1, y2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw glove
        if detections['glove'] is not None:
            x1, y1, x2, y2 = detections['glove']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Glove", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center of glove
            center_x, center_y = detections['glove']['center']
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
            
            # Add real-world coordinates if available
            if 'glove' in detections['real_world_coords']:
                real_x, real_y = detections['real_world_coords']['glove']
                cv2.putText(frame, f"Pos: ({real_x:.1f}, {real_y:.1f}) in", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw baseball
        if detections['baseball'] is not None:
            x1, y1, x2, y2 = detections['baseball']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Baseball", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw center of baseball
            center_x, center_y = detections['baseball']['center']
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            
            # Add real-world coordinates if available
            if 'baseball' in detections['real_world_coords']:
                real_x, real_y = detections['real_world_coords']['baseball']
                cv2.putText(frame, f"Pos: ({real_x:.1f}, {real_y:.1f}) in", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

    def _update_tracking_plot(self, ax, fig):
        """
        Update the 2D tracking plot with latest data, showing only glove movement.

        Args:
            ax: Matplotlib axis
            fig: Matplotlib figure
        """
        ax.clear()

        # Set up the plot
        ax.set_title('Glove Movement Tracking')
        ax.set_xlabel('X Position (inches from home plate)')
        ax.set_ylabel('Y Position (inches from home plate)')
        ax.grid(True)

        # Use advanced method to handle missing detections
        glove_x, glove_y = self._handle_missing_detections()

        # Draw home plate at origin
        if self.home_plate_reference is not None:
            # Simplified home plate shape at the origin
            home_plate_shape = np.array([[-8.5, 0], [8.5, 0], [0, 8.5], [-8.5, 0]])
            ax.fill(home_plate_shape[:, 0], home_plate_shape[:, 1], color='gray', alpha=0.5, label='Home Plate')

        # Plot tracking data (glove only)
        if glove_x and glove_y:
            ax.plot(glove_x, glove_y, 'g-', label='Glove Path')
            ax.scatter(glove_x[-1], glove_y[-1], color='green', s=100, marker='o', label='Current Glove Pos')

        # Set fixed axis limits as requested
        ax.set_xlim(-50, 50)
        ax.set_ylim(-10, 60)
        
        # Force exact 1:1 aspect ratio
        ax.set_aspect('equal', 'box')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Make figure tight
        fig.tight_layout()

        # Make sure the plot refreshes
        fig.canvas.draw()

        # Convert matplotlib figure to image
        plot_img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2RGB)

        return plot_img


    def _save_tracking_data(self, csv_path, video_path=None):
        """
        Save tracking data to a CSV file, including both raw detections and processed coordinates.

        Args:
            csv_path: Path to save the CSV file
            video_path: Path to the original video (for identification in batch mode)
        """
        if not self.tracking_data:
            self.logger.warning("No tracking data to save")
            return

        # Get total frame count from video if available
        total_frames = getattr(self, 'total_frames', None)
        if total_frames is None and video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        
        # If we still don't know total frames, estimate from tracking data
        if total_frames is None and self.tracking_data:
            total_frames = max(detection['frame_idx'] for detection in self.tracking_data) + 1
        
        # Get processed coordinates for all frames that have tracking data
        processed_x, processed_y = self._fill_missing_detections()
        
        # Create a mapping from frames that have tracking data to processed coordinates
        frame_indices = sorted(set(detection['frame_idx'] for detection in self.tracking_data))
        processed_coords = {}
        
        # Only map if we have both indices and coordinates
        if len(frame_indices) > 0 and len(processed_x) > 0:
            # Make sure they are the same length, or use the shorter length
            min_len = min(len(frame_indices), len(processed_x))
            for i in range(min_len):
                processed_coords[frame_indices[i]] = (processed_x[i], processed_y[i])
        
        # Create a map of frames with actual detections (not interpolated)
        frames_with_detections = set()
        for detection in self.tracking_data:
            if detection['glove'] is not None and 'glove' in detection['real_world_coords']:
                frames_with_detections.add(detection['frame_idx'])
        
        # Prepare data for CSV - create a record for every frame
        csv_data = []
        
        # Include all frames in the range, even if no detection
        for frame_idx in range(total_frames):
            # Find detection for this frame if it exists
            detection = next((d for d in self.tracking_data if d['frame_idx'] == frame_idx), None)
            
            # Base data - will be populated with None for missing frames
            row_data = {
                'frame_idx': frame_idx,
                'homeplate_center_x': None,
                'homeplate_center_y': None,
                'homeplate_width': None,
                'homeplate_confidence': None,
                'glove_center_x': None,
                'glove_center_y': None,
                'glove_confidence': None,
                'glove_real_x': None,
                'glove_real_y': None,
                'baseball_center_x': None,
                'baseball_center_y': None,
                'baseball_confidence': None,
                'baseball_real_x': None,
                'baseball_real_y': None,
                'pixels_per_inch': self.pixels_per_inch,
                # Add new columns for processed coordinates
                'glove_processed_x': None,
                'glove_processed_y': None,
                'is_interpolated': True  # Default to True, will be set to False for actual detections
            }
            
            # If we have detection data for this frame, populate it
            if detection:
                # Home plate data
                if detection['homeplate'] is not None:
                    row_data['homeplate_center_x'], row_data['homeplate_center_y'] = detection['homeplate']['center']
                    row_data['homeplate_width'] = detection['homeplate']['width']
                    row_data['homeplate_confidence'] = detection['homeplate']['confidence']

                # Glove data - raw detections
                if detection['glove'] is not None:
                    row_data['glove_center_x'], row_data['glove_center_y'] = detection['glove']['center']
                    row_data['glove_confidence'] = detection['glove']['confidence']

                    if 'glove' in detection['real_world_coords']:
                        row_data['glove_real_x'], row_data['glove_real_y'] = detection['real_world_coords']['glove']
                        row_data['is_interpolated'] = False  # This is an actual detection

                # Baseball data
                if detection['baseball'] is not None:
                    row_data['baseball_center_x'], row_data['baseball_center_y'] = detection['baseball']['center']
                    row_data['baseball_confidence'] = detection['baseball']['confidence']

                    if 'baseball' in detection['real_world_coords']:
                        row_data['baseball_real_x'], row_data['baseball_real_y'] = detection['real_world_coords']['baseball']
            
            # Add processed coordinates if available for this frame
            if frame_idx in processed_coords:
                row_data['glove_processed_x'], row_data['glove_processed_y'] = processed_coords[frame_idx]
                # Mark as not interpolated if this was an actual detection
                if frame_idx in frames_with_detections:
                    row_data['is_interpolated'] = False
                else:
                    row_data['is_interpolated'] = True
            
            # Add video path if in batch mode
            if video_path:
                row_data['video_filename'] = os.path.basename(video_path)
                
            csv_data.append(row_data)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

        # Logging information
        self.logger.info(f"Tracking data saved to {csv_path} with {len(csv_data)} frames")
        
        # For debugging
        processed_count = sum(1 for row in csv_data if row['glove_processed_x'] is not None)
        interpolated_count = sum(1 for row in csv_data if row['is_interpolated'] is True and row['glove_processed_x'] is not None)
        self.logger.info(f"Processed coordinates: {processed_count} frames, Interpolated: {interpolated_count} frames")
        
        if not os.path.exists(csv_path):
            self.logger.error(f"Failed to save CSV file at {csv_path}")
        else:
            self.logger.debug(f"CSV file saved successfully at {csv_path}")

        
    def analyze_glove_movement(self, csv_path: Optional[str] = None):
        """
        Analyze glove movement data and return statistics.
        
        Args:
            csv_path: Path to the tracking CSV (if None, uses the last tracking data)
            
        Returns:
            dict: Movement statistics
        """
        if csv_path is not None:
            df = pd.read_csv(csv_path)
        elif self.tracking_data:
            # Use the current tracking data
            return self._analyze_tracking_data()
        else:
            self.logger.warning("No tracking data available for analysis")
            return None
        
        # Calculate movement statistics
        stats = {
            'total_frames': len(df),
            'frames_with_glove': df['glove_center_x'].notna().sum(),
            'frames_with_baseball': df['baseball_center_x'].notna().sum(),
            'frames_with_homeplate': df['homeplate_center_x'].notna().sum(),
        }
        
        # Calculate glove movement distance (in real-world units)
        if stats['frames_with_glove'] > 1:
            # Get real-world coordinates where available
            real_coords = df[df['glove_real_x'].notna() & df['glove_real_y'].notna()]
            
            if len(real_coords) > 1:
                # Calculate the total distance travelled
                dx = real_coords['glove_real_x'].diff()
                dy = real_coords['glove_real_y'].diff()
                distances = np.sqrt(dx**2 + dy**2)
                
                stats['total_distance_inches'] = distances.sum()
                stats['max_distance_between_frames_inches'] = distances.max()
                stats['avg_distance_between_frames_inches'] = distances.mean()
                
                # Calculate the area covered by the glove (convex hull)
                if len(real_coords) > 2:  # Need at least 3 points for a convex hull
                    from scipy.spatial import ConvexHull
                    points = real_coords[['glove_real_x', 'glove_real_y']].values
                    try:
                        hull = ConvexHull(points)
                        stats['convex_hull_area_sq_inches'] = hull.volume  # In 2D, volume is area
                    except:
                        stats['convex_hull_area_sq_inches'] = None
                
                # Range of motion
                stats['x_range_inches'] = real_coords['glove_real_x'].max() - real_coords['glove_real_x'].min()
                stats['y_range_inches'] = real_coords['glove_real_y'].max() - real_coords['glove_real_y'].min()
                
                # Average position
                stats['avg_x_position_inches'] = real_coords['glove_real_x'].mean()
                stats['avg_y_position_inches'] = real_coords['glove_real_y'].mean()
        
        return stats
    
    def _analyze_tracking_data(self):
        """
        Analyze the current tracking data.
        
        Returns:
            dict: Movement statistics
        """
        if not self.tracking_data:
            return None
        
        # Convert tracking data to DataFrame for analysis
        data = []
        for detection in self.tracking_data:
            row = {
                'frame_idx': detection['frame_idx'],
                'glove_real_x': None,
                'glove_real_y': None,
                'baseball_real_x': None,
                'baseball_real_y': None
            }
            
            if 'glove' in detection['real_world_coords']:
                row['glove_real_x'], row['glove_real_y'] = detection['real_world_coords']['glove']
            
            if 'baseball' in detection['real_world_coords']:
                row['baseball_real_x'], row['baseball_real_y'] = detection['real_world_coords']['baseball']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate statistics
        stats = {
            'total_frames': len(self.tracking_data),
            'frames_with_glove': sum(1 for d in self.tracking_data if d['glove'] is not None),
            'frames_with_baseball': sum(1 for d in self.tracking_data if d['baseball'] is not None),
            'frames_with_homeplate': sum(1 for d in self.tracking_data if d['homeplate'] is not None),
        }
        
        # Calculate distances if we have real-world coordinates
        real_coords = df[df['glove_real_x'].notna() & df['glove_real_y'].notna()]
        
        if len(real_coords) > 1:
            dx = real_coords['glove_real_x'].diff()
            dy = real_coords['glove_real_y'].diff()
            distances = np.sqrt(dx**2 + dy**2)
            
            stats['total_distance_inches'] = distances.sum()
            stats['max_distance_between_frames_inches'] = distances.max()
            stats['avg_distance_between_frames_inches'] = distances.mean()
            
            # Range of motion
            stats['x_range_inches'] = real_coords['glove_real_x'].max() - real_coords['glove_real_x'].min()
            stats['y_range_inches'] = real_coords['glove_real_y'].max() - real_coords['glove_real_y'].min()
            
            # Average position
            stats['avg_x_position_inches'] = real_coords['glove_real_x'].mean()
            stats['avg_y_position_inches'] = real_coords['glove_real_y'].mean()
        
        return stats

    def _handle_missing_detections(self):
        """
        Enhanced method to handle missing glove detections using multiple strategies.
        Uses a combination of interpolation, filtering, and smoothing techniques
        to create a more natural and continuous glove movement trajectory.
        
        Returns:
            List[Tuple[float, float]]: Processed glove coordinates for all frames
        """
        if not self.tracking_data:
            return [], []
        
        # Extract frame indices and glove coordinates
        frame_indices = []
        glove_coords = []
        
        for detection in self.tracking_data:
            frame_indices.append(detection['frame_idx'])
            
            if 'glove' in detection['real_world_coords']:
                glove_coords.append(detection['real_world_coords']['glove'])
            else:
                glove_coords.append(None)
        
        # Find sequences of valid detections
        valid_sequences = []
        current_sequence = []
        current_frames = []
        
        for i, (coords, frame_idx) in enumerate(zip(glove_coords, frame_indices)):
            if coords is not None:
                current_sequence.append(coords)
                current_frames.append(frame_idx)
            elif current_sequence:
                if len(current_sequence) >= 2:  # Need at least 2 points for a valid sequence
                    valid_sequences.append((current_frames, current_sequence))
                current_sequence = []
                current_frames = []
        
        # Add the last sequence if it's not empty
        if len(current_sequence) >= 2:
            valid_sequences.append((current_frames, current_sequence))
        
        if not valid_sequences:
            self.logger.warning("No valid glove detection sequences found")
            return [], []
        
        # Process each valid sequence to remove outliers
        filtered_sequences = []
        for frames, coords in valid_sequences:
            if len(coords) <= 3:
                # For very short sequences, just keep them as is
                filtered_sequences.append((frames, coords))
            else:
                # For longer sequences, apply moving average to smooth the trajectory
                x_vals = [c[0] for c in coords]
                y_vals = [c[1] for c in coords]
                
                # Simple moving average with window size 3
                smoothed_x = []
                smoothed_y = []
                
                for i in range(len(x_vals)):
                    if i == 0 or i == len(x_vals) - 1:
                        # Keep endpoints unchanged
                        smoothed_x.append(x_vals[i])
                        smoothed_y.append(y_vals[i])
                    else:
                        # Average with neighboring points
                        smoothed_x.append((x_vals[i-1] + x_vals[i] + x_vals[i+1]) / 3)
                        smoothed_y.append((y_vals[i-1] + y_vals[i] + y_vals[i+1]) / 3)
                
                smoothed_coords = list(zip(smoothed_x, smoothed_y))
                filtered_sequences.append((frames, smoothed_coords))
        
        # Check if we need to interpolate between sequences
        if len(filtered_sequences) > 1:
            # Connect the sequences with smooth interpolation if the gaps aren't too large
            full_sequence_frames = []
            full_sequence_coords = []
            
            # Add the first sequence
            first_frames, first_coords = filtered_sequences[0]
            full_sequence_frames.extend(first_frames)
            full_sequence_coords.extend(first_coords)
            
            # Connect subsequent sequences with interpolation
            for i in range(1, len(filtered_sequences)):
                prev_frames, prev_coords = filtered_sequences[i-1]
                curr_frames, curr_coords = filtered_sequences[i]
                
                # Get the gap size in frames
                frame_gap = curr_frames[0] - prev_frames[-1]
                
                # Only interpolate if the gap isn't too large (less than 30 frames)
                if 1 < frame_gap < 30:
                    # Get the end coordinates of the previous sequence
                    start_x, start_y = prev_coords[-1]
                    # Get the start coordinates of the current sequence
                    end_x, end_y = curr_coords[0]
                    
                    # Create interpolated frames and coordinates
                    for j in range(1, frame_gap):
                        # Linear interpolation: position = start + (end - start) * fraction
                        fraction = j / frame_gap
                        interp_x = start_x + (end_x - start_x) * fraction
                        interp_y = start_y + (end_y - start_y) * fraction
                        interp_frame = prev_frames[-1] + j
                        
                        full_sequence_frames.append(interp_frame)
                        full_sequence_coords.append((interp_x, interp_y))
                
                # Add the current sequence
                full_sequence_frames.extend(curr_frames)
                full_sequence_coords.extend(curr_coords)
        else:
            # Just use the single sequence
            full_sequence_frames, full_sequence_coords = filtered_sequences[0]
        
        # Extract x and y coordinates
        x_coords = [coord[0] for coord in full_sequence_coords]
        y_coords = [coord[1] for coord in full_sequence_coords]
        
        # Apply a final velocity-based filter to remove any remaining jumps
        filtered_x = []
        filtered_y = []
        
        if len(x_coords) > 1:
            # Add the first point
            filtered_x.append(x_coords[0])
            filtered_y.append(y_coords[0])
            
            # Check each subsequent point
            for i in range(1, len(x_coords)):
                # Calculate displacement
                dx = x_coords[i] - filtered_x[-1]
                dy = y_coords[i] - filtered_y[-1]
                
                # Calculate distance
                distance = (dx**2 + dy**2)**0.5
                
                # If distance is too large (a jump), limit it
                max_distance = 10.0  # max allowed distance in inches
                if distance > max_distance:
                    # Scale the movement to the maximum allowed distance
                    scale_factor = max_distance / distance
                    new_x = filtered_x[-1] + dx * scale_factor
                    new_y = filtered_y[-1] + dy * scale_factor
                    
                    filtered_x.append(new_x)
                    filtered_y.append(new_y)
                else:
                    # Point is acceptable
                    filtered_x.append(x_coords[i])
                    filtered_y.append(y_coords[i])
        
        return filtered_x, filtered_y
    
    def plot_glove_heatmap(self, csv_path: Optional[str] = None, output_path: Optional[str] = None):
        """
        Create a heatmap of glove positions.
        
        Args:
            csv_path: Path to tracking CSV file
            output_path: Path to save the heatmap image
            
        Returns:
            str: Path to the saved heatmap image
        """
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            
            # Create filled version of the data from the CSV
            real_coords = df[df['glove_real_x'].notna() & df['glove_real_y'].notna()]
            
            if len(real_coords) > 1:
                # Fill missing values by forward-filling
                filled_df = df.copy()
                filled_df['glove_real_x'] = filled_df['glove_real_x'].fillna(method='ffill')
                filled_df['glove_real_y'] = filled_df['glove_real_y'].fillna(method='ffill')
                
                # Use the filled data
                glove_x = filled_df['glove_real_x'].dropna().tolist()
                glove_y = filled_df['glove_real_y'].dropna().tolist()
            else:
                self.logger.warning("Not enough data points in CSV for heatmap")
                return None
        elif self.tracking_data:
            # Use the current tracking data with filled detections
            glove_x, glove_y = self._fill_missing_detections()
            
            if not glove_x or not glove_y:
                self.logger.warning("No valid glove tracking data available for heatmap")
                return None
        else:
            self.logger.warning("No tracking data available for heatmap")
            return None
        
        # Generate heatmap
        plt.figure(figsize=(10, 8))
        
        # Create heatmap from glove positions
        if len(glove_x) > 1:
            plt.hist2d(glove_x, glove_y, bins=20, cmap='hot')
            plt.colorbar(label='Frequency')
            
            # Draw home plate at origin
            home_plate_shape = np.array([[-8.5, 0], [8.5, 0], [0, 8.5], [-8.5, 0]])
            plt.fill(home_plate_shape[:, 0], home_plate_shape[:, 1], color='gray', alpha=0.5)
            
            plt.title('Glove Position Heatmap')
            plt.xlabel('X Position (inches from home plate)')
            plt.ylabel('Y Position (inches from home plate)')
            plt.grid(True, alpha=0.3)
            
            # Set fixed axis limits
            plt.xlim(-50, 50)
            plt.ylim(-10, 60)
            
            # Force 1:1 aspect ratio
            plt.gca().set_aspect('equal', 'box')
            
            # Save the heatmap
            if output_path is None:
                output_path = os.path.join(self.results_dir, "glove_heatmap.png")
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Glove heatmap saved to {output_path}")
            return output_path
        else:
            self.logger.warning("Not enough glove position data for heatmap")
            plt.close()
            return None
        
    def _filter_glove_detection(self, prev_detection, current_detection, fps, max_velocity_inches_per_sec=120.0):
        """
        Filter out physically impossible glove movements based on velocity constraints.
        
        Args:
            prev_detection (dict): Previous frame's detection data
            current_detection (dict): Current frame's detection data
            fps (float): Frames per second of the video
            max_velocity_inches_per_sec (float): Maximum plausible velocity of the glove in inches per second
            
        Returns:
            bool: True if the detection passes the filter, False if it's an outlier
        """
        # If no previous detection, accept current one
        if prev_detection is None or 'glove' not in prev_detection['real_world_coords']:
            return True
        
        # If no current glove detection, nothing to filter
        if 'glove' not in current_detection['real_world_coords']:
            return True
        
        # Get real-world coordinates
        prev_x, prev_y = prev_detection['real_world_coords']['glove']
        curr_x, curr_y = current_detection['real_world_coords']['glove']
        
        # Calculate distance moved in real-world units (inches)
        distance = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
        
        # Calculate time between frames
        time_diff = (current_detection['frame_idx'] - prev_detection['frame_idx']) / fps
        
        # Calculate velocity in inches per second
        if time_diff > 0:
            velocity = distance / time_diff
        else:
            velocity = float('inf')  # Avoid division by zero
        
        # Check if velocity exceeds maximum plausible limit
        if velocity > max_velocity_inches_per_sec:
            self.logger.debug(f"Filtered outlier at frame {current_detection['frame_idx']}: velocity={velocity:.2f} in/s > {max_velocity_inches_per_sec} in/s")
            return False
        
        return True