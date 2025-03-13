import gc  # For garbage collection
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for better memory usage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch
import io
from contextlib import redirect_stdout
from datetime import datetime

from baseballcv.functions.savant_scraper import BaseballSavVideoScraper
from baseballcv.functions.load_tools import LoadTools

class GloveFramingTracker:
    """
    A class for tracking glove movements in baseball videos and analyzing framing.
    
    This class downloads baseball videos from Baseball Savant, uses computer vision models
    to detect and track the catcher's glove, and calculates metrics related to glove framing.
    It stores the tracking data and can generate visualization videos similar to the original
    glove_framing_tracking.py demo script.
    """
    
    def __init__(
        self, 
        glove_model: str = 'glove_tracking',
        ball_model: str = 'ball_trackingv4',
        homeplate_model: str = 'phc_detector',
        results_dir: str = "results/glove_framing",
        verbose: bool = True,
        device: str = None
    ):
        """
        Initialize the GloveFramingTracker.
        
        Args:
            glove_model (str): Model name for detecting gloves
            ball_model (str): Model name for detecting baseballs
            homeplate_model (str): Model name for detecting home plate
            results_dir (str): Directory to save results
            verbose (bool): Whether to print detailed progress information
            device (str): Device to run models on (cpu, cuda, etc.)
        """
        self.load_tools = LoadTools()
        
        # Load models
        self.glove_model = self.load_tools.load_model(glove_model)
        self.ball_model = self.load_tools.load_model(ball_model)
        self.homeplate_model = self.load_tools.load_model(homeplate_model)
        
        # Convert to YOLO models
        from ultralytics import YOLO
        self.glove_model = YOLO(self.glove_model)
        self.ball_model = YOLO(self.ball_model)
        self.homeplate_model = YOLO(self.homeplate_model)
        
        if verbose:
            print(f"Models loaded: {glove_model}, {ball_model}, {homeplate_model}")
        
        # Create results directory
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
    
    def analyze(
        self, 
        start_date: str, 
        end_date: str,
        team: str = None,
        pitch_call: str = None,
        max_videos: int = None,
        max_videos_per_game: int = None,
        create_video: bool = True,
        save_csv: bool = True,
        csv_path: str = None,
        delete_savant_videos: bool = True,  # Parameter to control deletion of downloaded videos
        delete_output_videos: bool = False  # Parameter to control deletion of output videos
    ) -> List[Dict]:
        """
        Analyze videos from a date range to track glove framing.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            team (str): Team abbreviation to filter by
            pitch_call (str): Pitch call to filter by (e.g., "Strike")
            max_videos (int): Maximum number of videos to process
            max_videos_per_game (int): Maximum videos per game
            create_video (bool): Whether to create annotated videos
            save_csv (bool): Whether to save analysis results to CSV
            csv_path (str): Custom path for CSV file
            delete_savant_videos (bool): Whether to delete downloaded videos after processing
            delete_output_videos (bool): Whether to delete created output videos after processing
            
        Returns:
            List[Dict]: List of analysis results per video
        """
        
        # Download videos
        savant_scraper = BaseballSavVideoScraper()
        download_folder = os.path.join(self.results_dir, "savant_videos")
        pitch_data = savant_scraper.run_statcast_pull_scraper(
            download_folder=download_folder, 
            start_date=start_date, 
            end_date=end_date, 
            team=team, 
            pitch_call=pitch_call, 
            max_videos=max_videos, 
            max_videos_per_game=max_videos_per_game,
            max_workers=(os.cpu_count() - 2) if os.cpu_count() > 3 else 1
        )

        video_files = [os.path.join(download_folder, f) for f in os.listdir(download_folder) if f.endswith('.mp4')]
        
        # Create list to store all results
        glove_tracking_results = []
        
        # Create a list to store all rows for the new CSV format
        all_frame_rows = []
        
        # Process each video
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            play_id = video_name.split('_')[-1]
            game_pk = video_name.split('_')[-2]
            
            # Find corresponding pitch data
            pitch_data_row = None
            for _, row in pitch_data.iterrows():
                if str(row["play_id"]) == play_id:
                    pitch_data_row = row
                    break
            
            if pitch_data_row is None:
                if self.verbose:
                    print(f"No pitch data found for play_id {play_id}, skipping...")
                continue
                
            output_path = os.path.join(self.results_dir, f"glove_framing_{play_id}.mp4")
            
            # Get strike zone information
            sz_top = float(pitch_data_row.get("sz_top", 3.5))  # Default to 3.5 feet if not available
            sz_bot = float(pitch_data_row.get("sz_bot", 1.5))  # Default to 1.5 feet if not available
            
            # Process video to track glove movement
            glove_positions_raw, glove_positions_transformed, glove_boxes, ball_glove_frame, homeplate_box, fps = self._track_glove_movement(
                video_path=video_path,
                output_path=output_path if create_video else None,
                sz_top=sz_top,
                sz_bot=sz_bot
            )
            
            # Calculate glove movement metrics
            smoothness, speed, distance_traveled = self._calculate_metrics(glove_positions_transformed, fps)
            
            # Extract key player IDs from Statcast data
            game_date = pitch_data_row.get("game_date", None)
            pitcher_id = pitch_data_row.get("pitcher", None)
            batter_id = pitch_data_row.get("batter", None)
            
            # Ensure we get catcher_id from fielder_2
            catcher_id = None
            # Try different ways to extract catcher ID
            if "fielder_2" in pitch_data_row:
                catcher_id = pitch_data_row["fielder_2"]
            elif "fielder_2_id" in pitch_data_row:
                catcher_id = pitch_data_row["fielder_2_id"]
            elif "statcast_fielder_2" in pitch_data_row:
                catcher_id = pitch_data_row["statcast_fielder_2"]
            
            # Compile basic results
            results = {
                "video_name": video_name,
                "play_id": play_id,
                "game_pk": game_pk,
                "game_date": game_date,
                "ball_glove_frame": ball_glove_frame,
                "glove_positions_raw": glove_positions_raw,
                "glove_positions_transformed": glove_positions_transformed,
                "glove_movement_smoothness": smoothness,
                "glove_movement_speed": speed,
                "glove_distance_traveled": distance_traveled,
                "fps": fps,
                "annotated_video": output_path if create_video else None,
                "catcher_id": catcher_id,  # Include catcher ID
                "pitcher_id": pitcher_id,
                "batter_id": batter_id,
                "sz_top": sz_top,
                "sz_bot": sz_bot
            }
            
            glove_tracking_results.append(results)
            
            # Create frame-by-frame rows for CSV with limited columns
            for frame_idx in range(len(glove_positions_raw)):
                # Get raw and transformed coordinates
                raw_x, raw_y = glove_positions_raw[frame_idx]
                transformed_x, transformed_y = glove_positions_transformed[frame_idx]
                
                row = {
                    # Only include the specified columns
                    "video_name": video_name,
                    "play_id": play_id,
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "pitcher_id": pitcher_id,
                    "batter_id": batter_id, 
                    "catcher_id": catcher_id,
                    "frame_number": frame_idx,
                    # Raw coordinates (in image space)
                    "glove_x": raw_x,
                    "glove_y": raw_y,
                    # Transformed coordinates (for the XY plane)
                    "glove_transformed_x": transformed_x,
                    "glove_transformed_y": transformed_y,
                    # Metrics
                    "glove_movement_smoothness": smoothness,
                    "glove_movement_speed": speed,
                    "glove_distance_traveled": distance_traveled,
                    "ball_glove_frame": ball_glove_frame,
                    "fps": fps,
                    # Home plate data
                    "homeplate_detected": homeplate_box is not None,
                    "homeplate_x1": homeplate_box[0] if homeplate_box else None,
                    "homeplate_y1": homeplate_box[1] if homeplate_box else None,
                    "homeplate_x2": homeplate_box[2] if homeplate_box else None,
                    "homeplate_y2": homeplate_box[3] if homeplate_box else None,
                    # Strike zone data
                    "sz_top": sz_top,
                    "sz_bot": sz_bot
                }
                
                all_frame_rows.append(row)
        
        # Save data to CSV if requested
        if save_csv and all_frame_rows:
            if csv_path is None:
                csv_path = os.path.join(self.results_dir, "glove_tracking_results.csv")
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            
            # Create DataFrame from all frame rows
            df = pd.DataFrame(all_frame_rows)
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            if self.verbose:
                print(f"Saved detailed results to {csv_path}")
                print(f"CSV contains {len(df)} rows with {len(df.columns)} columns of data")
        
        # Clean up downloaded and output videos if requested
        if delete_savant_videos:
            if self.verbose:
                print(f"Cleaning up downloaded videos from {download_folder}")
            savant_scraper.cleanup_savant_videos(download_folder)
        
        if delete_output_videos and create_video:
            if self.verbose:
                print("Cleaning up created output videos")
            for result in glove_tracking_results:
                output_video = result.get("annotated_video")
                if output_video and os.path.exists(output_video):
                    os.remove(output_video)
                    if self.verbose:
                        print(f"Deleted {output_video}")
        
        return glove_tracking_results
        
    def _detect_objects(self, video_path: str, model, object_name: str, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in every frame of the video.
        
        Args:
            video_path (str): Path to the video file
            model: YOLO model to use for detection
            object_name (str): Name of the object to detect
            conf_threshold (float): Confidence threshold for detection
            
        Returns:
            List[Dict]: List of detection dictionaries
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detections = []
        frame_number = 0
        
        pbar = tqdm(total=total_frames, desc=f"{object_name.capitalize()} Detection", 
                   disable=not self.verbose)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            with io.StringIO() as buf, redirect_stdout(buf):
                results = model.predict(frame, conf=conf_threshold, device=self.device, verbose=False)
                
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    cls_name = model.names[cls].lower()
                    
                    if cls_name == object_name.lower():
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf)
                        detections.append({
                            "frame": frame_number,
                            "frame_time": frame_number / fps,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "confidence": conf
                        })
            
            frame_number += 1
            pbar.update(1)
            
            # Clean up memory
            if frame_number % 30 == 0:  # Every 30 frames
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        pbar.close()
        cap.release()
        
        if self.verbose:
            print(f"Completed {object_name} detection. Found {len(detections)} detections")
        
        return detections
    
    def _detect_homeplate(self, homeplate_detections: List[Dict]) -> Optional[Tuple[int, int, int, int]]:
        """
        Process homeplate detections to find the best homeplate bounding box.
        
        Args:
            homeplate_detections (List[Dict]): List of homeplate detections
            
        Returns:
            Optional[Tuple[int, int, int, int]]: Homeplate box (x1, y1, x2, y2) or None if not found
        """
        if not homeplate_detections:
            return None
            
        # Group by frame
        homeplate_by_frame = {}
        for det in homeplate_detections:
            frame = det["frame"]
            if frame not in homeplate_by_frame:
                homeplate_by_frame[frame] = []
            homeplate_by_frame[frame].append(det)
            
        # Check multiple frames for stability
        homeplate_candidates = []
        
        for frame, detections in homeplate_by_frame.items():
            # Get highest confidence detection in this frame
            best_detection = max(detections, key=lambda d: d["confidence"])
            homeplate_candidates.append(best_detection)
        
        if not homeplate_candidates:
            return None
            
        # Use the detection with highest confidence across all frames
        best_homeplate = max(homeplate_candidates, key=lambda d: d["confidence"])
        
        return (best_homeplate["x1"], best_homeplate["y1"], 
                best_homeplate["x2"], best_homeplate["y2"])
    
    def _find_ball_reaches_glove(
        self, 
        ball_by_frame: Dict[int, List[Dict]], 
        glove_by_frame: Dict[int, List[Dict]], 
        tolerance: float = 0.1
    ) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
        """
        Find the frame where ball reaches the glove.

        Args:
            ball_by_frame (Dict): Ball detections grouped by frame
            glove_by_frame (Dict): Glove detections grouped by frame
            tolerance (float): Margin around glove box for ball detection

        Returns:
            Tuple[Optional[int], Optional[Tuple[float, float]]]:
                (frame index, ball center coordinates)
        """
        # Find all frames with both ball and glove
        common_frames = set(ball_by_frame.keys()) & set(glove_by_frame.keys())
        for frame in sorted(common_frames):
            for glove_det in glove_by_frame[frame]:
                # Add tolerance around glove box
                margin_x = tolerance * (glove_det["x2"] - glove_det["x1"])
                margin_y = tolerance * (glove_det["y2"] - glove_det["y1"])
                extended_x1 = glove_det["x1"] - margin_x
                extended_y1 = glove_det["y1"] - margin_y
                extended_x2 = glove_det["x2"] + margin_x
                extended_y2 = glove_det["y2"] + margin_y

                for ball_det in ball_by_frame[frame]:
                    # Calculate ball center
                    ball_center_x = (ball_det["x1"] + ball_det["x2"]) / 2
                    ball_center_y = (ball_det["y1"] + ball_det["y2"]) / 2

                    # Check if ball center is within extended glove box
                    if (extended_x1 <= ball_center_x <= extended_x2 and
                        extended_y1 <= ball_center_y <= extended_y2):
                        return frame, (ball_center_x, ball_center_y)

        # If no direct contact found, try to find closest approach
        min_distance = float('inf')
        closest_frame = None
        closest_ball_center = None
        
        for frame in sorted(common_frames):
            for glove_det in glove_by_frame[frame]:
                glove_center_x = (glove_det["x1"] + glove_det["x2"]) / 2
                glove_center_y = (glove_det["y1"] + glove_det["y2"]) / 2
                
                for ball_det in ball_by_frame[frame]:
                    ball_center_x = (ball_det["x1"] + ball_det["x2"]) / 2
                    ball_center_y = (ball_det["y1"] + ball_det["y2"]) / 2
                    
                    # Calculate Euclidean distance
                    distance = np.sqrt(
                        (ball_center_x - glove_center_x)**2 + 
                        (ball_center_y - glove_center_y)**2
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_frame = frame
                        closest_ball_center = (ball_center_x, ball_center_y)
        
        if closest_frame is not None:
            return closest_frame, closest_ball_center

        return None, None
    
    def _track_glove_movement(
        self, 
        video_path: str, 
        output_path: str = None,
        sz_top: float = 3.5,
        sz_bot: float = 1.5
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[int, int, int, int]], Optional[int], Optional[Tuple[int, int, int, int]], float]:
        """
        Track glove movement in a video and optionally create visualization.
        
        Args:
            video_path (str): Path to the input video
            output_path (str): Path to save the output video (None to skip)
            sz_top (float): Top of strike zone in feet
            sz_bot (float): Bottom of strike zone in feet
            
        Returns:
            Tuple containing:
            - List of raw glove positions (x, y)
            - List of transformed glove positions (x, y)
            - List of glove bounding boxes (x1, y1, x2, y2)
            - Frame where ball reaches glove
            - Home plate bounding box
            - Video FPS
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Detect objects in each frame
        glove_detections = self._detect_objects(video_path, self.glove_model, "glove")
        ball_detections = self._detect_objects(video_path, self.ball_model, "baseball")
        homeplate_detections = self._detect_objects(video_path, self.homeplate_model, "homeplate")

        # Group detections by frame
        glove_by_frame = {}
        for det in glove_detections:
            frame = det["frame"]
            if frame not in glove_by_frame:
                glove_by_frame[frame] = []
            glove_by_frame[frame].append(det)
        
        ball_by_frame = {}
        for det in ball_detections:
            frame = det["frame"]
            if frame not in ball_by_frame:
                ball_by_frame[frame] = []
            ball_by_frame[frame].append(det)
            
        # Find home plate
        homeplate_box = self._detect_homeplate(homeplate_detections)
        
        # Perspective transformation points
        # Use home plate to define the transformation if available
        if homeplate_box is not None:
            src_points = np.array([
                [homeplate_box[0], homeplate_box[3]],  # Bottom left
                [homeplate_box[2], homeplate_box[3]],  # Bottom right
                [homeplate_box[2], homeplate_box[1]],  # Top right
                [homeplate_box[0], homeplate_box[1]]   # Top left
            ], dtype=np.float32)
        else:
            # Use exact source points from example
            src_points = np.array([
                [frame_width * 0.3, frame_height * 0.8],
                [frame_width * 0.7, frame_height * 0.8],
                [frame_width * 0.7, frame_height * 0.2],
                [frame_width * 0.3, frame_height * 0.2]
            ], dtype=np.float32)

        # Define destination points preserving the pitcher's perspective
        # We're not trying to create a bird's eye view, but rather normalize the pitcher's view
        # while maintaining the correct scale based on the home plate width (17 inches)
        dst_width = frame_width  # Keep same width
        dst_height = frame_height  # Keep same height
            
        # Calculate destination points that normalize the home plate to be horizontal
        # and centered, but maintain the pitcher's perspective (not bird's eye)
        plate_width = 100  # Standardized width in pixels for home plate (will be calibrated to 17 inches)
        plate_height = plate_width * 0.4  # Approximate height ratio of home plate as seen from pitcher's view
        
        dst_points = np.array([
            [dst_width/2 - plate_width/2, dst_height * 0.7],  # Bottom left of home plate
            [dst_width/2 + plate_width/2, dst_height * 0.7],  # Bottom right of home plate
            [dst_width/2 + plate_width/2, dst_height * 0.7 - plate_height],  # Top right of home plate
            [dst_width/2 - plate_width/2, dst_height * 0.7 - plate_height]   # Top left of home plate
        ], dtype=np.float32)

        # Compute transformation matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Find when ball reaches glove
        ball_glove_frame, ball_center = self._find_ball_reaches_glove(
            ball_by_frame, glove_by_frame
        )

        # Track both raw and transformed glove positions
        glove_positions_raw = []  # Original image coordinates
        glove_positions_transformed = []  # Transformed to standardized plane
        glove_boxes = []
        sorted_frames = sorted(glove_by_frame.keys())
        
        for frame in sorted_frames:
            if frame not in glove_by_frame:
                continue
                
            for glove_det in glove_by_frame[frame]:
                x1, y1, x2, y2 = glove_det["x1"], glove_det["y1"], glove_det["x2"], glove_det["y2"]
                glove_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Store raw position
                glove_positions_raw.append(glove_center)
                
                # Transform to standardized coordinates
                try:
                    # Use perspective transform
                    flat_glove_point = cv2.perspectiveTransform(np.array([[glove_center]]), M)[0][0]
                    
                    # In the transformed view, y should increase upward (opposite of image coordinates)
                    # No need to reverse Y coordinate - we want to keep the pitcher's perspective
                    # where Y increases downward from the top of the frame
                    flat_glove_point = (flat_glove_point[0], flat_glove_point[1])
                    
                    glove_positions_transformed.append(flat_glove_point)
                    glove_boxes.append((x1, y1, x2, y2))
                except:
                    # Skip if transformation fails
                    if len(glove_positions_transformed) > 0:
                        # Use previous transformed point as fallback
                        glove_positions_transformed.append(glove_positions_transformed[-1])
                    else:
                        # First point - use a default in the middle of the transformed plane
                        glove_positions_transformed.append((dst_width/2, dst_height/2))
                
                break  # Take only the first glove per frame
        
        # Calculate pixels per inch using home plate width (17 inches)
        if homeplate_box is not None:
            # Calculate home plate width in pixels
            homeplate_width_px = homeplate_box[2] - homeplate_box[0]
            # 17 inches is standard width of home plate and strike zone
            pixels_per_inch = homeplate_width_px / 17
        else:
            # Fallback if home plate not detected
            pixels_per_inch = 10  # Rough estimate
        
        # Create visualization video if requested
        if output_path:
            self._create_visualization_video(
                video_path, 
                output_path,
                glove_by_frame,
                ball_by_frame,
                glove_positions_raw,
                glove_positions_transformed,
                M,
                dst_width,
                dst_height,
                ball_glove_frame,
                homeplate_box,
                sz_top,
                sz_bot,
                pixels_per_inch
            )
        
        # Force memory cleanup
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return glove_positions_raw, glove_positions_transformed, glove_boxes, ball_glove_frame, homeplate_box, fps
    
    def _calculate_metrics(
        self, 
        glove_positions: List[Tuple[float, float]], 
        fps: float
    ) -> Tuple[float, float, float]:
        """
        Calculate metrics for glove movement.
        
        Args:
            glove_positions (List[Tuple[float, float]]): List of glove positions
            fps (float): Frames per second
            
        Returns:
            Tuple[float, float, float]: 
                - Movement smoothness score (higher is smoother)
                - Average movement speed (pixels per second)
                - Total distance traveled (pixels)
        """
        if not glove_positions or len(glove_positions) < 3:
            return 0.0, 0.0, 0.0
            
        # Calculate velocities between consecutive frames
        velocities = []
        total_distance = 0.0
        
        for i in range(1, len(glove_positions)):
            prev_x, prev_y = glove_positions[i-1]
            curr_x, curr_y = glove_positions[i]
            
            # Calculate distance between consecutive positions
            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            total_distance += distance
            
            # Calculate velocity (distance per frame, convert to distance per second)
            velocity = distance * fps
            velocities.append(velocity)
        
        # Calculate average speed
        avg_speed = np.mean(velocities) if velocities else 0.0
        
        # Calculate smoothness (lower acceleration variation means smoother movement)
        # Calculate accelerations between consecutive velocity points
        accelerations = []
        for i in range(1, len(velocities)):
            acceleration = abs(velocities[i] - velocities[i-1])
            accelerations.append(acceleration)
            
        # Smoothness is inversely related to the standard deviation of acceleration
        if accelerations:
            accel_std = np.std(accelerations)
            smoothness = 1.0 / (1.0 + accel_std) if accel_std > 0 else 1.0
        else:
            smoothness = 0.0
        
        return smoothness, avg_speed, total_distance
    
    def _create_visualization_video(
        self,
        input_path: str,
        output_path: str,
        glove_by_frame: Dict[int, List[Dict]],
        ball_by_frame: Dict[int, List[Dict]],
        glove_positions_raw: List[Tuple[float, float]],
        glove_positions_transformed: List[Tuple[float, float]],
        transformation_matrix: np.ndarray,
        dst_width: int,
        dst_height: int,
        ball_glove_frame: Optional[int],
        homeplate_box: Optional[Tuple[int, int, int, int]],
        sz_top: float = 3.5,
        sz_bot: float = 1.5,
        pixels_per_inch: float = 10.0
    ) -> None:
        """
        Create a visualization video showing glove tracking and transformed view.
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video
            glove_by_frame (Dict): Glove detections by frame
            ball_by_frame (Dict): Ball detections by frame
            glove_positions_raw (List): Raw glove positions
            glove_positions_transformed (List): Transformed glove positions for visualization
            transformation_matrix (np.ndarray): Perspective transformation matrix
            dst_width (int): Width of transformed view
            dst_height (int): Height of transformed view
            ball_glove_frame (int): Frame where ball reaches glove
            homeplate_box (Tuple): Homeplate box coordinates
            sz_top (float): Top of strike zone in feet
            sz_bot (float): Bottom of strike zone in feet
            pixels_per_inch (float): Conversion factor from pixels to inches
        """
        # Always use CPU for video creation to avoid GPU memory issues
        if self.verbose:
            print("Creating visualization video...")
        
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Find the extents of the trajectory
        if glove_positions_transformed:
            min_x = min(pos[0] for pos in glove_positions_transformed)
            max_x = max(pos[0] for pos in glove_positions_transformed)
            min_y = min(pos[1] for pos in glove_positions_transformed)
            max_y = max(pos[1] for pos in glove_positions_transformed)
            
            # Add padding to ensure nothing is clipped
            padding = 100
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(dst_width, max_x + padding)
            max_y = min(dst_height, max_y + padding)
            
            # Ensure we maintain a 1:1 aspect ratio
            width_range = max_x - min_x
            height_range = max_y - min_y
            max_range = max(width_range, height_range)
            
            # Adjust to have equal ranges for x and y (maintain 1:1 aspect ratio)
            if width_range < max_range:
                center_x = (min_x + max_x) / 2
                min_x = center_x - max_range / 2
                max_x = center_x + max_range / 2
            
            if height_range < max_range:
                center_y = (min_y + max_y) / 2
                min_y = center_y - max_range / 2
                max_y = center_y + max_range / 2
        else:
            # Default to full view if no positions
            min_x, min_y = 0, 0
            max_x, max_y = dst_width, dst_height
        
        # Convert strike zone boundaries from feet to pixels
        # Determine pixels per foot
        pixels_per_foot = pixels_per_inch * 12
        
        # Calculate the center of the home plate for reference
        if homeplate_box:
            homeplate_center_x = (homeplate_box[0] + homeplate_box[2]) / 2
            homeplate_center_y = homeplate_box[3]  # Bottom of the home plate box
            
            # Transform home plate center to visualization coordinates
            hp_center_transformed = cv2.perspectiveTransform(
                np.array([[[homeplate_center_x, homeplate_center_y]]]), 
                transformation_matrix
            )[0][0]
            
            # Adjust y-coordinate to match our visualization convention (y increases upward)
            hp_center_transformed = (hp_center_transformed[0], dst_height - hp_center_transformed[1])
            
            # Calculate strike zone boundaries in pixels relative to home plate
            # Strike zone is centered on home plate horizontally
            strike_zone_width_px = 17 * pixels_per_inch  # Standard 17 inches
            strike_zone_left = hp_center_transformed[0] - strike_zone_width_px / 2
            strike_zone_right = hp_center_transformed[0] + strike_zone_width_px / 2
            
            # Strike zone height based on sz_top and sz_bot
            # From pitcher's perspective, sz_top is higher (smaller y value) than sz_bot
            strike_zone_height_px = (sz_top - sz_bot) * pixels_per_foot
            
            # Position the strike zone above home plate, with the bottom of zone at sz_bot height
            # In the pitcher's perspective, lower y values are higher in the frame
            strike_zone_bottom = hp_center_transformed[1] - 50  # Just above home plate
            strike_zone_top = strike_zone_bottom - strike_zone_height_px  # Higher = smaller y value
        else:
            # Fallback if home plate not detected
            strike_zone_left = dst_width / 2 - 8.5 * pixels_per_inch
            strike_zone_right = dst_width / 2 + 8.5 * pixels_per_inch
            
            # Assume a standard positioning from pitcher's perspective
            # where the strike zone is above home plate
            strike_zone_height_px = (sz_top - sz_bot) * pixels_per_foot
            strike_zone_bottom = dst_height * 0.65  # Position above where home plate would be
            strike_zone_top = strike_zone_bottom - strike_zone_height_px  # Higher = smaller y in pitcher's view
        
        # Update min_x, max_x, min_y, max_y to include the strike zone
        min_x = min(min_x, strike_zone_left - 20)
        max_x = max(max_x, strike_zone_right + 20)
        min_y = min(min_y, strike_zone_bottom - 20)
        max_y = max(max_y, strike_zone_top + 20)
        
        # Calculate scaling factor for visualization
        vis_width = 400  # Width of visualization panel
        vis_height = int(vis_width * (max_y - min_y) / (max_x - min_x)) if max_x > min_x else vis_width
        
        # Ensure visualization height doesn't exceed frame height
        if vis_height > frame_height:
            vis_height = frame_height
            vis_width = int(vis_height * (max_x - min_x) / (max_y - min_y)) if max_y > min_y else vis_height
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        combined_width = frame_width + vis_width + 20  # Add 20px for gap between views
        out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, frame_height))
        
        # Get glove image path (try different asset locations)
        asset_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "baseball_glove.png"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "baseball_glove.png")
        ]
        
        glove_img = None
        for path in asset_paths:
            if os.path.exists(path):
                try:
                    glove_img = plt.imread(path)
                    break
                except:
                    pass
        
        if glove_img is None:
            # Create a simple glove placeholder
            glove_img = np.ones((50, 50, 4), dtype=np.uint8) * 255
            glove_img[:, :, 3] = 255  # Alpha channel
        
        processed_positions = []
        frame_num = 0
        
        # Process video frames
        pbar = tqdm(total=total_frames, desc="Creating Video", disable=not self.verbose)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create a combined frame
            combined_frame = np.zeros((frame_height, combined_width, 3), dtype=np.uint8)
            
            # Original video with annotations on the left
            annotated_frame = frame.copy()
            
            # Draw home plate if detected
            if homeplate_box:
                cv2.rectangle(annotated_frame, 
                              (homeplate_box[0], homeplate_box[1]), 
                              (homeplate_box[2], homeplate_box[3]), 
                              (0, 128, 255), 2)
                cv2.putText(annotated_frame, "Home Plate", (homeplate_box[0], homeplate_box[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 2)
            
            # Draw glove detections
            if frame_num in glove_by_frame:
                for det in glove_by_frame[frame_num]:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)
                    cv2.circle(annotated_frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)  # Red dot
                    
                    if len(processed_positions) < len(glove_positions_transformed):
                        processed_positions.append(glove_positions_transformed[len(processed_positions)])
            
            # Draw ball detections
            if frame_num in ball_by_frame:
                for det in ball_by_frame[frame_num]:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            # Add frame number
            cv2.putText(annotated_frame, f"Frame: {frame_num}", (10, frame_height - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Ball-glove contact frame indicator
            if frame_num == ball_glove_frame:
                cv2.putText(annotated_frame, "BALL CONTACT", (frame_width // 2 - 80, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Copy to combined frame
            combined_frame[0:frame_height, 0:frame_width] = annotated_frame
            
            # Create the right-side visualization
            right_side = np.zeros((frame_height, vis_width, 3), dtype=np.uint8)
            right_side[:, :] = (0, 0, 0)  # Black background
            
            # Draw the glove path on the right side
            if processed_positions:
                # Draw title
                cv2.putText(
                    right_side,
                    "Glove Movement (Pitcher's Perspective)",
                    (vis_width // 2 - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),  # White
                    2
                )
                
                # Function to map from world to visualization coordinates
                def world_to_vis(x, y):
                    # Map from world coordinates to visualization panel
                    # Since we're using the pitcher's perspective, y increases downward
                    vis_x = int((x - min_x) / (max_x - min_x) * vis_width)
                    vis_y = int((y - min_y) / (max_y - min_y) * vis_height)
                    return vis_x, vis_y
                
                # Draw axes
                axis_color = (100, 100, 100)  # Gray
                origin_x, origin_y = world_to_vis(min_x, min_y)
                
                # X-axis
                cv2.line(right_side, (0, origin_y), (vis_width, origin_y), axis_color, 1)
                # Y-axis
                cv2.line(right_side, (origin_x, 0), (origin_x, frame_height), axis_color, 1)
                
                # Draw strike zone
                sz_left, sz_bottom = world_to_vis(strike_zone_left, strike_zone_bottom)
                sz_right, sz_top = world_to_vis(strike_zone_right, strike_zone_top)
                
                # Draw strike zone rectangle
                cv2.rectangle(
                    right_side,
                    (sz_left, sz_top),
                    (sz_right, sz_bottom),
                    (0, 255, 255),  # Yellow
                    2
                )
                
                # Label strike zone
                cv2.putText(
                    right_side,
                    "Strike Zone",
                    (sz_left, sz_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),  # Yellow
                    1
                )
                
                # Draw home plate at the bottom
                if homeplate_box:
                    hp_x, hp_y = world_to_vis(hp_center_transformed[0], hp_center_transformed[1])
                    cv2.circle(
                        right_side,
                        (hp_x, hp_y),
                        5,
                        (0, 128, 255),  # Orange
                        -1
                    )
                    cv2.putText(
                        right_side,
                        "Home Plate",
                        (hp_x - 40, hp_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 128, 255),  # Orange
                        1
                    )
                
                # Draw tracking trail
                for i in range(1, len(processed_positions)):
                    # Get current and previous positions
                    prev_pos = processed_positions[i-1]
                    curr_pos = processed_positions[i]
                    
                    # Convert to visualization coordinates
                    prev_x, prev_y = world_to_vis(prev_pos[0], prev_pos[1])
                    curr_x, curr_y = world_to_vis(curr_pos[0], curr_pos[1])
                    
                    # Draw line segment (with a color gradient based on frame number)
                    progress = i / len(processed_positions)
                    color = (
                        int(255 * (1 - progress)),  # Blue decreases
                        int(50 * progress),         # Green increases slightly
                        int(255 * progress)         # Red increases
                    )
                    
                    cv2.line(
                        right_side,
                        (prev_x, prev_y),
                        (curr_x, curr_y),
                        color,
                        2
                    )
                
                # Get the current glove position
                if len(processed_positions) > 0:
                    last_pos = processed_positions[-1]
                    last_x, last_y = world_to_vis(last_pos[0], last_pos[1])
                    
                    # Draw current glove position
                    cv2.circle(
                        right_side,
                        (last_x, last_y),
                        10,  # Radius
                        (0, 255, 255),  # Yellow
                        -1  # Filled circle
                    )
                    
                    # Display coordinates for reference
                    cv2.putText(
                        right_side,
                        f"X: {last_pos[0]:.1f}, Y: {last_pos[1]:.1f}",
                        (10, frame_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                
                # Add legend for scale
                scale_length_inches = 6  # Show a 6-inch scale
                scale_px = int(scale_length_inches * pixels_per_inch)
                scale_start = (vis_width - scale_px - 10, frame_height - 50)
                scale_end = (vis_width - 10, frame_height - 50)
                
                cv2.line(
                    right_side,
                    scale_start,
                    scale_end,
                    (255, 255, 255),  # White
                    2
                )
                
                cv2.putText(
                    right_side,
                    f"{scale_length_inches} inches",
                    (scale_start[0], scale_start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            # Copy right side visualization to combined frame
            combined_frame[0:frame_height, frame_width+20:] = right_side
            
            # Write to output
            out.write(combined_frame)
            
            # Update progress
            frame_num += 1
            pbar.update(1)
            
            # Cleanup memory
            if frame_num % 10 == 0:
                gc.collect()
        
        pbar.close()
        cap.release()
        out.release()
        
        # Force garbage collection
        gc.collect()
        if self.verbose:
            print(f"Video saved to {output_path}")