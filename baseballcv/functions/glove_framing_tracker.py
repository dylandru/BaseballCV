import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch
import io
from contextlib import redirect_stdout

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
        csv_path: str = None
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
        
        # Dictionary for detailed data collection
        detailed_results = []
        
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
            
            # Process video to track glove movement
            glove_positions, glove_boxes, ball_glove_frame, homeplate_box, fps = self._track_glove_movement(
                video_path=video_path,
                output_path=output_path if create_video else None
            )
            
            # Calculate glove movement metrics
            smoothness, speed, distance_traveled = self._calculate_metrics(glove_positions, fps)
            
            # Compile basic results
            results = {
                "video_name": video_name,
                "play_id": play_id,
                "game_pk": game_pk,
                "ball_glove_frame": ball_glove_frame,
                "glove_positions": glove_positions,
                "glove_movement_smoothness": smoothness,
                "glove_movement_speed": speed,
                "glove_distance_traveled": distance_traveled,
                "fps": fps,
                "annotated_video": output_path if create_video else None
            }
            
            glove_tracking_results.append(results)
            
            # Prepare detailed data for CSV
            detailed_data = {
                # Basic identification
                "video_name": video_name,
                "play_id": play_id,
                "game_pk": game_pk,
                
                # Glove movement data
                "glove_movement_smoothness": smoothness,
                "glove_movement_speed": speed,
                "glove_distance_traveled": distance_traveled,
                "fps": fps,
                
                # Home plate data
                "homeplate_detected": homeplate_box is not None,
                "homeplate_x1": homeplate_box[0] if homeplate_box else None,
                "homeplate_y1": homeplate_box[1] if homeplate_box else None,
                "homeplate_x2": homeplate_box[2] if homeplate_box else None,
                "homeplate_y2": homeplate_box[3] if homeplate_box else None,
                
                # Ball-glove contact frame
                "ball_glove_frame": ball_glove_frame,
                
                # Video output
                "video_output_path": output_path if create_video else None,
            }
            
            # Add glove position data
            if glove_positions:
                for i, pos in enumerate(glove_positions):
                    if i < 100:  # Limit to 100 frames to keep CSV manageable
                        detailed_data[f"glove_x_frame_{i}"] = pos[0]
                        detailed_data[f"glove_y_frame_{i}"] = pos[1]
            
            # Add any Statcast data that might be available
            if pitch_data_row is not None:
                for key, value in pitch_data_row.items():
                    detailed_data[f"statcast_{key}"] = value
            
            detailed_results.append(detailed_data)
        
        # Save detailed data to CSV if requested
        if save_csv and detailed_results:
            if csv_path is None:
                csv_path = os.path.join(self.results_dir, "glove_tracking_results.csv")
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            
            # Create DataFrame from detailed results
            df = pd.DataFrame(detailed_results)
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            if self.verbose:
                print(f"Saved detailed results to {csv_path}")
                print(f"CSV contains {len(df)} rows with {len(df.columns)} columns of data")
        
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
        output_path: str = None
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int, int]], Optional[int], Optional[Tuple[int, int, int, int]], float]:
        """
        Track glove movement in a video and optionally create visualization.
        
        Args:
            video_path (str): Path to the input video
            output_path (str): Path to save the output video (None to skip)
            
        Returns:
            Tuple containing:
            - List of glove positions (x, y)
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
            # Use default source points based on broadcast view
            src_points = np.array([
                [frame_width * 0.3, frame_height * 0.8],
                [frame_width * 0.7, frame_height * 0.8],
                [frame_width * 0.7, frame_height * 0.2],
                [frame_width * 0.3, frame_height * 0.2]
            ], dtype=np.float32)

        # Define destination points for a standard plane
        dst_height = frame_height
        dst_width = int(dst_height * 0.4)
        dst_points = np.array([
            [0, dst_height - 1],
            [dst_width - 1, dst_height - 1],
            [dst_width - 1, 0],
            [0, 0]
        ], dtype=np.float32)

        # Compute transformation matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        # Find when ball reaches glove
        ball_glove_frame, ball_center = self._find_ball_reaches_glove(
            ball_by_frame, glove_by_frame
        )

        # Transform glove positions
        glove_positions = []
        glove_boxes = []
        sorted_frames = sorted(glove_by_frame.keys())
        
        for frame in sorted_frames:
            if frame not in glove_by_frame:
                continue
                
            for glove_det in glove_by_frame[frame]:
                x1, y1, x2, y2 = glove_det["x1"], glove_det["y1"], glove_det["x2"], glove_det["y2"]
                glove_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                # Transform glove center to standardized coordinates
                try:
                    flat_glove_point = cv2.perspectiveTransform(np.array([[glove_center]]), M)[0][0]
                    glove_positions.append(flat_glove_point)
                    glove_boxes.append((x1, y1, x2, y2))
                except:
                    # Skip if transformation fails
                    continue
                
                break  # Take only the first glove per frame
        
        # Create visualization video if requested
        if output_path:
            self._create_visualization_video(
                video_path, 
                output_path,
                glove_by_frame,
                ball_by_frame,
                glove_positions,
                M,
                dst_width,
                dst_height,
                ball_glove_frame,
                homeplate_box
            )
        
        return glove_positions, glove_boxes, ball_glove_frame, homeplate_box, fps
    
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
        glove_positions: List[Tuple[float, float]],
        transformation_matrix: np.ndarray,
        dst_width: int,
        dst_height: int,
        ball_glove_frame: Optional[int],
        homeplate_box: Optional[Tuple[int, int, int, int]]
    ) -> None:
        """
        Create a visualization video showing glove tracking and transformed view.
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to output video
            glove_by_frame (Dict): Glove detections by frame
            ball_by_frame (Dict): Ball detections by frame
            glove_positions (List): Transformed glove positions
            transformation_matrix (np.ndarray): Perspective transformation matrix
            dst_width (int): Width of transformed view
            dst_height (int): Height of transformed view
            ball_glove_frame (int): Frame where ball reaches glove
            homeplate_box (Tuple): Homeplate box coordinates
        """
        cap = cv2.VideoCapture(input_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        combined_width = frame_width + dst_width + 20  # Add 20px for gap between views
        out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, frame_height))
        
        # Get glove image path (try different asset locations)
        asset_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "baseball_glove.png"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "baseball_glove.png")
        ]
        
        glove_img = None
        for path in asset_paths:
            if os.path.exists(path):
                glove_img = plt.imread(path)
                break
        
        if glove_img is None:
            # Create a simple glove placeholder
            glove_img = np.ones((50, 50, 4), dtype=np.uint8) * 255
            glove_img[:, :, 3] = 255  # Alpha channel
        
        processed_positions = []
        frame_num = 0
        
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
                    
                    if len(processed_positions) < len(glove_positions):
                        processed_positions.append(glove_positions[len(processed_positions)])
            
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
            
            # Create transformed view on the right
            # Set up plot
            fig = plt.figure(figsize=(dst_width/100, frame_height/100), dpi=100)
            ax = fig.add_subplot(111)
            
            # Setup the transformed plane view
            ax.set_facecolor('blue')
            ax.set_xlim(0, dst_width)
            ax.set_ylim(dst_height, 0)  # Invert y-axis
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Glove Movement throughout Play", fontsize=10, color='white')
            
            # Add rectangle for the background
            ax.add_patch(plt.Rectangle((0, 0), dst_width, dst_height, fill=True, color='blue'))
            
            # Plot glove path
            if processed_positions:
                x = [pos[0] for pos in processed_positions]
                y = [pos[1] for pos in processed_positions]
                ax.plot(x, y, 'r-', linewidth=2)
                
                # Add glove image at current position
                if processed_positions:
                    im = OffsetImage(glove_img, zoom=0.04)
                    ab = AnnotationBbox(im, (x[-1], y[-1]), xycoords='data', frameon=False)
                    ax.add_artist(ab)
            
            # Convert plot to image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            plot_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            
            # Resize to match dst dimensions
            plot_image = cv2.resize(plot_image, (dst_width, frame_height))
            
            # Copy to combined frame
            combined_frame[0:frame_height, frame_width+20:] = plot_image
            
            # Write to output
            out.write(combined_frame)
            
            frame_num += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()