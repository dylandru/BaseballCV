import logging
import cv2
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import pandas as pd
from ultralytics import YOLO
import io
from contextlib import redirect_stdout
from baseballcv.functions.load_tools import LoadTools
import math
import mediapipe as mp
from baseballcv.utilities import BaseballCVLogger

class DistanceToZone:
    """
    Class for calculating and visualizing the distance of a pitch to the strike zone.
    
    This class uses computer vision models to detect the catcher, glove, ball, and 
    strike zone in baseball videos, then calculates the distance of the pitch relative to the zone.
    """
    
    def __init__(
        self, 
        catcher_model: str = 'phc_detector',
        glove_model: str = 'glove_tracking',
        ball_model: str = 'ball_trackingv4',
        homeplate_model: str = 'glove_tracking',
        results_dir: str = "results",
        verbose: bool = True,
        device: str = None,
        zone_vertical_adjustment: float = 0.5,
        logger: logging.Logger = None
    ):
        """
        Initialize the DistanceToZone class.
        
        Args:
            catcher_model (str): Model name for detecting catchers
            glove_model (str): Model name for detecting gloves
            ball_model (str): Model name for detecting baseballs
            homeplate_model (str): Model name for detecting home plate
            results_dir (str): Directory to save results
            verbose (bool): Whether to print detailed progress information
            device (str): Device to run models on (cpu, cuda, etc.)
            zone_vertical_adjustment (float): Factor to adjust strike zone vertically as percentage 
                                             of elbow-to-hip distance. Positive values move zone 
                                             toward home plate, negative away from home plate.
            logger (logging.Logger): Logger instance for logging
        """
        self.load_tools = LoadTools()
        self.catcher_model = YOLO(self.load_tools.load_model(catcher_model))
        self.glove_model = YOLO(self.load_tools.load_model(glove_model))
        self.ball_model = YOLO(self.load_tools.load_model(ball_model))
        self.homeplate_model = YOLO(self.load_tools.load_model(homeplate_model))
        
        # Initialize MediaPipe pose for 3D pose detection
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.logger = logger if logger is not None else BaseballCVLogger.get_logger(__name__)
        
        if verbose:
            self.logger.info(f"Models loaded: {catcher_model}, {glove_model}, {ball_model}, {homeplate_model}")
        
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.verbose = verbose
        self.device = device
        self.zone_vertical_adjustment = zone_vertical_adjustment  # Store the adjustment factor
    
    def analyze(
        self, 
        start_date: str, 
        end_date: str,
        team_abbr: str = None,
        player: int = None,
        pitch_type: str = None,
        max_videos: int = None,
        max_videos_per_game: int = None,
        create_video: bool = True,
        save_csv: bool = True,  # New parameter for CSV output
        csv_path: str = None    # Path for the CSV file
    ) -> List[Dict]:
        """
        Analyze videos from a date range to calculate distances to strike zone.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            team_abbr (str): Team abbreviation to filter by
            player (int): Player ID to filter by
            pitch_type (str): Pitch type to filter by (e.g., "FF")
            max_videos (int): Maximum number of videos to process
            max_videos_per_game (int): Maximum videos per game
            create_video (bool): Whether to create annotated videos
            save_csv (bool): Whether to save analysis results to CSV
            csv_path (str): Custom path for CSV file (default: results/distance_to_zone_results.csv)
            
        Returns:
            List[Dict]: List of analysis results per video
        """
        from baseballcv.functions.savant_scraper import BaseballSavVideoScraper
        
        savant_scraper = BaseballSavVideoScraper(start_date, end_date,
                                                 player=player,
                                                 team_abbr=team_abbr, pitch_type=pitch_type,
                                                 max_return_videos=max_videos, 
                                                 max_videos_per_game=max_videos_per_game)
        
        download_folder = os.path.join(self.results_dir, "savant_videos")

        savant_scraper.run_executor()

        pitch_data = savant_scraper.get_play_ids_df()

        video_files = [os.path.join(download_folder, f) for f in os.listdir(download_folder) if f.endswith('.mp4')]
        
        dtoz_results = []
        
        # Create a detailed data collection list for CSV
        detailed_results = []
        
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            play_id = video_name.split('_')[-1]
            game_pk = video_name.split('_')[-2]
            
            # Find the corresponding row in pitch_data
            pitch_data_row = None
            for _, row in pitch_data.iterrows():
                if row["play_id"] == play_id:
                    pitch_data_row = row
                    break
            
            if pitch_data_row is None:
                if self.verbose:
                    self.logger.info(f"No pitch data found for play_id {play_id}, skipping...")
                continue
                
            output_path = os.path.join(self.results_dir, f"distance_to_zone_{play_id}.mp4")
            
            catcher_detections = self._detect_objects(video_path, self.catcher_model, "catcher")
            glove_detections = self._detect_objects(video_path, self.glove_model, "glove")
            ball_detections = self._detect_objects(video_path, self.ball_model, "baseball")
            
            # Also get pitcher and hitter detections for better filtering
            pitcher_detections = self._detect_objects(video_path, self.catcher_model, "pitcher")
            hitter_detections = self._detect_objects(video_path, self.catcher_model, "hitter", conf_threshold=0.3)  # Lower threshold for hitter
            
            ball_glove_frame, ball_center, ball_detection = self._find_ball_reaches_glove(video_path, glove_detections, ball_detections)
            
            # Get catcher position to help distinguish hitter from umpire
            catcher_position = self._get_catcher_position(catcher_detections, ball_glove_frame)
            
            # First detect the reliable bounding box for the hitter with improved detection
            hitter_box, hitter_frame, hitter_frame_idx = self._find_best_hitter_box(
                video_path=video_path,
                hitter_detections=hitter_detections,
                catcher_position=catcher_position,
                frame_idx_start=max(0, ball_glove_frame - 120) if ball_glove_frame else 0,  # Search more frames
                frame_search_range=120  # Expanded search range
            )
            
            # Detect both standard YOLO pose and MediaPipe 3D pose
            hitter_keypoints = None
            hitter_pose_3d = None
            if hitter_box is not None and hitter_frame is not None:
                hitter_keypoints = self._detect_pose_in_box(
                    frame=hitter_frame,
                    box=hitter_box
                )
                
                # Detect 3D pose with MediaPipe
                hitter_pose_3d = self._detect_3d_pose(
                    frame=hitter_frame,
                    box=hitter_box
                )
            
            # Detect home plate for strike zone positioning
            homeplate_box, homeplate_confidence, homeplate_frame = self._detect_homeplate(
                video_path, 
                reference_frame=ball_glove_frame
            )
            
            # Enhanced strike zone computation with consensus-based approach
            strike_zone, pixels_per_foot = self._compute_strike_zone(
                catcher_detections, 
                pitch_data_row, 
                ball_glove_frame, 
                video_path, 
                hitter_keypoints=hitter_keypoints,
                hitter_box=hitter_box,
                homeplate_box=homeplate_box,
                hitter_pose_3d=hitter_pose_3d  # Added 3D pose for adjustment
            )
            
            distance = None
            position = None
            closest_point = None  # Store closest point for visualization
            
            if ball_center is not None and strike_zone is not None and pixels_per_foot is not None:
                distance_pixels, distance_inches, position, closest_point = self._calculate_distance_to_zone(
                    ball_center, strike_zone, pixels_per_foot)
                distance = distance_inches

                if self.verbose:
                    self.logger.info(f"Distance to zone: {distance:.2f} inches")
                    self.logger.info(f"Position relative to zone: {position}")
            
            if create_video and output_path and strike_zone is not None:
                self._create_annotated_video(
                    video_path, 
                    output_path,
                    catcher_detections, 
                    glove_detections,
                    ball_detections,
                    strike_zone,
                    ball_glove_frame,
                    distance,
                    position,
                    hitter_keypoints=hitter_keypoints,
                    hitter_frame_idx=hitter_frame_idx,
                    hitter_box=hitter_box,
                    homeplate_box=homeplate_box,
                    hitter_pose_3d=hitter_pose_3d,  # Added 3D pose for visualization
                    closest_point=closest_point     # Added closest point for measurement visualization
                )
            
            # Collect all data for results
            results = {
                "video_name": video_name,
                "play_id": play_id,
                "game_pk": game_pk,
                "ball_glove_frame": ball_glove_frame,
                "ball_center": ball_center,
                "strike_zone": strike_zone,
                "distance_to_zone": distance,
                "position": position,
                "annotated_video": output_path if create_video else None,
                "in_zone": position == "In Zone" if position else None
            }
            
            # Add detailed data collection for CSV
            detailed_data = {
                # Basic identification
                "video_name": video_name,
                "play_id": play_id,
                "game_pk": game_pk,
                
                # Ball and glove information
                "ball_glove_frame": ball_glove_frame,
                "ball_center_x": ball_center[0] if ball_center else None,
                "ball_center_y": ball_center[1] if ball_center else None,
                
                # Strike zone information
                "zone_left": strike_zone[0] if strike_zone else None,
                "zone_top": strike_zone[1] if strike_zone else None,
                "zone_right": strike_zone[2] if strike_zone else None,
                "zone_bottom": strike_zone[3] if strike_zone else None,
                "zone_width_px": (strike_zone[2] - strike_zone[0]) if strike_zone else None,
                "zone_height_px": (strike_zone[3] - strike_zone[1]) if strike_zone else None,
                
                # Distance measurements
                "distance_to_zone_inches": distance,
                "position_description": position,
                "in_zone": position == "In Zone" if position else None,
                "pixels_per_foot": pixels_per_foot,
                
                # Closest point coordinates
                "closest_point_x": closest_point[0] if closest_point else None,
                "closest_point_y": closest_point[1] if closest_point else None,
                
                # Home plate information
                "homeplate_detected": homeplate_box is not None,
                "homeplate_confidence": homeplate_confidence if homeplate_box else None,
                "homeplate_x1": homeplate_box[0] if homeplate_box else None,
                "homeplate_y1": homeplate_box[1] if homeplate_box else None,
                "homeplate_x2": homeplate_box[2] if homeplate_box else None,
                "homeplate_y2": homeplate_box[3] if homeplate_box else None,
                
                # Hitter information
                "hitter_detected": hitter_box is not None,
                "hitter_frame": hitter_frame_idx,
                "hitter_box_x1": hitter_box[0] if hitter_box else None,
                "hitter_box_y1": hitter_box[1] if hitter_box else None,
                "hitter_box_x2": hitter_box[2] if hitter_box else None,
                "hitter_box_y2": hitter_box[3] if hitter_box else None,
                "pose_detected": hitter_keypoints is not None,
                "pose3d_detected": hitter_pose_3d is not None,
                
                # Detection counts
                "num_ball_detections": len(ball_detections),
                "num_glove_detections": len(glove_detections),
                "num_catcher_detections": len(catcher_detections),
                
                # Video output
                "video_output_path": output_path if create_video else None,
                
                # Vertical adjustment factor used
                "zone_vertical_adjustment": self.zone_vertical_adjustment
            }
            
            # Add any Statcast data that might be available
            if pitch_data_row is not None:
                for key, value in pitch_data_row.items():
                    detailed_data[f"statcast_{key}"] = value
            
            detailed_results.append(detailed_data)
            dtoz_results.append(results)
        
        # Save detailed data to CSV if requested
        if save_csv and detailed_results:
            if csv_path is None:
                csv_path = os.path.join(self.results_dir, "distance_to_zone_results.csv")
            
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
            
            # Create DataFrame from detailed results
            df = pd.DataFrame(detailed_results)
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            
            if self.verbose:
                self.logger.info(f"Saved detailed results to {csv_path}")
                self.logger.info(f"CSV contains {len(df)} rows with {len(df.columns)} columns of data")
        
        return dtoz_results
    
    def _get_catcher_position(self, catcher_detections: List[Dict], reference_frame: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the catcher position near the reference frame to help identify where the umpire is
        
        Args:
            catcher_detections (List[Dict]): List of catcher detections
            reference_frame (int): Reference frame (usually ball-glove contact)
            
        Returns:
            Optional[Tuple[int, int, int, int]]: Catcher box coordinates (x1, y1, x2, y2)
        """
        if not catcher_detections or reference_frame is None:
            return None
            
        # Group detections by frame
        catcher_by_frame = {}
        for det in catcher_detections:
            frame = det["frame"]
            if frame not in catcher_by_frame:
                catcher_by_frame[frame] = []
            catcher_by_frame[frame].append(det)
            
        # Find the closest frame to reference frame with catcher detections
        closest_frame = None
        min_distance = float('inf')
        
        for frame in catcher_by_frame.keys():
            dist = abs(frame - reference_frame)
            if dist < min_distance:
                min_distance = dist
                closest_frame = frame
                
        if closest_frame is None:
            return None
            
        # Use the largest catcher detection (most likely the actual catcher)
        largest_detection = None
        max_area = 0
        
        for det in catcher_by_frame[closest_frame]:
            area = (det["x2"] - det["x1"]) * (det["y2"] - det["y1"])
            if area > max_area:
                max_area = area
                largest_detection = det
                
        if largest_detection is None:
            return None
            
        return (largest_detection["x1"], largest_detection["y1"], 
                largest_detection["x2"], largest_detection["y2"])
    
    def _detect_objects(self, video_path: str, model: YOLO, object_name: str, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in every frame of the video.
        
        Args:
            video_path (str): Path to the video file
            model (YOLO): YOLO model to use for detection
            object_name (str): Name of the object to detect
            conf_threshold (float): Confidence threshold for detection
            
        Returns:
            List[Dict]: List of detection dictionaries
        """
        if self.verbose:
            self.logger.info(f"Detecting {object_name} in video: {os.path.basename(video_path)}")
        
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
            self.logger.info(f"Completed {object_name} detection. Found {len(detections)} detections")
        
        return detections
    
    def _find_ball_reaches_glove(self, video_path: str, glove_detections: List[Dict], ball_detections: List[Dict], tolerance: float = 0.1) -> Tuple[Optional[int], Optional[Tuple[float, float]], Optional[Dict]]:
        """
        Find the frame where ball reaches the glove with more robust detection validation.

        Args:
            video_path (str): Path to the video file
            glove_detections (List[Dict]): List of glove detection dictionaries
            ball_detections (List[Dict]): List of ball detection dictionaries
            tolerance (float): Margin around glove box for ball detection

        Returns:
            Tuple[Optional[int], Optional[Tuple[float, float]], Optional[Dict]]:
                (frame index, ball center coordinates, ball detection dictionary)
        """
        if self.verbose:
            self.logger.info("Detecting when ball reaches glove...")

        # Group detections by frame for easier processing
        glove_by_frame = {}
        for det in glove_detections:
            glove_by_frame.setdefault(det["frame"], []).append(det)

        ball_by_frame = {}
        for det in ball_detections:
            ball_by_frame.setdefault(det["frame"], []).append(det)

        # Identify continuous ball detection sequences
        ball_frames = sorted(ball_by_frame.keys())

        # Function to find continuous sequences
        def find_continuous_sequences(frames):
            sequences = []
            current_sequence = []

            for i in range(len(frames)):
                if not current_sequence or frames[i] == current_sequence[-1] + 1:
                    current_sequence.append(frames[i])
                else:
                    if len(current_sequence) > 5:  # Require at least 5 consecutive frames
                        sequences.append(current_sequence)
                    current_sequence = [frames[i]]

            # Check the last sequence
            if len(current_sequence) > 5:
                sequences.append(current_sequence)

            return sequences

        # Find continuous sequences of ball detections
        ball_detection_sequences = find_continuous_sequences(ball_frames)

        if self.verbose:
            self.logger.info(f"Found {len(ball_detection_sequences)} continuous ball detection sequences")

        # Search for ball-glove contact in the most significant detection sequences
        # Sort sequences by length, prioritizing longer sequences
        ball_detection_sequences.sort(key=len, reverse=True)

        for sequence in ball_detection_sequences:
            # Search through the sequence frames
            for frame in sequence:
                if frame not in glove_by_frame:
                    continue

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
                            if self.verbose:
                                self.logger.info(f"Ball reached glove at frame {frame}")
                            return frame, (ball_center_x, ball_center_y), ball_det

        # FALLBACK 1: Try with larger tolerance
        if self.verbose:
            self.logger.info("Standard detection failed, trying with larger tolerance...")
            
        larger_tolerance = 0.3  # 30% margin
        for sequence in ball_detection_sequences:
            for frame in sequence:
                if frame not in glove_by_frame:
                    continue

                for glove_det in glove_by_frame[frame]:
                    margin_x = larger_tolerance * (glove_det["x2"] - glove_det["x1"])
                    margin_y = larger_tolerance * (glove_det["y2"] - glove_det["y1"])
                    extended_x1 = glove_det["x1"] - margin_x
                    extended_y1 = glove_det["y1"] - margin_y
                    extended_x2 = glove_det["x2"] + margin_x
                    extended_y2 = glove_det["y2"] + margin_y

                    for ball_det in ball_by_frame[frame]:
                        ball_center_x = (ball_det["x1"] + ball_det["x2"]) / 2
                        ball_center_y = (ball_det["y1"] + ball_det["y2"]) / 2

                        if (extended_x1 <= ball_center_x <= extended_x2 and
                            extended_y1 <= ball_center_y <= extended_y2):
                            if self.verbose:
                                self.logger.info(f"Ball reached glove at frame {frame} (with larger tolerance)")
                            return frame, (ball_center_x, ball_center_y), ball_det

        # FALLBACK 2: Look for closest ball to any glove
        if self.verbose:
            self.logger.info("Expanded tolerance failed, trying closest approach method...")
            
        best_distance = float('inf')
        best_frame = None
        best_ball_center = None
        best_ball_det = None
        
        # Check each ball against each glove in the same frame
        for frame in ball_frames:
            if frame not in glove_by_frame:
                continue
                
            for glove_det in glove_by_frame[frame]:
                glove_center_x = (glove_det["x1"] + glove_det["x2"]) / 2
                glove_center_y = (glove_det["y1"] + glove_det["y2"]) / 2
                
                for ball_det in ball_by_frame[frame]:
                    ball_center_x = (ball_det["x1"] + ball_det["x2"]) / 2
                    ball_center_y = (ball_det["y1"] + ball_det["y2"]) / 2
                    
                    # Calculate Euclidean distance
                    distance = math.sqrt(
                        (ball_center_x - glove_center_x)**2 + 
                        (ball_center_y - glove_center_y)**2
                    )
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_frame = frame
                        best_ball_center = (ball_center_x, ball_center_y)
                        best_ball_det = ball_det
        
        if best_frame is not None:
            if self.verbose:
                self.logger.info(f"Found closest ball approach at frame {best_frame} (distance: {best_distance:.2f} pixels)")
            return best_frame, best_ball_center, best_ball_det
            
        # FALLBACK 3: Just use the middle frame with a ball detection
        if ball_frames:
            middle_idx = len(ball_frames) // 2
            middle_frame = ball_frames[middle_idx]
            middle_ball_det = ball_by_frame[middle_frame][0]
            middle_ball_center_x = (middle_ball_det["x1"] + middle_ball_det["x2"]) / 2
            middle_ball_center_y = (middle_ball_det["y1"] + middle_ball_det["y2"]) / 2
            
            if self.verbose:
                self.logger.info(f"Using middle ball frame {middle_frame} as fallback")
            return middle_frame, (middle_ball_center_x, middle_ball_center_y), middle_ball_det

        if self.verbose:
            self.logger.info("Could not detect when ball reaches glove")
        return None, None, None

    def _find_best_hitter_box(self, video_path: str, hitter_detections: List[Dict],
                            catcher_position: Optional[Tuple[int, int, int, int]] = None,
                            frame_idx_start: int = 0, frame_search_range: int = 90) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Optional[int]]:
        """
        Find the best bounding box for the hitter using robust filtering to avoid detecting the umpire.
        
        Args:
            video_path (str): Path to the video file
            hitter_detections (List[Dict]): Pre-computed hitter detections
            catcher_position (Tuple[int, int, int, int]): Position of the catcher
            frame_idx_start (int): Starting frame index
            frame_search_range (int): Number of frames to search
            
        Returns:
            Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Optional[int]]:
                (hitter box, frame containing the hitter, frame index)
        """
        if self.verbose:
            self.logger.info("Finding best hitter bounding box...")
        
        # Capture video
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define search range
        search_start = max(0, frame_idx_start)
        search_end = min(total_frames, search_start + frame_search_range)
        
        # Variables to store best detection
        best_hitter_box = None
        best_hitter_frame = None
        best_frame_idx = None
        best_confidence = 0
        
        # First, try to use pre-computed hitter detections if available
        if hitter_detections:
            hitter_by_frame = {}
            for det in hitter_detections:
                frame = det["frame"]
                if search_start <= frame <= search_end:
                    if frame not in hitter_by_frame:
                        hitter_by_frame[frame] = []
                    hitter_by_frame[frame].append(det)
            
            valid_frames = sorted(hitter_by_frame.keys())
            
            for frame_idx in valid_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                for det in hitter_by_frame[frame_idx]:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    conf = det["confidence"]
                    
                    # Apply heuristics to identify the actual hitter (not umpire):
                    # 1. Must be reasonably sized (minimum dimensions)
                    # 2. Typically on opposite side from catcher
                    is_valid_hitter = True
                    
                    # Size check - must be substantial
                    area = (x2 - x1) * (y2 - y1)
                    min_area = width * height * 0.015  # Reduced from 0.02 to 0.015
                    min_width = width * 0.04  # Reduced from 0.05 to 0.04
                    min_height = height * 0.08  # Reduced from 0.1 to 0.08
                    
                    if area < min_area or (x2 - x1) < min_width or (y2 - y1) < min_height:
                        is_valid_hitter = False
                    
                    # Position check relative to catcher - less strict now
                    if catcher_position:
                        catcher_center_x = (catcher_position[0] + catcher_position[2]) / 2
                        hitter_center_x = (x1 + x2) / 2
                        
                        # If catcher is on extreme right, hitter should be on left (and vice versa)
                        if ((catcher_center_x > width*0.75 and hitter_center_x > width*0.75) or
                            (catcher_center_x < width*0.25 and hitter_center_x < width*0.25)):
                            is_valid_hitter = False
                    
                    # If detection is valid and better than current best
                    if is_valid_hitter and conf > best_confidence:
                        best_hitter_box = (x1, y1, x2, y2)
                        best_hitter_frame = frame.copy()
                        best_frame_idx = frame_idx
                        best_confidence = conf
        
        # If we couldn't find a good hitter from existing detections, search with PHC model
        if best_hitter_box is None:
            if self.verbose:
                print("Looking for hitter using PHC model with reduced constraints...")
            
            for frame_idx in range(search_start, search_end, 5):  # Step by 5 frames for efficiency
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Detect hitter with PHC model
                with io.StringIO() as buf, redirect_stdout(buf):
                    phc_results = self.catcher_model.predict(frame, conf=0.3, verbose=False)  # Lower threshold
                
                for result in phc_results:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        
                        # Look specifically for "hitter" class
                        if self.catcher_model.names[cls].lower() == "hitter" and conf > best_confidence:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            # Apply less strict heuristics
                            is_valid_hitter = True
                            
                            # Size check with reduced thresholds
                            area = (x2 - x1) * (y2 - y1)
                            min_area = width * height * 0.015  # Reduced from 0.02 to 0.015
                            min_width = width * 0.04  # Reduced from 0.05 to 0.04
                            min_height = height * 0.08  # Reduced from 0.1 to 0.08
                            
                            if area < min_area or (x2 - x1) < min_width or (y2 - y1) < min_height:
                                is_valid_hitter = False
                            
                            # Position check relative to catcher - less strict
                            if catcher_position:
                                catcher_center_x = (catcher_position[0] + catcher_position[2]) / 2
                                hitter_center_x = (x1 + x2) / 2
                                
                                # Only exclude if hitter and catcher are in same extreme corner
                                if ((catcher_center_x > width*0.75 and hitter_center_x > width*0.75) or
                                    (catcher_center_x < width*0.25 and hitter_center_x < width*0.25)):
                                    is_valid_hitter = False
                            
                            if is_valid_hitter:
                                best_hitter_box = (x1, y1, x2, y2)
                                best_hitter_frame = frame.copy()
                                best_frame_idx = frame_idx
                                best_confidence = conf
        
        # FALLBACK: Use any detection that seems reasonable if we still don't have one
        if best_hitter_box is None:
            if self.verbose:
                print("Trying fallback method for hitter detection...")
            
            # Reset the video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Try to find any large bounding box that could be a player
            for frame_idx in range(0, min(total_frames, 200), 10):  # Check first 200 frames, step of 10
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Convert to grayscale for motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold to find potential player blobs
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it's large enough to be a player
                    if w * h > width * height * 0.01 and w > width * 0.03 and h > height * 0.08:
                        # Create a box with some margin
                        x1 = max(0, x - int(w * 0.1))
                        y1 = max(0, y - int(h * 0.1))
                        x2 = min(width, x + w + int(w * 0.1))
                        y2 = min(height, y + h + int(h * 0.1))
                        
                        # Check if it's in a reasonable position (not at the bottom of frame where umpire is)
                        if y < height * 0.6:  # Not too low in the frame
                            best_hitter_box = (x1, y1, x2, y2)
                            best_hitter_frame = frame.copy()
                            best_frame_idx = frame_idx
                            # Use a minimum confidence
                            best_confidence = 0.5
                            break
                
                if best_hitter_box is not None:
                    break
        
        cap.release()
        
        if best_hitter_box is not None:
            if self.verbose:
                print(f"Found valid hitter box at frame {best_frame_idx} with confidence {best_confidence:.2f}")
        else:
            if self.verbose:
                print("Could not find valid hitter box")
        
        return best_hitter_box, best_hitter_frame, best_frame_idx

    def _detect_pose_in_box(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Detect pose strictly within the given bounding box to avoid detecting umpire's pose.
        
        Args:
            frame (np.ndarray): Frame containing the hitter
            box (Tuple[int, int, int, int]): Bounding box for the hitter (x1, y1, x2, y2)
            
        Returns:
            Optional[np.ndarray]: Keypoints for the hitter's pose, or None if detection fails
        """
        if self.verbose:
            print("Detecting hitter's pose within bounding box...")
        
        # Load YOLO pose model
        try:
            pose_model = YOLO("yolov8n-pose.pt")
        except:
            # Try to download it
            os.system("wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt")
            pose_model = YOLO("yolov8n-pose.pt")
        
        # Extract only the region of interest to force pose detection in that area
        x1, y1, x2, y2 = box
        
        # Add a small margin (15%) - increased from 10%
        margin_x = int((x2 - x1) * 0.15)
        margin_y = int((y2 - y1) * 0.15)
        
        # Ensure the box stays within frame boundaries
        x1_margin = max(0, x1 - margin_x)
        y1_margin = max(0, y1 - margin_y)
        x2_margin = min(frame.shape[1], x2 + margin_x)
        y2_margin = min(frame.shape[0], y2 + margin_y)
        
        # Extract the hitter region with margin
        hitter_region = frame[y1_margin:y2_margin, x1_margin:x2_margin].copy()
        
        if hitter_region.size == 0:
            if self.verbose:
                print("Invalid hitter region (empty)")
            return None
        
        # Run pose detection on hitter region only
        with io.StringIO() as buf, redirect_stdout(buf):
            hitter_results = pose_model.predict(hitter_region, verbose=False, conf=0.3)  # Lower confidence threshold
        
        # Check if we got any results
        if (len(hitter_results) == 0 or 
            len(hitter_results[0].keypoints) == 0 or 
            hitter_results[0].keypoints.data.shape[1] < 17):
            if self.verbose:
                print("No pose detected within hitter region")
            
            # Try on full image with the box as a guide
            with io.StringIO() as buf, redirect_stdout(buf):
                full_results = pose_model.predict(frame, verbose=False, conf=0.3)
            
            if (len(full_results) == 0 or 
                len(full_results[0].keypoints) == 0 or 
                full_results[0].keypoints.data.shape[1] < 17):
                if self.verbose:
                    print("No pose detected in full image either")
                return None
            
            # Check if any pose has a significant overlap with our box
            best_overlap = 0
            best_keypoints = None
            
            for person_idx, keypoints in enumerate(full_results[0].keypoints.data):
                # Count how many keypoints are inside the box
                points_in_box = 0
                valid_points = 0
                
                for i in range(keypoints.shape[0]):
                    if keypoints[i, 2] > 0.3:  # If point is detected with reasonable confidence
                        valid_points += 1
                        if (x1 <= keypoints[i, 0] <= x2 and y1 <= keypoints[i, 1] <= y2):
                            points_in_box += 1
                
                # Calculate overlap ratio
                if valid_points > 0:
                    overlap = points_in_box / valid_points
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_keypoints = keypoints
            
            if best_overlap > 0.3:  # If at least 30% of points are in box
                return best_keypoints
            
            return None
        
        # Get keypoints from region
        keypoints = hitter_results[0].keypoints.data[0].clone()  # Clone to avoid modifying original
        
        # Check if we detected essential keypoints (knees and hips)
        has_essential_keypoints = False
        for knee_idx in [13, 14]:  # Left and right knee
            for hip_idx in [11, 12]:  # Left and right hip
                if keypoints[knee_idx, 2] > 0.3 and keypoints[hip_idx, 2] > 0.3:
                    has_essential_keypoints = True
                    break
        
        if not has_essential_keypoints:
            if self.verbose:
                print("Essential keypoints (knees, hips) not detected with confidence")
            
            # If essential keypoints missing, we'll try with the full image
            with io.StringIO() as buf, redirect_stdout(buf):
                full_results = pose_model.predict(frame, verbose=False, conf=0.3)
            
            if (len(full_results) > 0 and 
                len(full_results[0].keypoints) > 0 and 
                full_results[0].keypoints.data.shape[1] >= 17):
                
                # Find the pose with the most overlap with our box
                best_overlap = 0
                best_keypoints = None
                
                for person_idx, person_keypoints in enumerate(full_results[0].keypoints.data):
                    # Count how many keypoints are inside the box
                    points_in_box = 0
                    valid_points = 0
                    
                    for i in range(person_keypoints.shape[0]):
                        if person_keypoints[i, 2] > 0.3:  # If point is detected with reasonable confidence
                            valid_points += 1
                            if (x1 <= person_keypoints[i, 0] <= x2 and y1 <= person_keypoints[i, 1] <= y2):
                                points_in_box += 1
                    
                    # Calculate overlap ratio
                    if valid_points > 0:
                        overlap = points_in_box / valid_points
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_keypoints = person_keypoints
                
                if best_overlap > 0.3:  # If at least 30% of points are in box
                    if self.verbose:
                        print(f"Using pose from full image with {best_overlap:.2f} overlap with hitter box")
                    return best_keypoints
            
            return None
        
        # Validate pose is anatomically reasonable with relaxed constraints
        # Check knees are below hips
        left_knee_y = keypoints[13, 1].item() if keypoints[13, 2] > 0.3 else None
        right_knee_y = keypoints[14, 1].item() if keypoints[14, 2] > 0.3 else None
        left_hip_y = keypoints[11, 1].item() if keypoints[11, 2] > 0.3 else None
        right_hip_y = keypoints[12, 1].item() if keypoints[12, 2] > 0.3 else None
        
        valid_anatomy = True
        
        # Check left side with more tolerance
        if left_knee_y is not None and left_hip_y is not None:
            if left_knee_y < left_hip_y - 20:  # Knee should be below hip with some tolerance
                valid_anatomy = False
        
        # Check right side with more tolerance
        if right_knee_y is not None and right_hip_y is not None:
            if right_knee_y < right_hip_y - 20:  # Knee should be below hip with some tolerance
                valid_anatomy = False
        
        if not valid_anatomy:
            if self.verbose:
                print("Detected pose is anatomically invalid - will try with full image")
            
            # Try with full image if regional pose is invalid
            with io.StringIO() as buf, redirect_stdout(buf):
                full_results = pose_model.predict(frame, verbose=False, conf=0.3)
            
            if (len(full_results) > 0 and 
                len(full_results[0].keypoints) > 0 and 
                full_results[0].keypoints.data.shape[1] >= 17):
                
                # Find the pose with the most overlap with our box
                best_overlap = 0
                best_keypoints = None
                
                for person_idx, person_keypoints in enumerate(full_results[0].keypoints.data):
                    # Count how many keypoints are inside the box
                    points_in_box = 0
                    valid_points = 0
                    
                    for i in range(person_keypoints.shape[0]):
                        if person_keypoints[i, 2] > 0.3:  # If point is detected with reasonable confidence
                            valid_points += 1
                            if (x1 <= person_keypoints[i, 0] <= x2 and y1 <= person_keypoints[i, 1] <= y2):
                                points_in_box += 1
                    
                    # Calculate overlap ratio
                    if valid_points > 0:
                        overlap = points_in_box / valid_points
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_keypoints = person_keypoints
                
                if best_overlap > 0.3:  # If at least 30% of points are in box
                    if self.verbose:
                        print(f"Using pose from full image with {best_overlap:.2f} overlap with hitter box")
                    return best_keypoints
            
            return None
        
        # Shift keypoints back to full frame coordinates
        for i in range(keypoints.shape[0]):
            keypoints[i, 0] += x1_margin  # Add x offset
            keypoints[i, 1] += y1_margin  # Add y offset
        
        if self.verbose:
            print("Successfully detected valid pose for hitter")
        
        return keypoints
    
    def _detect_3d_pose(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[Dict]:
        """
        Detect 3D pose using MediaPipe strictly within the hitter's bounding box.
        
        Args:
            frame (np.ndarray): Frame containing the hitter
            box (Tuple[int, int, int, int]): Bounding box for the hitter
            
        Returns:
            Optional[Dict]: MediaPipe pose results or None if detection fails
        """
        if self.verbose:
            print("Detecting 3D pose with MediaPipe strictly within hitter box...")
        
        # Extract ROI with margin
        x1, y1, x2, y2 = box
        
        # Add margin (15%)
        margin_x = int((x2 - x1) * 0.15)
        margin_y = int((y2 - y1) * 0.15)
        
        # Ensure box stays within frame boundaries
        x1_margin = max(0, x1 - margin_x)
        y1_margin = max(0, y1 - margin_y)
        x2_margin = min(frame.shape[1], x2 + margin_x)
        y2_margin = min(frame.shape[0], y2 + margin_y)
        
        # Extract region
        hitter_region = frame[y1_margin:y2_margin, x1_margin:x2_margin].copy()
        
        if hitter_region.size == 0:
            if self.verbose:
                print("Invalid hitter region for 3D pose (empty)")
            return None
        
        # Configure MediaPipe Pose with higher detection confidence
        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,  # Medium complexity for balance
            enable_segmentation=False,
            min_detection_confidence=0.4
        ) as pose:
            # Process only the region of interest
            results = pose.process(cv2.cvtColor(hitter_region, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            if self.verbose:
                print("MediaPipe couldn't detect 3D pose in hitter region")
            
            # Try with a lower detection threshold as fallback
            with self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.3
            ) as pose:
                # Still only process the hitter region
                results = pose.process(cv2.cvtColor(hitter_region, cv2.COLOR_BGR2RGB))
            
            if not results.pose_landmarks:
                if self.verbose:
                    print("MediaPipe couldn't detect 3D pose even with lower threshold")
                return None
            
            if self.verbose:
                print("Successfully detected 3D pose with lower threshold")
        else:
            if self.verbose:
                print("Successfully detected 3D pose for hitter with MediaPipe")
        
        # Store the ROI offset to adjust landmarks when drawing
        offset = (x1_margin, y1_margin)
        return {"results": results, "offset": offset}
        
    def _detect_homeplate(self, video_path: str, reference_frame: int = None) -> Tuple[Optional[Tuple[int, int, int, int]], float, int]:
        """
        Detect the home plate in the video using the dedicated homeplate model.
        Returns the home plate bounding box, confidence score, and the frame used.
        
        Args:
            video_path (str): Path to the video file
            reference_frame (int): Reference frame to start search from
            
        Returns:
            Tuple[Optional[Tuple[int, int, int, int]], float, int]: 
                (home plate box, confidence, frame used)
        """
        if self.verbose:
            print("Detecting home plate...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define search frames around ball-glove contact (or use reference frame)
        search_frame = reference_frame if reference_frame is not None else total_frames // 2
        
        search_frames = [
            search_frame,             # The exact frame
            search_frame + 2,         # The adjusted frame (accounting for delay)
            search_frame - 2,
            search_frame + 4,
            search_frame - 4,
            search_frame - 8,
            search_frame - 12,
            search_frame - 16,
            search_frame - 20,
            search_frame - 30,
            search_frame - 45,
            search_frame - 60,
        ]
        # Filter out negative frames
        search_frames = [max(0, frame) for frame in search_frames]

        if self.verbose:
            print(f"Searching for home plate around reference frame {search_frame}...")

        # Dictionary to store all frames with home plate detections (for visualization)
        detection_frames = {}
        all_homeplate_detections = []

        for frame_idx in search_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Use the dedicated homeplate model
            with io.StringIO() as buf, redirect_stdout(buf):
                results = self.homeplate_model.predict(frame, conf=0.2, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    cls_name = self.homeplate_model.names[cls].lower()
                    
                    # Check for home plate class
                    if "homeplate" == cls_name:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detection = {
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "model": "glove_tracking",
                            "frame": frame_idx
                        }
                        all_homeplate_detections.append(detection)
                    
                    # Additionally, try alternative class names that might be used
                    elif "home_plate" == cls_name or "plate" == cls_name or "home" == cls_name:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detection = {
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "model": "glove_tracking",
                            "frame": frame_idx
                        }
                        all_homeplate_detections.append(detection)

            # Store frame for visualization if detections were found
            if all_homeplate_detections:
                detection_frames[frame_idx] = {
                    "frame": frame.copy(),
                    "detections": all_homeplate_detections
                }

        # Try a broader search if we didn't find enough detections
        if len(all_homeplate_detections) < 3:
            if self.verbose:
                print("Few home plate detections found. Scanning more frames...")
            # Try additional frames
            step_size = 10
            total_frames = min(300, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

            for frame_idx in range(0, total_frames, step_size):
                if frame_idx in search_frames:
                    continue  # Skip already searched frames

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Use dedicated homeplate model
                with io.StringIO() as buf, redirect_stdout(buf):
                    results = self.homeplate_model.predict(frame, conf=0.15, verbose=False)
                
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        cls_name = self.homeplate_model.names[cls].lower()
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        if cls_name == "homeplate":
                            detection = {
                                "box": (x1, y1, x2, y2),
                                "conf": conf,
                                "model": "glove_tracking",
                                "frame": frame_idx
                            }
                            all_homeplate_detections.append(detection)
                        
                        # Try alternative class names
                        elif cls_name in ["home_plate", "plate", "home"]:
                            detection = {
                                "box": (x1, y1, x2, y2),
                                "conf": conf,
                                "model": "glove_tracking",
                                "frame": frame_idx
                            }
                            all_homeplate_detections.append(detection)

                if all_homeplate_detections:
                    detection_frames[frame_idx] = {
                        "frame": frame.copy(),
                        "detections": all_homeplate_detections
                    }

                # Stop if we've found enough detections
                if len(all_homeplate_detections) >= 5:
                    if self.verbose:
                        print(f"Found {len(all_homeplate_detections)} home plate detections")
                    break

        # Create a consensus home plate detection from all detections
        consensus_homeplate_box = None
        homeplate_confidence = 0
        homeplate_frame_idx = None
        homeplate_frame = None

        if all_homeplate_detections:
            if self.verbose:
                print(f"Processing {len(all_homeplate_detections)} home plate detections for consensus...")

            # Extract coordinates
            all_x1 = [det["box"][0] for det in all_homeplate_detections]
            all_y1 = [det["box"][1] for det in all_homeplate_detections]
            all_x2 = [det["box"][2] for det in all_homeplate_detections]
            all_y2 = [det["box"][3] for det in all_homeplate_detections]

            # Compute medians for outlier detection
            median_x1 = np.median(all_x1)
            median_y1 = np.median(all_y1)
            median_x2 = np.median(all_x2)
            median_y2 = np.median(all_y2)

            # Get median width and height for threshold calculation
            median_width = median_x2 - median_x1
            median_height = median_y2 - median_y1

            # Define threshold for outlier detection (as a percentage of median width/height)
            threshold_x = 0.3 * median_width
            threshold_y = 0.3 * median_height

            # Filter out outliers
            valid_detections = []
            for det in all_homeplate_detections:
                box = det["box"]

                # Check if this detection deviates too much from the median
                if (abs(box[0] - median_x1) > threshold_x or
                    abs(box[1] - median_y1) > threshold_y or
                    abs(box[2] - median_x2) > threshold_x or
                    abs(box[3] - median_y2) > threshold_y):
                    if self.verbose:
                        print(f"Removing outlier detection in frame {det['frame']}: deviation too large")
                    continue

                valid_detections.append(det)

            if valid_detections:
                if self.verbose:
                    print(f"After filtering, {len(valid_detections)} valid detections remain")

                # Calculate confidence-weighted average
                total_weight = sum(det["conf"] for det in valid_detections)

                avg_x1 = sum(det["box"][0] * det["conf"] for det in valid_detections) / total_weight
                avg_y1 = sum(det["box"][1] * det["conf"] for det in valid_detections) / total_weight
                avg_x2 = sum(det["box"][2] * det["conf"] for det in valid_detections) / total_weight
                avg_y2 = sum(det["box"][3] * det["conf"] for det in valid_detections) / total_weight

                # Create the consensus box
                consensus_homeplate_box = (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2))

                # Use average confidence
                homeplate_confidence = total_weight / len(valid_detections)

                # Find the best frame to represent this consensus (closest to the average)
                best_match_score = float('inf')
                best_match_idx = None

                for det in valid_detections:
                    box = det["box"]
                    match_score = (
                        abs(box[0] - avg_x1) +
                        abs(box[1] - avg_y1) +
                        abs(box[2] - avg_x2) +
                        abs(box[3] - avg_y2)
                    )

                    if match_score < best_match_score:
                        best_match_score = match_score
                        best_match_idx = det["frame"]

                homeplate_frame_idx = best_match_idx

                if self.verbose:
                    print(f"Consensus home plate box: {consensus_homeplate_box}")
                    print(f"Average confidence: {homeplate_confidence:.2f}")
                    print(f"Best matching frame: {homeplate_frame_idx}")

                # Get the frame for visualization
                if homeplate_frame_idx in detection_frames:
                    homeplate_frame = detection_frames[homeplate_frame_idx]["frame"].copy()
            else:
                # Fall back to best single detection if all were outliers
                if self.verbose:
                    print("All detections were outliers, using the best single detection")
                best_detection = max(all_homeplate_detections, key=lambda det: det["conf"])
                consensus_homeplate_box = best_detection["box"]
                homeplate_confidence = best_detection["conf"]
                homeplate_frame_idx = best_detection["frame"]

                if homeplate_frame_idx in detection_frames:
                    homeplate_frame = detection_frames[homeplate_frame_idx]["frame"].copy()
        else:
            if self.verbose:
                print("No home plate detections found using the dedicated model")
                print("Attempting fallback with catcher model...")
                
            # Try catcher model as fallback
            for frame_idx in search_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                with io.StringIO() as buf, redirect_stdout(buf):
                    results = self.catcher_model.predict(frame, conf=0.2, verbose=False)
                
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        cls_name = self.catcher_model.names[cls].lower()
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        if "homeplate" == cls_name:
                            detection = {
                                "box": (x1, y1, x2, y2),
                                "conf": conf,
                                "model": "PHC",
                                "frame": frame_idx
                            }
                            all_homeplate_detections.append(detection)
                            
                        # Try alternative names
                        elif "home_plate" == cls_name or "plate" == cls_name or "home" == cls_name:
                            detection = {
                                "box": (x1, y1, x2, y2),
                                "conf": conf,
                                "model": "PHC",
                                "frame": frame_idx
                            }
                            all_homeplate_detections.append(detection)
                            
                if all_homeplate_detections:
                    detection_frames[frame_idx] = {
                        "frame": frame.copy(),
                        "detections": all_homeplate_detections
                    }
                    
            # If we found fallback detections, process them as before
            if all_homeplate_detections:
                # Get the best detection
                best_detection = max(all_homeplate_detections, key=lambda det: det["conf"])
                consensus_homeplate_box = best_detection["box"]
                homeplate_confidence = best_detection["conf"]
                homeplate_frame_idx = best_detection["frame"]
                
                if homeplate_frame_idx in detection_frames:
                    homeplate_frame = detection_frames[homeplate_frame_idx]["frame"].copy()
                    
                if self.verbose:
                    print(f"Found fallback homeplate detection with confidence {homeplate_confidence:.2f}")
            else:
                if self.verbose:
                    print("No home plate detections found with any model")

        cap.release()
        return consensus_homeplate_box, homeplate_confidence, homeplate_frame_idx
    
    def _compute_strike_zone(self, catcher_detections: List[Dict], pitch_data: pd.Series, 
                             ball_glove_frame: int, video_path: str, 
                             hitter_keypoints: Optional[np.ndarray] = None,
                             hitter_box: Optional[Tuple[int, int, int, int]] = None,
                             homeplate_box: Optional[Tuple[int, int, int, int]] = None,
                             hitter_pose_3d: Optional[Dict] = None) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[float]]:
        """
        Compute strike zone dimensions based on home plate detection and Statcast data.
        Uses a consensus-based approach for accurate home plate detection.
        
        Args:
            catcher_detections (List[Dict]): List of catcher detection dictionaries
            pitch_data (pd.Series): Pitch data containing strike zone information
            ball_glove_frame (int): Frame where ball reaches glove
            video_path (str): Path to the video file
            hitter_keypoints (Optional[np.ndarray]): Optional keypoints for hitter pose
            hitter_box (Optional[Tuple[int, int, int, int]]): Optional bounding box for hitter
            homeplate_box (Optional[Tuple[int, int, int, int]]): Optional home plate detection
            hitter_pose_3d (Optional[Dict]): Optional 3D pose landmarks
            
        Returns:
            Tuple[Optional[Tuple[int, int, int, int]], Optional[float]]:
                (strike zone coordinates (left, top, right, bottom), pixels per foot)
        """
        if self.verbose:
            self.logger.info("Computing strike zone using MLB official dimensions...")

        # Use provided home plate detection if available, otherwise detect it
        if homeplate_box is None:
            homeplate_box, homeplate_confidence, homeplate_frame = self._detect_homeplate(video_path, ball_glove_frame)

        # If home plate detection is successful
        if homeplate_box is not None:
            # Calibrate based on home plate width (17 inches)
            plate_width_pixels = homeplate_box[2] - homeplate_box[0]
            plate_width_inches = 17.0
            pixels_per_inch = plate_width_pixels / plate_width_inches
            pixels_per_foot = pixels_per_inch * 12
            
            # Center the strike zone on the home plate
            plate_center_x = (homeplate_box[0] + homeplate_box[2]) / 2
            
            # Get home plate Y position for reference
            plate_top_y = homeplate_box[1]
            plate_bottom_y = homeplate_box[3]
            plate_center_y = (plate_top_y + plate_bottom_y) / 2
            
            # Use Statcast data for vertical positioning
            sz_top = float(pitch_data["sz_top"])
            sz_bot = float(pitch_data["sz_bot"])
            
            # Calculate zone dimensions
            zone_height_inches = (sz_top - sz_bot) * 12  # Convert feet to inches
            zone_height_pixels = int(zone_height_inches * pixels_per_inch)
            zone_width_pixels = int(17.0 * pixels_per_inch)  # MLB standard width
            
            # Variables to store elbow and hip landmarks for adjustment
            elbow_y = None
            hip_y = None
            
            # Use hitter pose (YOLO) to refine strike zone if available
            if hitter_keypoints is not None:
                # Check for knee landmarks (bottom of strike zone)
                knee_y = None
                for knee_idx in [13, 14]:  # Left and right knee
                    if hitter_keypoints[knee_idx, 2] > 0.3:  # If detected with good confidence
                        if knee_y is None or hitter_keypoints[knee_idx, 1].item() < knee_y:
                            knee_y = hitter_keypoints[knee_idx, 1].item()
                
                # Check for midpoint between shoulders and hips (top of strike zone)
                top_y = None
                shoulder_y = None
                
                # Get shoulder position
                shoulder_idxs = [5, 6]  # Left and right shoulder
                shoulders_detected = 0
                shoulder_y_sum = 0
                for idx in shoulder_idxs:
                    if hitter_keypoints[idx, 2] > 0.3:
                        shoulder_y_sum += hitter_keypoints[idx, 1].item()
                        shoulders_detected += 1
                if shoulders_detected > 0:
                    shoulder_y = shoulder_y_sum / shoulders_detected
                
                # Get hip position
                hip_idxs = [11, 12]  # Left and right hip
                hips_detected = 0
                hip_y_sum = 0
                for idx in hip_idxs:
                    if hitter_keypoints[idx, 2] > 0.3:
                        hip_y_sum += hitter_keypoints[idx, 1].item()
                        hips_detected += 1
                if hips_detected > 0:
                    hip_y = hip_y_sum / hips_detected
                
                # Get elbow position (for zone adjustment)
                elbow_idxs = [7, 8]  # Left and right elbow
                elbows_detected = 0
                elbow_y_sum = 0
                for idx in elbow_idxs:
                    if hitter_keypoints[idx, 2] > 0.3:
                        elbow_y_sum += hitter_keypoints[idx, 1].item()
                        elbows_detected += 1
                if elbows_detected > 0:
                    elbow_y = elbow_y_sum / elbows_detected
                
                # Calculate midpoint for top of strike zone
                if shoulder_y is not None and hip_y is not None:
                    top_y = (shoulder_y + hip_y) / 2
                
                # If we have both landmarks, use them to define the strike zone height
                if knee_y is not None and top_y is not None:
                    # Verify that top is actually above knee (sanity check)
                    if top_y < knee_y - 50:  # At least 50 pixels difference
                        zone_top_y = int(top_y)
                        zone_bottom_y = int(knee_y)
                        zone_height_pixels = zone_bottom_y - zone_top_y
                        
                        # Calculate adjustment for vertical position using configurable factor
                        # Positive values move zone toward home plate (up in the image)
                        adjustment = 0
                        if elbow_y is not None and hip_y is not None:
                            # Use the configurable factor instead of fixed 0.5
                            adjustment = int((hip_y - elbow_y) * self.zone_vertical_adjustment)
                            if self.verbose:
                                self.logger.info(f"Adjusting strike zone by {adjustment} pixels ({self.zone_vertical_adjustment:.2f} * elbow-to-hip distance)")
                        
                        # Recalculate calibration based on zone height and Statcast data
                        if self.verbose:
                            self.logger.info(f"Using hitter pose to refine strike zone height: {zone_height_pixels}px")
                        
                        # Center the zone horizontally
                        zone_left_x = int(plate_center_x - (zone_width_pixels / 2))
                        zone_right_x = int(plate_center_x + (zone_width_pixels / 2))
                        
                        # Apply the vertical adjustment - move zone up (closer to home plate) if positive factor
                        if adjustment != 0:
                            zone_top_y -= adjustment
                            zone_bottom_y -= adjustment
                        
                        strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
                        
                        if self.verbose:
                            self.logger.info(f"Strike zone from home plate and pose: {strike_zone}")
                        
                        return strike_zone, pixels_per_foot
            
            # Try using MediaPipe 3D pose for better adjustment if available
            elif hitter_pose_3d is not None:
                landmarks = hitter_pose_3d["results"].pose_landmarks
                offset_x, offset_y = hitter_pose_3d["offset"]
                
                if landmarks:
                    # Get key landmarks from MediaPipe
                    # Knees (indices 25, 26 in MediaPipe)
                    left_knee = landmarks.landmark[25]
                    right_knee = landmarks.landmark[26]
                    
                    # Calculate knee position for bottom of zone
                    knee_y = None
                    if left_knee.visibility > 0.3 and right_knee.visibility > 0.3:
                        left_knee_y = int(left_knee.y * frame.shape[0]) + offset_y
                        right_knee_y = int(right_knee.y * frame.shape[0]) + offset_y
                        knee_y = min(left_knee_y, right_knee_y)
                    elif left_knee.visibility > 0.3:
                        knee_y = int(left_knee.y * frame.shape[0]) + offset_y
                    elif right_knee.visibility > 0.3:
                        knee_y = int(right_knee.y * frame.shape[0]) + offset_y
                    
                    # Shoulders (indices 11, 12 in MediaPipe)
                    left_shoulder = landmarks.landmark[11]
                    right_shoulder = landmarks.landmark[12]
                    
                    # Hips (indices 23, 24 in MediaPipe)
                    left_hip = landmarks.landmark[23]
                    right_hip = landmarks.landmark[24]
                    
                    # Elbows (indices 13, 14 in MediaPipe)
                    left_elbow = landmarks.landmark[13]
                    right_elbow = landmarks.landmark[14]
                    
                    # Calculate positions for landmarks
                    shoulder_y = None
                    hip_y = None
                    elbow_y = None
                    
                    if left_shoulder.visibility > 0.3 and right_shoulder.visibility > 0.3:
                        left_shoulder_y = int(left_shoulder.y * frame.shape[0]) + offset_y
                        right_shoulder_y = int(right_shoulder.y * frame.shape[0]) + offset_y
                        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                    elif left_shoulder.visibility > 0.3:
                        shoulder_y = int(left_shoulder.y * frame.shape[0]) + offset_y
                    elif right_shoulder.visibility > 0.3:
                        shoulder_y = int(right_shoulder.y * frame.shape[0]) + offset_y
                    
                    if left_hip.visibility > 0.3 and right_hip.visibility > 0.3:
                        left_hip_y = int(left_hip.y * frame.shape[0]) + offset_y
                        right_hip_y = int(right_hip.y * frame.shape[0]) + offset_y
                        hip_y = (left_hip_y + right_hip_y) / 2
                    elif left_hip.visibility > 0.3:
                        hip_y = int(left_hip.y * frame.shape[0]) + offset_y
                    elif right_hip.visibility > 0.3:
                        hip_y = int(right_hip.y * frame.shape[0]) + offset_y
                    
                    if left_elbow.visibility > 0.3 and right_elbow.visibility > 0.3:
                        left_elbow_y = int(left_elbow.y * frame.shape[0]) + offset_y
                        right_elbow_y = int(right_elbow.y * frame.shape[0]) + offset_y
                        elbow_y = (left_elbow_y + right_elbow_y) / 2
                    elif left_elbow.visibility > 0.3:
                        elbow_y = int(left_elbow.y * frame.shape[0]) + offset_y
                    elif right_elbow.visibility > 0.3:
                        elbow_y = int(right_elbow.y * frame.shape[0]) + offset_y
                    
                    # Calculate top of strike zone (midpoint between shoulders and hips)
                    top_y = None
                    if shoulder_y is not None and hip_y is not None:
                        top_y = (shoulder_y + hip_y) / 2
                    
                    # If we have key landmarks, define the strike zone
                    if knee_y is not None and top_y is not None:
                        # Verify anatomically correct
                        if top_y < knee_y - 50:
                            zone_top_y = int(top_y)
                            zone_bottom_y = int(knee_y)
                            zone_height_pixels = zone_bottom_y - zone_top_y
                            
                            # Calculate adjustment for vertical position using configurable factor
                            adjustment = 0
                            if elbow_y is not None and hip_y is not None:
                                # Use the configurable factor instead of fixed 0.5
                                adjustment = int((hip_y - elbow_y) * self.zone_vertical_adjustment)
                                if self.verbose:
                                    self.logger.info(f"Adjusting strike zone by {adjustment} pixels ({self.zone_vertical_adjustment:.2f} * elbow-to-hip distance)")
                            
                            # Center the zone horizontally
                            zone_left_x = int(plate_center_x - (zone_width_pixels / 2))
                            zone_right_x = int(plate_center_x + (zone_width_pixels / 2))
                            
                            # Apply the vertical adjustment
                            if adjustment != 0:
                                zone_top_y -= adjustment
                                zone_bottom_y -= adjustment
                            
                            strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
                            
                            if self.verbose:
                                self.logger.info(f"Strike zone from home plate and MediaPipe 3D pose: {strike_zone}")
                            
                            return strike_zone, pixels_per_foot
            
            # If pose detection wasn't available or reliable, use home plate and Statcast data
            # Get a reference to the video to read a frame for height estimation
            cap = cv2.VideoCapture(video_path)
            if ball_glove_frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, ball_glove_frame)
            ret, frame = cap.read()
            frame_height = frame.shape[0] if ret else 720  # Default if can't read frame
            cap.release()
            
            # Estimate ground level
            # In baseball, home plate is on the ground, so use its bottom edge
            ground_y = plate_bottom_y
            
            # Calculate vertical positions relative to estimated ground level
            sz_bot_pixels = int(sz_bot * pixels_per_foot)  # Convert feet to pixels
            sz_top_pixels = int(sz_top * pixels_per_foot)  # Convert feet to pixels
            
            # Position the strike zone relative to the ground (which is at home plate level)
            zone_bottom_y = int(ground_y - sz_bot_pixels)
            zone_top_y = int(ground_y - sz_top_pixels)
            
            # Center the zone horizontally on home plate
            zone_left_x = int(plate_center_x - (zone_width_pixels / 2))
            zone_right_x = int(plate_center_x + (zone_width_pixels / 2))
            
            # Apply a standard adjustment to move the zone up a bit (closer to home plate)
            # Use the configurable adjustment factor
            zone_height_pixels = zone_bottom_y - zone_top_y
            adjustment = int(zone_height_pixels * 0.15 * self.zone_vertical_adjustment)
            if adjustment != 0:
                zone_top_y -= adjustment
                zone_bottom_y -= adjustment
            
            strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
            
            if self.verbose:
                self.logger.info(f"Strike zone computed using home plate: {strike_zone}")
            
            return strike_zone, pixels_per_foot
        
        # Try using hitter's pose to estimate strike zone when home plate detection fails
        elif hitter_keypoints is not None and hitter_box is not None:
            # Use the hitter's pose to estimate strike zone
            if self.verbose:
                self.logger.info("Using hitter's pose to estimate strike zone...")
            
            # Extract relevant keypoints
            # Knees (bottom of zone)
            knee_y = None
            for knee_idx in [13, 14]:  # Left and right knee
                if hitter_keypoints[knee_idx, 2] > 0.3:
                    if knee_y is None or hitter_keypoints[knee_idx, 1].item() < knee_y:
                        knee_y = hitter_keypoints[knee_idx, 1].item()
            
            # Mid-point between shoulders and hips (top of strike zone)
            top_y = None
            shoulder_y = None
            hip_y = None
            elbow_y = None
            
            # Get shoulder position
            shoulder_idxs = [5, 6]  # Left and right shoulder
            shoulders_detected = 0
            shoulder_y_sum = 0
            for idx in shoulder_idxs:
                if hitter_keypoints[idx, 2] > 0.3:
                    shoulder_y_sum += hitter_keypoints[idx, 1].item()
                    shoulders_detected += 1
            if shoulders_detected > 0:
                shoulder_y = shoulder_y_sum / shoulders_detected
            
            # Get hip position
            hip_idxs = [11, 12]  # Left and right hip
            hips_detected = 0
            hip_y_sum = 0
            for idx in hip_idxs:
                if hitter_keypoints[idx, 2] > 0.3:
                    hip_y_sum += hitter_keypoints[idx, 1].item()
                    hips_detected += 1
            if hips_detected > 0:
                hip_y = hip_y_sum / hips_detected
            
            # Get elbow position
            elbow_idxs = [7, 8]  # Left and right elbow
            elbows_detected = 0
            elbow_y_sum = 0
            for idx in elbow_idxs:
                if hitter_keypoints[idx, 2] > 0.3:
                    elbow_y_sum += hitter_keypoints[idx, 1].item()
                    elbows_detected += 1
            if elbows_detected > 0:
                elbow_y = elbow_y_sum / elbows_detected
            
            # Calculate midpoint for top of strike zone
            if shoulder_y is not None and hip_y is not None:
                top_y = (shoulder_y + hip_y) / 2
            
            # Determine horizontal center of hitter
            hitter_center_x = (hitter_box[0] + hitter_box[2]) / 2
            
            # If we have critical landmarks, use them for the strike zone
            if knee_y is not None and top_y is not None and knee_y > top_y:
                # For calibration, estimate height of player and convert to pixels per inch
                # Average MLB player height is around 6'2" (74 inches)
                player_height_px = hitter_box[3] - hitter_box[1]  # Full body height in pixels
                estimated_player_height_inches = 74
                pixels_per_inch = player_height_px / estimated_player_height_inches
                pixels_per_foot = pixels_per_inch * 12
                
                # Zone dimensions (17 inches wide by rule)
                zone_width_pixels = int(17.0 * pixels_per_inch)
                zone_height_pixels = int(knee_y - top_y)
                
                # Calculate adjustment using configurable factor
                adjustment = 0
                if elbow_y is not None and hip_y is not None:
                    adjustment = int((hip_y - elbow_y) * self.zone_vertical_adjustment)
                    if self.verbose:
                        self.logger.info(f"Adjusting strike zone by {adjustment} pixels ({self.zone_vertical_adjustment:.2f} * elbow-to-hip distance)")
                
                # Construct the strike zone
                zone_left_x = int(hitter_center_x - (zone_width_pixels / 2))
                zone_right_x = int(hitter_center_x + (zone_width_pixels / 2))
                zone_top_y = int(top_y)
                zone_bottom_y = int(knee_y)
                
                # Apply the vertical adjustment
                if adjustment != 0:
                    zone_top_y -= adjustment
                    zone_bottom_y -= adjustment
                
                strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
                
                if self.verbose:
                    self.logger.info(f"Strike zone computed using hitter's pose: {strike_zone}")
                
                return strike_zone, pixels_per_foot
            
        # Fallback to catcher detection if home plate detection fails and no reliable pose
        if catcher_detections and ball_glove_frame is not None:
            if self.verbose:
                self.logger.info("Falling back to catcher detection for strike zone estimation...")
            
            # Group detections by frame
            catcher_by_frame = {}
            for det in catcher_detections:
                catcher_by_frame[det["frame"]] = det
            
            # Find catcher detection closest to ball-glove frame
            nearby_frames = []
            for frame, det in catcher_by_frame.items():
                nearby_frames.append((abs(frame - ball_glove_frame), frame, det))
            
            nearby_frames.sort()
            
            if nearby_frames:
                _, frame_used, catcher_det = nearby_frames[0]
                
                # Estimate plate width and position from catcher
                catcher_width = catcher_det["x2"] - catcher_det["x1"]
                # Typical catcher width compared to home plate (17 inches)
                plate_width_pixels = catcher_width / 1.8
                
                # Place the plate in front of catcher
                plate_center_x = (catcher_det["x1"] + catcher_det["x2"]) / 2
                # Plate is slightly in front of catcher
                plate_bottom_y = catcher_det["y2"] + catcher_width * 0.1
                
                # Use Statcast data for vertical positioning
                sz_top = float(pitch_data["sz_top"]) 
                sz_bot = float(pitch_data["sz_bot"])
                
                # Calibration based on home plate width (17 inches)
                plate_width_inches = 17.0
                pixels_per_inch = plate_width_pixels / plate_width_inches
                pixels_per_foot = pixels_per_inch * 12
                
                # Calculate zone dimensions
                zone_height_inches = (sz_top - sz_bot) * 12  # Convert feet to inches
                zone_height_pixels = int(zone_height_inches * pixels_per_inch)
                zone_width_pixels = int(17.0 * pixels_per_inch)  # MLB standard width
                
                # Estimate ground level
                ground_y = plate_bottom_y
                
                # Calculate vertical zone positions
                sz_bot_pixels = int(sz_bot * pixels_per_foot)  # Convert feet to pixels
                sz_top_pixels = int(sz_top * pixels_per_foot)  # Convert feet to pixels
                
                # Position the strike zone relative to ground level
                zone_bottom_y = int(ground_y - sz_bot_pixels)
                zone_top_y = int(ground_y - sz_top_pixels)
                
                # Center the zone horizontally
                zone_left_x = int(plate_center_x - (zone_width_pixels / 2))
                zone_right_x = int(plate_center_x + (zone_width_pixels / 2))
                
                # Apply adjustment based on configurable factor
                adjustment = int(zone_height_pixels * 0.15 * self.zone_vertical_adjustment)
                if adjustment != 0:
                    zone_top_y -= adjustment
                    zone_bottom_y -= adjustment
                
                strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
                
                if self.verbose:
                    self.logger.info(f"Strike zone computed using catcher (fallback): {strike_zone}")
                
                return strike_zone, pixels_per_foot
        
        # Last resort: use video dimensions and Statcast data for a rough estimate
        if self.verbose:
            self.logger.info("Using video dimensions for a rough strike zone estimate (last resort)")
        
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Use StatCast data for rough pixel conversion
        sz_top = float(pitch_data["sz_top"])
        sz_bot = float(pitch_data["sz_bot"])
        zone_height_inches = (sz_top - sz_bot) * 12
        
        # Estimate zone from video dimensions
        zone_width = width // 5
        pixels_per_inch = zone_width / 17.0
        pixels_per_foot = pixels_per_inch * 12
        
        zone_height = int(zone_height_inches * pixels_per_inch)
        zone_left_x = (width - zone_width) // 2
        zone_right_x = zone_left_x + zone_width
        zone_bottom_y = height * 3 // 4
        zone_top_y = zone_bottom_y - zone_height
        
        # Apply adjustment based on configurable factor
        adjustment = int(zone_height * 0.15 * self.zone_vertical_adjustment)
        if adjustment != 0:
            zone_top_y -= adjustment
            zone_bottom_y -= adjustment
        
        strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
        
        if self.verbose:
            self.logger.info(f"Using estimated strike zone (last resort): {strike_zone}")
        
        return strike_zone, pixels_per_foot

    def _calculate_distance_to_zone(self, ball_center: Tuple[float, float], 
                                    strike_zone: Tuple[int, int, int, int],
                                    pixels_per_foot: float) -> Tuple[float, float, str, Tuple[float, float]]:
        """
        Calculate the distance from the ball to the nearest point on the strike zone.
        Using the MLB standard 17-inch strike zone width for calibration.
        
        Args:
            ball_center (Tuple[float, float]): (x, y) coordinates of ball center
            strike_zone (Tuple[int, int, int, int]): Strike zone coordinates (left, top, right, bottom)
            pixels_per_foot (float): Conversion factor from pixels to feet
            
        Returns:
            Tuple[float, float, str, Tuple[float, float]]: 
                (distance in pixels, distance in inches, position description, closest point coordinates)
        """
        ball_x, ball_y = ball_center
        zone_left, zone_top, zone_right, zone_bottom = strike_zone
        
        # Define a small tolerance (1.5 inches in pixels) to avoid false positives at the boundary
        inches_per_pixel = 12 / pixels_per_foot
        tolerance_pixels = 1.5 / inches_per_pixel
        
        # Find closest point on strike zone boundary
        closest_x = max(zone_left, min(ball_x, zone_right))
        closest_y = max(zone_top, min(ball_y, zone_bottom))
        
        # Calculate distance in pixels
        dx = ball_x - closest_x
        dy = ball_y - closest_y
        distance_pixels = math.sqrt(dx**2 + dy**2)
        
        # If ball is inside strike zone or within tolerance distance, distance is 0
        inside_zone = (zone_left <= ball_x <= zone_right and zone_top <= ball_y <= zone_bottom)
        
        # Even if the coordinates suggest it's inside, double-check with distance calculation
        # This will catch edge cases where the ball is very close to the boundary
        if inside_zone and distance_pixels > tolerance_pixels:
            # Double verify since we're getting conflicting results
            if self.verbose:
                self.logger.info(f"Ball coordinates suggest inside zone but distance ({distance_pixels:.2f}px) > tolerance. Re-checking...")
            
            # If it's on the edge, force it to be outside
            inside_zone = False
        
        # Calculate position description
        position = ""
        if ball_y < zone_top:
            position = "High"
        elif ball_y > zone_bottom:
            position = "Low"
        
        if ball_x < zone_left:
            position += " Inside" if position else "Inside"
        elif ball_x > zone_right:
            position += " Outside" if position else "Outside"
        
        # If we're very close to being inside but distance > 0, make it explicit
        if distance_pixels <= tolerance_pixels and not inside_zone:
            position = "Borderline " + (position if position else "Edge")
        
        # If truly inside, set distance to 0 and position to "In Zone"
        if inside_zone:
            distance_pixels = 0
            position = "In Zone"
            
        # Convert to inches
        distance_inches = distance_pixels * inches_per_pixel
        
        # Return distance and the closest point coordinates for visualization
        return distance_pixels, distance_inches, position, (closest_x, closest_y)
    
    def _create_annotated_video(
        self, 
        video_path: str, 
        output_path: str,
        catcher_detections: List[Dict], 
        glove_detections: List[Dict],
        ball_detections: List[Dict],
        strike_zone: Tuple[int, int, int, int],
        ball_glove_frame: Optional[int],
        distance_inches: Optional[float] = None,
        position: Optional[str] = None,
        hitter_keypoints: Optional[np.ndarray] = None,
        hitter_frame_idx: Optional[int] = None,
        hitter_box: Optional[Tuple[int, int, int, int]] = None,
        homeplate_box: Optional[Tuple[int, int, int, int]] = None,
        hitter_pose_3d: Optional[Dict] = None,
        frames_before: int = 8,
        frames_after: int = 8,
        closest_point: Optional[Tuple[float, float]] = None
    ) -> str:
        """
        Create an annotated video showing detections and strike zone.
        Includes slow motion replay at the end.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save annotated video
            catcher_detections (List[Dict]): Catcher detection results
            glove_detections (List[Dict]): Glove detection results
            ball_detections (List[Dict]): Ball detection results
            strike_zone (Tuple[int, int, int, int]): Strike zone (left, top, right, bottom)
            ball_glove_frame (Optional[int]): Frame where ball reaches glove
            distance_inches (Optional[float]): Distance to zone in inches
            position (Optional[str]): Position relative to zone
            hitter_keypoints (Optional[np.ndarray]): Keypoints for hitter's pose
            hitter_frame_idx (Optional[int]): Frame where hitter was detected
            hitter_box (Optional[Tuple[int, int, int, int]]): Bounding box for hitter
            homeplate_box (Optional[Tuple[int, int, int, int]]): Home plate bounding box
            hitter_pose_3d (Optional[Dict]): MediaPipe 3D pose results
            frames_before (int): Number of frames before glove contact to show zone
            frames_after (int): Number of frames after glove contact to show zone
            closest_point (Optional[Tuple[float, float]]): Coordinates of closest point on strike zone
            
        Returns:
            str: Path to the output video
        """
        if self.verbose:
            logging.info(f"Creating annotated video: {output_path}")
        
        # Convert detections to frame-indexed dictionaries
        catcher_by_frame = {}
        for det in catcher_detections:
            frame = det["frame"]
            if frame not in catcher_by_frame:
                catcher_by_frame[frame] = []
            catcher_by_frame[frame].append(det)
        
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
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create a temp file for the normal-speed video
        temp_output_path = output_path.replace('.mp4', '_temp.mp4')
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        # Extract strike zone dimensions
        zone_left, zone_top, zone_right, zone_bottom = strike_zone
        zone_width = zone_right - zone_left
        zone_height = zone_bottom - zone_top
        zone_center_x = (zone_left + zone_right) // 2
        
        # Define the pose skeleton connections
        skeleton = [
            (5, 7), (7, 9),    # left arm
            (6, 8), (8, 10),   # right arm
            (5, 6),            # shoulders
            (5, 11), (6, 12),  # torso
            (11, 12),          # hips
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16)  # right leg
        ]
        
        # For storing ball trajectory for visualization
        ball_trajectory = []
        
        # Process each frame
        pbar = tqdm(total=total_frames, desc="Creating Video", disable=not self.verbose)
        
        # Store key frames for slow motion
        slow_motion_frames = []
        slow_motion_start = max(0, ball_glove_frame - 15) if ball_glove_frame is not None else 0
        slow_motion_end = min(total_frames, ball_glove_frame + 15) if ball_glove_frame is not None else min(30, total_frames)
        
        # Fix for potential None issues with ball_glove_frame
        if ball_glove_frame is None:
            ball_glove_frame = total_frames // 2  # Use middle frame as a fallback
            logging.warning("No ball-glove contact frame detected. Using middle frame as fallback.")
            
        for frame_idx in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                logging.error(f"Failed to read frame {frame_idx} from video {video_path}")
                break
            
            # Create a copy for annotations
            annotated_frame = frame.copy()
            
            # Draw the distance info box for the ENTIRE video (not just at contact)
            # First add a semi-transparent background for better visibility
            if distance_inches is not None:
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (width - 310, 20), (width - 10, 160), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                cv2.rectangle(annotated_frame, (width - 310, 20), (width - 10, 160), (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Distance: {distance_inches:.2f} inches", (width - 300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Position: {position}", (width - 300, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Frame: {frame_idx}", (width - 300, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # Add whether the pitch is inside/outside the zone
                inside_zone = position == "In Zone"
                cv2.putText(annotated_frame, f"In Zone: {'Yes' if inside_zone else 'No'}", (width - 300, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw catcher detections (green)
            if frame_idx in catcher_by_frame:
                for det in catcher_by_frame[frame_idx]:
                    cv2.rectangle(annotated_frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "Catcher", (det["x1"], det["y1"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw glove detections (blue)
            if frame_idx in glove_by_frame:
                for det in glove_by_frame[frame_idx]:
                    cv2.rectangle(annotated_frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, "Glove", (det["x1"], det["y1"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Draw ball detections (red) and track trajectory
            if frame_idx in ball_by_frame:
                for det in ball_by_frame[frame_idx]:
                    cv2.rectangle(annotated_frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 0, 255), 2)
                    ball_cx = int((det["x1"] + det["x2"]) / 2)
                    ball_cy = int((det["y1"] + det["y2"]) / 2)
                    cv2.circle(annotated_frame, (ball_cx, ball_cy), 3, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "Ball", (det["x1"], det["y1"] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Add to trajectory for visualization
                    ball_trajectory.append((frame_idx, ball_cx, ball_cy))
            
            # Draw home plate if detected (orange)
            if homeplate_box is not None:
                cv2.rectangle(annotated_frame, 
                            (homeplate_box[0], homeplate_box[1]), 
                            (homeplate_box[2], homeplate_box[3]), 
                            (0, 128, 255), 2)
                cv2.putText(annotated_frame, "Home Plate", (homeplate_box[0], homeplate_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 2)
                
                # Draw center of home plate
                home_center_x = (homeplate_box[0] + homeplate_box[2]) // 2
                home_center_y = (homeplate_box[1] + homeplate_box[3]) // 2
                cv2.circle(annotated_frame, (home_center_x, home_center_y), 5, (0, 128, 255), -1)
            
            # Always draw strike zone (not just around ball-glove contact)
            # Draw strike zone (yellow)
            cv2.rectangle(annotated_frame, (zone_left, zone_top), (zone_right, zone_bottom), (0, 255, 255), 2)
            cv2.putText(annotated_frame, "Strike Zone", (zone_left, zone_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw line from home plate center to strike zone center if available
            if homeplate_box is not None:
                home_center_x = (homeplate_box[0] + homeplate_box[2]) // 2
                home_center_y = (homeplate_box[1] + homeplate_box[3]) // 2
                zone_center_x = (zone_left + zone_right) // 2
                zone_center_y = (zone_top + zone_bottom) // 2
                cv2.line(annotated_frame, (home_center_x, home_center_y), 
                        (zone_center_x, zone_center_y), (0, 255, 255), 1, cv2.LINE_AA)
            
            # Ball crosses zone info at ball-glove contact frame
            if frame_idx == ball_glove_frame:
                cv2.putText(annotated_frame, "BALL CROSSES ZONE", (width // 2 - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # If ball is in this frame and closest_point is provided, draw the measurement
                if frame_idx in ball_by_frame and closest_point is not None:
                    ball_det = ball_by_frame[frame_idx][0]
                    ball_cx = int((ball_det["x1"] + ball_det["x2"]) / 2)
                    ball_cy = int((ball_det["y1"] + ball_det["y2"]) / 2)
                    
                    # Draw a larger, more visible marker at the ball position
                    cv2.circle(annotated_frame, (ball_cx, ball_cy), 8, (0, 0, 255), -1)
                    cv2.circle(annotated_frame, (ball_cx, ball_cy), 10, (255, 255, 255), 2)
                    
                    # Draw a marker at the closest point on the strike zone
                    closest_x, closest_y = closest_point
                    cv2.circle(annotated_frame, (int(closest_x), int(closest_y)), 8, (0, 255, 255), -1)
                    cv2.circle(annotated_frame, (int(closest_x), int(closest_y)), 10, (255, 255, 255), 2)
                    
                    # Draw line between ball and closest point
                    cv2.line(annotated_frame, (ball_cx, ball_cy), (int(closest_x), int(closest_y)),
                            (255, 255, 0), 3, cv2.LINE_AA)
                    
                    # Add distance text
                    midpoint_x = (ball_cx + int(closest_x)) // 2
                    midpoint_y = (ball_cy + int(closest_y)) // 2
                    text_position = (midpoint_x + 5, midpoint_y - 5)
                    
                    # Draw text with contrasting background
                    text = f"{distance_inches:.1f}\""
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    cv2.rectangle(annotated_frame, 
                                (text_position[0] - 5, text_position[1] - text_size[1] - 5),
                                (text_position[0] + text_size[0] + 5, text_position[1] + 5),
                                (0, 0, 0), -1)
                    cv2.putText(annotated_frame, text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw the hitter box and pose information when available
            if hitter_box is not None:
                # Draw the hitter box
                cv2.rectangle(annotated_frame,
                            (hitter_box[0], hitter_box[1]),
                            (hitter_box[2], hitter_box[3]),
                            (255, 192, 0), 2)  # Blue-green
                cv2.putText(annotated_frame, "Hitter",
                        (hitter_box[0], hitter_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 192, 0), 2)
            
            # Draw 2D pose skeleton
            if hitter_keypoints is not None:
                # Draw keypoints for hitter
                for k in range(hitter_keypoints.shape[0]):
                    if hitter_keypoints[k, 2] > 0.3:  # Confidence threshold
                        x, y = int(hitter_keypoints[k, 0].item()), int(hitter_keypoints[k, 1].item())
                        cv2.circle(annotated_frame, (x, y), 4, (255, 0, 255), -1)  # Magenta
                
                # Draw skeleton connections
                for pair in skeleton:
                    if (hitter_keypoints[pair[0], 2] > 0.3 and
                        hitter_keypoints[pair[1], 2] > 0.3):
                        pt1 = (int(hitter_keypoints[pair[0], 0].item()),
                            int(hitter_keypoints[pair[0], 1].item()))
                        pt2 = (int(hitter_keypoints[pair[1], 0].item()),
                            int(hitter_keypoints[pair[1], 1].item()))
                        cv2.line(annotated_frame, pt1, pt2, (255, 0, 255), 2)
            
            # Draw 3D pose overlay from MediaPipe (only if hitter box is also available)
            if hitter_pose_3d is not None and hitter_box is not None:
                # Draw the 3D pose
                mp_results = hitter_pose_3d["results"]
                offset_x, offset_y = hitter_pose_3d["offset"]
                
                # Create a copy for semi-transparent overlay
                pose_overlay = annotated_frame.copy()
                
                # Get hitter box coordinates to draw the pose only within the box
                x1, y1, x2, y2 = hitter_box
                
                # Create a mask for the hitter region
                mask = np.zeros_like(annotated_frame)
                cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
                
                # Draw the pose landmarks and connections on a temporary image
                temp_overlay = annotated_frame.copy()
                self.mp_drawing.draw_landmarks(
                    temp_overlay,
                    mp_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
                
                # Apply the mask to limit drawing to hitter box region
                pose_overlay = np.where(mask > 0, temp_overlay, pose_overlay)
                
                # Add overlay with transparency
                cv2.addWeighted(pose_overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                # Add "3D Pose" label
                cv2.putText(annotated_frame, "3D Pose Overlay", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw ball trajectory (last 10 positions)
            trajectory_to_draw = [point for point in ball_trajectory if point[0] <= frame_idx]
            if len(trajectory_to_draw) > 1:
                # Only show last 10 points
                trajectory_to_draw = trajectory_to_draw[-10:]
                # Draw trajectory line
                for i in range(1, len(trajectory_to_draw)):
                    prev_frame, prev_x, prev_y = trajectory_to_draw[i-1]
                    curr_frame, curr_x, curr_y = trajectory_to_draw[i]
                    # Only connect consecutive frames or frames that are close
                    if curr_frame - prev_frame < 5:  # Connect only if frames are close
                        cv2.line(annotated_frame, (prev_x, prev_y), (curr_x, curr_y), (255, 165, 0), 2)
            
            # Add frame number and timing info
            cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Safely check if ball_glove_frame is not None before comparison
            if ball_glove_frame is not None:
                if frame_idx == ball_glove_frame:
                    cv2.putText(annotated_frame, "GLOVE CONTACT FRAME", (10, height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Store frames for slow motion replay
            if slow_motion_start <= frame_idx <= slow_motion_end:
                slow_motion_frames.append(annotated_frame.copy())
            
            out.write(annotated_frame)
            pbar.update(1)
        
        # Add slow motion replay (at 1/8 speed)
        if slow_motion_frames:
            # Add a transition frame with text "SLOW MOTION REPLAY"
            transition_frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(transition_frame, "SLOW MOTION REPLAY (1/8 SPEED)", 
                    (width // 2 - 200, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add transition frame multiple times to create a pause
            for _ in range(int(fps)):  # Pause for 1 second
                out.write(transition_frame)
            
            # Add the slow motion frames (each frame 8 times for 1/8 speed)
            for frame in slow_motion_frames:
                # Add "SLOW MOTION" label
                cv2.putText(frame, "SLOW MOTION (1/8x)", (width - 240, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Repeat each frame 8 times for 1/8 speed
                for _ in range(8):
                    out.write(frame)
        
        # Close the writer and progress bar
        pbar.close()
        out.release()
        
        # Now we need to convert the temp file with ffmpeg to ensure compatibility
        # Some versions of OpenCV create videos that aren't widely compatible
        final_command = f"ffmpeg -y -i {temp_output_path} -c:v libx264 -preset medium -crf 23 {output_path}"
        exit_code = os.system(final_command)
        if exit_code != 0:
            logging.warning(f"FFmpeg conversion may have failed with exit code {exit_code}")
        
        # Remove the temporary file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        else:
            logging.warning(f"Could not find temporary file to remove: {temp_output_path}")
        
        if self.verbose:
            logging.info(f"Video saved to {output_path}")
        
        return output_path