import cv2
import numpy as np
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import pandas as pd
from ultralytics import YOLO
import io
from contextlib import redirect_stdout
from baseballcv.functions.savant_scraper import BaseballSavVideoScraper
from baseballcv.functions.load_tools import LoadTools
import math
import mediapipe as mp

class DistanceToZone:
    """
    Class for calculating and visualizing the distance of a pitch to the strike zone.
    
    This class uses computer vision models to detect the catcher, glove, ball, and 
    strike zone in baseball videos, then calculates the distance of the pitch to the zone.
    """
    
    def __init__(
        self, 
        catcher_model: str = 'phc_detector',
        glove_model: str = 'glove_tracking',
        ball_model: str = 'ball_trackingv4',
        homeplate_model: str = 'ball_trackingv4',
        results_dir: str = "results",
        verbose: bool = True,
        device: str = None
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
        """
        self.load_tools = LoadTools()
        self.catcher_model = YOLO(self.load_tools.load_model(catcher_model))
        self.glove_model = YOLO(self.load_tools.load_model(glove_model))
        self.ball_model = YOLO(self.load_tools.load_model(ball_model))
        self.homeplate_model = YOLO(self.load_tools.load_model(homeplate_model))
        
        # DEBUG: Print model class names to identify available classes
        print("\n--- DEBUG: Available Classes in Models ---")
        print(f"Catcher model ({catcher_model}) classes: {self.catcher_model.names}")
        print(f"Glove model ({glove_model}) classes: {self.glove_model.names}")
        print(f"Ball model ({ball_model}) classes: {self.ball_model.names}")
        print(f"Homeplate model ({homeplate_model}) classes: {self.homeplate_model.names}")
        
        # DEBUG: Check specifically for 'homeplate' or similar classes
        homeplate_classes = []
        for idx, cls_name in self.homeplate_model.names.items():
            if 'plate' in cls_name.lower() or 'home' in cls_name.lower():
                homeplate_classes.append((idx, cls_name))
        if homeplate_classes:
            print(f"Found potential homeplate classes: {homeplate_classes}")
        else:
            print("WARNING: No classes containing 'plate' or 'home' found in homeplate model!")
        
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.verbose = verbose
        self.device = device
    
    def analyze(
        self, 
        start_date: str, 
        end_date: str,
        team: str = None,
        pitch_call: str = None,
        max_videos: int = None,
        max_videos_per_game: int = None,
        create_video: bool = True
    ) -> List[Dict]:
        """
        Analyze videos from a date range to calculate distances to strike zone.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            team (str): Team abbreviation to filter by
            pitch_call (str): Pitch call to filter by (e.g., "Strike")
            max_videos (int): Maximum number of videos to process
            max_videos_per_game (int): Maximum videos per game
            create_video (bool): Whether to create annotated videos
            
        Returns:
            List[Dict]: List of analysis results per video
        """
        
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
        
        dtoz_results = []
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
                print(f"No pitch data found for play_id {play_id}, skipping...")
                continue
                
            output_path = os.path.join(self.results_dir, f"distance_to_zone_{play_id}.mp4")
            
            catcher_detections = self._detect_objects(video_path, self.catcher_model, "catcher")
            glove_detections = self._detect_objects(video_path, self.glove_model, "glove")
            ball_detections = self._detect_objects(video_path, self.ball_model, "baseball")
            
            # Also get pitcher and hitter detections for better filtering
            pitcher_detections = self._detect_objects(video_path, self.catcher_model, "pitcher")
            hitter_detections = self._detect_objects(video_path, self.catcher_model, "hitter")
            
            ball_glove_frame, ball_center, ball_detection = self._find_ball_reaches_glove(video_path, glove_detections, ball_detections)
            
            # Get catcher position to help distinguish hitter from umpire
            catcher_position = self._get_catcher_position(catcher_detections, ball_glove_frame)
            
            # First detect the reliable bounding box for the hitter
            hitter_box, hitter_frame, hitter_frame_idx = self._find_best_hitter_box(
                video_path=video_path,
                hitter_detections=hitter_detections,
                catcher_position=catcher_position,
                frame_idx_start=max(0, ball_glove_frame - 90) if ball_glove_frame else 0,
                frame_search_range=90
            )
            
            # Only detect pose WITHIN the hitter box to avoid getting the umpire
            hitter_keypoints = None
            if hitter_box is not None and hitter_frame is not None:
                hitter_keypoints = self._detect_pose_in_box(
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
                homeplate_box=homeplate_box
            )
            
            distance = None
            position = None
            
            if ball_center is not None and strike_zone is not None and pixels_per_foot is not None:
                distance_pixels, distance_inches, position = self._calculate_distance_to_zone(
                    ball_center, strike_zone, pixels_per_foot)
                distance = distance_inches

                if self.verbose:
                    print(f"Distance to zone: {distance:.2f} inches")
                    print(f"Position relative to zone: {position}")
            
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
                    homeplate_box=homeplate_box
                )
            
            results = {
                "video_name": video_name,
                "play_id": play_id,
                "game_pk": game_pk,
                "ball_glove_frame": ball_glove_frame,
                "ball_center": ball_center,
                "strike_zone": strike_zone,
                "distance_to_zone": distance,
                "position": position,
                "annotated_video": output_path if create_video else None
            }
            
            dtoz_results.append(results)
            
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
    
    def _detect_objects(self, video_path: str, model: YOLO, object_name: str) -> List[Dict]:
        """
        Detect objects in every frame of the video.
        
        Args:
            video_path (str): Path to the video file
            model (YOLO): YOLO model to use for detection
            object_name (str): Name of the object to detect
            
        Returns:
            List[Dict]: List of detection dictionaries
        """
        if self.verbose:
            print(f"\nDetecting {object_name} in video: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detections = []
        frame_number = 0
        
        pbar = tqdm(total=total_frames, desc=f"{object_name.capitalize()} Detection", 
                   disable=not self.verbose)
        
        # DEBUG: Check if the requested object is in the model's class list
        object_found = False
        for cls_idx, cls_name in model.names.items():
            if cls_name.lower() == object_name.lower():
                object_found = True
                break
        
        if not object_found:
            print(f"DEBUG WARNING: '{object_name}' not found in model classes: {model.names}")
            print(f"Available classes: {list(model.names.values())}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            with io.StringIO() as buf, redirect_stdout(buf):
                results = model.predict(frame, conf=0.5, device=self.device, verbose=False)
                
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    cls_name = model.names[cls].lower()
                    
                    # DEBUG: Print all detected classes in first few frames
                    if frame_number < 5:
                        print(f"DEBUG: Frame {frame_number}, detected class '{cls_name}' with conf {float(box.conf):.2f}")
                    
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
            print("\nDetecting when ball reaches glove...")

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
            print("Continuous ball detection sequences:")
            for seq in ball_detection_sequences:
                print(f"Sequence: {seq[0]} to {seq[-1]} (length: {len(seq)})")

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
                                print(f"Ball reached glove at frame {frame}")
                            return frame, (ball_center_x, ball_center_y), ball_det

        if self.verbose:
            print("Could not detect when ball reaches glove")
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
            print("\nFinding best hitter bounding box...")
        
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
                    # 3. Often higher in the frame than umpire
                    is_valid_hitter = True
                    
                    # Size check - must be substantial
                    area = (x2 - x1) * (y2 - y1)
                    min_area = width * height * 0.02  # At least 2% of frame
                    min_width = width * 0.05  # At least 5% of frame width
                    min_height = height * 0.1  # At least 10% of frame height
                    
                    if area < min_area or (x2 - x1) < min_width or (y2 - y1) < min_height:
                        is_valid_hitter = False
                    
                    # Position check relative to catcher
                    if catcher_position:
                        catcher_center_x = (catcher_position[0] + catcher_position[2]) / 2
                        hitter_center_x = (x1 + x2) / 2
                        
                        # If catcher is on right, hitter should be on left (and vice versa)
                        if ((catcher_center_x > width/2 and hitter_center_x > width/2) or
                            (catcher_center_x < width/2 and hitter_center_x < width/2)):
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
                print("Looking for hitter using PHC model...")
            
            for frame_idx in range(search_start, search_end, 5):  # Step by 5 frames for efficiency
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Detect hitter with PHC model
                with io.StringIO() as buf, redirect_stdout(buf):
                    phc_results = self.catcher_model.predict(frame, conf=0.4, verbose=False)
                
                for result in phc_results:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        
                        # Look specifically for "hitter" class
                        if self.catcher_model.names[cls].lower() == "hitter" and conf > best_confidence:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            # Apply the same heuristics as above
                            is_valid_hitter = True
                            
                            # Size check
                            area = (x2 - x1) * (y2 - y1)
                            min_area = width * height * 0.02  # At least 2% of frame
                            min_width = width * 0.05  # At least 5% of frame width
                            min_height = height * 0.1  # At least 10% of frame height
                            
                            if area < min_area or (x2 - x1) < min_width or (y2 - y1) < min_height:
                                is_valid_hitter = False
                            
                            # Position check relative to catcher
                            if catcher_position:
                                catcher_center_x = (catcher_position[0] + catcher_position[2]) / 2
                                hitter_center_x = (x1 + x2) / 2
                                
                                # If catcher is on right, hitter should be on left (and vice versa)
                                if ((catcher_center_x > width/2 and hitter_center_x > width/2) or
                                    (catcher_center_x < width/2 and hitter_center_x < width/2)):
                                    is_valid_hitter = False
                            
                            if is_valid_hitter:
                                best_hitter_box = (x1, y1, x2, y2)
                                best_hitter_frame = frame.copy()
                                best_frame_idx = frame_idx
                                best_confidence = conf
        
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
            print("\nDetecting hitter's pose strictly within bounding box...")
        
        # Load YOLO pose model
        try:
            pose_model = YOLO("yolov8n-pose.pt")
        except:
            # Try to download it
            os.system("wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt")
            pose_model = YOLO("yolov8n-pose.pt")
        
        # Extract only the region of interest to force pose detection in that area
        x1, y1, x2, y2 = box
        
        # Add a small margin (10%)
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)
        
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
            hitter_results = pose_model.predict(hitter_region, verbose=False)
        
        # Check if we got any results
        if (len(hitter_results) == 0 or 
            len(hitter_results[0].keypoints) == 0 or 
            hitter_results[0].keypoints.data.shape[1] < 17):
            if self.verbose:
                print("No pose detected within hitter region")
            return None
        
        # Get keypoints
        keypoints = hitter_results[0].keypoints.data[0].clone()  # Clone to avoid modifying original
        
        # Check if we detected essential keypoints (knees and hips)
        has_essential_keypoints = False
        for knee_idx in [13, 14]:  # Left and right knee
            for hip_idx in [11, 12]:  # Left and right hip
                if keypoints[knee_idx, 2] > 0.5 and keypoints[hip_idx, 2] > 0.5:
                    has_essential_keypoints = True
                    break
        
        if not has_essential_keypoints:
            if self.verbose:
                print("Essential keypoints (knees, hips) not detected with confidence")
            return None
        
        # Validate pose is anatomically reasonable
        # Check knees are below hips
        left_knee_y = keypoints[13, 1].item() if keypoints[13, 2] > 0.5 else None
        right_knee_y = keypoints[14, 1].item() if keypoints[14, 2] > 0.5 else None
        left_hip_y = keypoints[11, 1].item() if keypoints[11, 2] > 0.5 else None
        right_hip_y = keypoints[12, 1].item() if keypoints[12, 2] > 0.5 else None
        
        valid_anatomy = True
        
        # Check left side
        if left_knee_y is not None and left_hip_y is not None:
            if left_knee_y <= left_hip_y:  # Knee should be below hip
                valid_anatomy = False
        
        # Check right side
        if right_knee_y is not None and right_hip_y is not None:
            if right_knee_y <= right_hip_y:  # Knee should be below hip
                valid_anatomy = False
        
        if not valid_anatomy:
            if self.verbose:
                print("Detected pose is anatomically invalid (knees not below hips)")
            return None
        
        # Shift keypoints back to full frame coordinates
        for i in range(keypoints.shape[0]):
            keypoints[i, 0] += x1_margin  # Add x offset
            keypoints[i, 1] += y1_margin  # Add y offset
        
        if self.verbose:
            print("Successfully detected valid pose for hitter")
        
        return keypoints        
        
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
            print("\nDetecting home plate using specialized model...")
        
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
            print(f"Searching for home plate around ball-glove contact frame {search_frame}...")

        # Dictionary to store all frames with home plate detections (for visualization)
        detection_frames = {}
        all_homeplate_detections = []

        # DEBUG: Print homeplate model class names again for reference
        print("\n--- DEBUG: Homeplate Detection Class Check ---")
        print(f"Homeplate model classes: {self.homeplate_model.names}")
        
        # Check for all relevant classes that could be "homeplate"
        homeplate_class_candidates = []
        for idx, name in self.homeplate_model.names.items():
            if 'plate' in name.lower() or 'home' in name.lower():
                homeplate_class_candidates.append((idx, name))
        
        if homeplate_class_candidates:
            print(f"Found potential homeplate classes: {homeplate_class_candidates}")
        else:
            print("WARNING: No classes with 'plate' or 'home' found in model!")
        
        # Track all detected classes when processing frames
        detected_classes = set()

        for frame_idx in search_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Use the dedicated homeplate model
            with io.StringIO() as buf, redirect_stdout(buf):
                results = self.homeplate_model.predict(frame, conf=0.2, verbose=False)
            
            # DEBUG: Print all detected classes for diagnostic
            print(f"\nDEBUG: Frame {frame_idx} detections:")
            all_detections_in_frame = []
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    cls_name = self.homeplate_model.names[cls].lower()
                    detected_classes.add(cls_name)
                    
                    all_detections_in_frame.append((cls_name, conf))
                    
                    # DEBUG: Print all detections with confidence
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    print(f"  - Class: '{cls_name}', Conf: {conf:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")

                    # Check for home plate class
                    if "homeplate" == cls_name:
                        detection = {
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "model": "ball_trackingv4",
                            "frame": frame_idx
                        }
                        all_homeplate_detections.append(detection)
                    
                    # Additionally, try alternative class names that might be used
                    elif "home_plate" == cls_name or "plate" == cls_name or "home" == cls_name:
                        print(f"DEBUG: Found alternative homeplate class: {cls_name}")
                        detection = {
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "model": "ball_trackingv4",
                            "frame": frame_idx
                        }
                        all_homeplate_detections.append(detection)

            if all_detections_in_frame:
                print(f"  All detections in frame {frame_idx}: {all_detections_in_frame}")
            else:
                print(f"  No detections in frame {frame_idx}")

            # Store frame for visualization if detections were found
            if all_homeplate_detections:
                detection_frames[frame_idx] = {
                    "frame": frame.copy(),
                    "detections": all_homeplate_detections[-len(all_detections_in_frame):]
                }

        # Print summary of all detected classes for diagnostic
        print("\nDEBUG: All detected classes during scan:", detected_classes)
        print(f"DEBUG: Total homeplate detections found: {len(all_homeplate_detections)}")

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
                
                # DEBUG: Track detections in extended search
                extended_detections = []
                
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        cls_name = self.homeplate_model.names[cls].lower()
                        detected_classes.add(cls_name)
                        
                        extended_detections.append((cls_name, conf))
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        if cls_name == "homeplate":
                            detection = {
                                "box": (x1, y1, x2, y2),
                                "conf": conf,
                                "model": "ball_trackingv4",
                                "frame": frame_idx
                            }
                            all_homeplate_detections.append(detection)
                        
                        # Try alternative class names
                        elif cls_name in ["home_plate", "plate", "home"]:
                            print(f"DEBUG: Found alternative homeplate class in extended search: {cls_name}")
                            detection = {
                                "box": (x1, y1, x2, y2),
                                "conf": conf,
                                "model": "ball_trackingv4",
                                "frame": frame_idx
                            }
                            all_homeplate_detections.append(detection)

                if extended_detections:
                    print(f"DEBUG: Extended search frame {frame_idx} detections: {extended_detections}")

                if all_homeplate_detections:
                    detection_frames[frame_idx] = {
                        "frame": frame.copy(),
                        "detections": all_homeplate_detections[-len(extended_detections):]
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

                # DEBUG: Print catcher model classes
                print(f"\nDEBUG: Using catcher model as fallback, classes: {self.catcher_model.names}")
                
                with io.StringIO() as buf, redirect_stdout(buf):
                    results = self.catcher_model.predict(frame, conf=0.2, verbose=False)
                
                # DEBUG: Track all detections with catcher model
                catcher_detections = []
                
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        cls_name = self.catcher_model.names[cls].lower()
                        
                        catcher_detections.append((cls_name, conf))
                        
                        # DEBUG: Print all detections
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        print(f"  - Fallback detection: '{cls_name}', Conf: {conf:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")

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
                            print(f"DEBUG: Found alternative homeplate class in catcher model: {cls_name}")
                            detection = {
                                "box": (x1, y1, x2, y2),
                                "conf": conf,
                                "model": "PHC",
                                "frame": frame_idx
                            }
                            all_homeplate_detections.append(detection)
                            
                if catcher_detections:
                    print(f"  Catcher model detections in frame {frame_idx}: {catcher_detections}")
                            
                if all_homeplate_detections:
                    detection_frames[frame_idx] = {
                        "frame": frame.copy(),
                        "detections": all_homeplate_detections[-len(catcher_detections):]
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
                             homeplate_box: Optional[Tuple[int, int, int, int]] = None) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[float]]:
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
            
        Returns:
            Tuple[Optional[Tuple[int, int, int, int]], Optional[float]]:
                (strike zone coordinates (left, top, right, bottom), pixels per foot)
        """
        if self.verbose:
            print("\nComputing strike zone using MLB official dimensions...")

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
            
            # Use hitter pose to refine strike zone if available
            if hitter_keypoints is not None:
                # Check for knee landmarks (bottom of strike zone)
                knee_y = None
                for knee_idx in [13, 14]:  # Left and right knee
                    if hitter_keypoints[knee_idx, 2] > 0.5:  # If detected with good confidence
                        if knee_y is None or hitter_keypoints[knee_idx, 1].item() < knee_y:
                            knee_y = hitter_keypoints[knee_idx, 1].item()
                
                # Check for midpoint between shoulders and hips (top of strike zone)
                top_y = None
                shoulder_y = None
                hip_y = None
                
                # Get shoulder position
                shoulder_idxs = [5, 6]  # Left and right shoulder
                shoulders_detected = 0
                shoulder_y_sum = 0
                for idx in shoulder_idxs:
                    if hitter_keypoints[idx, 2] > 0.5:
                        shoulder_y_sum += hitter_keypoints[idx, 1].item()
                        shoulders_detected += 1
                if shoulders_detected > 0:
                    shoulder_y = shoulder_y_sum / shoulders_detected
                
                # Get hip position
                hip_idxs = [11, 12]  # Left and right hip
                hips_detected = 0
                hip_y_sum = 0
                for idx in hip_idxs:
                    if hitter_keypoints[idx, 2] > 0.5:
                        hip_y_sum += hitter_keypoints[idx, 1].item()
                        hips_detected += 1
                if hips_detected > 0:
                    hip_y = hip_y_sum / hips_detected
                
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
                        
                        # Recalculate calibration based on zone height and Statcast data
                        if self.verbose:
                            print(f"Using hitter pose to refine strike zone height: {zone_height_pixels}px")
                        
                        # Center the zone horizontally
                        zone_left_x = int(plate_center_x - (zone_width_pixels / 2))
                        zone_right_x = int(plate_center_x + (zone_width_pixels / 2))
                        
                        strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
                        
                        if self.verbose:
                            print(f"Strike zone from home plate and pose: {strike_zone}")
                            print(f"Home plate center: ({plate_center_x}, {plate_center_y})")
                            print(f"Calibration: {pixels_per_inch:.2f} pixels/inch, {pixels_per_foot:.2f} pixels/foot")
                            print(f"Zone width: {zone_width_pixels} pixels = 17 inches (MLB standard)")
                            print(f"Zone height: {zone_height_pixels} pixels")
                        
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
            
            # MLB strike zone is from hollow of knee to midpoint between shoulders and belt
            # Standard heights: knee (aprx 18-22" from ground), belt/midpoint(aprx 36-42" from ground)
            
            # We need to position the zone ABOVE the home plate, not directly on it
            # Calculate vertical distance from home plate to strike zone bottom
            # Use Statcast data for sz_bot (measured in feet from ground)
            # Typical sz_bot value is around 1.5-1.7 feet (18-20 inches)
            
            # Calculate vertical positions relative to estimated ground level
            sz_bot_pixels = int(sz_bot * pixels_per_foot)  # Convert feet to pixels
            sz_top_pixels = int(sz_top * pixels_per_foot)  # Convert feet to pixels
            
            # Position the strike zone relative to the ground (which is at home plate level)
            zone_bottom_y = int(ground_y - sz_bot_pixels)
            zone_top_y = int(ground_y - sz_top_pixels)
            
            # Center the zone horizontally on home plate
            zone_left_x = int(plate_center_x - (zone_width_pixels / 2))
            zone_right_x = int(plate_center_x + (zone_width_pixels / 2))
            
            strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
            
            if self.verbose:
                print(f"Strike zone computed using home plate: {strike_zone}")
                print(f"Home plate: {homeplate_box}")
                print(f"Home plate center: ({plate_center_x}, {plate_center_y})")
                print(f"Ground level estimate: {ground_y}")
                print(f"sz_bot: {sz_bot} feet = {sz_bot_pixels} pixels from ground")
                print(f"sz_top: {sz_top} feet = {sz_top_pixels} pixels from ground")
                print(f"Calibration: {pixels_per_inch:.2f} pixels/inch, {pixels_per_foot:.2f} pixels/foot")
                print(f"Zone width: {zone_width_pixels} pixels = 17 inches (MLB standard)")
                print(f"Zone height: {zone_height_pixels} pixels = {zone_height_inches:.1f} inches")
            
            return strike_zone, pixels_per_foot
        
        # Try using hitter's pose to estimate strike zone when home plate detection fails
        elif hitter_keypoints is not None and hitter_box is not None:
            # Use the hitter's pose to estimate strike zone
            if self.verbose:
                print("Using hitter's pose to estimate strike zone...")
            
            # Extract relevant keypoints
            # Knees (bottom of zone)
            knee_y = None
            for knee_idx in [13, 14]:  # Left and right knee
                if hitter_keypoints[knee_idx, 2] > 0.5:
                    if knee_y is None or hitter_keypoints[knee_idx, 1].item() < knee_y:
                        knee_y = hitter_keypoints[knee_idx, 1].item()
            
            # Mid-point between shoulders and hips (top of strike zone)
            top_y = None
            shoulder_y = None
            hip_y = None
            
            # Get shoulder position
            shoulder_idxs = [5, 6]  # Left and right shoulder
            shoulders_detected = 0
            shoulder_y_sum = 0
            for idx in shoulder_idxs:
                if hitter_keypoints[idx, 2] > 0.5:
                    shoulder_y_sum += hitter_keypoints[idx, 1].item()
                    shoulders_detected += 1
            if shoulders_detected > 0:
                shoulder_y = shoulder_y_sum / shoulders_detected
            
            # Get hip position
            hip_idxs = [11, 12]  # Left and right hip
            hips_detected = 0
            hip_y_sum = 0
            for idx in hip_idxs:
                if hitter_keypoints[idx, 2] > 0.5:
                    hip_y_sum += hitter_keypoints[idx, 1].item()
                    hips_detected += 1
            if hips_detected > 0:
                hip_y = hip_y_sum / hips_detected
            
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
                
                # Construct the strike zone
                zone_left_x = int(hitter_center_x - (zone_width_pixels / 2))
                zone_right_x = int(hitter_center_x + (zone_width_pixels / 2))
                zone_top_y = int(top_y)
                zone_bottom_y = int(knee_y)
                
                strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
                
                if self.verbose:
                    print(f"Strike zone computed using hitter's pose: {strike_zone}")
                    print(f"Calibration: {pixels_per_inch:.2f} pixels/inch, {pixels_per_foot:.2f} pixels/foot")
                    print(f"Zone width: {zone_width_pixels} pixels = 17 inches (MLB standard)")
                    print(f"Zone height: {zone_height_pixels} pixels")
                
                return strike_zone, pixels_per_foot
            
        # Fallback to catcher detection if home plate detection fails and no reliable pose
        if catcher_detections and ball_glove_frame is not None:
            if self.verbose:
                print("Falling back to catcher detection for strike zone estimation...")
            
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
                
                strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
                
                if self.verbose:
                    print(f"Strike zone computed using catcher (fallback): {strike_zone}")
                    print(f"Estimated ground level: {ground_y}")
                    print(f"Calibration: {pixels_per_inch:.2f} pixels/inch, {pixels_per_foot:.2f} pixels/foot")
                    print(f"Zone width: {zone_width_pixels} pixels = 17 inches (MLB standard)")
                    print(f"Zone height: {zone_height_pixels} pixels = {zone_height_inches:.1f} inches")
                
                return strike_zone, pixels_per_foot
        
        # Last resort: use video dimensions and Statcast data for a rough estimate
        if self.verbose:
            print("Using video dimensions for a rough strike zone estimate (last resort)")
        
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
        
        strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
        
        if self.verbose:
            print(f"Using estimated strike zone (last resort): {strike_zone}")
            print(f"Estimated calibration: {pixels_per_inch:.2f} pixels/inch")
        
        return strike_zone, pixels_per_foot

    def _calculate_distance_to_zone(self, ball_center: Tuple[float, float], 
                                    strike_zone: Tuple[int, int, int, int],
                                    pixels_per_foot: float) -> Tuple[float, float, str]:
        """
        Calculate the distance from the ball to the nearest point on the strike zone.
        Using the MLB standard 17-inch strike zone width for calibration.
        
        Args:
            ball_center (Tuple[float, float]): (x, y) coordinates of ball center
            strike_zone (Tuple[int, int, int, int]): Strike zone coordinates (left, top, right, bottom)
            pixels_per_foot (float): Conversion factor from pixels to feet
            
        Returns:
            Tuple[float, float, str]: (distance in pixels, distance in inches, position description)
        """
        ball_x, ball_y = ball_center
        zone_left, zone_top, zone_right, zone_bottom = strike_zone
        
        # Find closest point on strike zone boundary
        closest_x = max(zone_left, min(ball_x, zone_right))
        closest_y = max(zone_top, min(ball_y, zone_bottom))
        
        # If ball is inside strike zone, distance is 0
        if (zone_left <= ball_x <= zone_right and zone_top <= ball_y <= zone_bottom):
            return 0, 0, "In Zone"
        
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
        
        # Calculate distance in pixels
        dx = ball_x - closest_x
        dy = ball_y - closest_y
        distance_pixels = math.sqrt(dx**2 + dy**2)
        
        # Convert to inches
        inches_per_pixel = 12 / pixels_per_foot
        distance_inches = distance_pixels * inches_per_pixel
        
        return distance_pixels, distance_inches, position
    
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
        frames_before: int = 8,
        frames_after: int = 8
    ) -> str:
        """
        Create an annotated video showing detections and strike zone.
        
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
            frames_before (int): Number of frames before glove contact to show zone
            frames_after (int): Number of frames after glove contact to show zone
            
        Returns:
            str: Path to the output video
        """
        if self.verbose:
            print(f"\nCreating annotated video: {output_path}")
        
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
        
        # Create output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Extract strike zone dimensions
        zone_left, zone_top, zone_right, zone_bottom = strike_zone
        zone_width = zone_right - zone_left
        zone_height = zone_bottom - zone_top
        zone_center_x = (zone_left + zone_right) // 2
        
        # Cache for home plate detections
        homeplate_cache = {}
        
        # For saving the reference frame with ball crossing zone
        crossing_frame_path = os.path.join(self.results_dir, f"ball_crosses_zone_{os.path.basename(video_path)}.jpg")
        pose_frame_path = os.path.join(self.results_dir, f"hitter_pose_{os.path.basename(video_path)}.jpg")
        crossing_frame_saved = False
        saved_pose_frame = False
        
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
        
        # Calculate strike zone landmarks from pose if available
        strike_zone_lines = {}
        if hitter_keypoints is not None:
            # Calculate knee line (bottom of strike zone)
            left_knee = hitter_keypoints[13]
            right_knee = hitter_keypoints[14]
            if left_knee[2] > 0.5 and right_knee[2] > 0.5:
                knee_y = min(left_knee[1].item(), right_knee[1].item())
                strike_zone_lines["knee_y"] = knee_y
            elif left_knee[2] > 0.5:
                strike_zone_lines["knee_y"] = left_knee[1].item()
            elif right_knee[2] > 0.5:
                strike_zone_lines["knee_y"] = right_knee[1].item()
            
            # Calculate mid-torso line (top of strike zone)
            shoulders_detected = (hitter_keypoints[5][2] > 0.5 or hitter_keypoints[6][2] > 0.5)
            hips_detected = (hitter_keypoints[11][2] > 0.5 or hitter_keypoints[12][2] > 0.5)
            
            if shoulders_detected and hips_detected:
                shoulders_y = (hitter_keypoints[5][1].item() + hitter_keypoints[6][1].item()) / 2 if (
                    hitter_keypoints[5][2] > 0.5 and hitter_keypoints[6][2] > 0.5) else (
                    hitter_keypoints[5][1].item() if hitter_keypoints[5][2] > 0.5 else hitter_keypoints[6][1].item()
                )
                
                hips_y = (hitter_keypoints[11][1].item() + hitter_keypoints[12][1].item()) / 2 if (
                    hitter_keypoints[11][2] > 0.5 and hitter_keypoints[12][2] > 0.5) else (
                    hitter_keypoints[11][1].item() if hitter_keypoints[11][2] > 0.5 else hitter_keypoints[12][1].item()
                )
                
                strike_zone_lines["top_y"] = (shoulders_y + hips_y) / 2
        
        # Process each frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Creating Video", disable=not self.verbose)
        
        for frame_idx in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Create a copy for annotations
            annotated_frame = frame.copy()
            
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
            
            # Draw ball detections (red)
            if frame_idx in ball_by_frame:
                for det in ball_by_frame[frame_idx]:
                    cv2.rectangle(annotated_frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 0, 255), 2)
                    ball_cx = int((det["x1"] + det["x2"]) / 2)
                    ball_cy = int((det["y1"] + det["y2"]) / 2)
                    cv2.circle(annotated_frame, (ball_cx, ball_cy), 3, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "Ball", (det["x1"], det["y1"] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
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
            
            # Draw strike zone around ball-glove contact
            if ball_glove_frame is not None and ball_glove_frame - frames_before <= frame_idx <= ball_glove_frame + frames_after:
                # Draw strike zone (yellow)
                cv2.rectangle(annotated_frame, (zone_left, zone_top), (zone_right, zone_bottom), (0, 255, 255), 2)
                cv2.putText(annotated_frame, "Strike Zone", (zone_left, zone_top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Width: {zone_width}px (17in)", (zone_left, zone_bottom + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(annotated_frame, f"Height: {zone_height}px", (zone_left, zone_bottom + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Draw line from home plate center to strike zone center
                if homeplate_box is not None:
                    home_center_x = (homeplate_box[0] + homeplate_box[2]) // 2
                    home_center_y = (homeplate_box[1] + homeplate_box[3]) // 2
                    zone_center_x = (zone_left + zone_right) // 2
                    zone_center_y = (zone_top + zone_bottom) // 2
                    cv2.line(annotated_frame, (home_center_x, home_center_y), 
                            (zone_center_x, zone_center_y), (0, 255, 255), 1, cv2.LINE_AA)
                
                # Add distance info at ball-glove contact frame
                if frame_idx == ball_glove_frame and distance_inches is not None:
                    # Save this frame as the key moment when ball crosses zone
                    if not crossing_frame_saved:
                        cv2.imwrite(crossing_frame_path, annotated_frame)
                        crossing_frame_saved = True
                    
                    # Add background rectangle for better visibility
                    cv2.rectangle(annotated_frame, (width - 310, 20), (width - 10, 140), (0, 0, 0), -1)
                    cv2.rectangle(annotated_frame, (width - 310, 20), (width - 10, 140), (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Distance: {distance_inches:.2f} inches", (width - 300, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Position: {position}", (width - 300, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Frame: {frame_idx}", (width - 300, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, "BALL CROSSES ZONE", (width // 2 - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # If ball is in this frame, draw line to nearest point on strike zone
                    if frame_idx in ball_by_frame:
                        ball_det = ball_by_frame[frame_idx][0]
                        ball_cx = int((ball_det["x1"] + ball_det["x2"]) / 2)
                        ball_cy = int((ball_det["y1"] + ball_det["y2"]) / 2)
                        closest_x = max(zone_left, min(ball_cx, zone_right))
                        closest_y = max(zone_top, min(ball_cy, zone_bottom))
                        cv2.line(annotated_frame, (ball_cx, ball_cy), (closest_x, closest_y),
                                (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(annotated_frame, f"{distance_inches:.1f}\"",
                                  ((ball_cx + closest_x)//2, (ball_cy + closest_y)//2),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw the hitter box and pose information when available
            if hitter_box is not None:
                # Only draw within certain range of hitter_frame_idx to avoid clutter
                if hitter_frame_idx is None or abs(frame_idx - hitter_frame_idx) <= 10:
                    # Draw the hitter box
                    cv2.rectangle(annotated_frame,
                                (hitter_box[0], hitter_box[1]),
                                (hitter_box[2], hitter_box[3]),
                                (255, 192, 0), 2)  # Blue-green
                    cv2.putText(annotated_frame, "Hitter",
                              (hitter_box[0], hitter_box[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 192, 0), 2)
            
            # Draw pose skeleton
            if hitter_keypoints is not None and (hitter_frame_idx is None or abs(frame_idx - hitter_frame_idx) <= 10):
                # Draw keypoints for hitter
                for k in range(hitter_keypoints.shape[0]):
                    if hitter_keypoints[k, 2] > 0.5:  # Confidence threshold
                        x, y = int(hitter_keypoints[k, 0].item()), int(hitter_keypoints[k, 1].item())
                        cv2.circle(annotated_frame, (x, y), 4, (255, 0, 255), -1)  # Magenta
                
                # Draw skeleton connections
                for pair in skeleton:
                    if (hitter_keypoints[pair[0], 2] > 0.5 and
                        hitter_keypoints[pair[1], 2] > 0.5):
                        pt1 = (int(hitter_keypoints[pair[0], 0].item()),
                               int(hitter_keypoints[pair[0], 1].item()))
                        pt2 = (int(hitter_keypoints[pair[1], 0].item()),
                               int(hitter_keypoints[pair[1], 1].item()))
                        cv2.line(annotated_frame, pt1, pt2, (255, 0, 255), 2)
                
                # Draw strike zone reference lines from pose if available
                if "knee_y" in strike_zone_lines:
                    knee_y = int(strike_zone_lines["knee_y"])
                    cv2.line(annotated_frame, (0, knee_y), (width, knee_y),
                            (255, 255, 0), 1, cv2.LINE_AA)  # Yellow
                    cv2.putText(annotated_frame, "SZ Bottom (Knees)", (10, knee_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if "top_y" in strike_zone_lines:
                    top_y = int(strike_zone_lines["top_y"])
                    cv2.line(annotated_frame, (0, top_y), (width, top_y),
                            (255, 255, 0), 1, cv2.LINE_AA)  # Yellow
                    cv2.putText(annotated_frame, "SZ Top (Mid-Torso)", (10, top_y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Save one instance of the pose frame
                if frame_idx == hitter_frame_idx and not saved_pose_frame:
                    cv2.putText(annotated_frame, "HITTER POSE DETECTION", (width // 2 - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    cv2.imwrite(pose_frame_path, annotated_frame)
                    saved_pose_frame = True
            
            # Add frame number and timing info
            cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if frame_idx == ball_glove_frame:
                cv2.putText(annotated_frame, "GLOVE CONTACT FRAME", (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif frame_idx == ball_glove_frame - 2:
                cv2.putText(annotated_frame, "ORIGINAL CONTACT FRAME", (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif frame_idx == hitter_frame_idx:
                cv2.putText(annotated_frame, "HITTER POSE FRAME", (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            out.write(annotated_frame)
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        if self.verbose:
            print(f"Video saved to {output_path}")
            if crossing_frame_saved:
                print(f"Ball crossing frame saved to {crossing_frame_path}")
            if saved_pose_frame:
                print(f"Hitter pose frame saved to {pose_frame_path}")
        
        return output_path

# Helper method for debugging models - add this method to assist with checking class names
def inspect_model_classes(self, model_name: str):
    """
    Debug utility method to inspect class names in a specified model.
    
    Args:
        model_name (str): Name of the model to inspect
    """
    try:
        print(f"\n--- DEBUG: Inspecting classes in {model_name} model ---")
        model_path = self.load_tools.load_model(model_name)
        model = YOLO(model_path)
        
        print(f"Total classes in {model_name}: {len(model.names)}")
        print(f"Class mapping: {model.names}")
        
        # Check for potential homeplate classes
        homeplate_candidates = []
        for idx, name in model.names.items():
            if 'plate' in name.lower() or 'home' in name.lower():
                homeplate_candidates.append((idx, name))
        
        if homeplate_candidates:
            print(f"Potential homeplate classes: {homeplate_candidates}")
        else:
            print(f"WARNING: No classes with 'plate' or 'home' found in {model_name} model")
        
        # Print all available classes for reference
        print("All class names:")
        for idx, name in model.names.items():
            print(f"  {idx}: {name}")
            
        return model.names
    except Exception as e:
        print(f"Error inspecting model {model_name}: {str(e)}")
        return {}

# Add the debug method to the class
DistanceToZone.inspect_model_classes = inspect_model_classes