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
        results_dir: str = "results",
        verbose: bool = True,
        device: str = None
    ):
        """
        Initialize the DistanceToZone class.
        
        Args:
            catcher_model (YOLO): Model for detecting catchers
            glove_model (YOLO): Model for detecting gloves
            ball_model (YOLO): Model for detecting baseballs
            results_dir (str): Directory to save results
            verbose (bool): Whether to print detailed progress information
            device (str): Device to run models on (cpu, cuda, etc.)
        """
        self.load_tools = LoadTools()
        self.catcher_model = YOLO(self.load_tools.load_model(catcher_model))
        self.glove_model = YOLO(self.load_tools.load_model(glove_model))
        self.ball_model = YOLO(self.load_tools.load_model(ball_model))
        
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
        pitch_data = savant_scraper.run_statcast_pull_scraper(download_folder=download_folder, start_date=start_date, end_date=end_date, team=team, pitch_call=pitch_call, 
                                                 max_videos=max_videos, max_videos_per_game=max_videos_per_game,
                                                 max_workers=(os.cpu_count() - 2) if os.cpu_count() > 3 else 1)

        video_files = [os.path.join(download_folder, f) for f in os.listdir(download_folder) if f.endswith('.mp4')]
        
        dtoz_results = []
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            play_id = video_name.split('_')[-1]
            game_pk = video_name.split('_')[-2]
            pitch_data_row = pitch_data[pitch_data["play_id"] == play_id].iloc[0]
            output_path = os.path.join(self.results_dir, f"{video_name}_distance_to_zone.mp4") if create_video else None
            
            output_path = os.path.join(self.results_dir, f"distance_to_zone_{play_id}.mp4")
            
            catcher_detections = self._detect_objects(video_path, self.catcher_model, "catcher")
            glove_detections = self._detect_objects(video_path, self.glove_model, "glove")
            ball_detections = self._detect_objects(video_path, self.ball_model, "baseball")
            
            ball_glove_frame, ball_center = self._find_ball_reaches_glove(video_path, glove_detections, ball_detections)
            
            strike_zone_frame, strike_zone = self._compute_strikezone(
                video_path, pitch_data_row, catcher_detections, reference_frame=ball_glove_frame
            )
            
            distance = None
            position = None
            
            if ball_glove_frame is not None and ball_center is not None and strike_zone is not None:
                distance, position = self._calculate_distance_to_zone(pitch_data_row, ball_center, strike_zone)
                
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
                    position
                )
            
            results = {
                "video_name": video_name,
                "play_id": play_id,
                "game_pk": game_pk,
                "ball_glove_frame": ball_glove_frame,
                "ball_center": ball_center,
                "strike_zone_frame": strike_zone_frame,
                "strike_zone": strike_zone,
                "distance_to_zone": distance,
                "position": position,
                "annotated_video": output_path if create_video else None
            }
            
            dtoz_results.append(results)
            
        return dtoz_results
    
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
            print(f"\nDetecting {object_name} in video: {video_path}")
        
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
                results = model.predict(frame, conf=0.5, device=self.device, verbose=False)
                
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    if model.names[cls].lower() == object_name.lower():
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
    
    def _find_ball_reaches_glove(self, video_path: str, glove_detections: List[Dict], ball_detections: List[Dict], tolerance: float = 0.1) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
        """
        Find the first frame where a baseball's center is within a glove detection bounding box.
        
        Args:
            video_path (str): Path to the video file
            glove_detections (List[Dict]): List of glove detection dictionaries
            ball_detections (List[Dict]): List of ball detection dictionaries
            tolerance (float): Tolerance factor to expand the glove bounding box
            
        Returns:
            Tuple[Optional[int], Optional[Tuple[float, float]]]: 
                (frame index, ball center coordinates) if found, else (None, None)
        """
        if self.verbose:
            print(f"\nFinding when ball reaches glove in: {video_path}")
        
        # Group detections by frame for easier processing
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
        
        if self.verbose and ball_detection_sequences:
            print("Continuous ball detection sequences:")
            for seq in ball_detection_sequences:
                print(f"Sequence: {seq[0]} to {seq[-1]} (length: {len(seq)})")
        
        # Search for ball-glove contact in the most significant detection sequences
        # Sort sequences by length, prioritizing longer sequences
        ball_detection_sequences.sort(key=len, reverse=True)
        
        found_frame = None
        ball_center = None
        
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
                        ball_cx = (ball_det["x1"] + ball_det["x2"]) / 2
                        ball_cy = (ball_det["y1"] + ball_det["y2"]) / 2
                        
                        # Check if ball center is within extended glove box
                        if (extended_x1 <= ball_cx <= extended_x2 and
                            extended_y1 <= ball_cy <= extended_y2):
                            found_frame = frame
                            ball_center = (ball_cx, ball_cy)
                            break
                    
                    if found_frame is not None:
                        break
                
                if found_frame is not None:
                    break
            
            if found_frame is not None:
                break
        
        if self.verbose:
            if found_frame is not None:
                print(f"Ball reaches glove at frame {found_frame}")
            else:
                print("Could not detect when ball reaches glove")
        
        return found_frame, ball_center
        
    def _detect_homeplate(self, video_path: str, reference_frame: int = None) -> Tuple[Optional[Tuple[int, int, int, int]], float, int]:
        """
        Detect the home plate in the video using a consensus-based approach.
        Returns the home plate bounding box, confidence score, and the frame used.
        """
        if self.verbose:
            print("\nDetecting home plate using consensus-based approach...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define search frames around ball-glove contact (or use reference frame)
        search_frame = reference_frame if reference_frame is not None else total_frames // 2
        
        search_frames = [
            search_frame,             # The exact frame
            search_frame + 2,         # Some offset frames to improve detection
            search_frame - 2,
            search_frame + 4,
            search_frame - 4,
            search_frame - 8,
            search_frame - 16,
            search_frame - 30,
            search_frame - 45,
        ]
        # Filter out negative frames and frames beyond video length
        search_frames = [max(0, min(frame, total_frames-1)) for frame in search_frames]
        
        # Dictionary to track all home plate detections
        all_homeplate_detections = []
        
        # For visualization
        best_frame = None
        best_frame_idx = None
        
        # Try to detect home plate in search frames
        for frame_idx in search_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Try with PHC model first (might detect home plate)
            results = self.catcher_model.predict(frame, conf=0.15, verbose=False)
            
            frame_detections = []
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    
                    # Check for home plate in class names
                    if self.catcher_model.names[cls].lower() == "homeplate":
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detection = {
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "frame": frame_idx
                        }
                        frame_detections.append(detection)
                        all_homeplate_detections.append(detection)
                        
                        # Save best frame for visualization
                        if best_frame is None or conf > max(det["conf"] for det in all_homeplate_detections if det != detection):
                            best_frame = frame.copy()
                            best_frame_idx = frame_idx
        
        if all_homeplate_detections:
            if self.verbose:
                print(f"Found {len(all_homeplate_detections)} home plate detections")
            
            # Calculate consensus box (using median to avoid outliers)
            all_x1 = [det["box"][0] for det in all_homeplate_detections]
            all_y1 = [det["box"][1] for det in all_homeplate_detections]
            all_x2 = [det["box"][2] for det in all_homeplate_detections]
            all_y2 = [det["box"][3] for det in all_homeplate_detections]
            
            # Use median for robustness
            consensus_x1 = int(np.median(all_x1))
            consensus_y1 = int(np.median(all_y1))
            consensus_x2 = int(np.median(all_x2))
            consensus_y2 = int(np.median(all_y2))
            
            consensus_homeplate_box = (consensus_x1, consensus_y1, consensus_x2, consensus_y2)
            homeplate_confidence = np.mean([det["conf"] for det in all_homeplate_detections])
            
            # Save visualization of home plate detection
            if best_frame is not None and self.verbose:
                homeplate_debug = best_frame.copy()
                cv2.rectangle(homeplate_debug, 
                            (consensus_x1, consensus_y1), 
                            (consensus_x2, consensus_y2), 
                            (0, 255, 255), 2)
                
                center_x = (consensus_x1 + consensus_x2) // 2
                cv2.line(homeplate_debug, 
                       (center_x, consensus_y1 - 20), 
                       (center_x, consensus_y2 + 20), 
                       (0, 255, 255), 1)
                
                width_px = consensus_x2 - consensus_x1
                cv2.putText(homeplate_debug, 
                           f"Home Plate: {width_px} px = 17 inches", 
                           (consensus_x1, consensus_y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                os.makedirs(os.path.join(self.results_dir, "debug"), exist_ok=True)
                cv2.imwrite(os.path.join(self.results_dir, "debug", "homeplate_detection.jpg"), homeplate_debug)
                print(f"Home plate detection visualization saved to {os.path.join(self.results_dir, 'debug', 'homeplate_detection.jpg')}")
            
            cap.release()
            return consensus_homeplate_box, homeplate_confidence, best_frame_idx
        
        # If no home plate detection, return None
        cap.release()
        if self.verbose:
            print("No home plate detections found")
        return None, 0.0, None
    
    def _compute_strikezone(self, video_path: str, pitch_data: pd.Series, catcher_detections: List[Dict], reference_frame: Optional[int] = None) -> Tuple[int, Tuple[int, int, int, int]]:
        """
        Compute strike zone dimensions based on home plate detection and Statcast data.
        Uses a consensus-based approach for accurate home plate detection.
        
        Args:
            video_path (str): Path to the video file
            pitch_data (pd.Series): Pitch data containing strike zone information
            catcher_detections (List[Dict]): List of catcher detection dictionaries
            reference_frame (Optional[int]): Reference frame for computation
            
        Returns:
            Tuple[int, Tuple[int, int, int, int]]: (frame used, strike zone coordinates (left, top, right, bottom))
        """
        if self.verbose:
            print("\nComputing strike zone dimensions using enhanced method...")
        
        # Try to detect the home plate first
        homeplate_box, homeplate_confidence, homeplate_frame = self._detect_homeplate(video_path, reference_frame)
        
        # If home plate detection is successful
        if homeplate_box is not None:
            # Calibrate based on home plate width (17 inches)
            plate_width_pixels = homeplate_box[2] - homeplate_box[0]
            plate_width_inches = 17.0
            pixels_per_inch = plate_width_pixels / plate_width_inches
            pixels_per_foot = pixels_per_inch * 12
            
            # Center the strike zone on the home plate
            plate_center_x = (homeplate_box[0] + homeplate_box[2]) / 2
            
            # Use Statcast data for vertical positioning
            sz_top = float(pitch_data["sz_top"])
            sz_bot = float(pitch_data["sz_bot"])
            
            # Calculate zone dimensions
            zone_height_inches = (sz_top - sz_bot) * 12  # Convert feet to inches
            zone_height_pixels = int(zone_height_inches * pixels_per_inch)
            zone_width_pixels = int(17.0 * pixels_per_inch)  # MLB standard width
            
            # Estimate ground level from home plate
            ground_y = homeplate_box[3]  # Bottom of home plate
            
            # Calculate vertical zone positions
            zone_bottom_y = int(ground_y - (sz_bot * 12 * pixels_per_inch))
            zone_top_y = int(ground_y - (sz_top * 12 * pixels_per_inch))
            
            # Center the zone horizontally
            zone_left_x = int(plate_center_x - (zone_width_pixels / 2))
            zone_right_x = int(plate_center_x + (zone_width_pixels / 2))
            
            strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
            
            if self.verbose:
                print(f"Strike zone computed using home plate: {strike_zone}")
                print(f"Calibration: {pixels_per_inch:.2f} pixels/inch, {pixels_per_foot:.2f} pixels/foot")
                print(f"Zone width: {zone_width_pixels} pixels = 17 inches (MLB standard)")
                print(f"Zone height: {zone_height_pixels} pixels = {zone_height_inches:.1f} inches ({sz_bot:.2f}-{sz_top:.2f} feet)")
            
            return homeplate_frame, strike_zone
        
        # Fallback to catcher detection if home plate detection fails
        catcher_by_frame = {}
        for det in catcher_detections:
            catcher_by_frame[det["frame"]] = det
        
        if catcher_by_frame and reference_frame is not None:
            nearby_frames = []
            for frame, det in catcher_by_frame.items():
                nearby_frames.append((abs(frame - reference_frame), frame, det))
            
            nearby_frames.sort()
            
            if nearby_frames:
                _, frame_used, catcher_det = nearby_frames[0]
                
                # Estimate plate width and position from catcher (improved estimation)
                catcher_width = catcher_det["x2"] - catcher_det["x1"]
                # Typical catcher is about 1.5-2x the width of home plate 
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
                zone_bottom_y = int(ground_y - (sz_bot * 12 * pixels_per_inch))
                zone_top_y = int(zone_bottom_y - zone_height_pixels)
                
                # Center the zone horizontally
                zone_left_x = int(plate_center_x - (zone_width_pixels / 2))
                zone_right_x = int(plate_center_x + (zone_width_pixels / 2))
                
                strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
                
                if self.verbose:
                    print(f"Strike zone computed using catcher (fallback): {strike_zone}")
                    print(f"Calibration: {pixels_per_inch:.2f} pixels/inch, {pixels_per_foot:.2f} pixels/foot")
                    print(f"Zone width: {zone_width_pixels} pixels = 17 inches (MLB standard)")
                    print(f"Zone height: {zone_height_pixels} pixels = {zone_height_inches:.1f} inches")
                
                return frame_used, strike_zone
        
        # Last resort: use video dimensions for a rough estimate
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        zone_width = width // 5
        zone_height = int(zone_width * 1.5)  # Typical aspect ratio
        zone_left_x = (width - zone_width) // 2
        zone_right_x = zone_left_x + zone_width
        zone_bottom_y = height * 3 // 4
        zone_top_y = zone_bottom_y - zone_height
        
        strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
        
        if self.verbose:
            print(f"Using estimated strike zone (last resort): {strike_zone}")
        
        return reference_frame or 0, strike_zone
    
    def _calculate_distance_to_zone(self, pitch_data: pd.Series, ball_center: Tuple[float, float], strike_zone: Tuple[int, int, int, int]) -> Tuple[float, str]:
        """
        Calculate the distance from the ball to the nearest point on the strike zone.
        Using the MLB standard 17-inch strike zone width for calibration.
        
        Args:
            pitch_data (pd.Series): Pitch data containing information about the throw
            ball_center (Tuple[float, float]): (x, y) coordinates of ball center
            strike_zone (Tuple[int, int, int, int]): Strike zone coordinates (left, top, right, bottom)
            
        Returns:
            Tuple[float, str]: (distance in inches, position description)
        """
        ball_x, ball_y = ball_center
        zone_left, zone_top, zone_right, zone_bottom = strike_zone
        
        # Find closest point on strike zone boundary
        closest_x = max(zone_left, min(ball_x, zone_right))
        closest_y = max(zone_top, min(ball_y, zone_bottom))
        
        # If ball is inside strike zone, distance is 0
        if (zone_left <= ball_x <= zone_right and zone_top <= ball_y <= zone_bottom):
            return 0.0, "In Zone"
        
        # Calculate distance in pixels
        dx = ball_x - closest_x
        dy = ball_y - closest_y
        distance_pixels = math.sqrt(dx**2 + dy**2)
        
        # Convert to inches based on strike zone width (17 inches)
        zone_width_pixels = zone_right - zone_left
        zone_width_inches = 17.0
        inches_per_pixel = zone_width_inches / zone_width_pixels
        
        distance_inches = distance_pixels * inches_per_pixel
        
        # Determine position 
        vertical_position = "High" if ball_y < zone_top else "Low" if ball_y > zone_bottom else ""
        positions = []
        
        if vertical_position:
            positions.append(vertical_position)
            
        # Determine horizontal position (accounting for pitcher handedness)
        is_right_handed = pitch_data['p_throws'] == 'R'
        
        if ball_x < zone_left:
            # For right-handed pitchers, inside is to the left of zone
            positions.append("Inside" if is_right_handed else "Outside")
        elif ball_x > zone_right:
            # For right-handed pitchers, outside is to the right of zone
            positions.append("Outside" if is_right_handed else "Inside")
            
        position = " ".join(positions) or "Adjacent to Zone"
        
        return distance_inches, position
    
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
        frames_before: int = 8,
        frames_after: int = 8
    ):
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
            frames_before (int): Number of frames before glove contact to show zone
            frames_after (int): Number of frames after glove contact to show zone
        """
        if self.verbose:
            print(f"\nCreating annotated video: {output_path}")
        
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
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        zone_left, zone_top, zone_right, zone_bottom = strike_zone
        zone_width_pixels = zone_right - zone_left
        zone_center_x = (zone_left + zone_right) // 2
        zone_height = zone_bottom - zone_top
        
        homeplate_cache = {}
        homeplate_model = self.catcher_model
        use_model = homeplate_model is not None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Creating Video", disable=not self.verbose)

        # Render and save a frame for home plate detection visualization (for debugging)
        homeplate_box, _, _ = self._detect_homeplate(video_path, ball_glove_frame)
        
        # For saving the reference frame with ball crossing zone
        crossing_frame_path = os.path.join(self.results_dir, f"ball_crosses_zone_{os.path.basename(video_path)}.jpg")
        crossing_frame_saved = False
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            annotated_frame = frame.copy()
            
            near_contact = ball_glove_frame is not None and abs(frame_idx - ball_glove_frame) <= max(frames_before, frames_after)
            
            homeplate_box = None    
            homeplate_conf = 0
            
            if near_contact and use_model and frame_idx not in homeplate_cache:
                try:
                    with io.StringIO() as buf, redirect_stdout(buf):
                        results = homeplate_model.predict(frame, conf=0.2, device=self.device, verbose=False)
                    
                    for result in results:
                        for box in result.boxes:
                            cls = int(box.cls)
                            conf = float(box.conf)
                            cls_name = homeplate_model.names[cls].lower()
                            if "home" in cls_name and "plate" in cls_name or cls_name == "homeplate":
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                if conf > homeplate_conf:
                                    homeplate_box = (x1, y1, x2, y2)
                                    homeplate_conf = conf
                    homeplate_cache[frame_idx] = (homeplate_box, homeplate_conf)

                except Exception as e:
                    if self.verbose:
                        print(f"Error detecting home plate in frame {frame_idx}: {e}")
                    homeplate_cache[frame_idx] = (None, 0)
            elif frame_idx in homeplate_cache:
                homeplate_box, homeplate_conf = homeplate_cache[frame_idx]
            
            if frame_idx in catcher_by_frame:
                for det in catcher_by_frame[frame_idx]:
                    cv2.rectangle(annotated_frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "Catcher", (det["x1"], det["y1"] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if frame_idx in glove_by_frame:
                for det in glove_by_frame[frame_idx]:
                    cv2.rectangle(annotated_frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, "Glove", (det["x1"], det["y1"] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            if frame_idx in ball_by_frame:
                for det in ball_by_frame[frame_idx]:
                    cv2.rectangle(annotated_frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 0, 255), 2)
                    ball_cx = int((det["x1"] + det["x2"]) / 2)
                    ball_cy = int((det["y1"] + det["y2"]) / 2)
                    cv2.circle(annotated_frame, (ball_cx, ball_cy), 3, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, "Ball", (det["x1"], det["y1"] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if homeplate_box:
                cv2.rectangle(annotated_frame,
                             (homeplate_box[0], homeplate_box[1]),
                             (homeplate_box[2], homeplate_box[3]),
                             (0, 165, 255), 2)
                cv2.putText(annotated_frame, f"Home Plate ({homeplate_conf:.2f})",
                           (homeplate_box[0], homeplate_box[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                hp_center_x = (homeplate_box[0] + homeplate_box[2]) // 2
                cv2.line(annotated_frame, (hp_center_x, homeplate_box[1] - 20),
                        (hp_center_x, homeplate_box[3] + 20),
                        (0, 165, 255), 1, cv2.LINE_AA)
            
            if ball_glove_frame is not None and ball_glove_frame - frames_before <= frame_idx <= ball_glove_frame + frames_after:
                cv2.rectangle(annotated_frame, (zone_left, zone_top), (zone_right, zone_bottom), (0, 255, 255), 2)
                cv2.putText(annotated_frame, "Strike Zone", (zone_left, zone_top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"Width: {zone_width_pixels}px (17in)", (zone_left, zone_bottom + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(annotated_frame, f"Height: {zone_height}px", (zone_left, zone_bottom + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.line(annotated_frame, (zone_center_x, zone_top - 20), (zone_center_x, zone_bottom + 20),
                        (0, 255, 255), 1, cv2.LINE_AA)
                
                if frame_idx == ball_glove_frame and distance_inches is not None:
                    # Save this frame as the key moment when ball crosses zone
                    if not crossing_frame_saved:
                        cv2.imwrite(crossing_frame_path, annotated_frame)
                        crossing_frame_saved = True
                    
                    # Add info box with measurements
                    cv2.rectangle(annotated_frame, (width - 310, 20), (width - 10, 140), (0, 0, 0), -1)
                    cv2.rectangle(annotated_frame, (width - 310, 20), (width - 10, 140), (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Distance: {distance_inches:.2f} inches", (width - 300, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Position: {position}", (width - 300, 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, f"Frame: {frame_idx} (+2 delay)", (width - 300, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(annotated_frame, "BALL CROSSES ZONE", (width // 2 - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Draw line from ball to closest point on strike zone
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
            
            cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if frame_idx == ball_glove_frame:
                cv2.putText(annotated_frame, "GLOVE CONTACT FRAME", (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
            elif frame_idx == ball_glove_frame - 2:
                cv2.putText(annotated_frame, "ORIGINAL CONTACT FRAME", (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            out.write(annotated_frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        if self.verbose:
            print(f"Annotated video saved to {output_path}")
            if crossing_frame_saved:
                print(f"Ball crossing frame saved to {crossing_frame_path}")