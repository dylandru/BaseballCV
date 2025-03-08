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
            
            ball_glove_frame, ball_center = self._find_ball_reaches_glove(video_path, glove_detections)
            
            strike_zone_frame, strike_zone = self._compute_strikezone(
                video_path, pitch_data_row, catcher_detections, reference_frame=ball_glove_frame
            )
            
            distance = None
            position = None
            
            if ball_glove_frame is not None and ball_center is not None and strike_zone is not None:
                distance, position = self._calculate_distance_to_zone(pitch_data_row, ball_center, strike_zone)
                
                if self.verbose:
                    print(f"Distance to zone: {distance:.2f} pixels")
                    print(f"Position relative to zone: {position}")
            
            if create_video and output_path and strike_zone is not None:
                self._create_annotated_video(
                    video_path, 
                    output_path,
                    catcher_detections, 
                    glove_detections,
                    ball_detections,
                    strike_zone_frame,
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
    
    def _find_ball_reaches_glove(self, video_path: str, glove_detections: List[Dict], tolerance: float = 0.1) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
        """
        Find the first frame where a baseball's center is within a glove detection bounding box.
        
        Args:
            video_path (str): Path to the video file
            glove_detections (List[Dict]): List of glove detection dictionaries
            tolerance (float): Tolerance factor to expand the glove bounding box
            
        Returns:
            Tuple[Optional[int], Optional[Tuple[float, float]]]: 
                (frame index, ball center coordinates) if found, else (None, None)
        """
        if self.verbose:
            print(f"\nFinding when ball reaches glove in: {video_path}")
        
        glove_by_frame = {}
        for det in glove_detections:
            frame = det["frame"]
            if frame not in glove_by_frame:
                glove_by_frame[frame] = []
            glove_by_frame[frame].append(det)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        found_frame = None
        ball_center = None
        frame_index = 0
        
        pbar = tqdm(total=total_frames, desc="Ball Tracking", 
                   disable=not self.verbose)
        
        while cap.isOpened() and frame_index < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_index in glove_by_frame:
                with io.StringIO() as buf, redirect_stdout(buf):
                    results = self.ball_model.predict(frame, conf=0.5, device=self.device, verbose=False)
                    
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls)
                        if self.ball_model.names[cls].lower() == "baseball":
                            ball_box = box.xyxy[0].tolist()
                            ball_center_x = (ball_box[0] + ball_box[2]) / 2.0
                            ball_center_y = (ball_box[1] + ball_box[3]) / 2.0
                            
                            for glove_det in glove_by_frame[frame_index]:
                                margin_x = tolerance * (glove_det["x2"] - glove_det["x1"])
                                margin_y = tolerance * (glove_det["y2"] - glove_det["y1"])
                                extended_x1 = glove_det["x1"] - margin_x
                                extended_y1 = glove_det["y1"] - margin_y
                                extended_x2 = glove_det["x2"] + margin_x
                                extended_y2 = glove_det["y2"] + margin_y
                                
                                if (ball_center_x >= extended_x1 and ball_center_x <= extended_x2 and
                                    ball_center_y >= extended_y1 and ball_center_y <= extended_y2):
                                    found_frame = frame_index
                                    ball_center = (ball_center_x, ball_center_y)
                                    break
                            
                            if found_frame is not None:
                                break
                    
                    if found_frame is not None:
                        break
            
            if found_frame is not None:
                break
                
            frame_index += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        if self.verbose:
            if found_frame is not None:
                print(f"Ball reaches glove at frame {found_frame}")
            else:
                print("Could not detect when ball reaches glove")
        
        return found_frame, ball_center
    
    def _compute_strikezone(self, video_path: str, pitch_data: pd.Series, catcher_detections: List[Dict], reference_frame: Optional[int] = None) -> Tuple[int, Tuple[int, int, int, int]]:
        """
        Compute the strike zone dimensions based on catcher position and Statcast data.
        
        Args:
            video_path (str): Path to the video file
            pitch_data (pd.Series): Pitch data containing strike zone information
            catcher_detections (List[Dict]): List of catcher detection dictionaries
            reference_frame (Optional[int]): Reference frame for computation
            
        Returns:
            Tuple[int, Tuple[int, int, int, int]]: (frame used, strike zone coordinates (left, top, right, bottom))
        """
        if self.verbose:
            print("\nComputing strike zone dimensions...")
        
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
                
                catcher_center_x = (catcher_det["x1"] + catcher_det["x2"]) / 2
                catcher_bottom_y = catcher_det["y2"]
                
                plate_width_pixels = (catcher_det["x2"] - catcher_det["x1"]) * 0.8
                plate_center_x = catcher_center_x
                plate_bottom_y = catcher_bottom_y + plate_width_pixels * 0.3
                
                sz_top = float(pitch_data["sz_top"])
                sz_bot = float(pitch_data["sz_bot"])
                
                plate_width_feet = 1.42
                pixels_per_foot = plate_width_pixels / plate_width_feet
                
                zone_height = (sz_top - sz_bot) * pixels_per_foot
                zone_width = plate_width_pixels
                
                zone_bottom_y = int(plate_bottom_y - (sz_bot * pixels_per_foot))
                zone_top_y = int(zone_bottom_y - zone_height)
                zone_left_x = int(plate_center_x - (zone_width / 2))
                zone_right_x = int(plate_center_x + (zone_width / 2))
                
                strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
                
                if self.verbose:
                    print(f"Strike zone computed at frame {frame_used}: {strike_zone}")
                
                return frame_used, strike_zone
            
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        zone_width = width // 5
        zone_height = height // 4
        zone_left_x = (width - zone_width) // 2
        zone_right_x = zone_left_x + zone_width
        zone_bottom_y = height * 3 // 4
        zone_top_y = zone_bottom_y - zone_height
        
        strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
        
        if self.verbose:
            print(f"Using estimated strike zone at frame {reference_frame}: {strike_zone}")
        
        return reference_frame or 0, strike_zone
    
    def _calculate_distance_to_zone(self, pitch_data: pd.Series, ball_center: Tuple[float, float], strike_zone: Tuple[int, int, int, int]) -> Tuple[float, str]:
        """
        Calculate the distance from the ball to the nearest point on the strike zone.
        
        Args:
            ball_center (Tuple[float, float]): (x, y) coordinates of ball center
            strike_zone (Tuple[int, int, int, int]): Strike zone coordinates (left, top, right, bottom)
            
        Returns:
            Tuple[float, str]: (distance in inches, position description)
        """
        ball_x, ball_y = ball_center
        zone_left, zone_top, zone_right, zone_bottom = strike_zone
        
        inside_x = ball_x >= zone_left and ball_x <= zone_right
        inside_y = ball_y >= zone_top and ball_y <= zone_bottom
        
        inside_zone = inside_x and inside_y
        
        if inside_zone:
            return 0.0, "In Zone"
        
        closest_x = max(zone_left, min(ball_x, zone_right))
        closest_y = max(zone_top, min(ball_y, zone_bottom))
        
        distance_pixels = np.sqrt((ball_x - closest_x)**2 + (ball_y - closest_y)**2)
        
        zone_width_pixels = zone_right - zone_left
        zone_width_inches = 17.0
        inches_per_pixel = zone_width_inches / zone_width_pixels
        
        distance_inches = distance_pixels * inches_per_pixel
        
        positions = []
        if ball_y < zone_top:
            positions.append("High")
        elif ball_y > zone_bottom:
            positions.append("Low")
        
        if ball_x < zone_left:
            positions.append("Inside" if pitch_data['p_throws'] == 'R' else "Outside")
        elif ball_x > zone_right:
            positions.append("Outside" if pitch_data['p_throws'] == 'R' else "Inside")
        
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
        homeplate_model = self.glove_model
        use_model = homeplate_model is not None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Creating Video", disable=not self.verbose)

        
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
            
            # Add frame information
            cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Mark contact frames
            if frame_idx == ball_glove_frame:
                cv2.putText(annotated_frame, "GLOVE CONTACT FRAME", (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            elif frame_idx == ball_glove_frame - 2:
                cv2.putText(annotated_frame, "ORIGINAL CONTACT FRAME", (10, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Write frame to output
            out.write(annotated_frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        if self.verbose:
            print(f"Annotated video saved to {output_path}")
