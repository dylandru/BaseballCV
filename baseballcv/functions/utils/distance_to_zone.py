import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from baseballcv.functions.load_tools import LoadTools
from baseballcv.functions.savant_scraper import BaseballSavVideoScraper

class DistanceToZone:
    """
    Class for calculating and visualizing the distance of a pitch to the strike zone.
    
    This class uses computer vision models to detect the catcher, glove, ball, and 
    strike zone in baseball videos, then calculates the distance of the pitch to the zone.
    """
    
    def __init__(
        self, 
        device: str = 'cpu',
        phc_model: str = 'phc_detector',
        glove_model: str = 'glove_tracking',
        ball_model: str = 'ball_trackingv4',
        results_dir: str = "results",
        verbose: bool = True):
        """
        Initialize the DistanceToZone class.
        
        Args:
            catcher_model (YOLO): Model for detecting catchers
            glove_model (YOLO): Model for detecting gloves
            ball_model (YOLO): Model for detecting baseballs
            results_dir (str): Directory to save results
            verbose (bool): Whether to print detailed progress information
        """
        self.device = device
        self.load_tools = LoadTools()
        self.phc_model = YOLO(self.load_tools.load_model(phc_model))
        self.glove_model = YOLO(self.load_tools.load_model(glove_model))
        self.ball_model = YOLO(self.load_tools.load_model(ball_model))
        
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.verbose = verbose
    
    def analyze(
        self, 
        start_date: str, 
        end_date: str,
        team: str = None,
        pitch_call: str = None,
        max_videos: int = None,
        max_videos_per_game: int = None,
        create_video: bool = True) -> Dict:
        """
        Analyze a video to calculate the distance of a pitch to the strike zone.
        
        Args:
            video_path (str): Path to the video file
            pitch_data (pd.Series): Pitch data row containing 'sz_top' and 'sz_bot'
            create_video (bool): Whether to create an annotated video
            output_path (Optional[str]): Path to save the annotated video
            
        Returns:
            Dict: Analysis results including distance to zone
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
            
            catcher_detections = self._detect_objects(video_path, self.phc_model, "catcher")
            glove_detections = self._detect_objects(video_path, self.glove_model, "glove")
            ball_detections = self._detect_objects(video_path, self.ball_model, "baseball")
            
            ball_glove_frame, ball_center = self._find_ball_reaches_glove(video_path, glove_detections)
            
            strike_zone_frame, strike_zone = self._compute_strikezone(
                video_path, pitch_data_row, catcher_detections, reference_frame=ball_glove_frame
            )
            
            distance = None
            position = None
            
            if ball_glove_frame is not None and ball_center is not None:
                distance, position = self._calculate_distance_to_zone(ball_center, strike_zone)
                
                if self.verbose:
                    print(f"Distance to zone: {distance:.2f} pixels")
                    print(f"Position relative to zone: {position}")
            
            if create_video and output_path:
                self._create_annotated_video(
                    video_path, 
                    output_path,
                    catcher_detections, 
                    glove_detections,
                    ball_detections,
                    strike_zone_frame,
                    strike_zone,
                    ball_glove_frame
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
            }
            
            dtoz_results.append(results)
            
        return dtoz_results

    def _detect_objects(self, video_path: str, model: YOLO, object_name: str, conf: float = 0.5) -> List[Dict]:
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
                    disable=not self.verbose, dynamic_ncols=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model.predict(frame, conf=conf, device=self.device)
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
                glove_by_frame.setdefault(det["frame"], []).append(det)
            
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            found_frame = None
            ball_center = None
            frame_index = 0
            
            pbar = tqdm(total=total_frames, desc="Ball Tracking", 
                        disable=not self.verbose, dynamic_ncols=True)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_index in glove_by_frame:
                    results = self.ball_model.predict(frame, conf=0.5, device=self.device)
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
            Compute the strike zone based on catcher position and pitch data.
            
            Args:
                video_path (str): Path to the video file
                pitch_data (pd.Series): Pitch data row containing 'sz_top' and 'sz_bot'
                catcher_detections (List[Dict]): List of catcher detection dictionaries
                reference_frame (Optional[int]): Reference frame to compute strike zone near
                
            Returns:
                Tuple[int, Tuple[int, int, int, int]]: 
                    (frame used for strike zone, strike zone coordinates (left, top, right, bottom))
            """
            if self.verbose:
                print("\nComputing strike zone")
            
            if reference_frame is None:
                catcher_frames = sorted([det["frame"] for det in catcher_detections])
                if catcher_frames:
                    reference_frame = catcher_frames[len(catcher_frames) // 2]
                else:
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    reference_frame = total_frames // 2
            
            nearby_frames = []
            for det in catcher_detections:
                frame_diff = abs(det["frame"] - reference_frame)
                if frame_diff <= 10:  # Look within 10 frames
                    nearby_frames.append((frame_diff, det))
            
            nearby_frames.sort(key=lambda x: x[0])
            
            if nearby_frames:
                _, catcher_det = nearby_frames[0]
                frame_used = catcher_det["frame"]
                
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
                
                return frame_used, strike_zone,
            else:
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
                
                return reference_frame, strike_zone

    def _calculate_distance_to_zone(
            self, 
            ball_center: Tuple[float, float], 
            strike_zone: Tuple[int, int, int, int]
        ) -> Tuple[float, str]:
            """
            Calculate the distance from the ball to the strike zone.
            
            Args:
                ball_center (Tuple[float, float]): Ball center coordinates (x, y)
                strike_zone (Tuple[int, int, int, int]): Strike zone coordinates (left, top, right, bottom)
                
            Returns:
                Tuple[float, str]: (distance in pixels, position description)
            """
            ball_x, ball_y = ball_center
            zone_left, zone_top, zone_right, zone_bottom = strike_zone
            
            position = ""
            distance_x = 0
            distance_y = 0
            
            if ball_y < zone_top:
                position = "high"
                distance_y = zone_top - ball_y
            elif ball_y > zone_bottom:
                position = "low"
                distance_y = ball_y - zone_bottom
            else:
                distance_y = 0
            
            if ball_x < zone_left:
                position += "_inside" if position else "inside"
                distance_x = zone_left - ball_x
            elif ball_x > zone_right:
                position += "_outside" if position else "outside"
                distance_x = ball_x - zone_right
            else:
                distance_x = 0
            
            if distance_x > 0 and distance_y > 0:
                distance = np.sqrt(distance_x**2 + distance_y**2)
            else:
                distance = max(distance_x, distance_y)
            
            if not position:
                position = "inside"
            
            return distance, position

    def _create_annotated_video(
            self, 
            video_path: str, 
            output_path: str,
            catcher_detections: List[Dict], 
            glove_detections: List[Dict],
            ball_detections: List[Dict],
            strike_zone_frame: int,
            strike_zone: Tuple[int, int, int, int],
            ball_glove_frame: Optional[int] = None):
            """
            Create an annotated video with detections and strike zone.
            
            Args:
                video_path (str): Path to the input video
                output_path (str): Path to save the annotated video
                catcher_detections (List[Dict]): List of catcher detection dictionaries
                glove_detections (List[Dict]): List of glove detection dictionaries
                ball_detections (List[Dict]): List of ball detection dictionaries
                strike_zone_frame (int): Frame where strike zone was calculated
                strike_zone (Tuple[int, int, int, int]): Strike zone coordinates (left, top, right, bottom)
                ball_glove_frame (Optional[int]): Frame where ball reaches glove
            """
            if self.verbose:
                print(f"\nCreating annotated video: {output_path}")
            
            catcher_dict = {det["frame"]: det for det in catcher_detections}
            glove_dict = {det["frame"]: det for det in glove_detections}
            ball_dict = {det["frame"]: det for det in ball_detections}
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            pbar = tqdm(total=total_frames, desc="Creating Video", 
                        disable=not self.verbose, dynamic_ncols=True)
            
            frame_number = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                zone_left, zone_top, zone_right, zone_bottom = strike_zone
                cv2.rectangle(frame, (zone_left, zone_top), (zone_right, zone_bottom), (0, 255, 255), 2)
                cv2.putText(frame, "Strike Zone", (zone_left, zone_top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                if frame_number in catcher_dict:
                    det = catcher_dict[frame_number]
                    cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 255, 0), 2)
                    cv2.putText(frame, f"Catcher {det['confidence']:.2f}", (det["x1"], det["y1"] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if frame_number in glove_dict:
                    det = glove_dict[frame_number]
                    cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (255, 0, 0), 2)
                    cv2.putText(frame, f"Glove {det['confidence']:.2f}", (det["x1"], det["y1"] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                if frame_number in ball_dict:
                    det = ball_dict[frame_number]
                    cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), (0, 0, 255), 2)
                    ball_center_x = (det["x1"] + det["x2"]) / 2
                    ball_center_y = (det["y1"] + det["y2"]) / 2
                    
                    cv2.circle(frame, (int(ball_center_x), int(ball_center_y)), 3, (0, 0, 255), -1)
                    
                    distance, position = self._calculate_distance_to_zone(
                        (ball_center_x, ball_center_y), strike_zone
                    )
                    
                    cv2.putText(frame, f"Ball {det['confidence']:.2f}", (det["x1"], det["y1"] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f"Dist: {distance:.1f}px", (det["x1"], det["y1"] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f"Pos: {position}", (det["x1"], det["y1"] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                if frame_number == ball_glove_frame:
                    cv2.putText(frame, "BALL REACHES GLOVE", (frame_width // 2 - 150, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                cv2.putText(frame, f"Frame: {frame_number}", (10, frame_height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                out.write(frame)
                
                frame_number += 1
                pbar.update(1)

            
            pbar.close()
            cap.release()
            out.release()
            
            if self.verbose:
                print(f"Annotated video saved to {output_path}")
        
        
