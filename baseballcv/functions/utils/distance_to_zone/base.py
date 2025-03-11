import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from ultralytics import YOLO
import mediapipe as mp

from baseballcv.functions.savant_scraper import BaseballSavVideoScraper
from baseballcv.functions.load_tools import LoadTools

from .object_detector import ObjectDetector
from .pose_detector import PoseDetector
from .strike_zone import StrikeZoneCalculator
from .distance_calculator import DistanceCalculator
from .visualizer import VideoAnnotator

class DistanceToZone:
    """
    Orchestrator class for calculating and visualizing the distance of a pitch to the strike zone.
    
    This class coordinates the overall analysis process by leveraging specialized components:
    - Object detection (ball, glove, catcher, etc.)
    - Pose detection (player poses)
    - Strike zone calculation
    - Distance measurement
    - Visualization
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
        zone_vertical_adjustment: float = 0.5
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
            zone_vertical_adjustment (float): Factor to adjust strike zone vertically
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
        
        if verbose:
            print(f"Models loaded: {catcher_model}, {glove_model}, {ball_model}, {homeplate_model}")
        
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.verbose = verbose
        self.device = device
        self.zone_vertical_adjustment = zone_vertical_adjustment
        
        # Initialize specialized components
        self.object_detector = ObjectDetector(
            catcher_model=self.catcher_model,
            glove_model=self.glove_model,
            ball_model=self.ball_model,
            homeplate_model=self.homeplate_model,
            device=device,
            verbose=verbose
        )
        
        self.pose_detector = PoseDetector(
            mp_pose=self.mp_pose,
            mp_drawing=self.mp_drawing,
            mp_drawing_styles=self.mp_drawing_styles,
            catcher_model=self.catcher_model,
            verbose=verbose
        )
        
        self.strike_zone_calculator = StrikeZoneCalculator(
            zone_vertical_adjustment=zone_vertical_adjustment,
            verbose=verbose
        )
        
        self.distance_calculator = DistanceCalculator(verbose=verbose)
        
        self.visualizer = VideoAnnotator(
            mp_drawing=self.mp_drawing,
            mp_drawing_styles=self.mp_drawing_styles,
            verbose=verbose
        )
    
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
        Analyze videos from a date range to calculate distances to strike zone.
        
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
        detailed_results = []
        
        for video_path in video_files:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            play_id = video_name.split('_')[-1]
            game_pk = video_name.split('_')[-2]
            
            # Find corresponding pitch data
            pitch_data_row = None
            for _, row in pitch_data.iterrows():
                if row["play_id"] == play_id:
                    pitch_data_row = row
                    break
            
            if pitch_data_row is None:
                if self.verbose:
                    print(f"No pitch data found for play_id {play_id}, skipping...")
                continue
                
            output_path = os.path.join(self.results_dir, f"distance_to_zone_{play_id}.mp4")
            
            # Perform object detection
            catcher_detections = self.object_detector.detect_objects(video_path, "catcher")
            glove_detections = self.object_detector.detect_objects(video_path, "glove")
            ball_detections = self.object_detector.detect_objects(video_path, "baseball")
            pitcher_detections = self.object_detector.detect_objects(video_path, "pitcher")
            hitter_detections = self.object_detector.detect_objects(video_path, "hitter", conf_threshold=0.3)
            
            # Find when ball reaches glove
            ball_glove_frame, ball_center, ball_detection = self.object_detector.find_ball_reaches_glove(
                video_path, glove_detections, ball_detections
            )
            
            # Get catcher position
            catcher_position = self.object_detector.get_catcher_position(catcher_detections, ball_glove_frame)
            
            # Find best hitter box
            hitter_box, hitter_frame, hitter_frame_idx = self.object_detector.find_best_hitter_box(
                video_path=video_path,
                hitter_detections=hitter_detections,
                catcher_position=catcher_position,
                frame_idx_start=max(0, ball_glove_frame - 120) if ball_glove_frame else 0,
                frame_search_range=120
            )
            
            # Detect pose
            hitter_keypoints = None
            hitter_pose_3d = None
            if hitter_box is not None and hitter_frame is not None:
                hitter_keypoints = self.pose_detector.detect_pose_in_box(
                    frame=hitter_frame,
                    box=hitter_box
                )
                
                hitter_pose_3d = self.pose_detector.detect_3d_pose(
                    frame=hitter_frame,
                    box=hitter_box
                )
            
            # Detect home plate
            homeplate_box, homeplate_confidence, homeplate_frame = self.object_detector.detect_homeplate(
                video_path, 
                reference_frame=ball_glove_frame
            )
            
            # Compute strike zone
            strike_zone, pixels_per_foot = self.strike_zone_calculator.compute_strike_zone(
                catcher_detections=catcher_detections, 
                pitch_data=pitch_data_row, 
                ball_glove_frame=ball_glove_frame, 
                video_path=video_path, 
                hitter_keypoints=hitter_keypoints,
                hitter_box=hitter_box,
                homeplate_box=homeplate_box,
                hitter_pose_3d=hitter_pose_3d
            )
            
            # Calculate distance to zone
            distance = None
            position = None
            closest_point = None
            
            if ball_center is not None and strike_zone is not None and pixels_per_foot is not None:
                distance_pixels, distance_inches, position, closest_point = self.distance_calculator.calculate_distance_to_zone(
                    ball_center, strike_zone, pixels_per_foot
                )
                distance = distance_inches

                if self.verbose:
                    print(f"Distance to zone: {distance:.2f} inches")
                    print(f"Position relative to zone: {position}")
            
            # Create annotated video
            if create_video and output_path and strike_zone is not None:
                self.visualizer.create_annotated_video(
                    video_path=video_path, 
                    output_path=output_path,
                    catcher_detections=catcher_detections, 
                    glove_detections=glove_detections,
                    ball_detections=ball_detections,
                    strike_zone=strike_zone,
                    ball_glove_frame=ball_glove_frame,
                    distance=distance,
                    position=position,
                    hitter_keypoints=hitter_keypoints,
                    hitter_frame_idx=hitter_frame_idx,
                    hitter_box=hitter_box,
                    homeplate_box=homeplate_box,
                    hitter_pose_3d=hitter_pose_3d,
                    closest_point=closest_point
                )
            
            # Collect results
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
            
            # Add detailed data for CSV
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
                print(f"Saved detailed results to {csv_path}")
                print(f"CSV contains {len(df)} rows with {len(df.columns)} columns of data")
        
        return dtoz_results