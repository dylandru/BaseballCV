import cv2
import numpy as np
import os
from typing import Dict, Optional, Tuple
import io
from contextlib import redirect_stdout
from ultralytics import YOLO

class PoseDetector:
    """
    Class for detecting player poses in baseball videos.
    
    Handles both 2D pose detection with YOLO and 3D pose detection with MediaPipe.
    """
    
    def __init__(self, mp_pose, mp_drawing, mp_drawing_styles, catcher_model, verbose=True):
        """
        Initialize the PoseDetector.
        
        Args:
            mp_pose: MediaPipe pose solution
            mp_drawing: MediaPipe drawing utilities
            mp_drawing_styles: MediaPipe drawing styles
            catcher_model: YOLO model for detecting players (includes pose capabilities)
            verbose: Whether to print detailed progress information
        """
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        self.catcher_model = catcher_model
        self.verbose = verbose
    
    def detect_pose_in_box(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
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
            hitter_results = pose_model.predict(hitter_region, verbose=False, conf=0.3)
        
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
        
        # Shift keypoints back to full frame coordinates
        for i in range(keypoints.shape[0]):
            keypoints[i, 0] += x1_margin  # Add x offset
            keypoints[i, 1] += y1_margin  # Add y offset
        
        if self.verbose:
            print("Successfully detected valid pose for hitter")
        
        return keypoints
    
    def detect_3d_pose(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[Dict]:
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