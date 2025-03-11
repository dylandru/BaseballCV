import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

class StrikeZoneCalculator:
    """
    Class for computing and calibrating strike zones in baseball videos.
    
    Utilizes various inputs such as home plate detection, player pose, and Statcast data.
    """
    
    def __init__(self, zone_vertical_adjustment=0.5, verbose=True):
        """
        Initialize the StrikeZoneCalculator.
        
        Args:
            zone_vertical_adjustment (float): Factor to adjust strike zone vertically
            verbose (bool): Whether to print detailed progress information
        """
        self.zone_vertical_adjustment = zone_vertical_adjustment
        self.verbose = verbose
    
    def compute_strike_zone(self, catcher_detections: List[Dict], pitch_data, 
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
            pitch_data: Pitch data containing strike zone information
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
            print("Computing strike zone using MLB official dimensions...")

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
                
                # Rest of implementation condensed for brevity but would follow the original structure
                
                # Calculate midpoint for top of strike zone logic would be here
                
                # Calculate adjustment for vertical position using configurable factor
                # Positive values move zone toward home plate (up in the image)
                adjustment = 0
                if elbow_y is not None and hip_y is not None:
                    # Use the configurable factor instead of fixed 0.5
                    adjustment = int((hip_y - elbow_y) * self.zone_vertical_adjustment)
                    if self.verbose:
                        print(f"Adjusting strike zone by {adjustment} pixels ({self.zone_vertical_adjustment:.2f} * elbow-to-hip distance)")
                
                # And so on...
            
            # Additional implementation condensed for brevity
            # This would include logic for MediaPipe 3D pose and fallback to Statcast data
            
            return (0, 0, 0, 0), 12.0  # Placeholder return, would include actual computed values
        
        # Additional implementation for fallbacks when home plate detection fails
        # This would include logic for using hitter's pose and catcher detection
        
        return None, None