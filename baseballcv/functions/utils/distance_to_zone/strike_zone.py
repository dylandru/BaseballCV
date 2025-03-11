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
                                print(f"Adjusting strike zone by {adjustment} pixels ({self.zone_vertical_adjustment:.2f} * elbow-to-hip distance)")
                        
                        # Recalculate calibration based on zone height and Statcast data
                        if self.verbose:
                            print(f"Using hitter pose to refine strike zone height: {zone_height_pixels}px")
                        
                        # Center the zone horizontally
                        zone_left_x = int(plate_center_x - (zone_width_pixels / 2))
                        zone_right_x = int(plate_center_x + (zone_width_pixels / 2))
                        
                        # Apply the vertical adjustment - move zone up (closer to home plate) if positive factor
                        if adjustment != 0:
                            zone_top_y -= adjustment
                            zone_bottom_y -= adjustment
                        
                        strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
                        
                        if self.verbose:
                            print(f"Strike zone from home plate and pose: {strike_zone}")
                        
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
                        left_knee_y = int(left_knee.y * video_frame_height) + offset_y
                        right_knee_y = int(right_knee.y * video_frame_height) + offset_y
                        knee_y = min(left_knee_y, right_knee_y)
                    elif left_knee.visibility > 0.3:
                        knee_y = int(left_knee.y * video_frame_height) + offset_y
                    elif right_knee.visibility > 0.3:
                        knee_y = int(right_knee.y * video_frame_height) + offset_y
                    
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
                    
                    # Get video frame shape for coordinate calculations
                    cap = cv2.VideoCapture(video_path)
                    ret, frame = cap.read()
                    video_frame_height = frame.shape[0] if ret else 720
                    video_frame_width = frame.shape[1] if ret else 1280
                    cap.release()
                    
                    if left_shoulder.visibility > 0.3 and right_shoulder.visibility > 0.3:
                        left_shoulder_y = int(left_shoulder.y * video_frame_height) + offset_y
                        right_shoulder_y = int(right_shoulder.y * video_frame_height) + offset_y
                        shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                    elif left_shoulder.visibility > 0.3:
                        shoulder_y = int(left_shoulder.y * video_frame_height) + offset_y
                    elif right_shoulder.visibility > 0.3:
                        shoulder_y = int(right_shoulder.y * video_frame_height) + offset_y
                    
                    if left_hip.visibility > 0.3 and right_hip.visibility > 0.3:
                        left_hip_y = int(left_hip.y * video_frame_height) + offset_y
                        right_hip_y = int(right_hip.y * video_frame_height) + offset_y
                        hip_y = (left_hip_y + right_hip_y) / 2
                    elif left_hip.visibility > 0.3:
                        hip_y = int(left_hip.y * video_frame_height) + offset_y
                    elif right_hip.visibility > 0.3:
                        hip_y = int(right_hip.y * video_frame_height) + offset_y
                    
                    if left_elbow.visibility > 0.3 and right_elbow.visibility > 0.3:
                        left_elbow_y = int(left_elbow.y * video_frame_height) + offset_y
                        right_elbow_y = int(right_elbow.y * video_frame_height) + offset_y
                        elbow_y = (left_elbow_y + right_elbow_y) / 2
                    elif left_elbow.visibility > 0.3:
                        elbow_y = int(left_elbow.y * video_frame_height) + offset_y
                    elif right_elbow.visibility > 0.3:
                        elbow_y = int(right_elbow.y * video_frame_height) + offset_y
                    
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
                                    print(f"Adjusting strike zone by {adjustment} pixels ({self.zone_vertical_adjustment:.2f} * elbow-to-hip distance)")
                            
                            # Center the zone horizontally
                            zone_left_x = int(plate_center_x - (zone_width_pixels / 2))
                            zone_right_x = int(plate_center_x + (zone_width_pixels / 2))
                            
                            # Apply the vertical adjustment
                            if adjustment != 0:
                                zone_top_y -= adjustment
                                zone_bottom_y -= adjustment
                            
                            strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
                            
                            if self.verbose:
                                print(f"Strike zone from home plate and MediaPipe 3D pose: {strike_zone}")
                            
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
                print(f"Strike zone computed using home plate: {strike_zone}")
            
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
                        print(f"Adjusting strike zone by {adjustment} pixels ({self.zone_vertical_adjustment:.2f} * elbow-to-hip distance)")
                
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
                    print(f"Strike zone computed using hitter's pose: {strike_zone}")
                
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
                
                # Get video dimensions
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
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
                    print(f"Strike zone computed using catcher (fallback): {strike_zone}")
                
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
        
        # Apply adjustment based on configurable factor
        adjustment = int(zone_height * 0.15 * self.zone_vertical_adjustment)
        if adjustment != 0:
            zone_top_y -= adjustment
            zone_bottom_y -= adjustment
        
        strike_zone = (zone_left_x, zone_top_y, zone_right_x, zone_bottom_y)
        
        if self.verbose:
            print(f"Using estimated strike zone (last resort): {strike_zone}")
        
        return strike_zone, pixels_per_foot