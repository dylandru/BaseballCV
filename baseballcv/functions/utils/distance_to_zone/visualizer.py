import cv2
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

class VideoAnnotator:
    """
    Class for creating annotated videos from baseball analysis.
    
    Visualizes detections, strike zones, and distances.
    """
    
    def __init__(self, mp_drawing, mp_drawing_styles, verbose=True):
        """
        Initialize the VideoAnnotator.
        
        Args:
            mp_drawing: MediaPipe drawing utilities
            mp_drawing_styles: MediaPipe drawing styles
            verbose (bool): Whether to print detailed progress information
        """
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        self.verbose = verbose
    
    def create_annotated_video(
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
            print(f"Creating annotated video: {output_path}")
        
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
            
        for frame_idx in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
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
        try:
            final_command = f"ffmpeg -y -i {temp_output_path} -c:v libx264 -preset medium -crf 23 {output_path}"
            os.system(final_command)
            
            # Remove the temporary file
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        except Exception as e:
            if self.verbose:
                print(f"Error converting video with ffmpeg: {e}")
                print(f"Using original output file: {temp_output_path}")
            # Rename temp file to final name if conversion fails
            if os.path.exists(temp_output_path):
                os.rename(temp_output_path, output_path)
        
        if self.verbose:
            print(f"Video saved to {output_path}")
        
        return output_path