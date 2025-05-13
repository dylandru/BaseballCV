# In baseballcv/functions/utils/baseball_utils/event_detector.py

import cv2
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO # Assuming YOLO for ball/glove detection
from baseballcv.utilities import BaseballCVLogger, ProgressBar
from baseballcv.functions.load_tools import LoadTools

# Attempt to import MoviePy, provide instructions if missing
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    MoviePyAvailable = False
    # Log this information or raise it when _crop_video_segment_moviepy is called
else:
    MoviePyAvailable = True

class EventDetector:
    def __init__(self,
                 model_alias: str,
                 logger: Optional[BaseballCVLogger] = None,
                 verbose: bool = True,
                 device: str = 'cpu',
                 confidence_ball: float = 0.3,
                 confidence_glove: float = 0.3):
        self.logger = logger if logger else BaseballCVLogger.get_logger(self.__class__.__name__)
        self.verbose = verbose
        self.device = device
        self.load_tools = LoadTools()
        
        # Ensure the model alias is appropriate (e.g., 'glove_tracking' which detects ball and glove)
        self.model_path = self.load_tools.load_model(model_alias)
        self.model = YOLO(self.model_path)
        self.logger.info(f"EventDetector initialized with model: {model_alias} on device: {self.device}")

        self.confidence_ball = confidence_ball
        self.confidence_glove = confidence_glove

        # Get class IDs for ball and glove from the loaded model
        self.ball_class_id = None
        self.glove_class_id = None
        if self.model.names:
            for class_id, name in self.model.names.items():
                if 'baseball' in name.lower() or 'ball' in name.lower():
                    self.ball_class_id = class_id
                elif 'glove' in name.lower():
                    self.glove_class_id = class_id
        
        if self.ball_class_id is None:
            self.logger.warning("Could not determine 'ball' class ID from the model.")
        if self.glove_class_id is None:
            self.logger.warning("Could not determine 'glove' class ID from the model.")


    def _get_detections(self, frame: np.ndarray) -> Tuple[Optional[Dict], List[Dict]]:
        """Detects ball and glove in a single frame."""
        ball_detection = None
        glove_detections = [] # Could be multiple gloves, usually we care about one
        
        results = self.model.predict(frame, device=self.device, verbose=False, conf=0.1) # Lower initial conf for processing

        for res in results:
            for box in res.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                if class_id == self.ball_class_id and confidence >= self.confidence_ball:
                    current_ball_box = box.xyxy[0].cpu().numpy()
                    # If multiple balls, take the most confident or largest one (simple approach: first one above conf)
                    if ball_detection is None or confidence > ball_detection['confidence']:
                        ball_detection = {"box": current_ball_box, "center": box.xywh[0][:2].cpu().numpy(), "confidence": confidence}
                
                elif class_id == self.glove_class_id and confidence >= self.confidence_glove:
                    glove_detections.append({"box": box.xyxy[0].cpu().numpy(), "center": box.xywh[0][:2].cpu().numpy(), "confidence": confidence})
        
        # If multiple gloves, pick the most confident one for simplicity or one closest to expected catch zone
        if glove_detections:
            # Simple: highest confidence - can be refined
            primary_glove = max(glove_detections, key=lambda x: x['confidence'])
            return ball_detection, [primary_glove] 
        
        return ball_detection, []


    def _iou(self, boxA, boxB):
        """Calculate Intersection over Union (IoU) of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
        return iou

    def _find_pitch_flight_frames(self, video_path: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Identifies the frame of ball release and arrival at the glove.
        Returns (release_frame, arrival_frame)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Cannot open video: {video_path}")
            return None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        release_frame = None
        arrival_frame = None
        
        ball_positions_over_time = [] # Store (frame_idx, ball_center_x, ball_center_y)
        
        self.logger.info("Scanning video for pitch flight...")
        for frame_idx in ProgressBar(range(total_frames), desc="Analyzing Frames"):
            ret, frame = cap.read()
            if not ret:
                break

            ball_det, glove_dets = self._get_detections(frame)

            if ball_det:
                ball_positions_over_time.append((frame_idx, ball_det['center'][0], ball_det['center'][1]))
                
                # Simplified Release Detection: First time ball is confidently detected.
                # This assumes video starts near the pitching action.
                if release_frame is None:
                    # Add a small delay to ensure it's not just pitcher holding the ball
                    # This heuristic needs testing and refinement.
                    if len(ball_positions_over_time) > 3: # e.g. ball detected for 3 consecutive frames
                         # Check if ball is moving (e.g. away from initial position)
                        if len(ball_positions_over_time) > 5: # Need a few points to see movement
                            initial_x = ball_positions_over_time[0][1]
                            initial_y = ball_positions_over_time[0][2]
                            current_x = ball_positions_over_time[-1][1]
                            current_y = ball_positions_over_time[-1][2]
                            # Basic check for displacement (can be improved with velocity checks)
                            if np.sqrt((current_x - initial_x)**2 + (current_y - initial_y)**2) > 10: # Arbitrary pixel threshold
                                release_frame = ball_positions_over_time[0][0] # Mark release at the start of this movement
                                self.logger.info(f"Potential ball release detected at frame: {release_frame}")


                # Arrival Detection
                if glove_dets:
                    glove_box = glove_dets[0]['box'] # Assuming one primary glove
                    # Check if ball is near or overlapping with the glove
                    # Using IoU or distance between centers
                    if self._iou(ball_det['box'], glove_box) > 0.1: # Intersection threshold
                        arrival_frame = frame_idx # Candidate for arrival
                        # We typically want the *last* frame of contact, so we might overwrite this.
                        # For simplicity now, first contact after release.
                        if release_frame is not None and frame_idx > release_frame : # Ensure arrival is after release
                             self.logger.info(f"Potential ball arrival at glove detected at frame: {arrival_frame}")
                             # To get the *end* of flight, we'd continue searching or look for ball disappearing.
                             # For now, let's take this first clear contact post-release.


        cap.release()

        # Refine arrival_frame: often it's better to take the last detected interaction
        # if arrival_frame was set multiple times.
        # For this simplified version, we are taking the first significant interaction post "release".
        # More advanced: track the ball into the glove and find the frame where its motion stops or it's occluded by glove.
        
        if release_frame and arrival_frame and arrival_frame <= release_frame:
            self.logger.warning("Arrival detected before or at release, resetting arrival. Needs better release logic.")
            arrival_frame = None # Invalid state, might need to search for arrival again after release_frame

        # If arrival_frame is still None but release was found, maybe the ball was missed or went out of frame
        # Default to end of detected ball flight if no clear catch
        if release_frame is not None and arrival_frame is None and ball_positions_over_time:
             arrival_frame = ball_positions_over_time[-1][0]
             self.logger.info(f"No clear glove arrival, setting arrival to last seen ball frame: {arrival_frame}")


        if not release_frame: self.logger.warning("Could not detect ball release.")
        if not arrival_frame: self.logger.warning("Could not detect ball arrival at glove.")
        
        return release_frame, arrival_frame

    def _crop_video_segment_moviepy(self, video_path: str, start_seconds: float, end_seconds: float, output_segment_path: str):
        """Crops a video segment using MoviePy."""
        if not MoviePyAvailable:
            self.logger.error("MoviePy library is not installed. Please install it with 'pip install moviepy' to use this feature.")
            raise ImportError("MoviePy library not found.")
        
        try:
            os.makedirs(os.path.dirname(output_segment_path), exist_ok=True)
            with VideoFileClip(video_path) as video:
                # Ensure start_seconds and end_seconds are within video duration
                duration = video.duration
                start_seconds = max(0, start_seconds)
                end_seconds = min(duration, end_seconds)

                if start_seconds >= end_seconds:
                    self.logger.warning(f"Invalid time range for cropping: start {start_seconds}s, end {end_seconds}s. Skipping crop.")
                    return False
                
                subclip = video.subclip(start_seconds, end_seconds)
                subclip.write_videofile(output_segment_path, codec="libx264", audio_codec="aac", temp_audiofile=f'{output_segment_path}.m4a', remove_temp=True, logger=None if self.verbose else 'bar') # Use logger=None to suppress MoviePy's verbose output
            self.logger.info(f"Successfully cropped segment to {output_segment_path} from {start_seconds:.2f}s to {end_seconds:.2f}s")
            return True
        except Exception as e:
            self.logger.error(f"Failed to crop video segment {output_segment_path} using MoviePy: {e}")
            # Attempt to clean up temporary audio file if it exists
            temp_audio_path = f'{output_segment_path}.m4a'
            if os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except OSError:
                    pass # Ignore if removal fails
            return False


    def extract_pitch_flight_segment(self,
                                     video_path: str,
                                     output_path: str,
                                     pre_release_padding_frames: int,
                                     post_arrival_padding_frames: int,
                                     **kwargs) -> Dict:
        """
        Main method to extract the pitched ball flight segment from a video.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Input video not found or cannot be opened: {video_path}")
            return {"error": "Input video not found", "cropped_video_path": None, "release_frame": None, "arrival_frame": None}
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if fps == 0:
            self.logger.error(f"Could not determine FPS for video: {video_path}. Assuming 30 FPS.")
            fps = 30.0 # Default FPS if not readable

        release_frame, arrival_frame = self._find_pitch_flight_frames(video_path)

        if release_frame is None or arrival_frame is None:
            self.logger.warning(f"Could not determine complete pitch flight for {video_path}.")
            return {"error": "Could not determine complete pitch flight", "cropped_video_path": None, "release_frame": release_frame, "arrival_frame": arrival_frame}

        # Apply padding
        start_crop_frame = max(0, release_frame - pre_release_padding_frames)
        end_crop_frame = min(total_frames_video -1 , arrival_frame + post_arrival_padding_frames)
        
        start_time_seconds = start_crop_frame / fps
        end_time_seconds = end_crop_frame / fps

        if start_time_seconds >= end_time_seconds:
            self.logger.error(f"Calculated invalid crop times for {video_path}: start {start_time_seconds}s, end {end_time_seconds}s after padding.")
            return {"error": "Invalid crop time range after padding", "cropped_video_path": None, "release_frame": release_frame, "arrival_frame": arrival_frame}

        video_filename = os.path.basename(video_path)
        # Sanitize filename from output_path if it's a full path, or use it as directory
        if os.path.splitext(output_path)[1]: # If output_path looks like a file
            output_dir = os.path.dirname(output_path)
            base_output_filename = os.path.basename(output_path)
        else: # If output_path is a directory
            output_dir = output_path
            base_output_filename = f"flight_{os.path.splitext(video_filename)[0]}.mp4"
        
        os.makedirs(output_dir, exist_ok=True)
        cropped_video_full_path = os.path.join(output_dir, base_output_filename)

        success = self._crop_video_segment_moviepy(video_path, start_time_seconds, end_time_seconds, cropped_video_full_path)
        
        if success:
            return {
                "cropped_video_path": cropped_video_full_path,
                "release_frame": release_frame,
                "arrival_frame": arrival_frame,
                "cropped_start_frame": start_crop_frame,
                "cropped_end_frame": end_crop_frame,
                "fps": fps
            }
        else:
            return {
                "error": "Video cropping failed",
                "cropped_video_path": None,
                "release_frame": release_frame,
                "arrival_frame": arrival_frame,
                "fps": fps
            }