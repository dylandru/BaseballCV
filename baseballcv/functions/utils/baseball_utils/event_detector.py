# In baseballcv/functions/utils/baseball_utils/event_detector.py (or your Colab cell)

import cv2
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from ultralytics import YOLO
from baseballcv.utilities import BaseballCVLogger, ProgressBar # Assuming these are defined/imported
from baseballcv.functions.load_tools import LoadTools # Assuming this is defined/imported

try:
    from moviepy.editor import VideoFileClip
    MoviePyAvailable = True
except ImportError:
    MoviePyAvailable = False

class EventDetector:
    def __init__(self,
                 primary_model_alias: str, # For ball/glove
                 pitcher_model_alias: Optional[str] = 'phc_detector', # For pitcher detection
                 logger: Optional[BaseballCVLogger] = None,
                 verbose: bool = True,
                 device: str = 'cpu',
                 confidence_ball: float = 0.3,
                 confidence_glove: float = 0.3,
                 confidence_pitcher: float = 0.5): # Confidence for pitcher detection

        self.logger = logger if logger else BaseballCVLogger.get_logger(self.__class__.__name__)
        self.verbose = verbose
        self.device = device
        self.load_tools = LoadTools()

        # Load primary model (for ball, glove)
        self.primary_model_path = self.load_tools.load_model(primary_model_alias)
        if not self.primary_model_path:
            raise ValueError(f"Failed to load primary model weights: {primary_model_alias}")
        self.primary_model = YOLO(self.primary_model_path)
        self.logger.info(f"EventDetector: Primary model '{primary_model_alias}' loaded for ball/glove detection.")

        self.ball_class_id = self._get_class_id_from_model(self.primary_model, ['baseball', 'ball'])
        self.glove_class_id = self._get_class_id_from_model(self.primary_model, ['glove'])

        # Load pitcher detection model
        self.pitcher_model = None
        if pitcher_model_alias:
            self.pitcher_model_path = self.load_tools.load_model(pitcher_model_alias)
            if self.pitcher_model_path:
                self.pitcher_model = YOLO(self.pitcher_model_path)
                self.pitcher_class_id = self._get_class_id_from_model(self.pitcher_model, ['pitcher'])
                self.logger.info(f"EventDetector: Pitcher model '{pitcher_model_alias}' loaded.")
                if self.pitcher_class_id is None:
                    self.logger.warning(f"Could not determine 'pitcher' class ID from model '{pitcher_model_alias}'. Detected classes: {self.pitcher_model.names}")
            else:
                self.logger.warning(f"Could not load pitcher model '{pitcher_model_alias}'. Pitcher specific events might not work.")
        
        self.confidence_ball = confidence_ball
        self.confidence_glove = confidence_glove
        self.confidence_pitcher = confidence_pitcher


    def _get_class_id_from_model(self, model: YOLO, target_names: List[str]) -> Optional[int]:
        if model and model.names:
            for class_id, name in model.names.items():
                if name.lower() in target_names:
                    return class_id
        self.logger.warning(f"Class ID for any of '{target_names}' not found in model. Names: {model.names if model else 'No model'}")
        return None

    def _get_detections_from_model(self, frame: np.ndarray, model: YOLO, target_class_id: Optional[int], min_confidence: float) -> List[Dict]:
        detections_list = []
        if not model or target_class_id is None:
            return detections_list
            
        results = model.predict(frame, device=self.device, verbose=False, conf=min_confidence)
        for res in results:
            for box in res.boxes:
                if int(box.cls[0].item()) == target_class_id:
                    detections_list.append({
                        "box": box.xyxy[0].cpu().numpy(),
                        "center": box.xywh[0][:2].cpu().numpy(), # x_center, y_center
                        "confidence": float(box.conf[0].item())
                    })
        # If multiple, pick the most confident or largest one (can be refined)
        if detections_list:
            return [max(detections_list, key=lambda x: x['confidence'])] # Return only the primary one
        return []

    def _iou(self, boxA_coords, boxB_coords): # Same as before
        boxA = [min(boxA_coords[0], boxA_coords[2]), min(boxA_coords[1], boxA_coords[3]),
                max(boxA_coords[0], boxA_coords[2]), max(boxA_coords[1], boxA_coords[3])]
        boxB = [min(boxB_coords[0], boxB_coords[2]), min(boxB_coords[1], boxB_coords[3]),
                max(boxB_coords[0], boxB_coords[2]), max(boxB_coords[1], boxB_coords[3])]
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        denominator = float(boxAArea + boxBArea - interArea)
        return interArea / denominator if denominator > 0 else 0

    def _find_pitch_flight_frames(self, video_path: str) -> Tuple[Optional[int], Optional[int]]:
        # This method remains largely the same, but uses _get_detections_from_model
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): self.logger.error(f"Cannot open video: {video_path}"); return None, None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: self.logger.error(f"Video has no frames: {video_path}"); cap.release(); return None, None

        release_frame, arrival_frame = None, None
        ball_detected_frames_indices = [] # Store frame indices where ball is seen
        last_ball_box = None
        
        self.logger.info(f"Scanning for pitch flight in '{os.path.basename(video_path)}'...")
        for frame_idx in ProgressBar(range(total_frames), desc="Pitch Flight Scan", total=total_frames):
            ret, frame = cap.read()
            if not ret: break

            ball_dets = self._get_detections_from_model(frame, self.primary_model, self.ball_class_id, self.confidence_ball)
            glove_dets = self._get_detections_from_model(frame, self.primary_model, self.glove_class_id, self.confidence_glove)
            
            current_frame_has_ball = bool(ball_dets)
            if current_frame_has_ball:
                ball_detected_frames_indices.append(frame_idx)
                last_ball_box = ball_dets[0]['box'] # Assuming primary ball if multiple
                
                if release_frame is None and len(ball_detected_frames_indices) > 5:
                     # Check if the last 5 ball detections are somewhat consecutive
                    if (ball_detected_frames_indices[-1] - ball_detected_frames_indices[-5]) < 10: # within 10 frames
                        release_frame = ball_detected_frames_indices[0]
                        self.logger.info(f"Potential ball release at frame: {release_frame}")

            if release_frame is not None and current_frame_has_ball and glove_dets:
                if self._iou(last_ball_box, glove_dets[0]['box']) > 0.05: # IoU threshold
                    arrival_frame = frame_idx # Keep updating to get the end of interaction

            if release_frame is not None and arrival_frame is not None and not current_frame_has_ball and last_ball_box is not None:
                if frame_idx > arrival_frame: # Ball disappeared AFTER last glove interaction
                    self.logger.info(f"Ball disappeared post-glove interaction, confirming arrival at frame: {arrival_frame}")
                    break
        cap.release()

        if release_frame is not None and arrival_frame is not None:
             self.logger.info(f"Final ball-glove interaction (arrival) at frame: {arrival_frame}")
        elif release_frame is not None and not arrival_frame and ball_detected_frames_indices:
            arrival_frame = ball_detected_frames_indices[-1]
            self.logger.info(f"No clear glove arrival; using last seen ball frame: {arrival_frame}")

        if not release_frame: self.logger.warning("Ball release not detected.")
        if not arrival_frame: self.logger.warning("Ball arrival at glove not detected.")
        return release_frame, arrival_frame

    def _find_pitcher_mechanic_frames(self, video_path: str, fps: float) -> Tuple[Optional[int], Optional[int]]:
        if not self.pitcher_model or self.pitcher_class_id is None:
            self.logger.error("Pitcher detection model not available or pitcher class ID not set. Cannot detect mechanic.")
            return None, None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): self.logger.error(f"Cannot open video: {video_path}"); return None, None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: self.logger.error(f"Video has no frames: {video_path}"); cap.release(); return None, None

        # First, get the ball release frame using the pitch flight logic
        # This serves as a crucial anchor, typically the end of the main throwing motion
        # We run a simplified scan here or assume it's passed if already computed.
        # For now, let's assume we re-scan or have it.
        # To avoid full re-scan, this method could accept release_frame as optional input.
        self.logger.info(f"Determining ball release frame to anchor pitcher mechanic...")
        # Note: This is a simplified call to get release. A more optimized system
        # might share frame processing passes.
        actual_ball_release_frame, _ = self._find_pitch_flight_frames(video_path)

        if actual_ball_release_frame is None:
            self.logger.warning("Could not determine ball release frame. Mechanic detection will be less reliable.")
            # Fallback: use a heuristic if no release detected e.g. middle of video (less ideal)
            # For now, we'll proceed and the mechanic might be based on pitcher activity alone.
        
        mechanic_start_frame = None
        mechanic_end_frame = actual_ball_release_frame # Tentatively set end to ball release

        pitcher_active_frames = [] # (frame_idx, pitcher_box_center_x)
        last_pitcher_box = None
        min_pitcher_motion_frames = int(fps * 0.5) # Pitcher needs to be active for at least 0.5 seconds
        
        self.logger.info(f"Scanning for pitcher mechanic in '{os.path.basename(video_path)}'...")
        # Rewind video capture to process for pitcher
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_idx in ProgressBar(range(total_frames), desc="Pitcher Mechanic Scan", total=total_frames):
            ret, frame = cap.read()
            if not ret: break

            pitcher_dets = self._get_detections_from_model(frame, self.pitcher_model, self.pitcher_class_id, self.confidence_pitcher)

            if pitcher_dets:
                current_pitcher_box = pitcher_dets[0]['box'] # Assuming primary pitcher
                pitcher_active_frames.append(frame_idx)
                
                if mechanic_start_frame is None:
                    # Heuristic for start of mechanic:
                    # Pitcher is present, and if we have a release frame, we are before it.
                    # More advanced: look for initiation of significant motion.
                    # Simple start: first consistent pitcher detection leading up to release (if known)
                    if len(pitcher_active_frames) > min_pitcher_motion_frames:
                         # If release frame is known, ensure we are before it or it's our first major segment
                        if actual_ball_release_frame is None or frame_idx < actual_ball_release_frame:
                            # Consider the start of this consistent detection sequence as mechanic start
                            if (pitcher_active_frames[-1] - pitcher_active_frames[-min_pitcher_motion_frames]) < min_pitcher_motion_frames + int(fps*0.2): #fairly contiguous
                                mechanic_start_frame = pitcher_active_frames[-min_pitcher_motion_frames]
                                self.logger.info(f"Potential pitcher mechanic start at frame: {mechanic_start_frame}")
                
                last_pitcher_box = current_pitcher_box
            else: # Pitcher not detected
                if mechanic_start_frame is not None and mechanic_end_frame is None and actual_ball_release_frame is None:
                    # If pitcher disappears after starting mechanic and no ball release to guide, end mechanic here
                    mechanic_end_frame = pitcher_active_frames[-1] if pitcher_active_frames else frame_idx -1
                    self.logger.info(f"Pitcher disappeared, setting mechanic end at {mechanic_end_frame}")
                    break # Stop if pitcher disappears after mechanic started and no release anchor

            # If we have a release frame, and we've passed it, and we have a start, we can stop.
            if actual_ball_release_frame is not None and mechanic_start_frame is not None and frame_idx > actual_ball_release_frame + int(fps*0.2): # allow a little past release
                break
        
        cap.release()

        if mechanic_start_frame and not mechanic_end_frame: # If loop finished but end not set by disappearance
            if actual_ball_release_frame:
                mechanic_end_frame = actual_ball_release_frame
            elif pitcher_active_frames: # Fallback if no release, use last pitcher activity
                mechanic_end_frame = pitcher_active_frames[-1]
            self.logger.info(f"Setting mechanic end (fallback/release): {mechanic_end_frame}")


        if not mechanic_start_frame: self.logger.warning("Pitcher mechanic start not detected.")
        if not mechanic_end_frame: self.logger.warning("Pitcher mechanic end not detected.")
        
        # Ensure start is before end
        if mechanic_start_frame and mechanic_end_frame and mechanic_start_frame >= mechanic_end_frame:
            self.logger.warning(f"Mechanic start ({mechanic_start_frame}) is not before end ({mechanic_end_frame}). Invalidating mechanic segment.")
            return None, None
            
        return mechanic_start_frame, mechanic_end_frame

    def _crop_video_segment_moviepy(self, video_path: str, start_seconds: float, end_seconds: float, output_segment_path: str):
        # This method remains the same
        if not MoviePyAvailable:
            self.logger.error("MoviePy library is not installed. Please install it via '!pip install moviepy'.")
            raise ImportError("MoviePy library not found.")
        try:
            os.makedirs(os.path.dirname(output_segment_path), exist_ok=True)
            self.logger.info(f"Attempting to crop: {os.path.basename(video_path)} from {start_seconds:.2f}s to {end_seconds:.2f}s -> {os.path.basename(output_segment_path)}")
            with VideoFileClip(video_path) as video:
                duration = video.duration
                start_seconds = max(0, float(start_seconds))
                end_seconds = min(float(duration), float(end_seconds))
                if start_seconds >= end_seconds:
                    self.logger.warning(f"Invalid time range for cropping: start {start_seconds}s, end {end_seconds}s (duration {duration}s). Skipping crop.")
                    return False
                subclip = video.subclip(start_seconds, end_seconds)
                subclip.write_videofile(output_segment_path, codec="libx264", audio_codec="aac",
                                        temp_audiofile=f'{output_segment_path}.m4a', remove_temp=True,
                                        logger=None)
            self.logger.info(f"Successfully cropped segment to {output_segment_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to crop video segment {output_segment_path} using MoviePy: {e}")
            temp_audio_path = f'{output_segment_path}.m4a';
            if os.path.exists(temp_audio_path):
                try: os.remove(temp_audio_path)
                except OSError: pass
            return False

    def _process_and_crop(self, video_path: str, output_path_dir: str, 
                          event_frames: Tuple[Optional[int], Optional[int]], 
                          padding_pre_frames: int, padding_post_frames: int, 
                          event_name_for_file: str) -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return {"error": f"Cannot open video {video_path}", "cropped_video_path": None}
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps == 0: fps = 30.0; self.logger.warning(f"FPS 0 for {video_path}, defaulting to 30.")

        start_event_frame, end_event_frame = event_frames

        if start_event_frame is None or end_event_frame is None:
            msg = f"Could not determine complete {event_name_for_file} event for {os.path.basename(video_path)}."
            self.logger.warning(msg)
            return {"error": msg, "cropped_video_path": None, "start_event_frame": start_event_frame, "end_event_frame": end_event_frame, "fps": fps}

        if end_event_frame <= start_event_frame:
            msg = f"Detected end_frame ({end_event_frame}) is not after start_frame ({start_event_frame}) for {event_name_for_file} in {os.path.basename(video_path)}."
            self.logger.warning(msg)
            return {"error": msg, "cropped_video_path": None, "start_event_frame": start_event_frame, "end_event_frame": end_event_frame, "fps": fps}

        start_crop_frame = max(0, start_event_frame - padding_pre_frames)
        end_crop_frame = min(total_frames_video - 1, end_event_frame + padding_post_frames)
        
        start_time_seconds = start_crop_frame / fps
        end_time_seconds = end_crop_frame / fps

        if start_time_seconds >= end_time_seconds:
            msg = f"Invalid crop times for {os.path.basename(video_path)} ({event_name_for_file}): start {start_time_seconds:.2f}s, end {end_time_seconds:.2f}s after padding."
            self.logger.error(msg)
            return {"error": msg, "cropped_video_path": None, "start_event_frame": start_event_frame, "end_event_frame": end_event_frame, "fps": fps}

        video_filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        base_output_filename = f"{event_name_for_file}_{video_filename_no_ext}.mp4"
        
        os.makedirs(output_path_dir, exist_ok=True)
        cropped_video_name = base_output_filename
        count = 1
        while os.path.exists(os.path.join(output_path_dir, cropped_video_name)):
            cropped_video_name = f"{os.path.splitext(base_output_filename)[0]}_{count}{os.path.splitext(base_output_filename)[1]}"
            count += 1
        cropped_video_full_path = os.path.join(output_path_dir, cropped_video_name)

        success = self._crop_video_segment_moviepy(video_path, start_time_seconds, end_time_seconds, cropped_video_full_path)
        
        if success:
            return {
                "cropped_video_path": cropped_video_full_path,
                f"{event_name_for_file}_start_frame": start_event_frame,
                f"{event_name_for_file}_end_frame": end_event_frame,
                "cropped_start_frame": start_crop_frame,
                "cropped_end_frame": end_crop_frame,
                "cropped_start_time_s": start_time_seconds,
                "cropped_end_time_s": end_time_seconds,
                "fps": fps
            }
        else:
            return {"error": f"Video cropping failed for {event_name_for_file}", "cropped_video_path": None, 
                    f"{event_name_for_file}_start_frame": start_event_frame, f"{event_name_for_file}_end_frame": end_event_frame, "fps": fps}

    def extract_pitch_flight_segment(self, video_path: str, output_path_dir: str,
                                     pre_release_padding_frames: int, post_arrival_padding_frames: int, **kwargs) -> Dict:
        flight_frames = self._find_pitch_flight_frames(video_path)
        return self._process_and_crop(video_path, output_path_dir, flight_frames,
                                      pre_release_padding_frames, post_arrival_padding_frames, "pitch_flight")

    def extract_pitcher_mechanic_segment(self, video_path: str, output_path_dir: str,
                                         pre_mechanic_padding_frames: int, post_mechanic_padding_frames: int, **kwargs) -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return {"error": f"Cannot open video {video_path}"}
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps == 0: fps = 30.0

        mechanic_frames = self._find_pitcher_mechanic_frames(video_path, fps)
        return self._process_and_crop(video_path, output_path_dir, mechanic_frames,
                                      pre_mechanic_padding_frames, post_mechanic_padding_frames, "pitcher_mechanic")