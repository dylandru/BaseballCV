import cv2
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any # Added Any
from ultralytics import YOLO
from baseballcv.utilities import BaseballCVLogger, ProgressBar
from baseballcv.functions.load_tools import LoadTools

try:
    from moviepy.editor import VideoFileClip
    MoviePyAvailable = True
except ImportError:
    MoviePyAvailable = False

# Attempt to import RFDETR, handle if not available or only when needed
try:
    from rfdetr import RFDETRBase # Assuming 'base' type for glove_tracking_rfd.
                                 # If you have RFDETRLarge models, you might need to import that too
                                 # and add logic to select between Base and Large.
except ImportError:
    RFDETRBase = None # Set to None if not installed. Will raise error later if RFDETR type is used.


class EventDetector:
    def __init__(self,
                 primary_model_alias: str,
                 primary_model_type: str, # NEW: e.g., 'YOLO' or 'RFDETR'
                 pitcher_model_alias: Optional[str] = 'phc_detector', # Default is 'phc_detector'
                 pitcher_model_type: Optional[str] = 'YOLO', # NEW: Default type for phc_detector is YOLO
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

        self.primary_model_type = primary_model_type.upper()
        self.pitcher_model_type = pitcher_model_type.upper()

        # --- Load Primary Model (for ball, glove) ---
        self.primary_model_path = self.load_tools.load_model(primary_model_alias, model_type=self.primary_model_type)
        if not self.primary_model_path:
            raise ValueError(f"Failed to load primary model weights: {primary_model_alias} (Type: {self.primary_model_type})")

        if self.primary_model_type == 'YOLO':
            self.primary_model_engine = YOLO(self.primary_model_path)
            self.primary_model_labels = self.primary_model_engine.names # dict: {id: name}
        elif self.primary_model_type == 'RFDETR':
            if RFDETRBase is None:
                raise ImportError("RFDETR library is not installed. Please install it via 'pip install rfdetr' or similar to use RFDETR models.")
            # Assuming 'base' type for 'glove_tracking_rfd'. This could be made configurable if you use RFDETRLarge.
            self.primary_model_engine = RFDETRBase(device=self.device, pretrain_weights=self.primary_model_path)
            # IMPORTANT: Labels for RFDETR must be known and match the training order.
            # You need to define these based on how 'glove_tracking_rfd' was trained.
            if primary_model_alias == 'glove_tracking_rfd':
                # Replace with the actual labels and order for your 'glove_tracking_rfd' model.
                # Example: {0: 'glove', 1: 'baseball', 2: 'homeplate'}
                self.primary_model_labels = {0: 'glove', 1: 'ball', 2: 'homeplate'} # CRITICAL: Ensure this matches your model
                self.logger.info(f"Using predefined labels for RFDETR model '{primary_model_alias}': {self.primary_model_labels}")
            else:
                # If you support other RFDETR models, you'll need to add their label definitions here
                # or implement a more general way to get labels for RFDETR models.
                self.logger.error(f"Labels for RFDETR model alias '{primary_model_alias}' are not explicitly defined in EventDetector.")
                raise ValueError(f"Labels not defined for RFDETR model: {primary_model_alias}")
        else:
            raise ValueError(f"Unsupported primary_model_type: {self.primary_model_type}")
        self.logger.info(f"EventDetector: Primary model '{primary_model_alias}' (Type: {self.primary_model_type}) loaded.")

        # Get class IDs for primary model
        self.ball_class_id = self._get_class_id_from_labels(self.primary_model_labels, ['baseball', 'ball'])
        self.glove_class_id = self._get_class_id_from_labels(self.primary_model_labels, ['glove'])
        if self.ball_class_id is None:
             self.logger.warning(f"Could not find 'ball' or 'baseball' class in primary model '{primary_model_alias}' labels.")
        if self.glove_class_id is None:
             self.logger.warning(f"Could not find 'glove' class in primary model '{primary_model_alias}' labels.")


        # --- Load Pitcher Detection Model ---
        self.pitcher_model_engine = None
        self.pitcher_model_labels = None
        self.pitcher_class_id = None # Initialize
        if pitcher_model_alias:
            self.pitcher_model_path = self.load_tools.load_model(pitcher_model_alias, model_type=self.pitcher_model_type)
            if self.pitcher_model_path:
                if self.pitcher_model_type == 'YOLO':
                    self.pitcher_model_engine = YOLO(self.pitcher_model_path)
                    self.pitcher_model_labels = self.pitcher_model_engine.names
                elif self.pitcher_model_type == 'RFDETR':
                    if RFDETRBase is None:
                        raise ImportError("RFDETR library is not installed. Please install it to use RFDETR pitcher models.")
                    # self.pitcher_model_engine = RFDETRBase(device=self.device, pretrain_weights=self.pitcher_model_path)
                    # self.pitcher_model_labels = {0: 'pitcher', ...} # Define these for your RFDETR pitcher model
                    self.logger.warning(f"RFDETR pitcher model '{pitcher_model_alias}' loading logic needs specific labels. Assuming default 'phc_detector' is YOLO if types mismatch.")
                    # Fallback for default 'phc_detector' if somehow type was set to RFDETR but alias is still phc_detector
                    if pitcher_model_alias == 'phc_detector' and self.pitcher_model_type == 'RFDETR':
                        self.logger.warning(f"Pitcher model alias is 'phc_detector' (default YOLO) but type is RFDETR. Reverting to YOLO type for 'phc_detector'.")
                        self.pitcher_model_type = 'YOLO' # Correcting type for known default
                        self.pitcher_model_engine = YOLO(self.pitcher_model_path)
                        self.pitcher_model_labels = self.pitcher_model_engine.names
                    elif pitcher_model_alias == 'phc_detector' and self.pitcher_model_type == 'YOLO': # Standard default
                         self.pitcher_model_engine = YOLO(self.pitcher_model_path)
                         self.pitcher_model_labels = self.pitcher_model_engine.names
                    else:
                        raise ValueError(f"Labels for RFDETR pitcher model alias '{pitcher_model_alias}' are not defined in EventDetector.")


                if self.pitcher_model_engine:
                    self.pitcher_class_id = self._get_class_id_from_labels(self.pitcher_model_labels, ['pitcher'])
                    if self.pitcher_class_id is None:
                        self.logger.warning(f"Could not find 'pitcher' class in pitcher model '{pitcher_model_alias}' labels. Detected classes: {self.pitcher_model_labels}")
                    self.logger.info(f"EventDetector: Pitcher model '{pitcher_model_alias}' (Type: {self.pitcher_model_type}) loaded.")
                else:
                    self.logger.warning(f"Could not load pitcher model engine for '{pitcher_model_alias}' with type '{self.pitcher_model_type}'.")
            else:
                self.logger.warning(f"Could not find path for pitcher model '{pitcher_model_alias}'. Pitcher specific events might not work.")
        
        self.confidence_ball = confidence_ball
        self.confidence_glove = confidence_glove
        self.confidence_pitcher = confidence_pitcher

    def _get_class_id_from_labels(self, model_labels: Optional[Dict[int, str]], target_names: List[str]) -> Optional[int]:
        """Gets the class ID for a target name from a model's label mapping."""
        if model_labels:
            for class_id, name in model_labels.items():
                if name.lower() in [tn.lower() for tn in target_names]: # Case-insensitive match
                    return class_id
        # self.logger.debug(f"Class ID for any of '{target_names}' not found in model labels. Labels: {model_labels}")
        return None

    def _get_detections_from_model(self, frame: np.ndarray, model_engine: Any, model_type: str, target_class_id: Optional[int], min_confidence: float) -> List[Dict]:
        detections_list = []
        if not model_engine: # Check if the engine itself is None
            self.logger.debug(f"Model engine is None for type {model_type}, cannot get detections.")
            return detections_list
        if target_class_id is None: # If no target_class_id, we can't filter for it.
            self.logger.debug(f"Target class ID is None, cannot filter detections for model type {model_type}.")
            return detections_list
            
        if model_type == 'YOLO':
            results = model_engine.predict(frame, device=self.device, verbose=False, conf=min_confidence)
            for res in results:
                for box_obj in res.boxes:
                    if int(box_obj.cls[0].item()) == target_class_id:
                        detections_list.append({
                            "box": box_obj.xyxy[0].cpu().numpy(),
                            "center": box_obj.xywh[0][:2].cpu().numpy(), 
                            "confidence": float(box_obj.conf[0].item())
                        })
        elif model_type == 'RFDETR':
            # model_engine here is the RFDETRBase/Large instance from the rfdetr package
            detections_sv = model_engine.predict(frame, threshold=min_confidence) # detections_sv is a supervision.Detections object

            if detections_sv is not None and len(detections_sv.xyxy) > 0 : # Ensure detections_sv is not None and has items
                for i in range(len(detections_sv.xyxy)):
                    if detections_sv.class_id[i] == target_class_id:
                        xyxy = detections_sv.xyxy[i]
                        # RFDETR's sv.Detections might not always populate confidence if the threshold is applied in predict
                        # If confidence array is None or shorter than class_id array, handle appropriately
                        conf = min_confidence # Default to min_confidence if not available per detection
                        if detections_sv.confidence is not None and i < len(detections_sv.confidence):
                            conf = detections_sv.confidence[i]
                        
                        x_center = (xyxy[0] + xyxy[2]) / 2
                        y_center = (xyxy[1] + xyxy[3]) / 2

                        detections_list.append({
                            "box": np.array(xyxy), 
                            "center": np.array([x_center, y_center]),
                            "confidence": float(conf)
                        })
        else:
            self.logger.error(f"Unsupported model_type in _get_detections_from_model: {model_type}")
            return []

        # Pick the most confident one if multiple detections of the target class
        if detections_list: 
            return [max(detections_list, key=lambda x: x['confidence'])]
        return []

    def _iou(self, boxA_coords, boxB_coords):
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
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): self.logger.error(f"Cannot open video: {video_path}"); return None, None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: self.logger.error(f"Video has no frames: {video_path}"); cap.release(); return None, None

        release_frame, arrival_frame = None, None
        ball_detected_frames_indices = [] 
        last_ball_box = None
        
        self.logger.info(f"Scanning for pitch flight in '{os.path.basename(video_path)}'...")
        for frame_idx in ProgressBar(range(total_frames), desc="Pitch Flight Scan", total=total_frames):
            ret, frame = cap.read()
            if not ret: break

            ball_dets = self._get_detections_from_model(frame, self.primary_model_engine, self.primary_model_type, self.ball_class_id, self.confidence_ball)
            glove_dets = self._get_detections_from_model(frame, self.primary_model_engine, self.primary_model_type, self.glove_class_id, self.confidence_glove)
            
            current_frame_has_ball = bool(ball_dets)
            if current_frame_has_ball:
                ball_detected_frames_indices.append(frame_idx)
                last_ball_box = ball_dets[0]['box'] 
                
                if release_frame is None and len(ball_detected_frames_indices) > 5:
                    if (ball_detected_frames_indices[-1] - ball_detected_frames_indices[-5]) < 10: 
                        release_frame = ball_detected_frames_indices[0] # Potential release is the first frame in this consistent sequence
                        self.logger.info(f"Potential ball release at frame: {release_frame}")

            if release_frame is not None and current_frame_has_ball and glove_dets and last_ball_box is not None:
                if self._iou(last_ball_box, glove_dets[0]['box']) > 0.05: 
                    arrival_frame = frame_idx 

            if release_frame is not None and arrival_frame is not None and not current_frame_has_ball:
                # Ball disappeared AFTER last registered ball-glove interaction
                if frame_idx > arrival_frame:
                    self.logger.info(f"Ball disappeared post-glove interaction, confirming arrival at last interaction frame: {arrival_frame}")
                    break 
        cap.release()

        if release_frame is not None and arrival_frame is not None:
             self.logger.info(f"Final ball-glove interaction (arrival) at frame: {arrival_frame}")
        elif release_frame is not None and not arrival_frame and ball_detected_frames_indices:
            # If release was detected, but no clear glove arrival, use the last frame where the ball was seen.
            arrival_frame = ball_detected_frames_indices[-1]
            self.logger.info(f"No clear ball-glove interaction detected for arrival; using last seen ball frame: {arrival_frame}")

        if not release_frame: self.logger.warning("Ball release not detected.")
        if not arrival_frame: self.logger.warning("Ball arrival at glove not detected.")
        return release_frame, arrival_frame

    def _find_pitcher_mechanic_frames(self, video_path: str, fps: float) -> Tuple[Optional[int], Optional[int]]:
        if not self.pitcher_model_engine or self.pitcher_class_id is None:
            self.logger.error("Pitcher detection model not available or pitcher class ID not set. Cannot detect mechanic.")
            return None, None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): self.logger.error(f"Cannot open video: {video_path}"); return None, None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: self.logger.error(f"Video has no frames: {video_path}"); cap.release(); return None, None

        # Determine ball release frame to anchor pitcher mechanic
        # This uses the primary model (ball/glove)
        self.logger.info(f"Determining ball release frame to anchor pitcher mechanic using primary model ({self.primary_model_type})...")
        actual_ball_release_frame, _ = self._find_pitch_flight_frames(video_path) # Use existing method

        if actual_ball_release_frame is None:
            self.logger.warning("Could not determine ball release frame. Mechanic detection will be less reliable.")
        
        mechanic_start_frame = None
        mechanic_end_frame = actual_ball_release_frame # Tentatively set end to ball release if available
        
        pitcher_active_frames_indices = [] 
        min_pitcher_motion_frames = int(fps * 0.5) 
        
        self.logger.info(f"Scanning for pitcher mechanic in '{os.path.basename(video_path)}' using pitcher model ({self.pitcher_model_type})...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind

        for frame_idx in ProgressBar(range(total_frames), desc="Pitcher Mechanic Scan", total=total_frames):
            ret, frame = cap.read()
            if not ret: break

            pitcher_dets = self._get_detections_from_model(frame, self.pitcher_model_engine, self.pitcher_model_type, self.pitcher_class_id, self.confidence_pitcher)

            if pitcher_dets:
                pitcher_active_frames_indices.append(frame_idx)
                
                if mechanic_start_frame is None:
                    if len(pitcher_active_frames_indices) >= min_pitcher_motion_frames:
                         # Check for contiguous presence
                        if (pitcher_active_frames_indices[-1] - pitcher_active_frames_indices[-min_pitcher_motion_frames]) < (min_pitcher_motion_frames + int(fps*0.2)):
                            # If release frame is known, ensure we are before it or it's our first major segment
                            if actual_ball_release_frame is None or frame_idx < actual_ball_release_frame:
                                mechanic_start_frame = pitcher_active_frames_indices[-min_pitcher_motion_frames]
                                self.logger.info(f"Potential pitcher mechanic start at frame: {mechanic_start_frame}")
                
            elif mechanic_start_frame is not None and mechanic_end_frame is None: # Pitcher disappeared
                # If pitcher disappears after starting mechanic and no ball release to guide, end mechanic here
                if actual_ball_release_frame is None:
                    mechanic_end_frame = pitcher_active_frames_indices[-1] if pitcher_active_frames_indices else frame_idx -1
                    self.logger.info(f"Pitcher disappeared, setting mechanic end at {mechanic_end_frame}")
                    break 

            # If we have a release frame, and we've passed it by a margin, and we have a start, we can stop.
            if actual_ball_release_frame is not None and mechanic_start_frame is not None and frame_idx > actual_ball_release_frame + int(fps*0.2): 
                break
        
        cap.release()

        # Refine mechanic_end_frame
        if mechanic_start_frame and not mechanic_end_frame: # If loop finished but end not set
            if actual_ball_release_frame:
                mechanic_end_frame = actual_ball_release_frame
            elif pitcher_active_frames_indices: # Fallback if no release, use last pitcher activity
                mechanic_end_frame = pitcher_active_frames_indices[-1]
            self.logger.info(f"Mechanic end set to (fallback/release based): {mechanic_end_frame}")


        if not mechanic_start_frame: self.logger.warning("Pitcher mechanic start not detected.")
        if not mechanic_end_frame: self.logger.warning("Pitcher mechanic end not detected.")
        
        if mechanic_start_frame and mechanic_end_frame and mechanic_start_frame >= mechanic_end_frame:
            self.logger.warning(f"Mechanic start ({mechanic_start_frame}) is not before end ({mechanic_end_frame}). Invalidating segment.")
            return None, None
            
        return mechanic_start_frame, mechanic_end_frame

    def _crop_video_segment_moviepy(self, video_path: str, start_seconds: float, end_seconds: float, output_segment_path: str):
        if not MoviePyAvailable:
            self.logger.error("MoviePy library is not installed. Please install it via '!pip install moviepy'.")
            raise ImportError("MoviePy library not found.")
        try:
            # Ensure output_segment_path's directory exists
            os.makedirs(os.path.dirname(output_segment_path), exist_ok=True)

            self.logger.info(f"Attempting to crop: {os.path.basename(video_path)} from {start_seconds:.2f}s to {end_seconds:.2f}s -> {os.path.basename(output_segment_path)}")
            with VideoFileClip(video_path) as video:
                duration = video.duration
                # Ensure start_seconds and end_seconds are floats and valid
                start_seconds = max(0, float(start_seconds))
                end_seconds = min(float(duration), float(end_seconds))

                if start_seconds >= end_seconds:
                    self.logger.warning(f"Invalid time range for cropping: start {start_seconds}s, end {end_seconds}s (duration {duration}s). Original video duration is {duration}s. Skipping crop for {os.path.basename(output_segment_path)}.")
                    return False # Indicates failure or skip

                subclip = video.subclip(start_seconds, end_seconds)
                # Use a unique temp audio file name per process or timestamp if in parallel
                temp_audio_filename = f"{os.path.splitext(os.path.basename(output_segment_path))[0]}_temp_audio_{os.getpid()}.m4a"
                temp_audiofile_path = os.path.join(os.path.dirname(output_segment_path), temp_audio_filename)

                subclip.write_videofile(output_segment_path, codec="libx264", audio_codec="aac",
                                        temp_audiofile=temp_audiofile_path, remove_temp=True,
                                        logger=None) # MoviePy's default logger can be noisy
            self.logger.info(f"Successfully cropped segment to {output_segment_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to crop video segment {output_segment_path} using MoviePy: {e}")
            # Attempt to clean up temp audio file if it exists and an error occurred
            temp_audio_filename_on_error = f"{os.path.splitext(os.path.basename(output_segment_path))[0]}_temp_audio_{os.getpid()}.m4a"
            temp_audiofile_path_on_error = os.path.join(os.path.dirname(output_segment_path), temp_audio_filename_on_error)
            if os.path.exists(temp_audiofile_path_on_error):
                try:
                    os.remove(temp_audiofile_path_on_error)
                except OSError:
                    self.logger.warning(f"Could not remove temporary audio file on error: {temp_audiofile_path_on_error}")
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

        if fps is None or fps == 0: 
            self.logger.warning(f"FPS for {os.path.basename(video_path)} is {fps}. Defaulting to 30 FPS. This might affect crop accuracy.")
            fps = 30.0 
        if total_frames_video == 0:
            return {"error": f"Video {os.path.basename(video_path)} has 0 frames.", "cropped_video_path": None}


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
        end_crop_frame = min(total_frames_video -1 , end_event_frame + padding_post_frames) # Ensure not exceeding total_frames
        
        # Ensure end_crop_frame is not beyond the video length
        if end_crop_frame >= total_frames_video:
            self.logger.warning(f"Calculated end_crop_frame ({end_crop_frame}) exceeds total video frames ({total_frames_video-1}). Adjusting to video end.")
            end_crop_frame = total_frames_video - 1
            
        start_time_seconds = start_crop_frame / fps
        end_time_seconds = end_crop_frame / fps

        if start_time_seconds >= end_time_seconds:
            msg = (f"Invalid crop times for {os.path.basename(video_path)} ({event_name_for_file}): "
                   f"start {start_time_seconds:.2f}s (frame {start_crop_frame}), end {end_time_seconds:.2f}s (frame {end_crop_frame}) after padding. "
                   f"Original event frames: [{start_event_frame}-{end_event_frame}]. Video FPS: {fps}, Total Frames: {total_frames_video-1}.")
            self.logger.error(msg)
            return {"error": msg, "cropped_video_path": None, "start_event_frame": start_event_frame, "end_event_frame": end_event_frame, "fps": fps}


        video_filename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        base_output_filename = f"{event_name_for_file}_{video_filename_no_ext}.mp4"
        
        os.makedirs(output_path_dir, exist_ok=True)
        
        # Ensure unique filename for output, appending a counter if necessary
        cropped_video_name_candidate = base_output_filename
        count = 1
        while os.path.exists(os.path.join(output_path_dir, cropped_video_name_candidate)):
            cropped_video_name_candidate = f"{os.path.splitext(base_output_filename)[0]}_{count}{os.path.splitext(base_output_filename)[1]}"
            count += 1
        cropped_video_full_path = os.path.join(output_path_dir, cropped_video_name_candidate)


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
            return {"error": f"Video cropping failed for {event_name_for_file} on {os.path.basename(video_path)}", 
                    "cropped_video_path": None, 
                    f"{event_name_for_file}_start_frame": start_event_frame, 
                    f"{event_name_for_file}_end_frame": end_event_frame, 
                    "fps": fps}

    def extract_pitch_flight_segment(self, video_path: str, output_path_dir: str,
                                     pre_release_padding_frames: int, post_arrival_padding_frames: int, **kwargs) -> Dict:
        flight_frames = self._find_pitch_flight_frames(video_path)
        return self._process_and_crop(video_path, output_path_dir, flight_frames,
                                      pre_release_padding_frames, post_arrival_padding_frames, "pitch_flight")

    def extract_pitcher_mechanic_segment(self, video_path: str, output_path_dir: str,
                                         pre_mechanic_padding_frames: int, post_mechanic_padding_frames: int, **kwargs) -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return {"error": f"Cannot open video {video_path}", "cropped_video_path": None}
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps is None or fps == 0: 
            self.logger.warning(f"FPS for {os.path.basename(video_path)} is {fps}. Defaulting to 30 FPS for mechanic detection. This might affect accuracy.")
            fps = 30.0

        mechanic_frames = self._find_pitcher_mechanic_frames(video_path, fps)
        return self._process_and_crop(video_path, output_path_dir, mechanic_frames,
                                      pre_mechanic_padding_frames, post_mechanic_padding_frames, "pitcher_mechanic")