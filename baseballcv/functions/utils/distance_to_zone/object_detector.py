import cv2
import numpy as np
import os
import io
from contextlib import redirect_stdout
from typing import List, Dict, Tuple, Optional
import math
from tqdm import tqdm

class ObjectDetector:
    """
    Class for detecting objects in baseball videos.
    
    Handles detection of ball, glove, catcher, hitter, etc.
    """
    
    def __init__(self, catcher_model, glove_model, ball_model, homeplate_model, device=None, verbose=True):
        """
        Initialize the ObjectDetector.
        
        Args:
            catcher_model: Model for detecting catchers
            glove_model: Model for detecting gloves
            ball_model: Model for detecting baseballs
            homeplate_model: Model for detecting home plate
            device: Device to run models on
            verbose: Whether to print detailed progress information
        """
        self.catcher_model = catcher_model
        self.glove_model = glove_model
        self.ball_model = ball_model
        self.homeplate_model = homeplate_model
        self.device = device
        self.verbose = verbose
    
    def detect_objects(self, video_path: str, object_name: str, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in every frame of the video.
        
        Args:
            video_path (str): Path to the video file
            object_name (str): Name of the object to detect
            conf_threshold (float): Confidence threshold for detection
            
        Returns:
            List[Dict]: List of detection dictionaries
        """
        if self.verbose:
            print(f"Detecting {object_name} in video: {os.path.basename(video_path)}")
        
        model = self._get_model_for_object(object_name)
        
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
                results = model.predict(frame, conf=conf_threshold, device=self.device, verbose=False)
                
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    cls_name = model.names[cls].lower()
                    
                    if cls_name == object_name.lower():
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
    
    def _get_model_for_object(self, object_name: str):
        """
        Get the appropriate model for the object type.
        
        Args:
            object_name (str): Name of the object to detect
            
        Returns:
            The appropriate detection model
        """
        if object_name in ["catcher", "pitcher", "hitter"]:
            return self.catcher_model
        elif object_name == "glove":
            return self.glove_model
        elif object_name == "baseball":
            return self.ball_model
        elif object_name == "homeplate":
            return self.homeplate_model
        else:
            raise ValueError(f"Unknown object type: {object_name}")
    
    def get_catcher_position(self, catcher_detections: List[Dict], reference_frame: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the catcher position near the reference frame to help identify where the umpire is.
        
        Args:
            catcher_detections (List[Dict]): List of catcher detections
            reference_frame (int): Reference frame (usually ball-glove contact)
            
        Returns:
            Optional[Tuple[int, int, int, int]]: Catcher box coordinates (x1, y1, x2, y2)
        """
        if not catcher_detections or reference_frame is None:
            return None
            
        # Group detections by frame
        catcher_by_frame = {}
        for det in catcher_detections:
            frame = det["frame"]
            if frame not in catcher_by_frame:
                catcher_by_frame[frame] = []
            catcher_by_frame[frame].append(det)
            
        # Find the closest frame to reference frame with catcher detections
        closest_frame = None
        min_distance = float('inf')
        
        for frame in catcher_by_frame.keys():
            dist = abs(frame - reference_frame)
            if dist < min_distance:
                min_distance = dist
                closest_frame = frame
                
        if closest_frame is None:
            return None
            
        # Use the largest catcher detection (most likely the actual catcher)
        largest_detection = None
        max_area = 0
        
        for det in catcher_by_frame[closest_frame]:
            area = (det["x2"] - det["x1"]) * (det["y2"] - det["y1"])
            if area > max_area:
                max_area = area
                largest_detection = det
                
        if largest_detection is None:
            return None
            
        return (largest_detection["x1"], largest_detection["y1"], 
                largest_detection["x2"], largest_detection["y2"])
    
    def find_ball_reaches_glove(self, video_path: str, glove_detections: List[Dict], ball_detections: List[Dict], tolerance: float = 0.1) -> Tuple[Optional[int], Optional[Tuple[float, float]], Optional[Dict]]:
        """
        Find the frame where ball reaches the glove with robust detection validation.

        Args:
            video_path (str): Path to the video file
            glove_detections (List[Dict]): List of glove detection dictionaries
            ball_detections (List[Dict]): List of ball detection dictionaries
            tolerance (float): Margin around glove box for ball detection

        Returns:
            Tuple[Optional[int], Optional[Tuple[float, float]], Optional[Dict]]:
                (frame index, ball center coordinates, ball detection dictionary)
        """
        if self.verbose:
            print("Detecting when ball reaches glove...")

        # Group detections by frame for easier processing
        glove_by_frame = {}
        for det in glove_detections:
            glove_by_frame.setdefault(det["frame"], []).append(det)

        ball_by_frame = {}
        for det in ball_detections:
            ball_by_frame.setdefault(det["frame"], []).append(det)

        # Identify continuous ball detection sequences
        ball_frames = sorted(ball_by_frame.keys())

        # Function to find continuous sequences
        def find_continuous_sequences(frames):
            sequences = []
            current_sequence = []

            for i in range(len(frames)):
                if not current_sequence or frames[i] == current_sequence[-1] + 1:
                    current_sequence.append(frames[i])
                else:
                    if len(current_sequence) > 5:  # Require at least 5 consecutive frames
                        sequences.append(current_sequence)
                    current_sequence = [frames[i]]

            # Check the last sequence
            if len(current_sequence) > 5:
                sequences.append(current_sequence)

            return sequences

        # Find continuous sequences of ball detections
        ball_detection_sequences = find_continuous_sequences(ball_frames)

        if self.verbose:
            print(f"Found {len(ball_detection_sequences)} continuous ball detection sequences")

        # Search for ball-glove contact in the most significant detection sequences
        # Sort sequences by length, prioritizing longer sequences
        ball_detection_sequences.sort(key=len, reverse=True)

        for sequence in ball_detection_sequences:
            # Search through the sequence frames
            for frame in sequence:
                if frame not in glove_by_frame:
                    continue

                for glove_det in glove_by_frame[frame]:
                    # Add tolerance around glove box
                    margin_x = tolerance * (glove_det["x2"] - glove_det["x1"])
                    margin_y = tolerance * (glove_det["y2"] - glove_det["y1"])
                    extended_x1 = glove_det["x1"] - margin_x
                    extended_y1 = glove_det["y1"] - margin_y
                    extended_x2 = glove_det["x2"] + margin_x
                    extended_y2 = glove_det["y2"] + margin_y

                    for ball_det in ball_by_frame[frame]:
                        # Calculate ball center
                        ball_center_x = (ball_det["x1"] + ball_det["x2"]) / 2
                        ball_center_y = (ball_det["y1"] + ball_det["y2"]) / 2

                        # Check if ball center is within extended glove box
                        if (extended_x1 <= ball_center_x <= extended_x2 and
                            extended_y1 <= ball_center_y <= extended_y2):
                            if self.verbose:
                                print(f"Ball reached glove at frame {frame}")
                            return frame, (ball_center_x, ball_center_y), ball_det

        # FALLBACK 1: Try with larger tolerance
        if self.verbose:
            print("Standard detection failed, trying with larger tolerance...")
            
        larger_tolerance = 0.3  # 30% margin
        for sequence in ball_detection_sequences:
            for frame in sequence:
                if frame not in glove_by_frame:
                    continue

                for glove_det in glove_by_frame[frame]:
                    margin_x = larger_tolerance * (glove_det["x2"] - glove_det["x1"])
                    margin_y = larger_tolerance * (glove_det["y2"] - glove_det["y1"])
                    extended_x1 = glove_det["x1"] - margin_x
                    extended_y1 = glove_det["y1"] - margin_y
                    extended_x2 = glove_det["x2"] + margin_x
                    extended_y2 = glove_det["y2"] + margin_y

                    for ball_det in ball_by_frame[frame]:
                        ball_center_x = (ball_det["x1"] + ball_det["x2"]) / 2
                        ball_center_y = (ball_det["y1"] + ball_det["y2"]) / 2

                        if (extended_x1 <= ball_center_x <= extended_x2 and
                            extended_y1 <= ball_center_y <= extended_y2):
                            if self.verbose:
                                print(f"Ball reached glove at frame {frame} (with larger tolerance)")
                            return frame, (ball_center_x, ball_center_y), ball_det
            
        # FALLBACK 2: Look for closest ball to any glove
        if self.verbose:
            print("Expanded tolerance failed, trying closest approach method...")
            
        best_distance = float('inf')
        best_frame = None
        best_ball_center = None
        best_ball_det = None
        
        # Check each ball against each glove in the same frame
        for frame in ball_frames:
            if frame not in glove_by_frame:
                continue
                
            for glove_det in glove_by_frame[frame]:
                glove_center_x = (glove_det["x1"] + glove_det["x2"]) / 2
                glove_center_y = (glove_det["y1"] + glove_det["y2"]) / 2
                
                for ball_det in ball_by_frame[frame]:
                    ball_center_x = (ball_det["x1"] + ball_det["x2"]) / 2
                    ball_center_y = (ball_det["y1"] + ball_det["y2"]) / 2
                    
                    # Calculate Euclidean distance
                    distance = math.sqrt(
                        (ball_center_x - glove_center_x)**2 + 
                        (ball_center_y - glove_center_y)**2
                    )
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_frame = frame
                        best_ball_center = (ball_center_x, ball_center_y)
                        best_ball_det = ball_det
        
        if best_frame is not None:
            if self.verbose:
                print(f"Found closest ball approach at frame {best_frame} (distance: {best_distance:.2f} pixels)")
            return best_frame, best_ball_center, best_ball_det
            
        # FALLBACK 3: Just use the middle frame with a ball detection
        if ball_frames:
            middle_idx = len(ball_frames) // 2
            middle_frame = ball_frames[middle_idx]
            middle_ball_det = ball_by_frame[middle_frame][0]
            middle_ball_center_x = (middle_ball_det["x1"] + middle_ball_det["x2"]) / 2
            middle_ball_center_y = (middle_ball_det["y1"] + middle_ball_det["y2"]) / 2
            
            if self.verbose:
                print(f"Using middle ball frame {middle_frame} as fallback")
            return middle_frame, (middle_ball_center_x, middle_ball_center_y), middle_ball_det

        if self.verbose:
            print("Could not detect when ball reaches glove")
        return None, None, None
    
    def find_best_hitter_box(self, video_path: str, hitter_detections: List[Dict],
                            catcher_position: Optional[Tuple[int, int, int, int]] = None,
                            frame_idx_start: int = 0, frame_search_range: int = 90) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Optional[int]]:
        """
        Find the best bounding box for the hitter using robust filtering to avoid detecting the umpire.
        
        Args:
            video_path (str): Path to the video file
            hitter_detections (List[Dict]): Pre-computed hitter detections
            catcher_position (Tuple[int, int, int, int]): Position of the catcher
            frame_idx_start (int): Starting frame index
            frame_search_range (int): Number of frames to search
            
        Returns:
            Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Optional[int]]:
                (hitter box, frame containing the hitter, frame index)
        """
        if self.verbose:
            print("Finding best hitter bounding box...")
        
        # Capture video
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define search range
        search_start = max(0, frame_idx_start)
        search_end = min(total_frames, search_start + frame_search_range)
        
        # Variables to store best detection
        best_hitter_box = None
        best_hitter_frame = None
        best_frame_idx = None
        best_confidence = 0
        
        # First, try to use pre-computed hitter detections if available
        if hitter_detections:
            hitter_by_frame = {}
            for det in hitter_detections:
                frame = det["frame"]
                if search_start <= frame <= search_end:
                    if frame not in hitter_by_frame:
                        hitter_by_frame[frame] = []
                    hitter_by_frame[frame].append(det)
            
            valid_frames = sorted(hitter_by_frame.keys())
            
            for frame_idx in valid_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                for det in hitter_by_frame[frame_idx]:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    conf = det["confidence"]
                    
                    # Apply heuristics to identify the actual hitter (not umpire):
                    # 1. Must be reasonably sized (minimum dimensions)
                    # 2. Typically on opposite side from catcher
                    is_valid_hitter = True
                    
                    # Size check - must be substantial
                    area = (x2 - x1) * (y2 - y1)
                    min_area = width * height * 0.015  # Reduced from 0.02 to 0.015
                    min_width = width * 0.04  # Reduced from 0.05 to 0.04
                    min_height = height * 0.08  # Reduced from 0.1 to 0.08
                    
                    if area < min_area or (x2 - x1) < min_width or (y2 - y1) < min_height:
                        is_valid_hitter = False
                    
                    # Position check relative to catcher - less strict now
                    if catcher_position:
                        catcher_center_x = (catcher_position[0] + catcher_position[2]) / 2
                        hitter_center_x = (x1 + x2) / 2
                        
                        # If catcher is on extreme right, hitter should be on left (and vice versa)
                        if ((catcher_center_x > width*0.75 and hitter_center_x > width*0.75) or
                            (catcher_center_x < width*0.25 and hitter_center_x < width*0.25)):
                            is_valid_hitter = False
                    
                    # If detection is valid and better than current best
                    if is_valid_hitter and conf > best_confidence:
                        best_hitter_box = (x1, y1, x2, y2)
                        best_hitter_frame = frame.copy()
                        best_frame_idx = frame_idx
                        best_confidence = conf
        
        # If we couldn't find a good hitter from existing detections, search with PHC model
        if best_hitter_box is None:
            if self.verbose:
                print("Looking for hitter using PHC model with reduced constraints...")
            
            for frame_idx in range(search_start, search_end, 5):  # Step by 5 frames for efficiency
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Detect hitter with PHC model
                with io.StringIO() as buf, redirect_stdout(buf):
                    phc_results = self.catcher_model.predict(frame, conf=0.3, verbose=False)  # Lower threshold
                
                for result in phc_results:
                    for box in result.boxes:
                        cls = int(box.cls)
                        conf = float(box.conf)
                        
                        # Look specifically for "hitter" class
                        if self.catcher_model.names[cls].lower() == "hitter" and conf > best_confidence:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            # Apply less strict heuristics
                            is_valid_hitter = True
                            
                            # Size check with reduced thresholds
                            area = (x2 - x1) * (y2 - y1)
                            min_area = width * height * 0.015  # Reduced from 0.02 to 0.015
                            min_width = width * 0.04  # Reduced from 0.05 to 0.04
                            min_height = height * 0.08  # Reduced from 0.1 to 0.08
                            
                            if area < min_area or (x2 - x1) < min_width or (y2 - y1) < min_height:
                                is_valid_hitter = False
                            
                            # Position check relative to catcher - less strict
                            if catcher_position:
                                catcher_center_x = (catcher_position[0] + catcher_position[2]) / 2
                                hitter_center_x = (x1 + x2) / 2
                                
                                # Only exclude if hitter and catcher are in same extreme corner
                                if ((catcher_center_x > width*0.75 and hitter_center_x > width*0.75) or
                                    (catcher_center_x < width*0.25 and hitter_center_x < width*0.25)):
                                    is_valid_hitter = False
                            
                            if is_valid_hitter:
                                best_hitter_box = (x1, y1, x2, y2)
                                best_hitter_frame = frame.copy()
                                best_frame_idx = frame_idx
                                best_confidence = conf
        
        # FALLBACK: Use any detection that seems reasonable if we still don't have one
        if best_hitter_box is None:
            if self.verbose:
                print("Trying fallback method for hitter detection...")
            
            # Reset the video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Try to find any large bounding box that could be a player
            for frame_idx in range(0, min(total_frames, 200), 10):  # Check first 200 frames, step of 10
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Convert to grayscale for motion detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold to find potential player blobs
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if it's large enough to be a player
                    if w * h > width * height * 0.01 and w > width * 0.03 and h > height * 0.08:
                        # Create a box with some margin
                        x1 = max(0, x - int(w * 0.1))
                        y1 = max(0, y - int(h * 0.1))
                        x2 = min(width, x + w + int(w * 0.1))
                        y2 = min(height, y + h + int(h * 0.1))
                        
                        # Check if it's in a reasonable position (not at the bottom of frame where umpire is)
                        if y < height * 0.6:  # Not too low in the frame
                            best_hitter_box = (x1, y1, x2, y2)
                            best_hitter_frame = frame.copy()
                            best_frame_idx = frame_idx
                            # Use a minimum confidence
                            best_confidence = 0.5
                            break
                
                if best_hitter_box is not None:
                    break
        
        cap.release()
        
        if best_hitter_box is not None:
            if self.verbose:
                print(f"Found valid hitter box at frame {best_frame_idx} with confidence {best_confidence:.2f}")
        else:
            if self.verbose:
                print("Could not find valid hitter box")
        
        return best_hitter_box, best_hitter_frame, best_frame_idx
        
    def detect_homeplate(self, video_path: str, reference_frame: int = None) -> Tuple[Optional[Tuple[int, int, int, int]], float, int]:
        """
        Detect the home plate in the video using the dedicated homeplate model.
        Returns the home plate bounding box, confidence score, and the frame used.
        
        Args:
            video_path (str): Path to the video file
            reference_frame (int): Reference frame to start search from
            
        Returns:
            Tuple[Optional[Tuple[int, int, int, int]], float, int]: 
                (home plate box, confidence, frame used)
        """
        if self.verbose:
            print("Detecting home plate...")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define search frames around ball-glove contact (or use reference frame)
        search_frame = reference_frame if reference_frame is not None else total_frames // 2
        
        search_frames = [
            search_frame,             # The exact frame
            search_frame + 2,         # The adjusted frame (accounting for delay)
            search_frame - 2,
            search_frame + 4,
            search_frame - 4,
            search_frame - 8,
            search_frame - 12,
            search_frame - 16,
            search_frame - 20,
            search_frame - 30,
            search_frame - 45,
            search_frame - 60,
        ]
        # Filter out negative frames
        search_frames = [max(0, frame) for frame in search_frames]

        if self.verbose:
            print(f"Searching for home plate around reference frame {search_frame}...")

        # Dictionary to store all frames with home plate detections (for visualization)
        detection_frames = {}
        all_homeplate_detections = []

        for frame_idx in search_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Use the dedicated homeplate model
            with io.StringIO() as buf, redirect_stdout(buf):
                results = self.homeplate_model.predict(frame, conf=0.2, verbose=False)
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    cls_name = self.homeplate_model.names[cls].lower()
                    
                    # Check for home plate class
                    if "homeplate" == cls_name:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detection = {
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "model": "glove_tracking",
                            "frame": frame_idx
                        }
                        all_homeplate_detections.append(detection)
                    
                    # Additionally, try alternative class names that might be used
                    elif "home_plate" == cls_name or "plate" == cls_name or "home" == cls_name:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detection = {
                            "box": (x1, y1, x2, y2),
                            "conf": conf,
                            "model": "glove_tracking",
                            "frame": frame_idx
                        }
                        all_homeplate_detections.append(detection)

            # Store frame for visualization if detections were found
            if all_homeplate_detections:
                detection_frames[frame_idx] = {
                    "frame": frame.copy(),
                    "detections": all_homeplate_detections
                }

        # If we found any detections, use the best one
        if all_homeplate_detections:
            best_detection = max(all_homeplate_detections, key=lambda det: det["conf"])
            homeplate_box = best_detection["box"]
            homeplate_confidence = best_detection["conf"]
            homeplate_frame_idx = best_detection["frame"]
            
            cap.release()
            return homeplate_box, homeplate_confidence, homeplate_frame_idx
        
        # If no detections were found, return None
        cap.release()
        return None, 0.0, None