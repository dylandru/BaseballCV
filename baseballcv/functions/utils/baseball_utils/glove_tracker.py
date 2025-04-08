import os
import sys
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from baseballcv.utilities import BaseballCVLogger, ProgressBar
from baseballcv.functions.load_tools import LoadTools


class GloveTracker:
    """
    Class for tracking the catcher's glove, home plate, and baseball in videos.
    
    The tracker uses YOLO models to detect and track objects in baseball videos,
    providing real-time tracking, visualization, and data export capabilities for
    analysis of catcher movements and positioning.
    
    This class handles detection, filtering of outliers, coordinate transformation
    from pixel space to real-world measurements, and generates visualizations
    including tracking plots and heatmaps.
    """
    
    def __init__(
        self, 
        model_alias: str = 'glove_tracking',
        results_dir: str = "glove_tracking_results",
        device: str = None,
        confidence_threshold: float = 0.5,
        enable_filtering: bool = True,
        max_velocity_inches_per_sec: float = 120.0,
        logger: Optional[BaseballCVLogger] = None,
        suppress_detection_warnings: bool = True
    ):
        """
        Initialize the GloveTracker with model and configuration settings.
        
        Args:
            model_alias (str): The alias of the YOLO model to use for detection
            results_dir (str): Directory to save tracking results, videos, and visualizations
            device (str): Device to run the model on (cpu, cuda, mps)
            confidence_threshold (float): Minimum confidence threshold for accepting detections
            enable_filtering (bool): Whether to enable outlier filtering for glove detections
            max_velocity_inches_per_sec (float): Maximum plausible velocity for glove movement in inches/sec
            logger (BaseballCVLogger): Logger instance for logging messages and errors
            suppress_detection_warnings (bool): Whether to suppress warnings about missing detections
        """
        self.load_tools = LoadTools()
        self.logger = logger if logger else BaseballCVLogger.get_logger(self.__class__.__name__)
        self.model_path = self.load_tools.load_model(model_alias)
        self.model = YOLO(self.model_path)
        self.device = device if device else 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        self.confidence_threshold = confidence_threshold
        self.results_dir = results_dir
        self.suppress_detection_warnings = suppress_detection_warnings

        os.makedirs(self.results_dir, exist_ok=True)
        
        self.enable_filtering = enable_filtering
        self.max_velocity_inches_per_sec = max_velocity_inches_per_sec
        
        self.class_names = self.model.names
        self.logger.info(f"Model loaded with classes: {self.class_names}")
        
        self.glove_class_id = next((id for id, name in self.class_names.items() if 'glove' in name.lower()), None)
        self.homeplate_class_id = next((id for id, name in self.class_names.items() if 'homeplate' in name.lower() or 'home_plate' in name.lower() or 'home plate' in name.lower()), None)
        self.baseball_class_id = next((id for id, name in self.class_names.items() if 'baseball' in name.lower() or 'ball' in name.lower()), None)
        
        if not all([self.glove_class_id is not None, self.homeplate_class_id is not None, self.baseball_class_id is not None]):
            missing_classes = []
            if self.glove_class_id is None:
                missing_classes.append("glove")
            if self.homeplate_class_id is None:
                missing_classes.append("homeplate")
            if self.baseball_class_id is None:
                missing_classes.append("baseball")
            self.logger.warning(f"Some required classes not found in model: {missing_classes}")
        
        self.logger.info(f"Class IDs - Glove: {self.glove_class_id}, Home Plate: {self.homeplate_class_id}, Baseball: {self.baseball_class_id}")
        
        self.tracking_data = []
        self.home_plate_reference = None
        self.pixels_per_inch = None

    def track_video(self, video_path: str, output_path: Optional[str] = None,
                    show_plot: bool = True, create_video: bool = True, 
                    generate_heatmap: bool = True) -> str:
        """
        Track objects in a video and generate visualization with 2D tracking plot.

        Processes a video frame by frame, detecting the glove, home plate, and baseball
        in each frame. Calculates real-world coordinates based on home plate reference,
        filters outliers, and generates visualizations including tracking plots and heatmaps.

        Args:
            video_path (str): Path to input video file to be processed
            output_path (str): Path for output video (if None, auto-generated in results_dir)
            show_plot (bool): Whether to include the 2D tracking plot in the output video
            create_video (bool): Whether to create and save output videos
            generate_heatmap (bool): Whether to generate a heatmap visualization for this video

        Returns:
            str: Path to the output video file

        Raises:
            FileNotFoundError: If the input video file does not exist
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

        if output_path is None:
            video_filename = os.path.basename(video_path)
            output_path = os.path.join(self.results_dir, f"tracked_{video_filename}")

        csv_filename = os.path.splitext(os.path.basename(output_path))[0] + "_tracking.csv"
        csv_path = os.path.join(self.results_dir, csv_filename)

        self.tracking_data = []
        self.home_plate_reference = None
        self.pixels_per_inch = None

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if create_video else None

        out_width = int(width * 2) if show_plot else width
        out = None
        if create_video:
            out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        progress_bar = ProgressBar(
            total=total_frames, 
            desc=f"Processing video: {os.path.basename(video_path)}",
            disable=False,
            color="green",
            bar_format="Processing |{bar}| {percentage:3.0f}% [{n_fmt}/{total_fmt}] {rate_fmt} ETA: {remaining}"
        )

        frame_idx = 0

        with progress_bar as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model.predict(frame, conf=self.confidence_threshold, device=self.device, verbose=False)

                detections = self._process_detections(results, frame, frame_idx, fps)

                annotated_frame = self._annotate_frame(frame.copy(), detections, frame_idx)

                if show_plot:
                    self._update_tracking_plot(ax, fig)

                    fig.canvas.draw()
                    buf = fig.canvas.buffer_rgba()
                    plot_img = np.asarray(buf)[:, :, :3]  # Convert RGBA to RGB
                    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    plot_img = cv2.resize(plot_img, (width, height))

                    combined_frame = np.hstack((annotated_frame, plot_img))
                    if create_video:
                        out.write(combined_frame) 
                else:
                    if create_video:
                        out.write(annotated_frame) 

                frame_idx += 1
                pbar.update(1)
                sys.stdout.flush()

        cap.release()
        if create_video:
            out.release()
        plt.close(fig)

        self._save_tracking_data(csv_path, video_path)

        self.logger.info(f"Tracking completed. Output video saved to {output_path}")
        self.logger.info(f"Tracking data saved to {csv_path}")
        
        if generate_heatmap:
            video_name = os.path.basename(video_path)
            heatmap_path = self.plot_glove_heatmap(
                csv_path=csv_path,
                video_name=video_name,
                generate_heatmap=True
            )
            if heatmap_path:
                self.logger.info(f"Glove heatmap saved to {heatmap_path}")

        return output_path

    def _process_detections(self, results, frame, frame_idx, fps=30.0) -> Dict:
        """
        Process YOLO detection results for the current frame with outlier filtering.
        
        Extracts detection information from YOLO results, converts pixel coordinates
        to real-world coordinates based on home plate reference, and applies velocity-based
        filtering to remove physically impossible movements.
        
        Args:
            results: YOLO detection results containing bounding boxes and confidence scores
            frame: Current video frame being processed
            frame_idx: Index of the current frame in the video
            fps: Frames per second of the video for velocity calculations
                
        Returns:
            dict: Processed detections with glove, homeplate, and baseball information
                  including pixel coordinates, real-world coordinates, and confidence scores
        """
        detections = {
            'frame_idx': frame_idx,
            'glove': None,
            'homeplate': None,
            'baseball': None,
            'real_world_coords': {}
        }
        
        # Extract detections from results
        for result in results:
            for box_idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                
                x1, y1, x2, y2 = map(int, xyxy)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Store detection based on class
                if cls_id == self.glove_class_id:
                    if detections['glove'] is None or conf > detections['glove']['confidence']:
                        detections['glove'] = {
                            'box': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'confidence': conf
                        }
                
                elif cls_id == self.homeplate_class_id:
                    if detections['homeplate'] is None or conf > detections['homeplate']['confidence']:
                        detections['homeplate'] = {
                            'box': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'confidence': conf,
                            'width': x2 - x1  # Home plate width in pixels
                        }
                        
                        # Update home plate reference if needed
                        if self.home_plate_reference is None:
                            self.home_plate_reference = detections['homeplate']
                            # Calculate pixels per inch (home plate is 17 inches wide)
                            self.pixels_per_inch = detections['homeplate']['width'] / 17.0
                            self.logger.info(f"Home plate reference set: {self.pixels_per_inch:.2f} pixels per inch")
                
                elif cls_id == self.baseball_class_id:
                    if detections['baseball'] is None or conf > detections['baseball']['confidence']:
                        detections['baseball'] = {
                            'box': (x1, y1, x2, y2),
                            'center': (center_x, center_y),
                            'confidence': conf
                        }
        
        # Convert to real-world coordinates if home plate reference is available
        if self.home_plate_reference is not None and detections['glove'] is not None:
            # Use home plate center as origin
            origin_x, origin_y = self.home_plate_reference['center']
            
            # Calculate real-world coordinates (in inches) relative to home plate
            if detections['glove'] is not None:
                glove_x, glove_y = detections['glove']['center']
                real_x = (glove_x - origin_x) / self.pixels_per_inch
                real_y = (origin_y - glove_y) / self.pixels_per_inch  # Invert Y to match real-world coordinates
                detections['real_world_coords']['glove'] = (real_x, real_y)
            
            if detections['baseball'] is not None:
                ball_x, ball_y = detections['baseball']['center']
                real_x = (ball_x - origin_x) / self.pixels_per_inch
                real_y = (origin_y - ball_y) / self.pixels_per_inch
                detections['real_world_coords']['baseball'] = (real_x, real_y)
        
        # Apply filtering to glove detections
        if hasattr(self, 'enable_filtering') and self.enable_filtering and len(self.tracking_data) > 0:
            prev_detection = self.tracking_data[-1]
            if not self._filter_glove_detection(prev_detection, detections, fps, self.max_velocity_inches_per_sec):
                # If detection is an outlier, don't include glove data
                if 'glove' in detections['real_world_coords']:
                    del detections['real_world_coords']['glove']
                detections['glove'] = None
        
        # Add to tracking data if there are relevant detections
        if detections['glove'] is not None or detections['homeplate'] is not None or detections['baseball'] is not None:
            self.tracking_data.append(detections)
        
        return detections
    
    def _fill_missing_detections(self):
        """
        Fill in missing glove detection coordinates by repeating the last known position.
        
        Creates a continuous trajectory by filling gaps in detection with the last known
        valid position, preventing discontinuities in the tracking visualization.
        
        Returns:
            Tuple[List[float], List[float]]: Lists of x and y coordinates with gaps filled
        """
        if not self.tracking_data:
            return [], []
        
        # Extract frame indices and glove coordinates
        frame_indices = []
        glove_coords = []
        
        for detection in self.tracking_data:
            frame_indices.append(detection['frame_idx'])
            
            if 'glove' in detection['real_world_coords']:
                glove_coords.append(detection['real_world_coords']['glove'])
            else:
                glove_coords.append(None)
        
        # Fill in missing glove coordinates
        filled_x, filled_y = [], []
        last_valid_x, last_valid_y = None, None
        
        for i, coords in enumerate(glove_coords):
            if coords is not None:
                # Valid detection
                x, y = coords
                filled_x.append(x); filled_y.append(y)
                last_valid_x, last_valid_y = x, y
            elif last_valid_x is not None and last_valid_y is not None:
                # Missing detection, but we have previous valid coordinates
                filled_x.append(last_valid_x); filled_y.append(last_valid_y)
            else:
                # No valid previous coordinates to use
                # We'll skip this frame entirely
                pass
        
        return filled_x, filled_y

    def _annotate_frame(self, frame, detections, frame_idx):
        """
        Annotate the frame with detections and tracking information.
        
        Adds visual elements to the frame including bounding boxes, labels, and
        real-world coordinate information for detected objects.
        
        Args:
            frame: Current video frame to annotate
            detections: Dictionary of processed detections for the current frame
            frame_idx: Index of the current frame in the video
            
        Returns:
            numpy.ndarray: Annotated frame with visual elements added
        """
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if detections['homeplate'] is not None:
            x1, y1, x2, y2 = detections['homeplate']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"Home Plate", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            center_x, center_y = detections['homeplate']['center']
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 255, 0), -1)
            
            if self.pixels_per_inch is not None:
                cv2.putText(frame, f"Home Plate Width: 17.0 in", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, f"Scale: {self.pixels_per_inch:.2f} px/in", (x1, y2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        if detections['glove'] is not None:
            x1, y1, x2, y2 = detections['glove']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Glove", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            center_x, center_y = detections['glove']['center']
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
            
            if 'glove' in detections['real_world_coords']:
                real_x, real_y = detections['real_world_coords']['glove']
                cv2.putText(frame, f"Pos: ({real_x:.1f}, {real_y:.1f}) in", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if detections['baseball'] is not None:
            x1, y1, x2, y2 = detections['baseball']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Baseball", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            center_x, center_y = detections['baseball']['center']
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            
            if 'baseball' in detections['real_world_coords']:
                real_x, real_y = detections['real_world_coords']['baseball']
                cv2.putText(frame, f"Pos: ({real_x:.1f}, {real_y:.1f}) in", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

    def _update_tracking_plot(self, ax, fig):
        """
        Update the 2D tracking plot with latest glove movement data.

        Creates or updates a matplotlib plot showing the glove's movement trajectory
        in real-world coordinates (inches) relative to home plate.

        Args:
            ax: Matplotlib axis object to draw the plot on
            fig: Matplotlib figure object containing the axis
            
        Returns:
            numpy.ndarray: Image representation of the updated plot
        """
        ax.clear()
        ax.set_title('Glove Movement Tracking')
        ax.set_xlabel('X Position (inches from home plate)')
        ax.set_ylabel('Y Position (inches from home plate)')
        ax.grid(True)

        # Use advanced method to handle missing detections
        glove_x, glove_y = self._handle_missing_detections()
        self.logger.debug(f"Plotting glove data: {len(glove_x)} points")

        # Define valid_sequences based on whether the handling returned any points
        valid_sequences = bool(glove_x)

        # If the sequence processing finds no valid data
        if not valid_sequences:
            # Reconstruct glove_coords from tracking_data for logging stats
            glove_coords = [
                detection['real_world_coords'].get('glove')
                if 'real_world_coords' in detection and 'glove' in detection['real_world_coords'] else None
                for detection in self.tracking_data
            ]
            self.logger.debug(f"Detection stats: total={len(glove_coords)}, valid={sum(1 for c in glove_coords if c is not None)}")

        if self.home_plate_reference is not None:
            home_plate_shape = np.array([[-8.5, 0], [8.5, 0], [0, 8.5], [-8.5, 0]])
            ax.fill(home_plate_shape[:, 0], home_plate_shape[:, 1], color='gray', alpha=0.5, label='Home Plate')

        # Plot tracking data (glove only)
        if glove_x and glove_y:
            ax.plot(glove_x, glove_y, 'g-', label='Glove Path')
            ax.scatter(glove_x[-1], glove_y[-1], color='green', s=100, marker='o', label='Current Glove Pos')

        ax.set_xlim(-50, 50); ax.set_ylim(-10, 60)
        ax.set_aspect('equal', 'box')
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        fig.canvas.draw()

        # Convert matplotlib figure to image
        plot_img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2RGB)

        return plot_img


    def _save_tracking_data(self, csv_path, video_path=None):
        """
        Save tracking data to a CSV file with comprehensive information.

        Creates a CSV file containing frame-by-frame tracking data including raw pixel
        coordinates, real-world coordinates, detection confidence scores, and metadata
        about interpolated vs. actual detections.

        Args:
            csv_path: Path where the CSV file will be saved
            video_path: Path to the original video (for identification in batch mode)
        """
        if not self.tracking_data:
            self.logger.warning("No tracking data to save")
            return

        # Get total frame count from video if available
        total_frames = getattr(self, 'total_frames', None)
        if total_frames is None and video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        
        # If we still don't know total frames, estimate from tracking data
        if total_frames is None and self.tracking_data:
            total_frames = max(detection['frame_idx'] for detection in self.tracking_data) + 1
        
        # Get processed coordinates for all frames that have tracking data
        processed_x, processed_y = self._fill_missing_detections()
        
        # Create a mapping from frames that have tracking data to processed coordinates
        frame_indices = sorted(set(detection['frame_idx'] for detection in self.tracking_data))
        processed_coords = {}
        
        # Only map if we have both indices and coordinates
        if len(frame_indices) > 0 and len(processed_x) > 0:
            # Make sure they are the same length, or use the shorter length
            min_len = min(len(frame_indices), len(processed_x))
            for i in range(min_len):
                processed_coords[frame_indices[i]] = (processed_x[i], processed_y[i])
        
        # Create a map of frames with actual detections (not interpolated)
        frames_with_detections = set()
        for detection in self.tracking_data:
            if detection['glove'] is not None and 'glove' in detection['real_world_coords']:
                frames_with_detections.add(detection['frame_idx'])
        
        # Prepare data for CSV - create a record for every frame
        csv_data = []
        
        # Include all frames in the range, even if no detection
        for frame_idx in range(total_frames):
            # Find detection for this frame if it exists
            detection = next((d for d in self.tracking_data if d['frame_idx'] == frame_idx), None)
            
            # Base data - will be populated with None for missing frames
            row_data = {
                'frame_idx': frame_idx,
                'homeplate_center_x': None,
                'homeplate_center_y': None,
                'homeplate_width': None,
                'homeplate_confidence': None,
                'glove_center_x': None,
                'glove_center_y': None,
                'glove_confidence': None,
                'glove_real_x': None,
                'glove_real_y': None,
                'baseball_center_x': None,
                'baseball_center_y': None,
                'baseball_confidence': None,
                'baseball_real_x': None,
                'baseball_real_y': None,
                'pixels_per_inch': self.pixels_per_inch,
                # Add new columns for processed coordinates
                'glove_processed_x': None,
                'glove_processed_y': None,
                'is_interpolated': True  # Default to True, will be set to False for actual detections
            }
            
            # If we have detection data for this frame, populate it
            if detection:
                # Home plate data
                if detection['homeplate'] is not None:
                    row_data['homeplate_center_x'], row_data['homeplate_center_y'] = detection['homeplate']['center']
                    row_data['homeplate_width'] = detection['homeplate']['width']
                    row_data['homeplate_confidence'] = detection['homeplate']['confidence']

                # Glove data - raw detections
                if detection['glove'] is not None:
                    row_data['glove_center_x'], row_data['glove_center_y'] = detection['glove']['center']
                    row_data['glove_confidence'] = detection['glove']['confidence']

                    if 'glove' in detection['real_world_coords']:
                        row_data['glove_real_x'], row_data['glove_real_y'] = detection['real_world_coords']['glove']
                        row_data['is_interpolated'] = False  # This is an actual detection

                # Baseball data
                if detection['baseball'] is not None:
                    row_data['baseball_center_x'], row_data['baseball_center_y'] = detection['baseball']['center']
                    row_data['baseball_confidence'] = detection['baseball']['confidence']

                    if 'baseball' in detection['real_world_coords']:
                        row_data['baseball_real_x'], row_data['baseball_real_y'] = detection['real_world_coords']['baseball']
            
            # Add processed coordinates if available for this frame
            if frame_idx in processed_coords:
                row_data['glove_processed_x'], row_data['glove_processed_y'] = processed_coords[frame_idx]
                # Mark as not interpolated if this was an actual detection
                if frame_idx in frames_with_detections:
                    row_data['is_interpolated'] = False
                else:
                    row_data['is_interpolated'] = True
            
            if video_path:
                row_data['video_filename'] = os.path.basename(video_path)
                
            csv_data.append(row_data)

        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

        self.logger.info(f"Tracking data saved to {csv_path} with {len(csv_data)} frames")
        
        processed_count = sum(1 for row in csv_data if row['glove_processed_x'] is not None)
        interpolated_count = sum(1 for row in csv_data if row['is_interpolated'] is True and row['glove_processed_x'] is not None)
        self.logger.info(f"Processed coordinates: {processed_count} frames, Interpolated: {interpolated_count} frames")
        
        if not os.path.exists(csv_path):
            self.logger.error(f"Failed to save CSV file at {csv_path}")
        else:
            self.logger.info(f"CSV file saved successfully at {csv_path}")

        
    def plot_glove_heatmap(self, csv_path: Optional[str] = None, output_path: Optional[str] = None, 
                         video_name: Optional[str] = None, generate_heatmap: bool = True) -> Optional[str]:
        """
        Create a heatmap of glove positions.
        
        Args:
            csv_path: Path to tracking CSV file
            output_path: Path to save the heatmap image
            video_name: Name of the video file to include in the heatmap filename
            generate_heatmap: Whether to generate the heatmap
            
        Returns:
            str: Path to the saved heatmap image or None if not generated
        """
        if not generate_heatmap:
            return None
            
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            
            real_coords = df[df['glove_real_x'].notna() & df['glove_real_y'].notna()]
            
            if len(real_coords) > 1:
                # Fill missing values by forward-filling
                filled_df = df.copy()
                filled_df['glove_real_x'] = filled_df['glove_real_x'].fillna(method='ffill')
                filled_df['glove_real_y'] = filled_df['glove_real_y'].fillna(method='ffill')
                glove_x = filled_df['glove_real_x'].dropna().tolist()
                glove_y = filled_df['glove_real_y'].dropna().tolist()
            else:
                self.logger.warning("Not enough data points in CSV for heatmap")
                return None
        elif self.tracking_data:
            # Use the current tracking data with filled detections
            glove_x, glove_y = self._fill_missing_detections()
            
            if not glove_x or not glove_y:
                self.logger.warning("No valid glove tracking data available for heatmap")
                return None
        else:
            self.logger.warning("No tracking data available for heatmap")
            return None
        
        plt.figure(figsize=(10, 8))
        
        if len(glove_x) > 1:
            plt.hist2d(glove_x, glove_y, bins=20, cmap='hot')
            plt.colorbar(label='Frequency')
            
            home_plate_shape = np.array([[-8.5, 0], [8.5, 0], [0, 8.5], [-8.5, 0]])
            plt.fill(home_plate_shape[:, 0], home_plate_shape[:, 1], color='gray', alpha=0.5)
            
            title = 'Glove Position Heatmap'
            if video_name:
                title += f' - {video_name}'

            plt.title(title)
            plt.xlabel('X Position (inches from home plate)'); plt.ylabel('Y Position (inches from home plate)')
            plt.grid(True, alpha=0.3)
            plt.xlim(-50, 50); plt.ylim(-10, 60)
            plt.gca().set_aspect('equal', 'box')
            
            if output_path is None:
                if video_name:
                    heatmap_filename = f"glove_heatmap_{os.path.splitext(video_name)[0]}.png"
                else:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    heatmap_filename = f"glove_heatmap_{timestamp}.png"
                output_path = os.path.join(self.results_dir, heatmap_filename)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Glove heatmap saved to {output_path}")
            return output_path
        else:
            self.logger.warning("Not enough glove position data for heatmap")
            plt.close()
            return None

    def analyze_glove_movement(self, csv_path: Optional[str] = None) -> Optional[Dict]:
        """
        Analyze glove movement data and return statistics.
        
        Args:
            csv_path: Path to the tracking CSV (if None, uses the last tracking data)
            
        Returns:
            dict: Movement statistics
        """
        if csv_path is not None:
            df = pd.read_csv(csv_path)
        elif self.tracking_data:
            return self._analyze_tracking_data()
        else:
            self.logger.warning("No tracking data available for analysis")
            return None
        
        stats = {
            'total_frames': len(df),
            'frames_with_glove': df['glove_center_x'].notna().sum(),
            'frames_with_baseball': df['baseball_center_x'].notna().sum(),
            'frames_with_homeplate': df['homeplate_center_x'].notna().sum(),
        }
        
        # Calculate glove movement distance (in real-world units)
        if stats['frames_with_glove'] > 1:
            real_coords = df[df['glove_real_x'].notna() & df['glove_real_y'].notna()]
            
            if len(real_coords) > 1:
                # Calculate the total distance
                dx = real_coords['glove_real_x'].diff()
                dy = real_coords['glove_real_y'].diff()
                distances = np.sqrt(dx**2 + dy**2)
                
                stats['total_distance_inches'] = distances.sum()
                stats['max_distance_between_frames_inches'] = distances.max()
                stats['avg_distance_between_frames_inches'] = distances.mean()
                
                # Calculate the area covered by the glove (convex hull)
                if len(real_coords) > 2:  # Need at least 3 points for a convex hull
                    from scipy.spatial import ConvexHull
                    points = real_coords[['glove_real_x', 'glove_real_y']].values
                    try:
                        hull = ConvexHull(points)
                        stats['convex_hull_area_sq_inches'] = hull.volume  # In 2D, volume is area
                    except:
                        stats['convex_hull_area_sq_inches'] = None
                
                # Range of motion
                stats['x_range_inches'] = real_coords['glove_real_x'].max() - real_coords['glove_real_x'].min()
                stats['y_range_inches'] = real_coords['glove_real_y'].max() - real_coords['glove_real_y'].min()
                
                # Average position
                stats['avg_x_position_inches'] = real_coords['glove_real_x'].mean()
                stats['avg_y_position_inches'] = real_coords['glove_real_y'].mean()
        
        return stats
    
    def _analyze_tracking_data(self) -> Optional[Dict]:
        """
        Analyze the current tracking data.
        
        Returns:
            dict: Movement statistics
        """
        if not self.tracking_data:
            return None
        
        data = []
        for detection in self.tracking_data:
            row = {
                'frame_idx': detection['frame_idx'],
                'glove_real_x': None,
                'glove_real_y': None,
                'baseball_real_x': None,
                'baseball_real_y': None
            }
            
            if 'glove' in detection['real_world_coords']:
                row['glove_real_x'], row['glove_real_y'] = detection['real_world_coords']['glove']
            
            if 'baseball' in detection['real_world_coords']:
                row['baseball_real_x'], row['baseball_real_y'] = detection['real_world_coords']['baseball']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        stats = {
            'total_frames': len(self.tracking_data),
            'frames_with_glove': sum(1 for d in self.tracking_data if d['glove'] is not None),
            'frames_with_baseball': sum(1 for d in self.tracking_data if d['baseball'] is not None),
            'frames_with_homeplate': sum(1 for d in self.tracking_data if d['homeplate'] is not None),
        }
        
        real_coords = df[df['glove_real_x'].notna() & df['glove_real_y'].notna()]
        
        if len(real_coords) > 1:
            dx = real_coords['glove_real_x'].diff()
            dy = real_coords['glove_real_y'].diff()
            distances = np.sqrt(dx**2 + dy**2)
            
            stats['total_distance_inches'] = distances.sum()
            stats['max_distance_between_frames_inches'] = distances.max()
            stats['avg_distance_between_frames_inches'] = distances.mean()
            
            # Range of motion
            stats['x_range_inches'] = real_coords['glove_real_x'].max() - real_coords['glove_real_x'].min()
            stats['y_range_inches'] = real_coords['glove_real_y'].max() - real_coords['glove_real_y'].min()
            
            # Average position
            stats['avg_x_position_inches'] = real_coords['glove_real_x'].mean()
            stats['avg_y_position_inches'] = real_coords['glove_real_y'].mean()
        
        return stats

    def _handle_missing_detections(self) -> Tuple[List[float], List[float]]:
        """
        Enhanced method to handle missing glove detections using multiple strategies.
        Uses a combination of interpolation, filtering, and smoothing techniques
        to create a more natural and continuous glove movement trajectory.
        
        Returns:
            List[Tuple[float, float]]: Processed glove coordinates for all frames
        """
        if not self.tracking_data:
            return [], []
        
        frame_indices = []
        glove_coords = []
        
        for detection in self.tracking_data:
            frame_indices.append(detection['frame_idx'])
            
            if 'glove' in detection['real_world_coords']:
                glove_coords.append(detection['real_world_coords']['glove'])
            else:
                glove_coords.append(None)
        
        # Find sequences of valid detections with improved gap handling
        valid_sequences = []
        current_sequence = []
        current_frames = []
        last_valid_frame = -999  # Initialize with an impossible frame index
        max_gap = 3  # Allow gaps of up to 3 frames within a sequence
        
        for i, (coords, frame_idx) in enumerate(zip(glove_coords, frame_indices)):
            if coords is not None:
                # If this is a continuation of the current sequence or a small gap
                if not current_sequence or frame_idx - last_valid_frame <= max_gap:
                    # Fill any gap with interpolated values
                    if current_sequence and frame_idx - last_valid_frame > 1:
                        # Interpolate missing frames
                        prev_coords = current_sequence[-1]
                        for gap_frame in range(last_valid_frame + 1, frame_idx):
                            # Linear interpolation
                            fraction = (gap_frame - last_valid_frame) / (frame_idx - last_valid_frame)
                            interp_x = prev_coords[0] + (coords[0] - prev_coords[0]) * fraction
                            interp_y = prev_coords[1] + (coords[1] - prev_coords[1]) * fraction
                            current_sequence.append((interp_x, interp_y))
                            current_frames.append(gap_frame)
                    
                    current_sequence.append(coords)
                    current_frames.append(frame_idx)
                    last_valid_frame = frame_idx
                else:
                    # Gap too large, end current sequence and start a new one
                    if len(current_sequence) >= 2:
                        valid_sequences.append((current_frames, current_sequence))
                    current_sequence = [coords]
                    current_frames = [frame_idx]
                    last_valid_frame = frame_idx
            elif current_sequence and i == len(glove_coords) - 1:
                # End of data, add the current sequence if valid
                if len(current_sequence) >= 2:
                    valid_sequences.append((current_frames, current_sequence))
                current_sequence = []
                current_frames = []
            elif current_sequence and frame_idx - last_valid_frame > max_gap:
                # Too many missing frames, end the sequence
                if len(current_sequence) >= 2:
                    valid_sequences.append((current_frames, current_sequence))
                current_sequence = []
                current_frames = []
        
        # Add the last sequence if it's not empty
        if len(current_sequence) >= 2:
            valid_sequences.append((current_frames, current_sequence))
        
        if not valid_sequences:
            if not self.suppress_detection_warnings: # <-- Add this check
                self.logger.warning("No valid glove detection sequences found")
            return [], []
        
        # Post-process: Connect sequences that are close in time
        if len(valid_sequences) > 1:
            max_gap_between_sequences = 10  # Allow up to 10 frames between sequences
            connected_sequences = []
            current_frames, current_coords = valid_sequences[0]
            
            for i in range(1, len(valid_sequences)):
                next_frames, next_coords = valid_sequences[i]
                frame_gap = next_frames[0] - current_frames[-1]
                
                if frame_gap <= max_gap_between_sequences:
                    # Connect sequences with interpolation
                    last_coords = current_coords[-1]
                    first_coords = next_coords[0]
                    
                    # Interpolate the gap
                    for j in range(1, frame_gap):
                        gap_frame = current_frames[-1] + j
                        fraction = j / frame_gap
                        interp_x = last_coords[0] + (first_coords[0] - last_coords[0]) * fraction
                        interp_y = last_coords[1] + (first_coords[1] - last_coords[1]) * fraction
                        current_coords.append((interp_x, interp_y))
                        current_frames.append(gap_frame)
                    
                    # Add the next sequence
                    current_frames.extend(next_frames)
                    current_coords.extend(next_coords)
                else:
                    # Gap too large, store current sequence and start new one
                    connected_sequences.append((current_frames, current_coords))
                    current_frames, current_coords = next_frames, next_coords
            
            # Add the last connected sequence
            connected_sequences.append((current_frames, current_coords))
            valid_sequences = connected_sequences
        
        # Extract x and y coordinates from the first valid sequence
        # (or we could combine all sequences if there are multiple)
        frames, coords = valid_sequences[0]
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        
        # Apply a final velocity-based filter to remove any jumps
        filtered_x = []
        filtered_y = []
        
        if len(x_coords) > 1:
            # Add the first point
            filtered_x.append(x_coords[0])
            filtered_y.append(y_coords[0])
            
            for i in range(1, len(x_coords)):
                dx = x_coords[i] - filtered_x[-1]
                dy = y_coords[i] - filtered_y[-1]
                
                distance = (dx**2 + dy**2)**0.5
                
                # If distance is too large (a jump), limit it
                max_distance = 10.0  # max allowed distance in inches
                if distance > max_distance:
                    # Scale the movement to the maximum allowed distance
                    scale_factor = max_distance / distance
                    new_x = filtered_x[-1] + dx * scale_factor
                    new_y = filtered_y[-1] + dy * scale_factor
                    
                    filtered_x.append(new_x)
                    filtered_y.append(new_y)
                else:
                    filtered_x.append(x_coords[i])
                    filtered_y.append(y_coords[i])
        
        return filtered_x, filtered_y
    
        
    def _filter_glove_detection(self, prev_detection, current_detection, fps, max_velocity_inches_per_sec=120.0) -> bool:
        """
        Filter out physically impossible glove movements based on velocity constraints.
        
        Args:
            prev_detection (dict): Previous frame's detection data
            current_detection (dict): Current frame's detection data
            fps (float): Frames per second of the video
            max_velocity_inches_per_sec (float): Maximum plausible velocity of the glove in inches per second
            
        Returns:
            bool: True if the detection passes the filter, False if it's an outlier
        """
        # If no previous detection, accept current one
        if prev_detection is None or 'glove' not in prev_detection['real_world_coords']:
            return True
        
        # If no current glove detection, nothing to filter
        if 'glove' not in current_detection['real_world_coords']:
            return True
        
        prev_x, prev_y = prev_detection['real_world_coords']['glove']
        curr_x, curr_y = current_detection['real_world_coords']['glove']
        distance = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
        
        time_diff = (current_detection['frame_idx'] - prev_detection['frame_idx']) / fps
        
        # Calculate velocity in inches per second
        if time_diff > 0:
            velocity = distance / time_diff
        else:
            velocity = float('inf')  # Avoid division by zero
        
        if velocity > max_velocity_inches_per_sec:
            self.logger.debug(f"Filtered outlier at frame {current_detection['frame_idx']}: velocity={velocity:.2f} in/s > {max_velocity_inches_per_sec} in/s")
            return False
        
        return True
    
    def _generate_batch_summary(self, combined_df: pd.DataFrame, summary_path: str) -> None:
        """
        Generate summary statistics from batch processing.
        
        Args:
            combined_df (pd.DataFrame): Combined tracking data
            summary_path (str): Path to save summary CSV
        """
        if combined_df.empty:
            return
        
        # Group by video filename
        video_groups = combined_df.groupby('video_filename')
        
        summary_data = []
        for video_name, group in video_groups:
            # Filter to glove positions
            glove_data = group[group['glove_real_x'].notna() & group['glove_real_y'].notna()]
            
            if not glove_data.empty:
                # Calculate movement
                dx = glove_data['glove_real_x'].diff().dropna()
                dy = glove_data['glove_real_y'].diff().dropna()
                distances = np.sqrt(dx**2 + dy**2)
                
                summary = {
                    'video_filename': video_name,
                    'frame_count': len(group),
                    'frames_with_glove': len(glove_data),
                    'frames_with_baseball': group['baseball_real_x'].notna().sum(),
                    'frames_with_homeplate': group['homeplate_center_x'].notna().sum(),
                    'total_distance_inches': distances.sum(),
                    'max_glove_movement_inches': distances.max() if not distances.empty else None,
                    'avg_glove_movement_inches': distances.mean() if not distances.empty else None,
                    'glove_x_range_inches': glove_data['glove_real_x'].max() - glove_data['glove_real_x'].min() if len(glove_data) > 1 else None,
                    'glove_y_range_inches': glove_data['glove_real_y'].max() - glove_data['glove_real_y'].min() if len(glove_data) > 1 else None,
                    'avg_glove_x_position': glove_data['glove_real_x'].mean() if not glove_data.empty else None,
                    'avg_glove_y_position': glove_data['glove_real_y'].mean() if not glove_data.empty else None
                }
                summary_data.append(summary)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_path, index=False)
            self.logger.info(f"Batch summary saved to {summary_path}")

    def _generate_combined_heatmap(self, combined_df: pd.DataFrame, output_path: str) -> Optional[str]:
        """
        Generate a combined heatmap from all video tracking data.
        
        Args:
            combined_df (pd.DataFrame): Combined tracking data
            output_path (str): Path to save the heatmap
            
        Returns:
            str: Path to the saved heatmap or None if generation failed
        """
        if combined_df.empty:
            self.logger.warning("Empty DataFrame, cannot generate heatmap")
            return None
        
        glove_data = combined_df[combined_df['glove_real_x'].notna() & combined_df['glove_real_y'].notna()]
        
        if len(glove_data) < 2:
            self.logger.warning("Not enough glove position data for heatmap")
            return None
        
        plt.figure(figsize=(10, 8))
        
        plt.hist2d(glove_data['glove_real_x'], glove_data['glove_real_y'], bins=20, cmap='hot')
        plt.colorbar(label='Frequency')
        
        home_plate_shape = np.array([[-8.5, 0], [8.5, 0], [0, 8.5], [-8.5, 0]])
        plt.fill(home_plate_shape[:, 0], home_plate_shape[:, 1], color='gray', alpha=0.5)
        
        plt.title('Combined Glove Position Heatmap')
        plt.xlabel('X Position (inches from home plate)'); plt.ylabel('Y Position (inches from home plate)')
        plt.grid(True, alpha=0.3)
        plt.xlim(-50, 50); plt.ylim(-10, 60)
        
        plt.gca().set_aspect('equal', 'box')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Combined heatmap saved to {output_path}")
        return output_path
    