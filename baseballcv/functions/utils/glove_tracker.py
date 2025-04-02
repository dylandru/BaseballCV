# baseballcv/functions/utils/glove_tracker.py
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from ultralytics import YOLO
import supervision as sv
from baseballcv.utilities import BaseballCVLogger, ProgressBar
from baseballcv.functions.load_tools import LoadTools

class GloveTracker:
    """
    Class for tracking the catcher's glove, home plate, and baseball in videos.
    
    This class uses the YOLO model to detect and track these objects, and provides
    visualization and data export capabilities for analysis.
    """
    
    def __init__(
        self, 
        model_alias: str = 'glove_tracking',
        results_dir: str = "glove_tracking_results",
        device: str = None,
        confidence_threshold: float = 0.5,
        enable_filtering: bool = True,
        max_velocity_inches_per_sec: float = 120.0,
        logger: Optional[BaseballCVLogger] = None
    ):
        """
        Initialize the GloveTracker.
        
        Args:
            model_alias (str): The alias of the model to use for detection
            results_dir (str): Directory to save results
            device (str): Device to run the model on (cpu, cuda, mps)
            confidence_threshold (float): Confidence threshold for detections
            enable_filtering (bool): Whether to enable outlier filtering for glove detections
            max_velocity_inches_per_sec (float): Maximum plausible velocity for glove movement (for filtering)
            logger (BaseballCVLogger): Logger instance for logging
        """
        self.load_tools = LoadTools()
        self.logger = logger if logger else BaseballCVLogger.get_logger(self.__class__.__name__)
        self.model_path = self.load_tools.load_model(model_alias)
        self.model = YOLO(self.model_path)
        self.device = device if device else 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        self.confidence_threshold = confidence_threshold
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Enable filtering and set velocity threshold
        self.enable_filtering = enable_filtering
        self.max_velocity_inches_per_sec = max_velocity_inches_per_sec
        
        # Get class names from the model
        self.class_names = self.model.names
        self.logger.info(f"Model loaded with classes: {self.class_names}")
        
        # Define important class indices
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
        
        # For storing tracking data
        self.tracking_data = []
        
        # Home plate reference (to be set during tracking)
        self.home_plate_reference = None
        self.pixels_per_inch = None

    def track_video(self, video_path: str, output_path: Optional[str] = None,
                    show_plot: bool = True) -> str:
        """
        Track objects in a video and generate visualization with 2D tracking plot.

        Args:
            video_path (str): Path to input video
            output_path (str): Path for output video (if None, auto-generated in results_dir)
            show_plot (bool): Whether to show the 2D plot in the output video

        Returns:
            str: Path to the output video
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

        # Create output paths if not provided
        if output_path is None:
            video_filename = os.path.basename(video_path)
            output_path = os.path.join(self.results_dir, f"tracked_{video_filename}")

        # Construct the CSV path correctly
        csv_filename = os.path.splitext(os.path.basename(output_path))[0] + "_tracking.csv"
        csv_path = os.path.join(self.results_dir, csv_filename)

        # Open video
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # If show_plot is True, we'll create a wider output to accommodate the plot
        out_width = width * 2 if show_plot else width
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

        # Reset tracking data
        self.tracking_data = []
        self.home_plate_reference = None
        self.pixels_per_inch = None

        # Create plot for glove tracking
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Process frames
        progress_bar = ProgressBar(total=total_frames, desc="Processing Video")

        frame_idx = 0

        with progress_bar as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Run YOLO detection on the frame
                results = self.model.predict(frame, conf=self.confidence_threshold, device=self.device, verbose=False)

                # Process and extract detections - now passing fps
                detections = self._process_detections(results, frame, frame_idx, fps)

                # Draw annotations on the frame
                annotated_frame = self._annotate_frame(frame.copy(), detections, frame_idx)

                # Create the 2D tracking plot if needed
                if show_plot:
                    self._update_tracking_plot(ax, fig)

                    # Convert matplotlib figure to image
                    fig.canvas.draw()
                    buf = fig.canvas.buffer_rgba()
                    plot_img = np.asarray(buf)[:, :, :3]  # Convert RGBA to RGB by taking only first 3 channels
                    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    # Resize plot to match frame height
                    plot_img = cv2.resize(plot_img, (width, height))

                    # Create combined frame with original and plot side by side
                    combined_frame = np.hstack((annotated_frame, plot_img))
                    out.write(combined_frame)
                else:
                    out.write(annotated_frame)

                frame_idx += 1
                pbar.update(1)

        # Clean up
        cap.release()
        out.release()
        plt.close(fig)

        # Save tracking data to CSV
        self._save_tracking_data(csv_path)

        self.logger.info(f"Tracking completed. Output video saved to {output_path}")
        self.logger.info(f"Tracking data saved to {csv_path}")

        return output_path


    def _process_detections(self, results, frame, frame_idx, fps=30.0):
        """
        Process YOLO detection results for the current frame with outlier filtering.
        
        Args:
            results: YOLO detection results
            frame: Current video frame
            frame_idx: Frame index
            fps: Frames per second of the video
                
        Returns:
            dict: Processed detections with glove, homeplate, and baseball info
        """
        detections = {
            'frame_idx': frame_idx,
            'glove': None,
            'homeplate': None,
            'baseball': None,
            'real_world_coords': {}
        }
        
        frame_height, frame_width = frame.shape[:2]
        
        # Extract detections from results
        for result in results:
            for box_idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()  # Convert to numpy for easier handling
                
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

    def _annotate_frame(self, frame, detections, frame_idx):
        """
        Annotate the frame with detections and tracking information.
        
        Args:
            frame: Current video frame
            detections: Processed detections
            frame_idx: Frame index
            
        Returns:
            annotated_frame: Frame with annotations
        """
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw home plate
        if detections['homeplate'] is not None:
            x1, y1, x2, y2 = detections['homeplate']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"Home Plate", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw center of home plate
            center_x, center_y = detections['homeplate']['center']
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 255, 0), -1)
            
            # Add home plate dimensions if pixels_per_inch is available
            if self.pixels_per_inch is not None:
                cv2.putText(frame, f"Home Plate Width: 17.0 in", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, f"Scale: {self.pixels_per_inch:.2f} px/in", (x1, y2 + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw glove
        if detections['glove'] is not None:
            x1, y1, x2, y2 = detections['glove']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Glove", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center of glove
            center_x, center_y = detections['glove']['center']
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
            
            # Add real-world coordinates if available
            if 'glove' in detections['real_world_coords']:
                real_x, real_y = detections['real_world_coords']['glove']
                cv2.putText(frame, f"Pos: ({real_x:.1f}, {real_y:.1f}) in", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw baseball
        if detections['baseball'] is not None:
            x1, y1, x2, y2 = detections['baseball']['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Baseball", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw center of baseball
            center_x, center_y = detections['baseball']['center']
            cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            
            # Add real-world coordinates if available
            if 'baseball' in detections['real_world_coords']:
                real_x, real_y = detections['real_world_coords']['baseball']
                cv2.putText(frame, f"Pos: ({real_x:.1f}, {real_y:.1f}) in", (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

    def _update_tracking_plot(self, ax, fig):
        """
        Update the 2D tracking plot with latest data, showing only glove movement.

        Args:
            ax: Matplotlib axis
            fig: Matplotlib figure
        """
        ax.clear()

        # Set up the plot
        ax.set_title('Glove Movement Tracking')
        ax.set_xlabel('X Position (inches from home plate)')
        ax.set_ylabel('Y Position (inches from home plate)')
        ax.grid(True)

        # Get glove tracking points only
        glove_x = []
        glove_y = []

        for detection in self.tracking_data:
            if 'glove' in detection['real_world_coords']:
                x, y = detection['real_world_coords']['glove']
                glove_x.append(x)
                glove_y.append(y)

        # Draw home plate at origin
        if self.home_plate_reference is not None:
            # Simplified home plate shape at the origin
            home_plate_shape = np.array([[-8.5, 0], [8.5, 0], [0, 8.5], [-8.5, 0]])
            ax.fill(home_plate_shape[:, 0], home_plate_shape[:, 1], color='gray', alpha=0.5, label='Home Plate')

        # Plot tracking data (glove only)
        if glove_x and glove_y:
            ax.plot(glove_x, glove_y, 'g-', label='Glove Path')
            ax.scatter(glove_x[-1], glove_y[-1], color='green', s=100, marker='o', label='Current Glove Pos')

        # Set fixed axis limits as requested
        ax.set_xlim(-50, 50)
        ax.set_ylim(-10, 60)
        
        # Set aspect ratio to equal to maintain 1:1 ratio
        ax.set_aspect('equal')

        # Add legend
        ax.legend(loc='upper right')

        # Make sure the plot refreshes
        fig.canvas.draw()

        # Convert matplotlib figure to image
        plot_img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2RGB)

        return plot_img


    def _save_tracking_data(self, csv_path):
        """
        Save tracking data to a CSV file.

        Args:
            csv_path: Path to save the CSV file
        """
        if not self.tracking_data:
            self.logger.warning("No tracking data to save")
            return

        # Prepare data for CSV
        csv_data = []

        for detection in self.tracking_data:
            frame_idx = detection['frame_idx']

            # Home plate data
            homeplate_center_x = None
            homeplate_center_y = None
            homeplate_width = None
            homeplate_confidence = None

            if detection['homeplate'] is not None:
                homeplate_center_x, homeplate_center_y = detection['homeplate']['center']
                homeplate_width = detection['homeplate']['width']
                homeplate_confidence = detection['homeplate']['confidence']

            # Glove data
            glove_center_x = None
            glove_center_y = None
            glove_confidence = None
            glove_real_x = None
            glove_real_y = None

            if detection['glove'] is not None:
                glove_center_x, glove_center_y = detection['glove']['center']
                glove_confidence = detection['glove']['confidence']

                if 'glove' in detection['real_world_coords']:
                    glove_real_x, glove_real_y = detection['real_world_coords']['glove']

            # Baseball data
            baseball_center_x = None
            baseball_center_y = None
            baseball_confidence = None
            baseball_real_x = None
            baseball_real_y = None

            if detection['baseball'] is not None:
                baseball_center_x, baseball_center_y = detection['baseball']['center']
                baseball_confidence = detection['baseball']['confidence']

                if 'baseball' in detection['real_world_coords']:
                    baseball_real_x, baseball_real_y = detection['real_world_coords']['baseball']

            # Add row to CSV data
            csv_data.append({
                'frame_idx': frame_idx,
                'homeplate_center_x': homeplate_center_x,
                'homeplate_center_y': homeplate_center_y,
                'homeplate_width': homeplate_width,
                'homeplate_confidence': homeplate_confidence,
                'glove_center_x': glove_center_x,
                'glove_center_y': glove_center_y,
                'glove_confidence': glove_confidence,
                'glove_real_x': glove_real_x,
                'glove_real_y': glove_real_y,
                'baseball_center_x': baseball_center_x,
                'baseball_center_y': baseball_center_y,
                'baseball_confidence': baseball_confidence,
                'baseball_real_x': baseball_real_x,
                'baseball_real_y': baseball_real_y,
                'pixels_per_inch': self.pixels_per_inch
            })

        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False)

        # Logging information
        self.logger.info(f"Tracking data saved to {csv_path}")
        
        # For debugging
        if not os.path.exists(csv_path):
            self.logger.error(f"Failed to save CSV file at {csv_path}")
        else:
            self.logger.debug(f"CSV file saved successfully at {csv_path}")

        
    def analyze_glove_movement(self, csv_path: Optional[str] = None):
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
            # Use the current tracking data
            return self._analyze_tracking_data()
        else:
            self.logger.warning("No tracking data available for analysis")
            return None
        
        # Calculate movement statistics
        stats = {
            'total_frames': len(df),
            'frames_with_glove': df['glove_center_x'].notna().sum(),
            'frames_with_baseball': df['baseball_center_x'].notna().sum(),
            'frames_with_homeplate': df['homeplate_center_x'].notna().sum(),
        }
        
        # Calculate glove movement distance (in real-world units)
        if stats['frames_with_glove'] > 1:
            # Get real-world coordinates where available
            real_coords = df[df['glove_real_x'].notna() & df['glove_real_y'].notna()]
            
            if len(real_coords) > 1:
                # Calculate the total distance travelled
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
    
    def _analyze_tracking_data(self):
        """
        Analyze the current tracking data.
        
        Returns:
            dict: Movement statistics
        """
        if not self.tracking_data:
            return None
        
        # Convert tracking data to DataFrame for analysis
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
        
        # Calculate statistics
        stats = {
            'total_frames': len(self.tracking_data),
            'frames_with_glove': sum(1 for d in self.tracking_data if d['glove'] is not None),
            'frames_with_baseball': sum(1 for d in self.tracking_data if d['baseball'] is not None),
            'frames_with_homeplate': sum(1 for d in self.tracking_data if d['homeplate'] is not None),
        }
        
        # Calculate distances if we have real-world coordinates
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
    
    def plot_glove_heatmap(self, csv_path: Optional[str] = None, output_path: Optional[str] = None):
        """
        Create a heatmap of glove positions.
        
        Args:
            csv_path: Path to tracking CSV file
            output_path: Path to save the heatmap image
            
        Returns:
            str: Path to the saved heatmap image
        """
        if csv_path is not None:
            df = pd.read_csv(csv_path)
        elif self.tracking_data:
            # Convert tracking data to DataFrame
            data = []
            for detection in self.tracking_data:
                if 'glove' in detection['real_world_coords']:
                    x, y = detection['real_world_coords']['glove']
                    data.append({'glove_real_x': x, 'glove_real_y': y})
            df = pd.DataFrame(data)
        else:
            self.logger.warning("No tracking data available for heatmap")
            return None
        
        # Generate heatmap
        plt.figure(figsize=(10, 8))
        
        # Create heatmap from glove positions
        if 'glove_real_x' in df.columns and 'glove_real_y' in df.columns:
            real_coords = df[df['glove_real_x'].notna() & df['glove_real_y'].notna()]
            
            if len(real_coords) > 1:
                plt.hist2d(real_coords['glove_real_x'], real_coords['glove_real_y'], 
                        bins=20, cmap='hot')
                plt.colorbar(label='Frequency')
                
                # Draw home plate at origin
                home_plate_shape = np.array([[-8.5, 0], [8.5, 0], [0, 8.5], [-8.5, 0]])
                plt.fill(home_plate_shape[:, 0], home_plate_shape[:, 1], color='gray', alpha=0.5)
                
                plt.title('Glove Position Heatmap')
                plt.xlabel('X Position (inches from home plate)')
                plt.ylabel('Y Position (inches from home plate)')
                plt.grid(True, alpha=0.3)
                
                # Save the heatmap
                if output_path is None:
                    output_path = os.path.join(self.results_dir, "glove_heatmap.png")
                
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Glove heatmap saved to {output_path}")
                return output_path
            else:
                self.logger.warning("Not enough glove position data for heatmap")
                plt.close()
                return None
        else:
            self.logger.warning("No real-world glove coordinates available for heatmap")
            plt.close()
            return None
        
    def _filter_glove_detection(self, prev_detection, current_detection, fps, max_velocity_inches_per_sec=120.0):
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
        
        # Get real-world coordinates
        prev_x, prev_y = prev_detection['real_world_coords']['glove']
        curr_x, curr_y = current_detection['real_world_coords']['glove']
        
        # Calculate distance moved in real-world units (inches)
        distance = ((curr_x - prev_x)**2 + (curr_y - prev_y)**2)**0.5
        
        # Calculate time between frames
        time_diff = (current_detection['frame_idx'] - prev_detection['frame_idx']) / fps
        
        # Calculate velocity in inches per second
        if time_diff > 0:
            velocity = distance / time_diff
        else:
            velocity = float('inf')  # Avoid division by zero
        
        # Check if velocity exceeds maximum plausible limit
        if velocity > max_velocity_inches_per_sec:
            self.logger.debug(f"Filtered outlier at frame {current_detection['frame_idx']}: velocity={velocity:.2f} in/s > {max_velocity_inches_per_sec} in/s")
            return False
        
        return True