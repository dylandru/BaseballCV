import cv2
import torch
import numpy as np
import io
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import pandas as pd
import math
from tqdm import tqdm
from contextlib import redirect_stdout
from baseballcv.functions.utils.distance_to_zone import DistanceToZone

class GloveFramingTracker(DistanceToZone):
    """
    Class for tracking and visualizing catcher's glove framing movements relative 
    to the strike zone with enhanced coordinate handling.
    """
    
    def __init__(self, 
                 catcher_model: str = 'phc_detector',
                 glove_model: str = 'glove_tracking',
                 ball_model: str = 'ball_trackingv4',
                 homeplate_model: str = 'glove_tracking',
                 results_dir: str = "results",
                 verbose: bool = True,
                 device: str = None,
                 zone_vertical_adjustment: float = 0.5):
        """Initialize the GloveFramingTracker class extending DistanceToZone."""
        super().__init__(catcher_model, glove_model, ball_model, homeplate_model,
                         results_dir, verbose, device, zone_vertical_adjustment)
        self.glove_positions = []  # List to store glove positions over time
        self.glove_physical_positions = []  # List to store physical coordinates
        
    def track_glove_movement(self, video_path: str, start_frame: int = 0, 
                           end_frame: int = None) -> List[Dict]:
        """Track glove movement through a sequence of frames."""
        self.glove_positions = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None or end_frame > total_frames:
            end_frame = total_frames
            
        if self.verbose:
            print(f"Tracking glove movement from frame {start_frame} to {end_frame}")
            
        for frame_idx in tqdm(range(start_frame, end_frame), 
                              desc="Tracking Glove", disable=not self.verbose):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect glove in frame
            with io.StringIO() as buf, redirect_stdout(buf):
                results = self.glove_model.predict(frame, conf=0.5, device=self.device, verbose=False)
                
            glove_detected = False
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls)
                    cls_name = self.glove_model.names[cls].lower()
                    
                    if cls_name == "glove":
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf)
                        glove_center_x = (x1 + x2) / 2
                        glove_center_y = (y1 + y2) / 2
                        
                        self.glove_positions.append({
                            "frame": frame_idx,
                            "center_x": glove_center_x,
                            "center_y": glove_center_y,
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "confidence": conf
                        })
                        glove_detected = True
                        break
                        
                if glove_detected:
                    break
        
        cap.release()
        
        if self.verbose:
            print(f"Tracked glove in {len(self.glove_positions)} frames")
            
        return self.glove_positions
        
    def convert_to_physical_coordinates(self, pitch_data: pd.Series, 
                                        homeplate_box: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Convert pixel coordinates to physical coordinates (inches).
        Uses home plate width (17 inches) as reference for calibration.
        """
        if not self.glove_positions:
            if self.verbose:
                print("No glove positions to convert. Run track_glove_movement first.")
            return []
            
        self.glove_physical_positions = []
        plate_width_pixels = homeplate_box[2] - homeplate_box[0]
        plate_width_inches = 17.0  # Standard home plate width
        
        # Calculate pixels per inch
        pixels_per_inch = plate_width_pixels / plate_width_inches
        
        # Calculate plate center
        plate_center_x = (homeplate_box[0] + homeplate_box[2]) / 2
        plate_bottom_y = homeplate_box[3]  # Bottom of home plate (ground level)
        
        for pos in self.glove_positions:
            pixel_x, pixel_y = pos["center_x"], pos["center_y"]
            
            # Convert to physical coordinates (inches)
            # X: positive is right of plate center, negative is left
            physical_x = (pixel_x - plate_center_x) / pixels_per_inch
            
            # Y: positive is up from ground, negative is below ground
            # Need to flip the y-axis since pixel coordinates increase downward
            physical_y = (plate_bottom_y - pixel_y) / pixels_per_inch
            
            # Add to the list
            phys_pos = pos.copy()
            phys_pos["physical_x"] = physical_x
            phys_pos["physical_y"] = physical_y
            self.glove_physical_positions.append(phys_pos)
            
        return self.glove_physical_positions
    
    def visualize_glove_framing(self, pitch_data: pd.Series, 
                              homeplate_box: Tuple[int, int, int, int],
                              output_path: str = None, show: bool = True) -> plt.Figure:
        """
        Creates an enhanced visualization of glove framing with proper coordinate handling.
        
        Args:
            pitch_data: Pitch data containing strike zone information
            homeplate_box: Home plate bounding box (x1, y1, x2, y2)
            output_path: Path to save the visualization
            show: Whether to show the plot
            
        Returns:
            Figure object of the visualization
        """
        if not self.glove_positions:
            if self.verbose:
                print("No glove positions to visualize. Run track_glove_movement first.")
            return None
            
        # Convert to physical coordinates if not already done
        if not self.glove_physical_positions:
            self.convert_to_physical_coordinates(pitch_data, homeplate_box)
            
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get strike zone dimensions from Statcast data
        sz_top = float(pitch_data["sz_top"])  # in feet
        sz_bot = float(pitch_data["sz_bot"])  # in feet
        
        # Convert to inches
        sz_top_inches = sz_top * 12
        sz_bot_inches = sz_bot * 12
        sz_width_inches = 17.0  # Standard strike zone width
        
        # Plot glove positions
        x_vals = [pos["physical_x"] for pos in self.glove_physical_positions]
        y_vals = [pos["physical_y"] for pos in self.glove_physical_positions]
        
        # Draw glove path
        ax.plot(x_vals, y_vals, 'b-', linewidth=1.5, alpha=0.7)
        
        # Draw points for each glove position
        sc = ax.scatter(x_vals, y_vals, c=range(len(x_vals)), 
                       cmap='viridis', s=30, zorder=3)
        
        # Add colorbar to show frame progression
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('Frame Progression')
        
        # Draw strike zone
        sz_left = -sz_width_inches / 2
        sz_right = sz_width_inches / 2
        strike_zone = plt.Rectangle(
            (sz_left, sz_bot_inches), 
            sz_width_inches, 
            sz_top_inches - sz_bot_inches,
            fill=False, 
            edgecolor='r', 
            linewidth=2, 
            zorder=2,
            label='Strike Zone'
        )
        ax.add_patch(strike_zone)
        
        # Draw home plate
        plate_half_width = 8.5  # Half width of home plate in inches
        plate = plt.Rectangle(
            (-plate_half_width, 0), 
            2*plate_half_width, 
            plate_half_width,
            fill=True, 
            color='gray', 
            alpha=0.3, 
            zorder=1,
            label='Home Plate'
        )
        ax.add_patch(plate)
        
        # Determine plot limits to avoid clipping
        # Include strike zone and all glove positions
        data_x_min, data_x_max = min(x_vals + [sz_left]), max(x_vals + [sz_right])
        data_y_min, data_y_max = min(y_vals + [sz_bot_inches]), max(y_vals + [sz_top_inches])
        
        # Calculate padding to add around the data
        x_padding = (data_x_max - data_x_min) * 0.15
        y_padding = (data_y_max - data_y_min) * 0.15
        
        # Ensure minimum padding
        min_padding = 5.0  # inches
        x_padding = max(x_padding, min_padding)
        y_padding = max(y_padding, min_padding)
        
        # Set plot limits with padding
        x_min, x_max = data_x_min - x_padding, data_x_max + x_padding
        y_min, y_max = data_y_min - y_padding, data_y_max + y_padding
        
        # Ensure equal scaling for X and Y axes (aspect ratio 1:1)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        if x_range > y_range:
            # Make y-range match x-range
            center_y = (y_max + y_min) / 2
            y_min = center_y - x_range / 2
            y_max = center_y + x_range / 2
        else:
            # Make x-range match y-range
            center_x = (x_max + x_min) / 2
            x_min = center_x - y_range / 2
            x_max = center_x + y_range / 2
        
        # Set final plot limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Set aspect ratio to be equal
        ax.set_aspect('equal')
        
        # Add axis labels and title
        ax.set_xlabel('X-Position (inches from plate center)')
        ax.set_ylabel('Y-Position (inches from ground)')
        ax.set_title('Catcher Glove Framing Visualization')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend()
        
        # Add ball location if available
        if 'plate_x' in pitch_data and 'plate_z' in pitch_data:
            # Ball location in feet from center of plate
            # plate_x: negative is left of center from catcher's perspective
            # plate_z: height above ground
            plate_x = float(pitch_data["plate_x"]) * 12  # Convert to inches
            plate_z = float(pitch_data["plate_z"]) * 12  # Convert to inches
            
            ax.scatter(plate_x, plate_z, color='r', s=100, zorder=4, marker='o', 
                      label='Ball Location')
            
            # Draw line from glove end position to ball location
            if self.glove_physical_positions:
                end_pos = self.glove_physical_positions[-1]
                ax.plot([end_pos["physical_x"], plate_x], 
                       [end_pos["physical_y"], plate_z], 
                       'r--', linewidth=1.5, alpha=0.7)
                
                # Add distance annotation
                distance = math.sqrt(
                    (end_pos["physical_x"] - plate_x)**2 + 
                    (end_pos["physical_y"] - plate_z)**2
                )
                midpoint_x = (end_pos["physical_x"] + plate_x) / 2
                midpoint_y = (end_pos["physical_y"] + plate_z) / 2
                ax.annotate(
                    f"{distance:.1f}\"", 
                    (midpoint_x, midpoint_y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7)
                )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if self.verbose:
                print(f"Visualization saved to {output_path}")
        
        if show:
            plt.show()
            
        return fig
    
    def create_framing_video(self, video_path: str, pitch_data: pd.Series, 
                                output_path: str, homeplate_box: Optional[Tuple[int, int, int, int]] = None) -> str:
            """
            Create a video with both original footage and glove framing visualization.
            
            Args:
                video_path: Path to the input video
                pitch_data: Pitch data containing strike zone information
                output_path: Path to save the output
                homeplate_box: Optional home plate bounding box. If None, will be detected.
                
            Returns:
                Path to the created video
            """
            if not self.glove_positions:
                self.track_glove_movement(video_path)
            
            # Detect homeplate if not provided
            if homeplate_box is None:
                homeplate_box, _, _ = self._detect_homeplate(video_path)
                
            if homeplate_box is None:
                raise ValueError("Could not detect home plate. Please provide it manually.")
                
            # Convert to physical coordinates
            if not self.glove_physical_positions:
                self.convert_to_physical_coordinates(pitch_data, homeplate_box)
            
            # Open the input video
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_width = width + 500  # Extra space for the visualization
            output_height = height
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
            
            # Get strike zone dimensions
            sz_top = float(pitch_data["sz_top"]) * 12  # Convert to inches
            sz_bot = float(pitch_data["sz_bot"]) * 12  # Convert to inches
            sz_width_inches = 17.0  # Standard strike zone width
            
            # Calculate plot boundaries
            x_vals = [pos["physical_x"] for pos in self.glove_physical_positions]
            y_vals = [pos["physical_y"] for pos in self.glove_physical_positions]
            
            # Include strike zone in boundaries
            sz_left = -sz_width_inches / 2
            sz_right = sz_width_inches / 2
            
            data_x_min, data_x_max = min(x_vals + [sz_left]), max(x_vals + [sz_right])
            data_y_min, data_y_max = min(y_vals + [sz_bot]), max(y_vals + [sz_top])
            
            # Calculate padding
            x_padding = (data_x_max - data_x_min) * 0.15
            y_padding = (data_y_max - data_y_min) * 0.15
            
            # Ensure minimum padding
            min_padding = 5.0  # inches
            x_padding = max(x_padding, min_padding)
            y_padding = max(y_padding, min_padding)
            
            # Set plot limits with padding
            x_min, x_max = data_x_min - x_padding, data_x_max + x_padding
            y_min, y_max = data_y_min - y_padding, data_y_max + y_padding
            
            # Ensure equal scaling
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            if x_range > y_range:
                # Make y-range match x-range
                center_y = (y_max + y_min) / 2
                y_min = center_y - x_range / 2
                y_max = center_y + x_range / 2
            else:
                # Make x-range match y-range
                center_x = (x_max + x_min) / 2
                x_min = center_x - y_range / 2
                x_max = center_x + y_range / 2
                
            # Add ball location if available
            ball_x, ball_y = None, None
            if 'plate_x' in pitch_data and 'plate_z' in pitch_data:
                ball_x = float(pitch_data["plate_x"]) * 12  # Convert to inches
                ball_y = float(pitch_data["plate_z"]) * 12  # Convert to inches
            
            # Frame index to glove position mapping
            frame_to_glove = {pos["frame"]: pos for pos in self.glove_physical_positions}
            current_glove_positions = []
            
            # Process each frame
            pbar = tqdm(total=total_frames, desc="Creating Video", disable=not self.verbose)
            
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Create visualization canvas
                output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                
                # Copy original frame
                output_frame[:height, :width] = frame
                
                # Create plot for right side
                fig, ax = plt.figure(figsize=(5, 5), dpi=100), plt.gca()
                
                # Draw strike zone
                strike_zone = plt.Rectangle(
                    (sz_left, sz_bot), 
                    sz_width_inches, 
                    sz_top - sz_bot,
                    fill=False, 
                    edgecolor='r', 
                    linewidth=2, 
                    zorder=2
                )
                ax.add_patch(strike_zone)
                
                # Draw home plate
                plate_half_width = 8.5  # Half width of home plate in inches
                plate = plt.Rectangle(
                    (-plate_half_width, 0), 
                    2*plate_half_width, 
                    plate_half_width,
                    fill=True, 
                    color='gray', 
                    alpha=0.3, 
                    zorder=1
                )
                ax.add_patch(plate)
                
                # Update current glove positions
                if frame_idx in frame_to_glove:
                    pos = frame_to_glove[frame_idx]
                    current_glove_positions.append((pos["physical_x"], pos["physical_y"]))
                    
                # Draw glove path up to current frame
                if current_glove_positions:
                    x_path, y_path = zip(*current_glove_positions)
                    ax.plot(x_path, y_path, 'b-', linewidth=1.5, alpha=0.7)
                    
                    # Draw current glove position
                    ax.scatter([x_path[-1]], [y_path[-1]], s=100, c='blue', zorder=4)
                    
                    # Draw line from glove to ball if both are available
                    if ball_x is not None and ball_y is not None:
                        ax.scatter([ball_x], [ball_y], s=100, c='red', zorder=4)
                        
                        # Draw line from current glove to ball
                        ax.plot([x_path[-1], ball_x], [y_path[-1], ball_y], 'r--', linewidth=1.5, alpha=0.7)
                        
                        # Calculate and display distance
                        distance = math.sqrt((x_path[-1] - ball_x)**2 + (y_path[-1] - ball_y)**2)
                        midpoint_x = (x_path[-1] + ball_x) / 2
                        midpoint_y = (y_path[-1] + ball_y) / 2
                        ax.annotate(
                            f"{distance:.1f}\"", 
                            (midpoint_x, midpoint_y),
                            xytext=(5, 5),
                            textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7)
                        )
                
                # Set axis labels and limits
                ax.set_xlabel('X-Position (inches)')
                ax.set_ylabel('Y-Position (inches)')
                ax.set_title('Glove Movement')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_aspect('equal')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Convert plot to image
                fig.canvas.draw()
                plot_img = np.array(fig.canvas.renderer.buffer_rgba())
                plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
                
                # Resize plot to fit in the output frame
                plot_h, plot_w = plot_img.shape[:2]
                plot_aspect = plot_w / plot_h
                
                # Target dimensions for plot area
                target_h = output_height
                target_w = int(target_h * plot_aspect)
                
                # Resize plot
                plot_img = cv2.resize(plot_img, (target_w, target_h))
                
                # Calculate position to place the plot
                plot_x = width
                
                # Place plot in output frame
                if plot_x + target_w <= output_width:
                    output_frame[:target_h, plot_x:plot_x+target_w] = plot_img
                else:
                    # If plot is too wide, resize to fit
                    available_w = output_width - plot_x
                    resized_plot = cv2.resize(plot_img, (available_w, target_h))
                    output_frame[:target_h, plot_x:output_width] = resized_plot
                
                # Add frame to video
                out.write(output_frame)
                
                # Clear the figure to avoid memory leaks
                plt.close(fig)
                
                pbar.update(1)
                
            pbar.close()
            cap.release()
            out.release()
            
            if self.verbose:
                print(f"Video saved to {output_path}")
                
            return output_path
        
        def analyze_glove_framing(self, video_path: str, pitch_data: pd.Series) -> Dict:
            """
            Analyze glove framing metrics from the tracked positions.
            
            Args:
                video_path: Path to the input video
                pitch_data: Pitch data containing strike zone information
                
            Returns:
                Dictionary with framing metrics
            """
            if not self.glove_positions:
                self.track_glove_movement(video_path)
            
            # Detect homeplate
            homeplate_box, _, _ = self._detect_homeplate(video_path)
            
            if homeplate_box is None:
                raise ValueError("Could not detect home plate for framing analysis.")
                
            # Convert to physical coordinates
            if not self.glove_physical_positions:
                self.convert_to_physical_coordinates(pitch_data, homeplate_box)
            
            # Get ball location
            ball_x, ball_y = None, None
            if 'plate_x' in pitch_data and 'plate_z' in pitch_data:
                ball_x = float(pitch_data["plate_x"]) * 12  # Convert to inches
                ball_y = float(pitch_data["plate_z"]) * 12  # Convert to inches
            else:
                return {"error": "Ball location not available in pitch data"}
            
            # Get strike zone dimensions
            sz_top = float(pitch_data["sz_top"]) * 12  # Convert to inches
            sz_bot = float(pitch_data["sz_bot"]) * 12  # Convert to inches
            sz_width_inches = 17.0  # Standard strike zone width
            sz_left = -sz_width_inches / 2
            sz_right = sz_width_inches / 2
            
            # Check if ball is in strike zone
            ball_in_zone = (sz_left <= ball_x <= sz_right and sz_bot <= ball_y <= sz_top)
            
            # Get final glove position
            if not self.glove_physical_positions:
                return {"error": "No glove positions tracked"}
                
            final_pos = self.glove_physical_positions[-1]
            final_x, final_y = final_pos["physical_x"], final_pos["physical_y"]
            
            # Calculate distance from glove to ball
            distance_to_ball = math.sqrt((final_x - ball_x)**2 + (final_y - ball_y)**2)
            
            # Calculate distance from ball to nearest edge of strike zone
            if ball_in_zone:
                distance_to_zone_edge = min(
                    ball_x - sz_left,
                    sz_right - ball_x,
                    ball_y - sz_bot,
                    sz_top - ball_y
                )
            else:
                # Ball outside zone, find closest point on zone boundary
                closest_x = max(sz_left, min(ball_x, sz_right))
                closest_y = max(sz_bot, min(ball_y, sz_top))
                
                # If ball is already on the boundary in one dimension
                if closest_x == ball_x or closest_y == ball_y:
                    distance_to_zone_edge = math.sqrt(
                        (closest_x - ball_x)**2 + (closest_y - ball_y)**2
                    )
                else:
                    distance_to_zone_edge = -math.sqrt(
                        (closest_x - ball_x)**2 + (closest_y - ball_y)**2
                    )
            
            # Calculate total glove movement distance
            movement_distance = 0
            prev_x, prev_y = None, None
            
            for pos in self.glove_physical_positions:
                x, y = pos["physical_x"], pos["physical_y"]
                
                if prev_x is not None and prev_y is not None:
                    segment_distance = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                    movement_distance += segment_distance
                    
                prev_x, prev_y = x, y
            
            # Calculate average movement speed (inches per frame)
            num_frames = len(self.glove_physical_positions)
            avg_speed = movement_distance / num_frames if num_frames > 1 else 0
            
            # Return metrics
            return {
                "ball_location_x": ball_x,
                "ball_location_y": ball_y,
                "ball_in_strike_zone": ball_in_zone,
                "distance_to_zone_edge": distance_to_zone_edge,
                "final_glove_x": final_x,
                "final_glove_y": final_y,
                "glove_to_ball_distance": distance_to_ball,
                "total_movement_distance": movement_distance,
                "average_movement_speed": avg_speed,
                "num_frames_tracked": num_frames
            }