import os
from typing import Dict, List
from .utils import DistanceToZone, GloveTracker
from baseballcv.utilities import BaseballCVLogger

class BaseballTools:
    """
    Class for analyzing baseball videos with Computer Vision.
    
    Enhanced with multi-mode glove tracking capabilities.
    """
    def __init__(self, device: str = 'cpu', verbose: bool = True):
        """
        Initialize the BaseballTools class.

        Args:
            device (str): Device to use for the analysis (default is 'cpu')
            verbose (bool): Whether to print verbose output (default is True)
        """
        self.device = device
        self.verbose = verbose
        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__)

    def distance_to_zone(self, start_date: str = "2024-05-01", end_date: str = "2024-05-01", team_abbr: str = None, 
                         pitch_type: str = None, player: int = None,
                         max_videos: int = None, max_videos_per_game: int = None, create_video: bool = True, 
                         catcher_model: str = 'phc_detector', glove_model: str = 'glove_tracking', 
                         ball_model: str = 'ball_trackingv4', zone_vertical_adjustment: float = 0.5,
                         save_csv: bool = True, csv_path: str = None) -> list:
        """
        The DistanceToZone function calculates the distance of a pitch to the strike zone in a video.
        
        Args:
            start_date (str): Start date of the analysis
            end_date (str): End date of the analysis
            team_abbr (str): Team to analyze
            pitch_type (str): Pitch type to analyze
            player (int): Player to analyze
            max_videos (int): Maximum number of videos to analyze
            max_videos_per_game (int): Maximum number of videos per game to analyze
            create_video (bool): Whether to create a video of the analysis
            catcher_model (str): Path to the PHCDetector model
            glove_model (str): Path to the GloveTracking model
            ball_model (str): Path to the BallTracking model
            zone_vertical_adjustment (float): Factor to adjust strike zone vertically
            save_csv (bool): Whether to save analysis results to CSV
            csv_path (str): Custom path for CSV file
            
        Returns:
            list: List of results from the DistanceToZone class for each video analyzed.
        """
        from baseballcv.functions.utils import DistanceToZone
        
        dtoz = DistanceToZone(
            device=self.device, 
            verbose=self.verbose, 
            catcher_model=catcher_model, 
            glove_model=glove_model, 
            ball_model=ball_model,
            zone_vertical_adjustment=zone_vertical_adjustment,
            logger=self.logger
        )
        
        results = dtoz.analyze(
            start_date=start_date, 
            end_date=end_date, 
            team_abbr=team_abbr, 
            player=player,
            pitch_type=pitch_type, 
            max_videos=max_videos, 
            max_videos_per_game=max_videos_per_game, 
            create_video=create_video,
            save_csv=save_csv,
            csv_path=csv_path
        )
        
        return results

    def track_glove(self, video_path: str = None, output_path: str = None, 
                confidence_threshold: float = 0.5, device: str = None, 
                show_plot: bool = True, generate_heatmap: bool = True,
                enable_filtering: bool = True, max_velocity_inches_per_sec: float = 120.0,
                create_video: bool = True) -> dict:
        """
        Track the catcher's glove, home plate, and baseball in a video.
        
        This function uses the GloveTracker class to track these objects and 
        creates visualizations showing their movement throughout the video.
        
        Args:
            video_path (str): Path to the input video file
            output_path (str): Path to save the tracked output video (optional)
            confidence_threshold (float): Confidence threshold for detections
            device (str): Device to run the model on (cpu, cuda, mps)
            show_plot (bool): Whether to show the 2D tracking plot in the output video
            generate_heatmap (bool): Whether to generate a heatmap of glove positions
            enable_filtering (bool): Whether to enable filtering of outlier glove detections
            max_velocity_inches_per_sec (float): Maximum plausible velocity for filtering
            
        Returns:
            Dict: Results containing paths to output files and movement statistics
        """
        # Input validation
        if video_path is None or not os.path.exists(video_path):
            self.logger.error(f"Video file not found at {video_path}")
            return {"error": f"Video file not found at {video_path}"}
        
        # Initialize GloveTracker in regular mode
        tracker = GloveTracker(
            model_alias='glove_tracking',
            device=device if device else self.device,
            confidence_threshold=confidence_threshold,
            enable_filtering=enable_filtering,
            max_velocity_inches_per_sec=max_velocity_inches_per_sec,
            logger=self.logger,
            mode="regular"  # Using regular mode for single video processing
        )
        
        # Track the video
        # Pass create_video parameter
        output_video = tracker.track_video(
            video_path=video_path,
            output_path=output_path,
            show_plot=show_plot,
            create_video=create_video
        )
        
        # Get the CSV path
        csv_filename = os.path.splitext(os.path.basename(output_video))[0] + "_tracking.csv"
        csv_path = os.path.join(tracker.results_dir, csv_filename)
        
        # Analyze movement
        movement_stats = tracker.analyze_glove_movement(csv_path)
        
        # Generate heatmap if requested
        heatmap_path = None
        if generate_heatmap:
            heatmap_path = tracker.plot_glove_heatmap(csv_path)
        
        results = {
            "output_video": output_video,
            "tracking_data": csv_path,
            "movement_stats": movement_stats,
            "heatmap": heatmap_path,
            "filtering_applied": enable_filtering,
            "max_velocity_threshold": max_velocity_inches_per_sec if enable_filtering else None
        }
        
        return results
    
    def batch_track_gloves(self, input_folder: str, 
                      delete_after_processing: bool = False, 
                      skip_confirmation: bool = False,
                      confidence_threshold: float = 0.5, 
                      device: str = None,
                      show_plot: bool = True, 
                      generate_heatmap: bool = True,
                      enable_filtering: bool = True, 
                      max_velocity_inches_per_sec: float = 120.0,
                      max_workers: int = 1,
                      create_video: bool = True) -> dict:
        """
        Batch process multiple videos to track gloves and analyze movement.
        
        Args:
            input_folder (str): Folder containing videos to process
            delete_after_processing (bool): Whether to delete videos after processing
            skip_confirmation (bool): Skip deletion confirmation dialog
            confidence_threshold (float): Confidence threshold for detections
            device (str): Device to run the model on (cpu, cuda, mps)
            show_plot (bool): Whether to show the 2D tracking plot in the output videos
            generate_heatmap (bool): Whether to generate heatmaps of glove positions
            enable_filtering (bool): Whether to enable filtering of outlier glove detections
            max_velocity_inches_per_sec (float): Maximum plausible velocity for filtering
            max_workers (int): Maximum number of parallel workers (1 = sequential)
            
        Returns:
            Dict: Results containing paths to output files and batch statistics
        """
        # Initialize GloveTracker in batch mode
        tracker = GloveTracker(
            model_alias='glove_tracking',
            device=device if device else self.device,
            confidence_threshold=confidence_threshold,
            enable_filtering=enable_filtering,
            max_velocity_inches_per_sec=max_velocity_inches_per_sec,
            logger=self.logger,
            mode="batch"  # Using batch mode for processing multiple videos
        )
        
        # Run batch processing
        # Pass create_video parameter
        combined_csv = tracker.batch_process(
            input_folder=input_folder,
            delete_after_processing=delete_after_processing,
            skip_confirmation=skip_confirmation,
            show_plot=show_plot,
            create_video=create_video,
            max_workers=max_workers
        )
        
        if combined_csv:
            # Generate a combined heatmap if requested
            combined_heatmap = None
            if generate_heatmap:
                combined_heatmap = tracker.plot_glove_heatmap(combined_csv)
            
            results = {
                "combined_csv": combined_csv,
                "combined_heatmap": combined_heatmap,
                "results_dir": tracker.results_dir,
                "num_videos_processed": sum(1 for line in open(combined_csv) if line.strip()) - 1  # Subtract header
            }
        else:
            results = {
                "error": "Batch processing failed or no valid videos were found",
                "results_dir": tracker.results_dir
            }
        
        return results
    
    def scrape_and_track_gloves(self, 
                           start_date: str, 
                           end_date: str = None,
                           team_abbr: str = None,
                           player: int = None,
                           pitch_type: str = None,
                           max_videos: int = 10,
                           max_videos_per_game: int = None,
                           delete_after_processing: bool = True,
                           skip_confirmation: bool = False,
                           confidence_threshold: float = 0.5, 
                           device: str = None,
                           show_plot: bool = True, 
                           generate_heatmap: bool = True,
                           enable_filtering: bool = True, 
                           max_velocity_inches_per_sec: float = 120.0,
                           create_video: bool = True) -> dict:
        """
        Scrape videos from Baseball Savant and track gloves in them.
        
        This function combines the Savant scraper with glove tracking to provide
        a complete pipeline from video acquisition to movement analysis.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format (optional)
            team_abbr (str): Team abbreviation (optional)
            player (int): Player ID (optional)
            pitch_type (str): Pitch type (optional)
            max_videos (int): Maximum number of videos to process
            max_videos_per_game (int): Maximum videos per game
            delete_after_processing (bool): Whether to delete videos after processing
            skip_confirmation (bool): Skip confirmation for deletion
            confidence_threshold (float): Confidence threshold for detections
            device (str): Device to run the model on (cpu, cuda, mps)
            show_plot (bool): Whether to show the 2D tracking plot in the output videos
            generate_heatmap (bool): Whether to generate heatmaps of glove positions
            enable_filtering (bool): Whether to enable filtering of outlier glove detections
            max_velocity_inches_per_sec (float): Maximum plausible velocity for filtering
            
        Returns:
            Dict: Results containing paths to output files and analysis statistics
        """
        # Initialize GloveTracker in scrape mode
        tracker = GloveTracker(
            model_alias='glove_tracking',
            device=device if device else self.device,
            confidence_threshold=confidence_threshold,
            enable_filtering=enable_filtering,
            max_velocity_inches_per_sec=max_velocity_inches_per_sec,
            logger=self.logger,
            mode="scrape"  # Using scrape mode for integration with BaseballSavVideoScraper
        )
        
        # Run scrape and process workflow
        # Pass create_video parameter
        combined_csv = tracker.scrape_and_process(
            start_date=start_date,
            end_date=end_date,
            team_abbr=team_abbr,
            player=player,
            pitch_type=pitch_type,
            max_videos=max_videos,
            max_videos_per_game=max_videos_per_game,
            delete_after_processing=delete_after_processing,
            skip_confirmation=skip_confirmation,
            show_plot=show_plot,
            create_video=create_video
        )
        
        if combined_csv:
            # Generate a combined heatmap if requested
            combined_heatmap = None
            if generate_heatmap:
                combined_heatmap = tracker.plot_glove_heatmap(combined_csv)
            
            # Read CSV to get statistics
            import pandas as pd
            df = pd.read_csv(combined_csv)
            
            results = {
                "combined_csv": combined_csv,
                "combined_heatmap": combined_heatmap,
                "results_dir": tracker.results_dir,
                "num_videos_processed": len(df['video_filename'].unique()),
                "total_frames_analyzed": len(df),
                "frames_with_glove": df['glove_real_x'].notna().sum(),
                "frames_with_baseball": df['baseball_real_x'].notna().sum(),
                "statcast_data_available": any(col.startswith('statcast_') for col in df.columns)
            }
        else:
            results = {
                "error": "Scraping and processing failed or no valid videos were found",
                "results_dir": tracker.results_dir
            }
        
        return results