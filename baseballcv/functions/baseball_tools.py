import tempfile
import pandas as pd
import glob
import time
import shutil
import os
from typing import Dict, List, Optional
import concurrent.futures
# Updated import to reflect the new EventDetector that handles model types
from .utils.baseball_utils import DistanceToZone, GloveTracker, CommandAnalyzer, EventDetector
from baseballcv.utilities import BaseballCVLogger
from baseballcv.functions.savant_scraper import BaseballSavVideoScraper

class BaseballTools:
    """
    Class for analyzing baseball videos with Computer Vision.
    
    Enhanced with multi-mode glove tracking, command analysis, and event-based video cutting.
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
                         save_csv: bool = True, csv_path: str = None) -> List:
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

    def track_gloves(self,
                    mode: str = "regular",
                    # Common parameters
                    device: str = None, # Specific device for this tool, overrides class default if set
                    confidence_threshold: float = 0.5,
                    enable_filtering: bool = True,
                    max_velocity_inches_per_sec: float = 120.0,
                    show_plot: bool = True,
                    generate_heatmap: bool = True,
                    create_video: bool = True,
                    # Regular mode parameters
                    video_path: str = None,
                    output_path: str = None, # For regular mode, this is the output video file path
                    # Batch mode parameters
                    input_folder: str = None,
                    delete_after_processing: bool = False,
                    skip_confirmation: bool = False,
                    max_workers: int = 1,
                    generate_batch_info: bool = True,
                    # Scrape mode parameters
                    start_date: str = None,
                    end_date: str = None,
                    team_abbr: str = None,
                    player: int = None,
                    pitch_type: str = None,
                    max_videos: int = 10,
                    suppress_detection_warnings: bool = False,
                    max_videos_per_game: int = None) -> Dict:
        """
        Track the catcher's glove, home plate, and baseball in videos using one of three modes.
        
        Modes:
            "regular": Process a single video
            "batch": Process multiple videos in a folder
            "scrape": Download videos from Baseball Savant and process them
        
        Args:
            mode (str): Processing mode - "regular", "batch", or "scrape"
            
            # Common parameters for all modes
            device (str): Device to run the model on (cpu, cuda, mps). Overrides class default if set.
            confidence_threshold (float): Confidence threshold for detections
            enable_filtering (bool): Whether to enable filtering of outlier glove detections
            max_velocity_inches_per_sec (float): Maximum plausible velocity for filtering
            show_plot (bool): Whether to show the 2D tracking plot in the output video
            generate_heatmap (bool): Whether to generate a heatmap of glove positions
            create_video (bool): Whether to create an output video file
            
            # Regular mode parameters
            video_path (str): Path to the input video file (for regular mode)
            output_path (str): Path to save the tracked output video (for regular mode, optional)
            
            # Batch mode parameters
            input_folder (str): Folder containing videos to process (for batch mode)
            delete_after_processing (bool): Whether to delete videos after processing (for batch mode)
            skip_confirmation (bool): Skip deletion confirmation dialog (for batch mode)
            max_workers (int): Maximum number of parallel workers for batch processing
            generate_batch_info (bool): Whether to generate combined CSV, summary, and heatmap for batch.
            
            # Scrape mode parameters
            start_date (str): Start date in YYYY-MM-DD format (for scrape mode)
            end_date (str): End date in YYYY-MM-DD format (for scrape mode, optional)
            team_abbr (str): Team abbreviation (for scrape mode, optional)
            player (int): Player ID (for scrape mode, optional)
            pitch_type (str): Pitch type (for scrape mode, optional)
            max_videos (int): Maximum number of videos to download (for scrape mode)
            max_videos_per_game (int): Maximum videos per game (for scrape mode)
            suppress_detection_warnings (bool): Suppress warnings about missing detections during tracking.
            
        Returns:
            Dict: Results containing paths to output files and analysis statistics
        """
        valid_modes = ["regular", "batch", "scrape"]
        if mode not in valid_modes:
            self.logger.error(f"Invalid mode: {mode}. Must be one of {valid_modes}")
            return {"error": f"Invalid mode: {mode}. Must be one of {valid_modes}"}
        
        if mode == "regular" and (video_path is None or not os.path.exists(video_path)):
            self.logger.error(f"Video file not found at {video_path}")
            return {"error": f"Video file not found at {video_path}"}
        elif mode == "batch" and (input_folder is None or not os.path.isdir(input_folder)): # Changed to isdir
            self.logger.error(f"Input folder not found or not a directory: {input_folder}")
            return {"error": f"Input folder not found or not a directory: {input_folder}"}
        elif mode == "scrape" and start_date is None:
            self.logger.error("start_date is required for scrape mode")
            return {"error": "start_date is required for scrape mode"}
        
        tracker_device = device if device else self.device # Use specific device if provided, else class default

        tracker = GloveTracker(
            model_alias='glove_tracking', # This is fixed for GloveTracker
            device=tracker_device,
            confidence_threshold=confidence_threshold,
            enable_filtering=enable_filtering,
            max_velocity_inches_per_sec=max_velocity_inches_per_sec,
            logger=self.logger,
            suppress_detection_warnings=suppress_detection_warnings
        )
        
        results_dir = tracker.results_dir # This is where GloveTracker saves its outputs
        
        if mode == "regular":
            # For regular mode, output_path is the direct output video file.
            # GloveTracker's track_video handles the naming within results_dir if output_path is None.
            output_video_path = tracker.track_video(
                video_path=video_path,
                output_path=output_path, # Pass the user-specified output_path
                show_plot=show_plot,
                create_video=create_video,
                generate_heatmap=generate_heatmap
            )
            
            # Construct CSV path based on the actual output video name (handled by GloveTracker)
            csv_filename = os.path.splitext(os.path.basename(output_video_path))[0] + "_tracking.csv"
            csv_data_path = os.path.join(results_dir, csv_filename) # GloveTracker saves CSVs in its results_dir
            
            movement_stats = tracker.analyze_glove_movement(csv_data_path) if os.path.exists(csv_data_path) else None
            
            heatmap_file_path = None
            if generate_heatmap and os.path.exists(csv_data_path):
                # Heatmap path will also be inside tracker.results_dir
                video_name_for_heatmap = os.path.basename(video_path)
                heatmap_file_path = tracker.plot_glove_heatmap(
                    csv_path=csv_data_path, 
                    video_name=video_name_for_heatmap, 
                    generate_heatmap=True # Ensure it's called
                )

            return {
                "output_video": output_video_path,
                "tracking_data": csv_data_path if os.path.exists(csv_data_path) else None,
                "movement_stats": movement_stats,
                "heatmap": heatmap_file_path,
                "filtering_applied": enable_filtering,
                "max_velocity_threshold": max_velocity_inches_per_sec if enable_filtering else None,
                "results_dir": results_dir
            }
        
        elif mode == "batch":
            video_files = sum([glob.glob(os.path.join(input_folder, f"*{ext}")) + glob.glob(os.path.join(input_folder, f"*{ext.upper()}")) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.mts']], [])
            if not video_files:
                self.logger.error(f"No video files found in {input_folder}")
                return {"error": f"No video files found in {input_folder}"}
            
            self.logger.info(f"Found {len(video_files)} videos to process in {input_folder}")
            
            if delete_after_processing and not skip_confirmation:
                confirm = input(f"WARNING: {len(video_files)} videos in '{input_folder}' will be DELETED after processing. Continue? (y/n): ")
                if confirm.lower() != 'y':
                    self.logger.info("Batch processing cancelled by user.")
                    return {"error": "Batch processing cancelled by user"}
            
            all_results_summary = []
            all_dataframes = []

            def _process_video_batch(vid_path):
                try:
                    # Output path for this specific video will be determined by GloveTracker inside its results_dir
                    processed_output_video = tracker.track_video(
                        video_path=vid_path,
                        output_path=None, # Let GloveTracker decide name in its results_dir
                        show_plot=show_plot,
                        create_video=create_video,
                        generate_heatmap=generate_heatmap
                    )
                    
                    csv_name = os.path.splitext(os.path.basename(processed_output_video))[0] + "_tracking.csv"
                    csv_data_path = os.path.join(results_dir, csv_name) # Saved by GloveTracker
                    
                    df = None
                    if os.path.exists(csv_data_path):
                        df = pd.read_csv(csv_data_path)
                        df['video_filename'] = os.path.basename(vid_path) # Add original filename for reference
                    
                    heatmap_p = None
                    if generate_heatmap and df is not None:
                         heatmap_p = tracker.plot_glove_heatmap(csv_path=csv_data_path, video_name=os.path.basename(vid_path), generate_heatmap=True)

                    return {
                        "original_video_path": vid_path,
                        "output_video": processed_output_video,
                        "tracking_data_csv": csv_data_path if df is not None else None,
                        "heatmap_path": heatmap_p,
                        "dataframe_obj": df # For combining later
                    }
                except Exception as e_batch:
                    self.logger.error(f"Error processing {os.path.basename(vid_path)} in batch: {str(e_batch)}")
                    return {"original_video_path": vid_path, "error": str(e_batch)}

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_video = {executor.submit(_process_video_batch, vp): vp for vp in video_files}
                for future in concurrent.futures.as_completed(future_to_video):
                    video_file_path = future_to_video[future]
                    try:
                        res = future.result()
                        all_results_summary.append(res)
                        if res and "dataframe_obj" in res and res["dataframe_obj"] is not None:
                            all_dataframes.append(res["dataframe_obj"])
                        if delete_after_processing and res and "error" not in res : # Only delete if processed successfully
                            os.remove(video_file_path)
                            self.logger.info(f"Deleted processed video: {os.path.basename(video_file_path)}")
                    except Exception as exc_thread:
                        self.logger.error(f"{os.path.basename(video_file_path)} generated an exception: {exc_thread}")
                        all_results_summary.append({"original_video_path": video_file_path, "error": str(exc_thread)})
            
            combined_csv_path = None
            summary_file_path = None
            combined_heatmap_path = None

            if generate_batch_info and all_dataframes:
                combined_df = pd.concat(all_dataframes, ignore_index=True)
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                combined_csv_path = os.path.join(results_dir, f"batch_glove_tracking_ALL_{timestamp}.csv")
                combined_df.to_csv(combined_csv_path, index=False)
                self.logger.info(f"Combined tracking data for batch saved to {combined_csv_path}")
                
                summary_file_path = os.path.join(results_dir, f"batch_glove_tracking_SUMMARY_{timestamp}.csv")
                tracker._generate_batch_summary(combined_df, summary_file_path) # Internal method of GloveTracker
                
                if generate_heatmap and not combined_df.empty:
                    combined_heatmap_path = os.path.join(results_dir, f"batch_glove_tracking_COMBINED_HEATMAP_{timestamp}.png")
                    tracker._generate_combined_heatmap(combined_df, combined_heatmap_path)
            
            return {
                "processed_summary": all_results_summary,
                "combined_csv_path": combined_csv_path,
                "summary_file_path": summary_file_path,
                "combined_heatmap_path": combined_heatmap_path,
                "results_dir": results_dir # Main directory where all outputs from GloveTracker are saved
            }

        elif mode == "scrape":
            temp_download_folder = tempfile.mkdtemp(prefix="savant_videos_temp_")
            self.logger.info(f"Temporary download folder for scrape mode: {temp_download_folder}")

            try:
                savant_scraper = BaseballSavVideoScraper(
                    start_dt=start_date, end_dt=end_date,
                    team_abbr=team_abbr, player=player, pitch_type=pitch_type,
                    download_folder=temp_download_folder, # Use temp folder
                    max_return_videos=max_videos, max_videos_per_game=max_videos_per_game
                )
                self.logger.info("Scraping videos from Baseball Savant...")
                savant_scraper.run_executor()
                
                scraped_play_ids_df = savant_scraper.get_play_ids_df()
                if scraped_play_ids_df.empty:
                    self.logger.warning("No videos downloaded from Baseball Savant.")
                    shutil.rmtree(temp_download_folder)
                    return {"error": "No videos downloaded from Baseball Savant.", "results_dir": results_dir}

                self.logger.info(f"Downloaded {len(scraped_play_ids_df)} videos to temporary folder.")

                # Now run batch processing on the downloaded videos
                # Note: delete_after_processing for this internal batch call should manage the temp files
                batch_processing_results = self.track_gloves(
                    mode="batch",
                    input_folder=temp_download_folder, # Process from temp folder
                    delete_after_processing=True, # Clean up temp videos after they are processed
                    skip_confirmation=True, # No need to confirm deletion of temp files
                    device=tracker_device, 
                    confidence_threshold=confidence_threshold, enable_filtering=enable_filtering,
                    max_velocity_inches_per_sec=max_velocity_inches_per_sec, show_plot=show_plot,
                    generate_heatmap=generate_heatmap, create_video=create_video,
                    max_workers=max_workers, generate_batch_info=generate_batch_info,
                    suppress_detection_warnings=suppress_detection_warnings
                )
                
                # Augment batch results with scrape info and Statcast data
                if "combined_csv_path" in batch_processing_results and batch_processing_results["combined_csv_path"]:
                    combined_df = pd.read_csv(batch_processing_results["combined_csv_path"])
                    
                    # Merge Statcast data
                    # Ensure 'video_filename' in combined_df matches the format from temp_download_folder
                    # Create a mapping from original filename to Statcast row for easier lookup
                    statcast_map = {}
                    for _, row in scraped_play_ids_df.iterrows():
                         # Filename format from BaseballSavVideoScraper: f"{game_pk}_{play_id}.mp4"
                        filename_key = f"{row['game_pk']}_{row['play_id']}.mp4"
                        statcast_map[filename_key] = row.to_dict()
                    
                    # Add new columns to combined_df for each piece of statcast info
                    if statcast_map:
                        sample_statcast_keys = list(next(iter(statcast_map.values())).keys())
                        for skey in sample_statcast_keys:
                            combined_df[f'statcast_{skey}'] = None

                        for index, row in combined_df.iterrows():
                            original_vid_filename = row['video_filename'] # This should be the key like "GAMEPK_PLAYID.mp4"
                            if original_vid_filename in statcast_map:
                                for skey, sval in statcast_map[original_vid_filename].items():
                                    combined_df.loc[index, f'statcast_{skey}'] = sval
                        
                        combined_df.to_csv(batch_processing_results["combined_csv_path"], index=False)
                        self.logger.info(f"Statcast data merged into combined CSV: {batch_processing_results['combined_csv_path']}")

                batch_processing_results["scrape_details"] = {
                    "start_date": start_date, "end_date": end_date, "team_abbr": team_abbr,
                    "player": player, "pitch_type": pitch_type, "max_videos_scraped": len(scraped_play_ids_df)
                }
                return batch_processing_results

            finally:
                if os.path.exists(temp_download_folder): # Ensure temp folder is cleaned up
                    shutil.rmtree(temp_download_folder)
                    self.logger.info(f"Cleaned up temporary download folder: {temp_download_folder}")
        
        return {"error": f"Mode '{mode}' logic incomplete or error.", "results_dir": results_dir}


    def analyze_pitcher_command(self,
                                csv_input_dir: str, # Directory with GloveTracker CSVs
                                output_csv_name: str = "pitcher_command_summary.csv", # Name for the main detailed output CSV
                                create_overlay_videos: bool = False,
                                video_overlays_output_dir: Optional[str] = None, # Specific dir for overlays
                                debug_mode: bool = False,
                                aggregate_group_by: List[str] = ['pitcher', 'pitch_type'],
                                command_threshold_inches: float = 6.0
                                ) -> pd.DataFrame:
        """
        Analyzes pitcher command by comparing intended target (from catcher's glove)
        to actual pitch location (from Statcast), using CommandAnalyzer.

        Args:
            csv_input_dir (str): Directory containing GloveTracker CSV files.
                                 CommandAnalyzer will look for CSVs here.
            output_csv_name (str): Filename for the detailed output CSV. This file will be
                                   saved inside `csv_input_dir`.
            create_overlay_videos (bool): Whether to create overlay videos.
            video_overlays_output_dir (Optional[str]): Directory to save overlay videos.
                                                       If None and create_overlay_videos is True,
                                                       defaults to a 'command_video_overlays'
                                                       subdirectory within `csv_input_dir`.
            debug_mode (bool): Enable additional debugging features from CommandAnalyzer.
            aggregate_group_by (List[str]): Columns to group by for the aggregate summary.
            command_threshold_inches (float): Threshold in inches for "commanded" pitch definition.

        Returns:
            pd.DataFrame: DataFrame with detailed pitch-by-pitch command analysis results.
                          Aggregate results are saved to a separate CSV.
        """
        self.logger.info(f"Initializing Pitcher Command Analysis. Input CSVs from: {csv_input_dir}")

        analyzer = CommandAnalyzer(
            csv_input_dir=csv_input_dir, # CommandAnalyzer reads from here
            logger=self.logger,
            verbose=self.verbose,
            device=self.device, # Pass device from BaseballTools
            debug_mode=debug_mode
        )

        # Determine video output directory for overlays
        if create_overlay_videos and video_overlays_output_dir is None:
            video_overlays_output_dir = os.path.join(csv_input_dir, "command_video_overlays")
            self.logger.info(f"Overlay videos will be saved to: {video_overlays_output_dir}")
        
        if create_overlay_videos:
             os.makedirs(video_overlays_output_dir, exist_ok=True)


        detailed_results_df = analyzer.analyze_folder(
            output_csv=output_csv_name, # analyze_folder saves this in csv_input_dir
            create_overlay=create_overlay_videos,
            video_output_dir=video_overlays_output_dir # Pass the determined/created dir
        )

        if detailed_results_df.empty:
            self.logger.warning("Command analysis (analyze_folder) returned no detailed results.")
            return pd.DataFrame()

        # Generate and save aggregate metrics
        if aggregate_group_by:
            aggregate_results_df = analyzer.calculate_aggregate_metrics(
                results_df=detailed_results_df,
                group_by=aggregate_group_by,
                cmd_threshold_inches=command_threshold_inches
            )
            if not aggregate_results_df.empty:
                aggregate_csv_name = f"aggregate_{output_csv_name}"
                aggregate_output_path = os.path.join(csv_input_dir, aggregate_csv_name) # Save in csv_input_dir
                try:
                    aggregate_results_df.to_csv(aggregate_output_path, index=False)
                    self.logger.info(f"Aggregate command metrics saved to: {aggregate_output_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save aggregate command metrics: {str(e)}")
            else:
                self.logger.warning("Could not calculate or save aggregate command metrics.")
        
        return detailed_results_df

    def cut_video_on_pitch_flight(self,
                                  video_path: str,
                                  output_path_dir: str, # Changed from output_path to output_path_dir for clarity
                                  model_alias: str = 'glove_tracking', 
                                  pre_release_padding_frames: int = 10,
                                  post_arrival_padding_frames: int = 5,
                                  confidence_ball: float = 0.3,
                                  confidence_glove: float = 0.3,
                                  **kwargs) -> Dict:
        """
        Cuts a video to show the pitched ball flight, from release to arrival at the glove,
        with customizable frame padding.

        Args:
            video_path (str): Path to the input video file.
            output_path_dir (str): Directory to save the cropped video. The filename will be auto-generated.
            model_alias (str): Alias of the model to use for ball and glove detection.
                               Can be a YOLO model (e.g., 'glove_tracking') or RFDETR (e.g., 'glove_tracking_rfd').
            pre_release_padding_frames (int): Number of frames to include before detected ball release.
            post_arrival_padding_frames (int): Number of frames to include after detected ball arrival.
            confidence_ball (float): Minimum confidence for ball detection.
            confidence_glove (float): Minimum confidence for glove detection.
            **kwargs: Additional parameters for the EventDetector.

        Returns:
            Dict: Results including the path to the cropped video, and detected event frame numbers.
        """
        self.logger.info(f"Initializing pitch flight detection for video: {video_path}")

        if not os.path.exists(video_path):
            self.logger.error(f"Input video not found: {video_path}")
            return {"error": "Input video not found", "cropped_video_path": None}

        # Determine model type based on alias
        primary_model_type_to_pass = 'RFDETR' if 'rfd' in model_alias.lower() else 'YOLO'
        # Default pitcher model ('phc_detector') is YOLO. Its type is needed for EventDetector init.
        default_pitcher_model_type = 'YOLO' # Assuming 'phc_detector' remains YOLO

        try:
            event_detector_tool = EventDetector(
                primary_model_alias=model_alias,
                primary_model_type=primary_model_type_to_pass,
                pitcher_model_alias='phc_detector', # Default for EventDetector
                pitcher_model_type=default_pitcher_model_type,
                logger=self.logger,
                verbose=self.verbose,
                device=self.device,
                confidence_ball=confidence_ball,
                confidence_glove=confidence_glove
            )
        except Exception as e_init:
            self.logger.error(f"Failed to initialize EventDetector: {e_init}")
            return {"error": f"EventDetector initialization failed: {e_init}", "cropped_video_path": None}


        results = event_detector_tool.extract_pitch_flight_segment(
            video_path=video_path,
            output_path_dir=output_path_dir, # Pass the directory here
            pre_release_padding_frames=pre_release_padding_frames,
            post_arrival_padding_frames=post_arrival_padding_frames,
            **kwargs
        )
        return results
    
    def cut_video_on_pitcher_mechanic(self,
                                     video_path: str,
                                     output_path_dir: str, # Changed from output_path for clarity
                                     primary_model_alias: str = 'ball_trackingv4', # Model for ball release anchoring
                                     pitcher_model_alias: str = 'phc_detector',   # Model for pitcher detection
                                     pre_mechanic_padding_frames: int = 20, 
                                     post_mechanic_padding_frames: int = 5, 
                                     confidence_pitcher: float = 0.5,
                                     confidence_ball: float = 0.3,
                                     **kwargs) -> Dict:
        """
        Cuts a video to focus on the pitcher's throwing mechanic, anchored by ball release.

        Args:
            video_path (str): Path to the input video file.
            output_path_dir (str): Directory to save the cropped video. Filename auto-generated.
            primary_model_alias (str): Alias for the model used for ball detection (to find release point).
                                       Can be YOLO or RFDETR.
            pitcher_model_alias (str): Alias for the model used for pitcher detection.
                                       Can be YOLO or RFDETR.
            pre_mechanic_padding_frames (int): Frames to include before the detected start of the mechanic.
            post_mechanic_padding_frames (int): Frames to include after the detected end of the mechanic (e.g., ball release).
            confidence_pitcher (float): Confidence for pitcher detection.
            confidence_ball (float): Confidence for ball detection (by primary model).
            **kwargs: Additional parameters for EventDetector.

        Returns:
            Dict: Results including path to cropped video and detected event frames.
        """
        self.logger.info(f"Initializing pitcher mechanic detection for video: {video_path}")

        if not os.path.exists(video_path):
            self.logger.error(f"Input video not found: {video_path}")
            return {"error": "Input video not found", "cropped_video_path": None}

        # Determine model types based on aliases
        primary_m_type_to_pass = 'RFDETR' if 'rfd' in primary_model_alias.lower() else 'YOLO'
        pitcher_m_type_to_pass = 'RFDETR' if 'rfd' in pitcher_model_alias.lower() else 'YOLO'
        
        try:
            event_detector_tool = EventDetector(
                primary_model_alias=primary_model_alias,
                primary_model_type=primary_m_type_to_pass,
                pitcher_model_alias=pitcher_model_alias,
                pitcher_model_type=pitcher_m_type_to_pass,
                logger=self.logger,
                verbose=self.verbose,
                device=self.device,
                confidence_pitcher=confidence_pitcher,
                confidence_ball=confidence_ball,
                # confidence_glove is not directly a parameter here, but EventDetector might use its default
            )
        except ValueError as e_val: 
            self.logger.error(f"ValueError initializing EventDetector for pitcher mechanic: {e_val}")
            return {"error": f"EventDetector init failed (ValueError): {e_val}", "cropped_video_path": None}
        except ImportError as e_imp:
            self.logger.error(f"ImportError initializing EventDetector for pitcher mechanic: {e_imp}")
            return {"error": f"EventDetector init failed (ImportError): {e_imp}", "cropped_video_path": None}
        except Exception as e_gen: 
            self.logger.error(f"General error initializing EventDetector for pitcher mechanic: {e_gen}")
            return {"error": f"EventDetector init failed (General Error): {e_gen}", "cropped_video_path": None}


        results = event_detector_tool.extract_pitcher_mechanic_segment(
            video_path=video_path,
            output_path_dir=output_path_dir, # Pass directory here
            pre_mechanic_padding_frames=pre_mechanic_padding_frames,
            post_mechanic_padding_frames=post_mechanic_padding_frames,
            **kwargs
        )
        return results