import tempfile
import pandas as pd
import glob
import time
import shutil
import os
from typing import Dict, List
import concurrent.futures
from .utils import DistanceToZone, GloveTracker
from baseballcv.utilities import BaseballCVLogger
from baseballcv.functions.savant_scraper import BaseballSavVideoScraper

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
                    mode: str = "regular",  # Mode: "regular", "batch", or "scrape"
                    # Common parameters
                    device: str = None,
                    confidence_threshold: float = 0.5,
                    enable_filtering: bool = True, 
                    max_velocity_inches_per_sec: float = 120.0,
                    show_plot: bool = True,
                    generate_heatmap: bool = True,
                    create_video: bool = True,
                    # Regular mode parameters
                    video_path: str = None,
                    output_path: str = None,
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
            device (str): Device to run the model on (cpu, cuda, mps)
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
            
            # Scrape mode parameters
            start_date (str): Start date in YYYY-MM-DD format (for scrape mode)
            end_date (str): End date in YYYY-MM-DD format (for scrape mode, optional)
            team_abbr (str): Team abbreviation (for scrape mode, optional)
            player (int): Player ID (for scrape mode, optional)
            pitch_type (str): Pitch type (for scrape mode, optional)
            max_videos (int): Maximum number of videos to download (for scrape mode)
            max_videos_per_game (int): Maximum videos per game (for scrape mode)
            
        Returns:
            Dict: Results containing paths to output files and analysis statistics
        """
        # Validate mode parameter
        valid_modes = ["regular", "batch", "scrape"]
        if mode not in valid_modes:
            self.logger.error(f"Invalid mode: {mode}. Must be one of {valid_modes}")
            return {"error": f"Invalid mode: {mode}. Must be one of {valid_modes}"}
        
        # Parameter validation based on mode
        if mode == "regular" and (video_path is None or not os.path.exists(video_path)):
            self.logger.error(f"Video file not found at {video_path}")
            return {"error": f"Video file not found at {video_path}"}
        elif mode == "batch" and (input_folder is None or not os.path.exists(input_folder)):
            self.logger.error(f"Input folder not found: {input_folder}")
            return {"error": f"Input folder not found: {input_folder}"}
        elif mode == "scrape" and start_date is None:
            self.logger.error("start_date is required for scrape mode")
            return {"error": "start_date is required for scrape mode"}
        
        tracker = GloveTracker(
            model_alias='glove_tracking',
            device=device if device else self.device,
            confidence_threshold=confidence_threshold,
            enable_filtering=enable_filtering,
            max_velocity_inches_per_sec=max_velocity_inches_per_sec,
            logger=self.logger,
            suppress_detection_warnings=suppress_detection_warnings
        )
        
        results_dir = tracker.results_dir
        
        if mode == "regular":
            output_video = tracker.track_video(
                video_path=video_path,
                output_path=output_path,
                show_plot=show_plot,
                create_video=create_video,
                generate_heatmap=generate_heatmap
            )
            
            csv_filename = os.path.splitext(os.path.basename(output_video))[0] + "_tracking.csv"
            csv_path = os.path.join(results_dir, csv_filename)
            
            movement_stats = tracker.analyze_glove_movement(csv_path)
            
            if generate_heatmap:
                video_filename = os.path.basename(video_path)
                heatmap_filename = f"glove_heatmap_{os.path.splitext(video_filename)[0]}.png"
                heatmap_path = os.path.join(results_dir, heatmap_filename)
                if not os.path.exists(heatmap_path):
                    heatmap_path = None
            
            return {
                "output_video": output_video,
                "tracking_data": csv_path,
                "movement_stats": movement_stats,
                "heatmap": heatmap_path if generate_heatmap else None,
                "filtering_applied": enable_filtering,
                "max_velocity_threshold": max_velocity_inches_per_sec if enable_filtering else None
            }
        
        elif mode == "batch":

            video_files = sum([glob.glob(os.path.join(input_folder, f"*{ext}")) + glob.glob(os.path.join(input_folder, f"*{ext.upper()}")) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.mts']], [])
            if not video_files:
                self.logger.error(f"No video files found in {input_folder}")
                return {"error": f"No video files found in {input_folder}"}
            
            self.logger.info(f"Found {len(video_files)} videos to process in {input_folder}")
            
            # Warn about deletion if enabled
            if delete_after_processing and not skip_confirmation:
                confirm = input(f"WARNING: {len(video_files)} videos will be DELETED after processing. Continue? (y/n): ")
                if confirm.lower() != 'y':
                    self.logger.error("Batch processing cancelled")
                    return {"error": "Batch processing cancelled by user"}
            
            # We'll store all results and DataFrame data
            all_results = []
            combined_df = pd.DataFrame()

            def _process_video(video_path):
                try:
                    # Process the video
                    output_video = tracker.track_video(
                        video_path=video_path,
                        show_plot=show_plot,
                        create_video=create_video,
                        generate_heatmap=generate_heatmap
                    )
                    
                    # Get the CSV data
                    csv_filename = os.path.splitext(os.path.basename(output_video))[0] + "_tracking.csv"
                    csv_path = os.path.join(results_dir, csv_filename)
                    
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        df['video_filename'] = os.path.basename(video_path)
                        
                        if generate_heatmap:
                            video_filename = os.path.basename(video_path)
                            heatmap_filename = f"glove_heatmap_{os.path.splitext(video_filename)[0]}.png"
                            heatmap_path = os.path.join(results_dir, heatmap_filename)
                            if not os.path.exists(heatmap_path):
                                heatmap_path = None
                        else:
                            heatmap_path = None
                        
                        result = {
                            "video_path": video_path,
                            "output_video": output_video,
                            "tracking_data": csv_path,
                            "heatmap": heatmap_path,
                            "dataframe": df
                        }
                        return result
                    else:
                        self.logger.warning(f"No tracking data found for {video_path}")
                        return None
                except Exception as e:
                    self.logger.error(f"Error processing {video_path}: {str(e)}")
                    return None
            
            # Process the videos
            if max_workers > 1:
                self.logger.info(f"Processing videos in parallel with {max_workers} workers")
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(_process_video, path): path for path in video_files}
                    
                    for future in concurrent.futures.as_completed(futures):
                        path = futures[future]
                        try:
                            if result := future.result():
                                all_results.append(result)
                                combined_df = pd.concat([combined_df, result["dataframe"]], ignore_index=True)
                                
                                if delete_after_processing:
                                    os.remove(path)
                                    self.logger.info(f"Deleted video: {os.path.basename(path)}")
                        except Exception as e:
                            self.logger.error(f"Error processing {os.path.basename(path)}: {str(e)}")
            else:
                self.logger.info("Processing videos sequentially")
                for video_path in video_files:
                    if result := _process_video(video_path):
                        all_results.append(result)
                        combined_df = pd.concat([combined_df, result["dataframe"]], ignore_index=True)
                        if delete_after_processing:
                            os.remove(video_path)
                            self.logger.info(f"Deleted video: {os.path.basename(video_path)}")
            
            # If we didn't process any videos successfully
            if not all_results:
                self.logger.warning("No videos were processed successfully")
                return {"error": "No videos were processed successfully"}
            
            #Generate combos and summary if wanted
            if generate_batch_info:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                combined_csv_path = os.path.join(results_dir, f"batch_tracking_results_{timestamp}.csv")
                combined_df.to_csv(combined_csv_path, index=False)
                self.logger.info(f"Combined tracking data saved to {combined_csv_path}")
                
                summary_path = os.path.join(results_dir, f"batch_summary_{timestamp}.csv")
                tracker._generate_batch_summary(combined_df, summary_path)
                
                combined_heatmap_path = os.path.join(results_dir, f"combined_heatmap_{timestamp}.png") if generate_heatmap and not combined_df.empty else None
                if combined_heatmap_path:
                    tracker._generate_combined_heatmap(combined_df, combined_heatmap_path)
            
            return {
                "processed_videos": len(all_results),
                "individual_results": all_results,
                "combined_csv": combined_csv_path if generate_batch_info else None,
                "summary_file": summary_path if generate_batch_info else None,
                "combined_heatmap": combined_heatmap_path if generate_heatmap and not combined_df.empty else None,
                "results_dir": results_dir
            }
        
        elif mode == "scrape":
            download_folder = tempfile.mkdtemp(prefix="savant_videos_")
    
            try:
                scraper = BaseballSavVideoScraper(
                    start_dt=start_date,
                    end_dt=end_date,
                    team_abbr=team_abbr,  # Optional
                    player=player,  # Optional
                    pitch_type=pitch_type,  # Optional
                    download_folder=download_folder,
                    max_return_videos=max_videos,
                    max_videos_per_game=max_videos_per_game
                )
                
                self.logger.info(f"Scraping videos from Baseball Savant...")
                scraper.run_executor()
                
                play_ids_df = scraper.get_play_ids_df()
                
                if len(play_ids_df) == 0:
                    self.logger.error("No videos were downloaded from Baseball Savant")
                    return {"error": "No videos were downloaded from Baseball Savant"}
                
                self.logger.info(f"Successfully scraped {len(play_ids_df)} videos")
                
                statcast_data = {}
                for _, row in play_ids_df.iterrows():
                    game_pk, play_id = str(row['game_pk']), str(row['play_id'])
                    filename = f"{game_pk}_{play_id}.mp4"
                    statcast_data[filename] = {col: row[col] for col in row.index}
                
                # Process the downloaded videos using batch logic
                # Reuse the batch logic by calling this method recursively with mode='batch'
                batch_results = self.track_gloves(
                    mode="batch",
                    input_folder=download_folder,
                    delete_after_processing=delete_after_processing,
                    skip_confirmation=skip_confirmation,
                    confidence_threshold=confidence_threshold,
                    device=device,
                    show_plot=show_plot,
                    generate_heatmap=generate_heatmap,
                    enable_filtering=enable_filtering,
                    max_velocity_inches_per_sec=max_velocity_inches_per_sec,
                    create_video=create_video,
                    max_workers=max_workers
                )
                
                if "combined_csv" in batch_results and os.path.exists(batch_results["combined_csv"]):
                    combined_df = pd.read_csv(batch_results["combined_csv"])
                    statcast_columns = {f'statcast_{k}': [] for k in next(iter(statcast_data.values())).keys()} if statcast_data else {}
                    for col in statcast_columns:
                        combined_df[col] = None
                    for filename, data in statcast_data.items():
                        mask = combined_df['video_filename'] == filename
                        for k, v in data.items():
                            combined_df.loc[mask, f'statcast_{k}'] = v
                    combined_df.to_csv(batch_results["combined_csv"], index=False)
                    batch_results["statcast_data_added"] = True
                
                # Clean up temp directory if it wasn't already deleted during batch processing
                if not delete_after_processing and os.path.exists(download_folder):
                    if not skip_confirmation:
                        confirm = input(f"Delete downloaded videos? (y/n): ")
                        if confirm.lower() == 'y' or confirm.lower() == 'yes':
                            shutil.rmtree(download_folder)
                            self.logger.info(f"Deleted downloaded videos")
                    else:
                        shutil.rmtree(download_folder)
                        self.logger.info(f"Deleted downloaded videos")
                
                batch_results["scrape_info"] = {
                    "start_date": start_date,
                    "end_date": end_date,
                    "team_abbr": team_abbr,
                    "player": player,
                    "pitch_type": pitch_type,
                    "videos_requested": max_videos,
                    "videos_downloaded": len(play_ids_df)
                }
                
                return batch_results
            
            except Exception as e:
                self.logger.error(f"Error in scrape mode: {str(e)}")
                if os.path.exists(download_folder):
                    shutil.rmtree(download_folder)
                return {"error": f"Error in scrape mode: {str(e)}"}
        
        #Should never reach here, but just in case
        return {"error": f"Unknown error processing in mode: {mode}"}