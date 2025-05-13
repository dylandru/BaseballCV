# baseballcv/functions/utils/baseball_utils/command_analyzer.py

import pandas as pd
import numpy as np
import os
import glob
import math
import re
import shutil
import cv2
from typing import List, Dict, Tuple, Optional, Union
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from baseballcv.utilities import BaseballCVLogger, ProgressBar
#from baseballcv.functions.savant_scraper import BaseballSavVideoScraper
#from baseballcv.functions.utils.savant_utils import GamePlayIDScraper

class CommandAnalyzer:
    """
    Analyzes pitcher command by comparing intended target (from catcher's glove) 
    to actual pitch location (from Statcast).
    
    Redesigned to:
    1. Accurately determine the pitcher's intended target frame (pre-release)
    2. Correctly calculate deviation in inches using proper coordinate transformations
    3. Generate visual overlays showing target vs actual pitch location
    
    Core workflow:
    - Uses GloveTracker CSV files for glove positioning
    - Fetches Statcast data internally for actual pitch locations
    - Aligns coordinate systems by using MLBam standards
    - Optionally creates overlay visualizations
    """
    # Regular expression for extracting Play ID from filenames
    PLAY_ID_REGEX = re.compile(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})')
    
    # Physical constants based on MLBam standards (https://baseballsavant.mlb.com/csv-docs)
    # Approx. home plate dimensions (17" wide, half that is ~8.5") - this is accurate for MLB
    PLATE_WIDTH_INCHES = 17.0
    PLATE_HALF_WIDTH_INCHES = 8.5  # For reference, x=0 is center of plate
    
    # Defined height locations based on standard MLB measurements
    # Height of plate from ground - measured to the *top* of the plate (not center)
    PLATE_HEIGHT_FROM_GROUND_INCHES = 10.0 
    
    # Pitcher release point is ~60 feet, 6 inches (MLB mound distance), or 726 inches from the plate
    TYPICAL_RELEASE_POINT_DISTANCE = 726.0  
    
    # Timing constants for intent detection
    MAX_FRAMES_BEFORE_CROSSING = 24  # Approx max frames to look backwards from crossing
    MIN_STABILITY_DURATION_FRAMES = 3  # Min number of frames glove should be stable
    
    # Constants for filtering out unrealistic target positions
    MIN_TARGET_HEIGHT_FROM_GROUND = 12.0  # Minimum plausible target height in inches from ground
    MAX_TARGET_HEIGHT_FROM_GROUND = 60.0  # Maximum plausible target height in inches from ground
    
    def __init__(
        self,
        csv_input_dir: str,
        logger: Optional[BaseballCVLogger] = None,
        verbose: bool = True,
        device: str = 'cpu',
        debug_mode: bool = False
    ):
        """
        Initialize the CommandAnalyzer.
        
        Args:
            csv_input_dir: Directory containing GloveTracker CSV files
            logger: Optional logger instance (will create one if not provided)
            verbose: Whether to print verbose logs
            device: Computing device (for potential future model integration)
            debug_mode: Enable additional debugging features and plots
        """
        self.csv_input_dir = csv_input_dir
        self.logger = logger if logger else BaseballCVLogger.get_logger(self.__class__.__name__)
        self.verbose = verbose
        self.device = device
        self.debug_mode = debug_mode
        self.statcast_cache = {}
        self._video_downloader_instance = None
        
        # Debug directory for analysis artifacts
        self.debug_dir = os.path.join(csv_input_dir, "command_debug") if debug_mode else None
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)
            
        if not os.path.isdir(self.csv_input_dir):
            raise FileNotFoundError(f"CSV input directory not found: {self.csv_input_dir}")

        self.logger.info(f"CommandAnalyzer initialized. Reading CSVs from: {self.csv_input_dir}")

    # --- File and Data Handling Methods ---
    
    def _extract_ids_from_filename(self, filename: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Extract game_pk and play_id from a GloveTracker CSV filename.
        
        Args:
            filename: Path to the CSV file
            
        Returns:
            Tuple of (game_pk, play_id) if found, otherwise (None, None)
        """
        basename = os.path.basename(filename)
        
        # Try the standard format first: tracked_GAMEPK_PLAYID_tracking.csv
        match = re.search(r'tracked_(\d+)_([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', basename)
        if match:
            try:
                return int(match.group(1)), str(match.group(2))
            except (ValueError, IndexError):
                pass
                
        # Fallback: Look for any play_id UUID and try to extract game_pk from surrounding text
        play_id_match = self.PLAY_ID_REGEX.search(basename)
        if play_id_match:
            play_id = str(play_id_match.group(1))
            try:
                # Extract the first number before the play_id
                prefix = basename.split(play_id)[0]
                game_pk_match = re.findall(r'\d+', prefix)
                if game_pk_match:
                    return int(game_pk_match[-1]), play_id
            except (ValueError, IndexError):
                pass
                
        self.logger.warning(f"Could not extract valid game_pk and play_id from filename: {basename}")
        return None, None
        
    def _fetch_statcast_for_game(self, game_pk: int) -> Optional[pd.DataFrame]:
        """
        Fetch Statcast data for a specific game.
        
        Args:
            game_pk: MLB Game ID number
            
        Returns:
            DataFrame containing Statcast data or None if unsuccessful
        """
        from baseballcv.functions.utils.savant_utils import GamePlayIDScraper

        # Return cached data if available
        if game_pk in self.statcast_cache:
            return self.statcast_cache[game_pk]
            
        self.logger.info(f"Fetching Statcast data for game_pk: {game_pk}...")
        
        try:
            # Initialize a temporary fetcher with a valid date
            temp_fetcher = GamePlayIDScraper(
                start_dt="2024-05-01",
                logger=self.logger
            )
            
            # Override the game_pks attribute to just get our specific game
            temp_fetcher.game_pks = [game_pk]
            temp_fetcher.home_team = ["N/A"]
            temp_fetcher.away_team = ["N/A"]
            
            # Use the _get_play_ids method to fetch data for this game
            pl_df = temp_fetcher._get_play_ids(game_pk, "N/A", "N/A")
            del temp_fetcher
            
            if pl_df is not None and not pl_df.is_empty():
                # Convert to pandas DataFrame
                pd_df = pl_df.to_pandas()
                
                # Check for required columns
                required_cols = ['play_id', 'game_pk', 'plate_x', 'plate_z']
                if not all(col in pd_df.columns for col in required_cols):
                    missing = [col for col in required_cols if col not in pd_df.columns]
                    self.logger.error(f"Fetched Statcast game {game_pk} missing cols: {missing}")
                    self.statcast_cache[game_pk] = None
                    return None
                
                # Ensure consistent types
                pd_df['play_id'] = pd_df['play_id'].astype(str)
                pd_df['game_pk'] = pd_df['game_pk'].astype(int)
                
                # Cache the result
                self.statcast_cache[game_pk] = pd_df
                self.logger.info(f"Cached {len(pd_df)} pitches for game {game_pk}")
                return pd_df
            else:
                self.logger.warning(f"No pitch data returned for game_pk: {game_pk}")
                self.statcast_cache[game_pk] = None
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching Statcast for game {game_pk}: {str(e)}", exc_info=self.verbose)
            self.statcast_cache[game_pk] = None
            return None
    
    def _download_video_for_play(self, game_pk: int, play_id: str, video_output_dir: str) -> Optional[str]:
        """
        Download video for a specific play from Baseball Savant.
        
        Args:
            game_pk: MLB Game ID number
            play_id: MLB Play ID string
            video_output_dir: Directory to save the video
            
        Returns:
            Path to the downloaded video or None if unsuccessful
        """
        from baseballcv.functions.savant_scraper import BaseballSavVideoScraper

        os.makedirs(video_output_dir, exist_ok=True)
        output_path = os.path.join(video_output_dir, f"{game_pk}_{play_id}.mp4")
        
        # Return existing video if already downloaded
        if os.path.exists(output_path):
            self.logger.debug(f"Video already exists: {output_path}")
            return output_path
            
        self.logger.info(f"Downloading video for play {play_id} (Game {game_pk})...")
        
        try:
            # Initialize or get the video downloader
            if self._video_downloader_instance is None:
                # Need dummy date that has games for GamePlayIDScraper init
                dummy_start = "2024-04-01"
                temp_dl_folder = os.path.join(video_output_dir, "temp_scraper_init_video_dl")
                
                try:
                    self._video_downloader_instance = BaseballSavVideoScraper(
                        start_dt=dummy_start,
                        download_folder=temp_dl_folder,
                        max_return_videos=1
                    )
                    self.logger.debug("Video downloader instance created")
                    
                    # Clean up temp folder right away
                    if os.path.exists(temp_dl_folder):
                        shutil.rmtree(temp_dl_folder)
                        
                except Exception as e:
                    self.logger.error(f"Failed to create video downloader: {str(e)}")
                    if os.path.exists(temp_dl_folder):
                        shutil.rmtree(temp_dl_folder)
                    return None
            
            # Configure the downloader to save in our target directory
            downloader = self._video_downloader_instance
            downloader.download_folder = video_output_dir
            
            # Get the video page URL
            video_page_url = downloader.SAVANT_VIDEO_URL.format(play_id)
            video_page_response = downloader.requests_with_retry(video_page_url)
            
            if not video_page_response:
                raise ValueError(f"Failed to get video page for {play_id}")
                
            # Parse the page to find the video URL
            soup = BeautifulSoup(video_page_response.content, 'html.parser')
            video_container = soup.find('div', class_='video-box')
            
            if not video_container:
                raise ValueError(f"Video container not found on page for {play_id}")
                
            source_tag = video_container.find('video').find('source', type='video/mp4')
            if not source_tag or not source_tag.get('src'):
                raise ValueError(f"MP4 source URL not found for {play_id}")
                
            video_url = source_tag['src']
            
            # Get the video content
            video_response = downloader.requests_with_retry(video_url, stream=True)
            if not video_response:
                raise ValueError(f"Failed to get video stream for {play_id}")
                
            # Write the video file
            downloader._write_content(game_pk, play_id, video_response)
            self.logger.info(f"Video downloaded successfully: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to download video for play {play_id}: {str(e)}", exc_info=self.verbose)
            return None

    # --- Core Intent Detection and Coordinate Analysis Methods ---
    
    def _find_intent_frame(self, df: pd.DataFrame, ball_crossing_frame: Optional[int] = None) -> Optional[int]:
        """
        Identify the frame where the pitcher shows intent (target) via catcher's glove position.
        
        IMPROVED to specifically identify the stable glove position BEFORE the catcher
        lowers the glove, which is the true intended target location.
        
        Args:
            df: GloveTracker CSV data as DataFrame
            ball_crossing_frame: Optional frame where ball crosses plate
            
        Returns:
            Frame number representing pitcher's intent or None if not found
        """
        # Get valid glove tracking data (non-interpolated, with coordinates)
        valid_data = df[
            df['glove_processed_x'].notna() & 
            df['glove_processed_y'].notna() & 
            (df['is_interpolated'] == False)
        ].copy()
        
        if len(valid_data) < 5:  # Need more frames for reliable pattern detection
            self.logger.warning("Not enough valid glove data points to detect intent")
            return None
            
        # Sort by frame index (ensuring chronological order)
        valid_data = valid_data.sort_values(by='frame_idx').reset_index(drop=True)
        
        # Calculate time differentials between frames
        valid_data['dt'] = valid_data['frame_idx'].diff().fillna(1.0)
        valid_data.loc[valid_data['dt'] <= 0, 'dt'] = 1.0  # Ensure positive dt
        
        # Calculate differentials in glove position
        valid_data['dx'] = valid_data['glove_processed_x'].diff().fillna(0)
        valid_data['dy'] = valid_data['glove_processed_y'].diff().fillna(0)
        
        # Calculate glove velocity in inches per frame
        valid_data['velocity'] = np.sqrt(
            valid_data['dx']**2 + valid_data['dy']**2
        ) / valid_data['dt']
        
        # Apply smoothing to velocity to reduce noise
        if len(valid_data) >= 5:  # Need sufficient points for smoothing
            try:
                # Savitzky-Golay filter: window_length=5, polyorder=2
                valid_data['velocity_smooth'] = savgol_filter(
                    valid_data['velocity'].fillna(0), 
                    window_length=min(5, len(valid_data) - (len(valid_data) % 2 - 1)), 
                    polyorder=min(2, min(5, len(valid_data) - (len(valid_data) % 2 - 1)) - 1)
                )
            except Exception as e:
                self.logger.warning(f"Smoothing failed, using raw velocity: {str(e)}")
                valid_data['velocity_smooth'] = valid_data['velocity']
        else:
            valid_data['velocity_smooth'] = valid_data['velocity']
        
        # Calculate glove height from ground for each frame
        # glove_processed_y is in inches from plate center
        # Convert to height from ground using plate height
        valid_data['glove_height_from_ground'] = self.PLATE_HEIGHT_FROM_GROUND_INCHES + valid_data['glove_processed_y']
        
        # Calculate height differentials to detect lowering
        valid_data['height_diff'] = valid_data['glove_height_from_ground'].diff()
        
        # Define criteria for valid target frames
        velocity_threshold = 2.5  # inches per frame
        valid_data['is_stable'] = valid_data['velocity_smooth'] < velocity_threshold
        valid_data['is_reasonable_height'] = (
            (valid_data['glove_height_from_ground'] >= self.MIN_TARGET_HEIGHT_FROM_GROUND) &
            (valid_data['glove_height_from_ground'] <= self.MAX_TARGET_HEIGHT_FROM_GROUND)
        )
        valid_data['is_valid_target'] = valid_data['is_stable'] & valid_data['is_reasonable_height']
        
        # ===== IMPROVED DETECTION FOR PATTERN: STABLE → LOWERING =====
        
        # Define a moving window to calculate stability over multiple frames
        window_size = 3  # Number of frames to consider stable
        valid_data['stable_window'] = valid_data['is_stable'].rolling(window=window_size, min_periods=window_size).sum() >= window_size
        
        # Define significant lowering (negative height_diff over multiple frames)
        lowering_threshold = -3.0  # Inches per frame
        valid_data['is_lowering'] = valid_data['height_diff'] < lowering_threshold
        
        # Find stable windows followed by lowering
        # Step 1: Mark frames where we transition from stable to lowering
        valid_data['stable_to_lowering'] = (valid_data['stable_window'].shift(1) == True) & (valid_data['is_lowering'] == True)
        
        # Find all major lowering events
        lowering_indices = valid_data.index[valid_data['stable_to_lowering']].tolist()
        
        # For each lowering event, get the stable frame immediately before
        intent_candidates = []
        
        for lowering_idx in lowering_indices:
            if lowering_idx > 0:
                # Look back to find the center of the preceding stable window
                lookback_start = max(0, lowering_idx - (window_size * 2))
                lookback_frames = valid_data.iloc[lookback_start:lowering_idx]
                
                # Find stable frames at reasonable heights
                stable_frames = lookback_frames[
                    lookback_frames['is_stable'] & 
                    lookback_frames['is_reasonable_height']
                ]
                
                if not stable_frames.empty:
                    # Take the middle of the stable sequence - this is likely the target frame
                    intent_frame = int(stable_frames.iloc[len(stable_frames)//2]['frame_idx'])
                    height = stable_frames.iloc[len(stable_frames)//2]['glove_height_from_ground']
                    confidence = min(len(stable_frames) / 5.0, 1.0)  # Confidence based on sequence length
                    
                    intent_candidates.append({
                        'frame': intent_frame,
                        'height': height,
                        'confidence': confidence,
                        'lowering_idx': lowering_idx,
                        'stable_frames': len(stable_frames)
                    })
        
        # If we have a ball crossing frame, prioritize candidates that appear before it
        if intent_candidates:
            if ball_crossing_frame is not None:
                # Filter candidates that appear before ball crossing
                before_crossing = [c for c in intent_candidates if c['frame'] < ball_crossing_frame]
                if before_crossing:
                    # Use the candidate with highest confidence before crossing
                    best_candidate = sorted(before_crossing, key=lambda x: (x['confidence'], -x['lowering_idx']), reverse=True)[0]
                    self.logger.debug(f"Found intent frame {best_candidate['frame']} from stable→lowering pattern "
                                f"(height: {best_candidate['height']:.1f}in, {best_candidate['stable_frames']} stable frames)")
                    return best_candidate['frame']
            
            # If no ball crossing or no candidates before crossing, use highest confidence
            best_candidate = sorted(intent_candidates, key=lambda x: (x['confidence'], -x['lowering_idx']), reverse=True)[0]
            self.logger.debug(f"Found intent frame {best_candidate['frame']} from stable→lowering pattern "
                        f"(height: {best_candidate['height']:.1f}in, {best_candidate['stable_frames']} stable frames)")
            return best_candidate['frame']
        
        # ===== DETECTION BASED ON STABLE SEQUENCES AT REASONABLE HEIGHTS =====
        
        # If no stable→lowering pattern found, look for the best stable sequence at a reasonable height
        # Group consecutive stable frames at reasonable heights
        valid_data['is_valid_sequence'] = valid_data['is_valid_target']
        valid_data['sequence_group'] = (valid_data['is_valid_sequence'].diff() != 0).cumsum()
        
        # Get statistics for each sequence
        sequence_stats = valid_data[valid_data['is_valid_sequence']].groupby('sequence_group').agg({
            'frame_idx': ['first', 'last', 'count'],
            'velocity_smooth': 'mean',
            'glove_height_from_ground': 'mean'
        })
        
        if not sequence_stats.empty:
            # Flatten the MultiIndex columns
            sequence_stats.columns = ['_'.join(col).strip() for col in sequence_stats.columns.values]
            
            # Add sequence length
            sequence_stats['sequence_length'] = sequence_stats['frame_idx_count']
            
            # Filter to sequences of sufficient length
            min_sequence_length = 3
            valid_sequences = sequence_stats[sequence_stats['sequence_length'] >= min_sequence_length]
            
            if not valid_sequences.empty:
                # Calculate a score based on:
                # 1. Sequence length (longer is better)
                # 2. Stability (lower velocity is better)
                # 3. Height reasonableness (closer to strike zone center ~30" is better)
                valid_sequences['stability_score'] = 1.0 - (valid_sequences['velocity_smooth_mean'] / 5.0)
                valid_sequences['height_score'] = 1.0 - (abs(valid_sequences['glove_height_from_ground_mean'] - 30.0) / 20.0)
                valid_sequences['length_score'] = valid_sequences['sequence_length'] / 10.0
                
                # Calculate total score (weighted)
                valid_sequences['total_score'] = (
                    (0.4 * valid_sequences['stability_score']) + 
                    (0.4 * valid_sequences['height_score']) + 
                    (0.2 * valid_sequences['length_score'])
                )
                
                # Prioritize earlier sequences if the ball crossing frame is known
                if ball_crossing_frame is not None:
                    # Penalize sequences that are too close to crossing
                    valid_sequences['crossing_proximity'] = 1.0 - np.minimum(
                        np.maximum(0, ball_crossing_frame - valid_sequences['frame_idx_last']) / 30.0, 1.0
                    )
                    valid_sequences['total_score'] *= valid_sequences['crossing_proximity']
                
                # Find best sequence
                best_sequence_idx = valid_sequences['total_score'].idxmax()
                best_sequence = valid_sequences.loc[best_sequence_idx]
                
                # Calculate the middle frame of the best sequence
                best_group = best_sequence.name
                sequence_frames = valid_data[
                    (valid_data['sequence_group'] == best_group) & 
                    (valid_data['is_valid_sequence'] == True)
                ]['frame_idx'].values
                
                if len(sequence_frames) > 0:
                    # Take a frame toward the beginning of the sequence
                    # (about 1/3 of the way through rather than middle)
                    frame_idx = sequence_frames[len(sequence_frames) // 3]
                    
                    self.logger.debug(f"Found intent frame {frame_idx} from stable sequence "
                                f"(height: {best_sequence['glove_height_from_ground_mean']:.1f}in, "
                                f"{best_sequence['sequence_length']} frames)")
                    return int(frame_idx)
        
        # ===== FALLBACK: FIND ANY REASONABLE STABLE FRAME =====
        
        # If we reach here, we couldn't find a good pattern
        # Look for frames before crossing (if known) that are stable at reasonable heights
        reasonable_frames = valid_data[valid_data['is_reasonable_height']]
        
        if ball_crossing_frame is not None:
            before_crossing = reasonable_frames[reasonable_frames['frame_idx'] < ball_crossing_frame]
            
            if not before_crossing.empty:
                # Take the most stable frame at a reasonable height
                min_vel_idx = before_crossing['velocity_smooth'].idxmin()
                intent_frame = int(before_crossing.loc[min_vel_idx, 'frame_idx'])
                height = before_crossing.loc[min_vel_idx, 'glove_height_from_ground']
                self.logger.debug(f"Using most stable frame before crossing: {intent_frame} "
                            f"(height: {height:.1f}in, velocity: {before_crossing.loc[min_vel_idx, 'velocity_smooth']:.2f})")
                return intent_frame
        
        # Last resort: most stable frame among reasonable heights
        if not reasonable_frames.empty:
            min_vel_idx = reasonable_frames['velocity_smooth'].idxmin()
            intent_frame = int(reasonable_frames.loc[min_vel_idx, 'frame_idx'])
            height = reasonable_frames.loc[min_vel_idx, 'glove_height_from_ground']
            self.logger.debug(f"Using most stable frame at reasonable height: {intent_frame} "
                        f"(height: {height:.1f}in, velocity: {reasonable_frames.loc[min_vel_idx, 'velocity_smooth']:.2f}) as fallback")
            return intent_frame
        
        # Absolute last resort: just use the minimum velocity frame regardless of height
        if not valid_data.empty:
            min_vel_idx = valid_data['velocity_smooth'].idxmin()
            intent_frame = int(valid_data.loc[min_vel_idx, 'frame_idx'])
            self.logger.warning(f"Using minimum velocity frame {intent_frame} as last resort "
                        f"(height: {valid_data.loc[min_vel_idx, 'glove_height_from_ground']:.1f}in)")
            return intent_frame
        
        self.logger.warning("Could not determine a reliable intent frame")
        return None
    
    def _find_ball_crossing_frame(self, df: pd.DataFrame) -> Optional[int]:
        """
        Find the frame where the ball crosses (or is closest to) home plate.
        
        Args:
            df: GloveTracker CSV data
            
        Returns:
            Frame number or None if not found
        """
        # Look for frames with baseball detected and real-world coords
        ball_frames = df[
            df['baseball_real_x'].notna() & 
            df['baseball_real_y'].notna()
        ].copy()
        
        if ball_frames.empty:
            self.logger.debug("No baseball tracking data found")
            return None
        
        # Calculate distance to plate (x=0) for each frame
        ball_frames['distance_to_plate'] = np.abs(ball_frames['baseball_real_x'])
        
        # Find frame with minimum distance to plate
        min_dist_idx = ball_frames['distance_to_plate'].idxmin() 
        
        if pd.isna(min_dist_idx):
            return None
            
        crossing_frame = int(ball_frames.loc[min_dist_idx, 'frame_idx'])
        return crossing_frame
        
    def _transform_coordinates(self, 
                               target_x_inches: float, 
                               target_y_inches: float,
                               actual_x_ft: float, 
                               actual_z_ft: float,
                               sz_top_ft: Optional[float] = None,
                               sz_bot_ft: Optional[float] = None) -> Tuple[float, float, float, float]:
        """
        Transform GloveTracker and Statcast coordinates to a common system.
        
        Important: Corrects for the reversed x-axis in Statcast data.
        
        Args:
            target_x_inches: Horizontal target position from plate center (inches)
            target_y_inches: Vertical target position from plate center (inches)
            actual_x_ft: Horizontal actual position from plate center (feet)
            actual_z_ft: Vertical actual position from ground (feet)
            sz_top_ft: Top of strike zone in feet from ground (optional)
            sz_bot_ft: Bottom of strike zone in feet from ground (optional)
            
        Returns:
            Tuple of (target_x_inches, target_z_inches, actual_x_inches, actual_z_inches)
            All in a compatible coordinate system
        """
        # Convert feet to inches for Statcast values
        # IMPORTANT: Apply -1 multiplier to Statcast X to correct for reversed axis
        actual_x_inches = -1.0 * actual_x_ft * 12.0  # Reverse x-axis direction
        actual_z_inches = actual_z_ft * 12.0
        
        # Get strike zone info to help with coordinate transformation
        sz_top_inches = None if sz_top_ft is None else sz_top_ft * 12.0
        sz_bot_inches = None if sz_bot_ft is None else sz_bot_ft * 12.0
        
        # Get typical strike zone height if not provided
        if sz_top_inches is None or sz_bot_inches is None:
            typical_sz_top = 42.0  # ~3.5 feet from ground
            typical_sz_bot = 18.0  # ~1.5 feet from ground
            sz_height = typical_sz_top - typical_sz_bot
        else:
            sz_height = sz_top_inches - sz_bot_inches
        
        # Calculate plate center height from ground 
        # (MLB standard: plate is ~10" off ground at the top, so center is ~10" + half of plate thickness)
        plate_center_height = self.PLATE_HEIGHT_FROM_GROUND_INCHES + 0.5  # Add half inch for plate thickness
        
        # GloveTracker Y coord is from plate center, Statcast Z is from ground
        # Transform target Y to compatible Z coordinate
        target_z_inches = plate_center_height + target_y_inches
        
        # For diagnostic purposes, calculate strike zone center
        sz_center_from_ground = (
            sz_top_inches + sz_bot_inches
        ) / 2.0 if sz_top_inches is not None and sz_bot_inches is not None else None
        
        # Log detailed coordinate transformation if debugging
        if self.debug_mode:
            self.logger.debug(f"Coordinate Transform Details:")
            self.logger.debug(f"  Target X: {target_x_inches:.2f}\" (from plate center)")
            self.logger.debug(f"  Target Y: {target_y_inches:.2f}\" (from plate center)")
            self.logger.debug(f"  Transformed to Z: {target_z_inches:.2f}\" (from ground)")
            self.logger.debug(f"  Raw Statcast X: {actual_x_ft:.2f} ft (reversed)")
            self.logger.debug(f"  Actual X: {actual_x_inches:.2f}\" (from plate center, direction corrected)")
            self.logger.debug(f"  Actual Z: {actual_z_inches:.2f}\" (from ground)")
            self.logger.debug(f"  Plate center height: {plate_center_height:.2f}\" (from ground)")
            if sz_center_from_ground is not None:
                self.logger.debug(f"  Strike zone center: {sz_center_from_ground:.2f}\" (from ground)")
            self.logger.debug(f"  Strike zone height: {sz_height:.2f}\"")
        
        return target_x_inches, target_z_inches, actual_x_inches, actual_z_inches
    
    def _calculate_deviation(self, 
                            target_x: float, 
                            target_z: float, 
                            actual_x: float, 
                            actual_z: float) -> Tuple[float, float, float]:
        """
        Calculate deviation between target and actual pitch location.
        
        Args:
            target_x: Target X position in inches
            target_z: Target Z position in inches
            actual_x: Actual X position in inches
            actual_z: Actual Z position in inches
            
        Returns:
            Tuple of (deviation_inches, dev_x, dev_z)
        """
        # Calculate component deviations
        dev_x = actual_x - target_x
        dev_z = actual_z - target_z
        
        # Calculate Euclidean distance
        deviation_inches = np.sqrt(dev_x**2 + dev_z**2)
        
        return deviation_inches, dev_x, dev_z
    
    def _pixel_coords_from_inches(self, 
                                 x_inches: float, 
                                 z_inches: float,
                                 plate_center_px: Tuple[float, float],
                                 ppi: float,
                                 plate_height_from_ground: float) -> Tuple[int, int]:
        """
        Convert real-world coordinates to pixel coordinates for visualization.
        
        Args:
            x_inches: X position in inches from plate center
            z_inches: Z position in inches from ground
            plate_center_px: Pixel coordinates of plate center (x, y)
            ppi: Pixels per inch
            plate_height_from_ground: Height of plate from ground in inches
            
        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        plate_cx_px, plate_cy_px = plate_center_px
        
        # Convert x_inches (from plate center) directly
        pixel_x = int(plate_cx_px + (x_inches * ppi))
        
        # Convert z_inches (from ground) to y_pixels
        # 1. Convert to inches from plate center
        y_inches_from_plate_center = z_inches - plate_height_from_ground
        # 2. Convert to pixels, noting that y increases downward in pixel space
        pixel_y = int(plate_cy_px - (y_inches_from_plate_center * ppi))
        
        return pixel_x, pixel_y
    
    # --- Main Analysis Methods ---
    
    def calculate_command_metrics(self, csv_path: str, statcast_row: pd.Series) -> Optional[Dict]:
        """
        Calculate command metrics for a pitch using GloveTracker CSV and Statcast data.
        
        Args:
            csv_path: Path to GloveTracker CSV file
            statcast_row: Pandas Series with Statcast data for this pitch
            
        Returns:
            Dictionary with command metrics or None if analysis fails
        """
        # Extract game and play IDs
        try:
            game_pk = int(statcast_row['game_pk'])
            play_id = str(statcast_row['play_id'])
        except (KeyError, ValueError):
            self.logger.warning("Missing game_pk or play_id in Statcast data")
            return None
        
        # Load CSV data
        try:
            # Check required columns
            track_df = pd.read_csv(csv_path)
            required_cols = [
                'frame_idx', 'glove_processed_x', 'glove_processed_y', 
                'is_interpolated', 'pixels_per_inch', 
                'homeplate_center_x', 'homeplate_center_y'
            ]
            
            if not all(col in track_df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in track_df.columns]
                self.logger.warning(f"CSV {os.path.basename(csv_path)} missing columns: {missing}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to read CSV {csv_path}: {str(e)}")
            return None
            
        # Find ball crossing frame if possible
        crossing_frame = self._find_ball_crossing_frame(track_df)
        
        # Find intent frame where pitcher aimed (target)
        intent_frame = self._find_intent_frame(track_df, crossing_frame)
        
        if intent_frame is None:
            self.logger.warning(f"No reliable intent frame found for {play_id}")
            return None
            
        # Get data from the identified intent frame
        intent_row = track_df[track_df['frame_idx'] == intent_frame]
        if intent_row.empty:
            self.logger.warning(f"Intent frame {intent_frame} not found in CSV for {play_id}")
            return None
            
        intent_data = intent_row.iloc[0]
        
        # Extract target position (where catcher placed glove)
        target_x_inches = intent_data['glove_processed_x'] 
        target_y_inches = intent_data['glove_processed_y']
        
        # Get plate pixel information for later visualization
        plate_center_px = (
            intent_data['homeplate_center_x'],
            intent_data['homeplate_center_y']
        )
        ppi = intent_data['pixels_per_inch']
        
        if pd.isna(target_x_inches) or pd.isna(target_y_inches):
            self.logger.warning(f"Missing target coords in intent frame {intent_frame}")
            return None
            
        if pd.isna(ppi) or ppi <= 0:
            self.logger.warning(f"Invalid pixels per inch: {ppi}")
            return None
            
        if pd.isna(plate_center_px[0]) or pd.isna(plate_center_px[1]):
            self.logger.warning(f"Missing plate center coordinates")
            return None
            
        # Get actual pitch location from Statcast
        try:
            actual_x_ft = statcast_row['plate_x']
            actual_z_ft = statcast_row['plate_z']
            sz_top_ft = statcast_row.get('sz_top')
            sz_bot_ft = statcast_row.get('sz_bot')
        except KeyError:
            self.logger.warning(f"Missing pitch location in Statcast data")
            return None
            
        if pd.isna(actual_x_ft) or pd.isna(actual_z_ft):
            self.logger.warning(f"NaN values in Statcast pitch location")
            return None
            
        # Transform coordinates to compatible system
        target_x, target_z, actual_x, actual_z = self._transform_coordinates(
            target_x_inches, target_y_inches,
            actual_x_ft, actual_z_ft,
            sz_top_ft, sz_bot_ft
        )
        
        # Calculate deviation
        deviation_inches, dev_x, dev_z = self._calculate_deviation(
            target_x, target_z, actual_x, actual_z
        )
        
        # Generate pixel coordinates for visualization if needed
        target_px = self._pixel_coords_from_inches(
            target_x_inches, 
            target_z, 
            plate_center_px, 
            ppi,
            self.PLATE_HEIGHT_FROM_GROUND_INCHES
        )
        
        actual_px = self._pixel_coords_from_inches(
            actual_x, 
            actual_z, 
            plate_center_px, 
            ppi,
            self.PLATE_HEIGHT_FROM_GROUND_INCHES
        )
        
        # Create diagnostic visualization if debug mode enabled
        if self.debug_mode:
            self._create_debug_visualization(
                csv_path, play_id, 
                target_x, target_z, 
                actual_x, actual_z,
                intent_frame, crossing_frame,
                deviation_inches
            )
        
        # Calculate glove height for additional validation
        glove_height_from_ground = self.PLATE_HEIGHT_FROM_GROUND_INCHES + target_y_inches
        
        # Assemble result dictionary
        result = {
            # Core Info
            "game_pk": game_pk,
            "play_id": play_id,
            
            # Analysis Results 
            "intent_frame": intent_frame,
            "crossing_frame": crossing_frame,
            
            # Target (where pitcher aimed)
            "target_x_inches": target_x_inches,
            "target_y_inches": target_y_inches,
            "target_z_inches": target_z,
            "glove_height_from_ground": glove_height_from_ground,
            
            # Actual (where pitch went)
            "actual_x_inches": actual_x,
            "actual_z_inches": actual_z,
            
            # Deviation
            "deviation_inches": deviation_inches,
            "dev_x_inches": dev_x,
            "dev_z_inches": dev_z,
            
            # Visualization
            "target_px": target_px,
            "actual_px": actual_px,
            "plate_center_px": plate_center_px,
            "ppi": ppi,
        }
        
        # Add additional Statcast metrics if available
        statcast_fields = [
            'pitcher', 'pitch_type', 'p_throws', 'stand', 
            'balls', 'strikes', 'outs_when_up', 
            'sz_top', 'sz_bot', 'release_speed',
            'release_pos_x', 'release_pos_z'
        ]
        
        for field in statcast_fields:
            if field in statcast_row.index:
                result[field] = statcast_row[field]
        
        return result
        
    def _create_debug_visualization(self, 
                                   csv_path: str,
                                   play_id: str,
                                   target_x: float,
                                   target_z: float,
                                   actual_x: float,
                                   actual_z: float,
                                   intent_frame: int,
                                   crossing_frame: Optional[int],
                                   deviation_inches: float):
        """
        Create diagnostic visualizations for debugging purposes.
        
        Args:
            csv_path: CSV file path
            play_id: Play ID
            target_x, target_z: Target coordinates
            actual_x, actual_z: Actual coordinates
            intent_frame: Intent frame number
            crossing_frame: Crossing frame number
            deviation_inches: Calculated deviation
            
        Side effects:
            Saves visualization files to debug directory
        """
        if not self.debug_dir:
            return
            
        # Create a plot directory for this play
        play_debug_dir = os.path.join(self.debug_dir, f"play_{play_id}")
        os.makedirs(play_debug_dir, exist_ok=True)
        
        # Velocity analysis plot
        try:
            df = pd.read_csv(csv_path)
            
            # Get glove velocity data
            valid_data = df[
                df['glove_processed_x'].notna() & 
                df['glove_processed_y'].notna()
            ].copy()
            
            if not valid_data.empty:
                valid_data = valid_data.sort_values(by='frame_idx')
                
                # Calculate velocity if needed
                if 'velocity' not in valid_data.columns:
                    valid_data['dx'] = valid_data['glove_processed_x'].diff()
                    valid_data['dy'] = valid_data['glove_processed_y'].diff()
                    valid_data['velocity'] = np.sqrt(valid_data['dx']**2 + valid_data['dy']**2)
                
                # Calculate glove height from ground
                valid_data['glove_height'] = self.PLATE_HEIGHT_FROM_GROUND_INCHES + valid_data['glove_processed_y']
                
                # Create velocity plot
                plt.figure(figsize=(10, 6))
                plt.plot(valid_data['frame_idx'], valid_data['velocity'], 'b-', label='Glove Velocity')
                
                # Add markers for key frames
                plt.axvline(x=intent_frame, color='g', linestyle='--', label=f'Intent Frame ({intent_frame})')
                if crossing_frame is not None:
                    plt.axvline(x=crossing_frame, color='r', linestyle='--', label=f'Crossing Frame ({crossing_frame})')
                
                # Add height threshold references
                plt.axhline(y=2.5, color='gray', linestyle=':', label=f'Velocity Threshold')
                
                plt.title(f'Glove Velocity Analysis - Play {play_id}')
                plt.xlabel('Frame')
                plt.ylabel('Velocity (inches/frame)')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(play_debug_dir, f'velocity_analysis.png'), dpi=100)
                plt.close()
                
                # Create height plot
                plt.figure(figsize=(10, 6))
                plt.plot(valid_data['frame_idx'], valid_data['glove_height'], 'b-', label='Glove Height')
                
                # Add markers for key frames
                plt.axvline(x=intent_frame, color='g', linestyle='--', label=f'Intent Frame ({intent_frame})')
                if crossing_frame is not None:
                    plt.axvline(x=crossing_frame, color='r', linestyle='--', label=f'Crossing Frame ({crossing_frame})')
                
                # Add height threshold references
                plt.axhline(y=self.MIN_TARGET_HEIGHT_FROM_GROUND, color='orange', linestyle=':', 
                            label=f'Min Height ({self.MIN_TARGET_HEIGHT_FROM_GROUND} in)')
                plt.axhline(y=self.MAX_TARGET_HEIGHT_FROM_GROUND, color='orange', linestyle=':', 
                            label=f'Max Height ({self.MAX_TARGET_HEIGHT_FROM_GROUND} in)')
                
                plt.title(f'Glove Height Analysis - Play {play_id}')
                plt.xlabel('Frame')
                plt.ylabel('Height from Ground (inches)')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(play_debug_dir, f'height_analysis.png'), dpi=100)
                plt.close()
                
                # Create position trace plot
                plt.figure(figsize=(10, 6))
                plt.plot(valid_data['glove_processed_x'], valid_data['glove_processed_y'], 'b-o', alpha=0.5, markersize=3)
                
                # Mark intent frame
                intent_data = valid_data[valid_data['frame_idx'] == intent_frame]
                if not intent_data.empty:
                    plt.plot(intent_data['glove_processed_x'], intent_data['glove_processed_y'], 'go', markersize=8, label=f'Intent Frame ({intent_frame})')
                
                # Mark crossing frame if available
                if crossing_frame is not None:
                    crossing_data = valid_data[valid_data['frame_idx'] == crossing_frame]
                    if not crossing_data.empty:
                        plt.plot(crossing_data['glove_processed_x'], crossing_data['glove_processed_y'], 'ro', markersize=8, label=f'Crossing Frame ({crossing_frame})')
                
                plt.title(f'Glove Position Trace - Play {play_id}')
                plt.xlabel('X Position (inches from plate center)')
                plt.ylabel('Y Position (inches from plate center)')
                plt.axis('equal')
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(play_debug_dir, f'position_trace.png'), dpi=100)
                plt.close()
                
                # Create strike zone plot
                plt.figure(figsize=(8, 10))
                # Draw strike zone rectangle (17" wide x 24" tall, average)
                sz_left = -self.PLATE_HALF_WIDTH_INCHES
                sz_right = self.PLATE_HALF_WIDTH_INCHES
                sz_bottom = 18  # typical bottom of strike zone
                sz_top = 42     # typical top of strike zone
                
                # Plot strike zone
                plt.plot([sz_left, sz_right, sz_right, sz_left, sz_left], 
                         [sz_bottom, sz_bottom, sz_top, sz_top, sz_bottom], 'k-', linewidth=2)
                
                # Plot home plate
                plate_top = self.PLATE_HEIGHT_FROM_GROUND_INCHES
                plt.fill([sz_left, sz_right, 0, sz_left], 
                         [plate_top, plate_top, plate_top+4, plate_top], 'gray', alpha=0.3)
                
                # Plot target and actual locations
                plt.plot(target_x, target_z, 'go', markersize=12, label='Target')
                plt.plot(actual_x, actual_z, 'ro', markersize=12, label='Actual')
                
                # Draw line between them
                plt.plot([target_x, actual_x], [target_z, actual_z], 'y--', linewidth=2)
                
                # Add text annotation
                plt.text(0, sz_top + 5, 
                         f'Deviation: {deviation_inches:.1f} inches',
                         horizontalalignment='center', fontsize=12)
                
                plt.title(f'Target vs Actual - Play {play_id}')
                plt.xlabel('Horizontal Position (in)')
                plt.ylabel('Height from Ground (in)')
                plt.axis('equal')
                plt.grid(True)
                plt.legend()
                
                # Set reasonable axis limits
                plt.xlim(sz_left - 10, sz_right + 10)
                plt.ylim(plate_top - 5, sz_top + 15)
                
                plt.savefig(os.path.join(play_debug_dir, f'target_vs_actual.png'), dpi=100)
                plt.close()
        
        except Exception as e:
            self.logger.warning(f"Failed to create debug visualization: {str(e)}")
    
    def _create_overlay_video(self, video_path: str, analysis_results: Dict, output_path: str):
        """
        Create an overlay video with visualization of target vs actual pitch location.
        
        Args:
            video_path: Path to input video
            analysis_results: Command analysis results
            output_path: Path to save output video
            
        Side effects:
            Saves overlay video to output_path
        """
        self.logger.info(f"Creating overlay video for {analysis_results['play_id']} -> {os.path.basename(output_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create side panel for visualization
        panel_width = 300
        panel_height = height
        total_width = width + panel_width
        
        # Create base panel with strike zone reference
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Setup plot area in panel
        plot_margin = 40
        plot_origin_x = plot_margin
        plot_origin_y = panel_height - plot_margin
        plot_width = panel_width - 2 * plot_margin
        plot_height = plot_width
        plot_top_y = plot_origin_y - plot_height
        
        # Scale for inches to panel pixels conversion
        plot_scale = plot_width / 48.0  # Scale to fit ~+/-24 inches
        
        # Function to convert inches to panel pixels
        def inches_to_panel_pixels(x_in, z_in):
            # X increases right, Z increases up
            px = int(plot_origin_x + (plot_width / 2) + (x_in * plot_scale))
            py = int(plot_origin_y - (z_in * plot_scale))
            return px, py
            
        # Draw static elements on panel
        # Strike zone outline
        sz_left = -self.PLATE_HALF_WIDTH_INCHES
        sz_right = self.PLATE_HALF_WIDTH_INCHES
        sz_bottom = 18  # typical bottom
        sz_top = 42     # typical top
        
        sz_left_px, sz_bottom_px = inches_to_panel_pixels(sz_left, sz_bottom)
        sz_right_px, sz_top_px = inches_to_panel_pixels(sz_right, sz_top)
        
        # Draw strike zone
        cv2.rectangle(panel, (sz_left_px, sz_top_px), (sz_right_px, sz_bottom_px), (180, 180, 180), 1)
        
        # Draw home plate
        plate_top = 10  # plate height from ground
        plate_points = np.array([
            inches_to_panel_pixels(sz_left, plate_top),
            inches_to_panel_pixels(sz_right, plate_top),
            inches_to_panel_pixels(0, plate_top + 4),
            inches_to_panel_pixels(sz_left, plate_top)
        ], np.int32)
        
        cv2.fillPoly(panel, [plate_points], (80, 80, 80))
        
        # Panel header
        cv2.putText(panel, "Target vs Actual", 
                   (plot_margin, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                   
        # Deviation and analysis data
        deviation = analysis_results.get('deviation_inches', 0)
        cv2.putText(panel, f"Deviation: {deviation:.1f} in",
                   (plot_margin, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                   
        # Additional info if available
        pitch_type = analysis_results.get('pitch_type', 'N/A')
        pitcher = analysis_results.get('pitcher', 'N/A')
        
        # Add pitch info
        y_pos = 90
        cv2.putText(panel, f"Pitcher: {pitcher}",
                   (plot_margin, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                   
        y_pos += 25
        cv2.putText(panel, f"Pitch: {pitch_type}",
                   (plot_margin, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                   
        # Add glove height info
        y_pos += 25
        glove_height = analysis_results.get('glove_height_from_ground', 0)
        cv2.putText(panel, f"Target Height: {glove_height:.1f} in",
                   (plot_margin, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                   
        # Draw target and actual locations in panel
        target_x = analysis_results['target_x_inches']
        target_z = analysis_results['target_z_inches']
        actual_x = analysis_results['actual_x_inches']
        actual_z = analysis_results['actual_z_inches']
        
        target_panel_px = inches_to_panel_pixels(target_x, target_z)
        actual_panel_px = inches_to_panel_pixels(actual_x, actual_z)
        
        # Draw connection line
        cv2.line(panel, target_panel_px, actual_panel_px, (0, 255, 255), 2)
        
        # Draw target marker (blue circle)
        cv2.circle(panel, target_panel_px, 6, (255, 150, 0), -1)
        cv2.putText(panel, "T", 
                   (target_panel_px[0] + 8, target_panel_px[1] + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1, cv2.LINE_AA)
                   
        # Draw actual marker (red cross)
        cv2.drawMarker(panel, actual_panel_px, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.putText(panel, "A", 
                   (actual_panel_px[0] + 8, actual_panel_px[1] + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, height))
        
        # Get key frames
        intent_frame = analysis_results.get('intent_frame')
        crossing_frame = analysis_results.get('crossing_frame')
        
        # Get pixel coordinates for overlay
        target_px = analysis_results.get('target_px')
        actual_px = analysis_results.get('actual_px')
        
        # Process frames
        target_visible = False
        
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create combined frame with side panel
            combined = np.zeros((height, total_width, 3), dtype=np.uint8)
            combined[:, :width] = frame
            combined[:, width:] = panel
            
            # Draw frame counter
            cv2.putText(combined, f"Frame: {frame_idx}", 
                       (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Show target from intent frame onwards
            if intent_frame is not None and frame_idx >= intent_frame:
                target_visible = True
                
            if target_visible and target_px:
                cv2.drawMarker(combined, target_px, (255, 150, 0), cv2.MARKER_TILTED_CROSS, 20, 2)
                
            # Show actual location at crossing frame
            if crossing_frame is not None and frame_idx >= crossing_frame and actual_px:
                cv2.drawMarker(combined, actual_px, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                
                # Draw connection line between target and actual
                if target_px:
                    cv2.line(combined, target_px, actual_px, (0, 255, 255), 1)
            
            # Add frame-specific markers
            if frame_idx == intent_frame:
                cv2.putText(combined, "INTENT FRAME", 
                           (width // 2 - 80, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 150, 0), 2, cv2.LINE_AA)
                           
            if frame_idx == crossing_frame:
                cv2.putText(combined, "BALL CROSSING", 
                           (width // 2 - 80, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Write the frame
            out.write(combined)
            
        # Clean up
        cap.release()
        out.release()
        
        self.logger.info(f"Overlay video saved to {output_path}")
        
    def analyze_folder(self, output_csv: str = "command_analysis_results.csv",
                       create_overlay: bool = False,
                       video_output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze all GloveTracker CSVs in the input directory.
        
        Args:
            output_csv: Filename for output CSV with results
            create_overlay: Whether to create overlay videos
            video_output_dir: Directory for video output (if None, uses subdirectory of csv_input_dir)
            
        Returns:
            DataFrame with analysis results
        """
        # Setup output paths
        if create_overlay and video_output_dir is None:
            video_output_dir = os.path.join(self.csv_input_dir, "command_videos")
            self.logger.info(f"Video output directory defaulting to: {video_output_dir}")
        
        if create_overlay:
            os.makedirs(video_output_dir, exist_ok=True)
            
        # Find all GloveTracker CSV files
        csv_files = glob.glob(os.path.join(self.csv_input_dir, "**", "tracked_*_tracking.csv"), recursive=True)
        csv_files = list(set(csv_files))
        
        if not csv_files:
            self.logger.error(f"No GloveTracker CSV files found in {self.csv_input_dir}")
            return pd.DataFrame()
            
        self.logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Process each CSV file
        all_results = []
        
        for csv_path in ProgressBar(iterable=csv_files, desc="Analyzing Pitch Command"):
            # Extract IDs from filename
            game_pk, play_id = self._extract_ids_from_filename(csv_path)
            if game_pk is None or play_id is None:
                continue
                
            # Get Statcast data
            statcast_game_df = self._fetch_statcast_for_game(game_pk)
            if statcast_game_df is None:
                continue
                
            # Find matching play in Statcast data
            statcast_row_df = statcast_game_df[statcast_game_df['play_id'] == play_id]
            if statcast_row_df.empty:
                continue
                
            statcast_row = statcast_row_df.iloc[0]
            
            # Download video if needed for overlay
            video_path = None
            if create_overlay:
                video_path = self._download_video_for_play(game_pk, play_id, video_output_dir)
                if not video_path:
                    self.logger.warning(f"Could not download video for {play_id}, skipping overlay")
            
            # Calculate metrics
            pitch_metrics = self.calculate_command_metrics(csv_path, statcast_row)
            
            if pitch_metrics:
                all_results.append(pitch_metrics)
                
                # Create overlay if requested
                if create_overlay and video_path:
                    overlay_filename = f"cmd_overlay_{game_pk}_{play_id}.mp4"
                    overlay_path = os.path.join(video_output_dir, overlay_filename)
                    
                    try:
                        self._create_overlay_video(video_path, pitch_metrics, overlay_path)
                    except Exception as e:
                        self.logger.error(f"Failed to create overlay for {play_id}: {str(e)}", exc_info=self.verbose)
        
        # Compile results
        if not all_results:
            self.logger.warning("No valid pitches were analyzed")
            return pd.DataFrame()
            
        results_df = pd.DataFrame(all_results)
        
        # Save results
        output_path = os.path.join(self.csv_input_dir, output_csv)
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            self.logger.info(f"Analysis results saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results CSV: {str(e)}")
            
        return results_df
        
    def calculate_aggregate_metrics(self, results_df: pd.DataFrame, 
                                   group_by: List[str] = ['pitcher'], 
                                   cmd_threshold_inches: float = 6.0) -> pd.DataFrame:
        """
        Calculate aggregate command metrics grouped by specified columns.
        
        Args:
            results_df: DataFrame with command analysis results
            group_by: List of columns to group by (e.g., ['pitcher', 'pitch_type'])
            cmd_threshold_inches: Threshold in inches to be considered "commanded"
            
        Returns:
            DataFrame with aggregated metrics
        """
        if results_df is None or results_df.empty:
            self.logger.error("No data available for aggregation")
            return pd.DataFrame()
            
        if 'deviation_inches' not in results_df.columns:
            self.logger.error("Missing 'deviation_inches' column in results")
            return pd.DataFrame()
            
        # Validate grouping columns
        valid_cols = [col for col in group_by if col in results_df.columns]
        if len(valid_cols) != len(group_by):
            missing = [col for col in group_by if col not in results_df.columns]
            self.logger.warning(f"Columns not found for grouping: {missing}")
            group_by = valid_cols
            
        if not group_by:
            self.logger.error("No valid columns for grouping")
            return pd.DataFrame()
            
        # Filter data to rows with valid deviation and grouping values
        filtered_df = results_df.dropna(subset=['deviation_inches'] + group_by)
        
        if filtered_df.empty:
            self.logger.warning("No valid data after filtering")
            return pd.DataFrame()
            
        # Add commanded flag
        filtered_df = filtered_df.copy()
        filtered_df['is_commanded'] = filtered_df['deviation_inches'] <= cmd_threshold_inches
        
        # Define aggregation functions
        agg_funcs = {
            'AvgDev_inches': pd.NamedAgg(column='deviation_inches', aggfunc='mean'),
            'StdDev_inches': pd.NamedAgg(column='deviation_inches', aggfunc='std'),
            'CmdPct': pd.NamedAgg(column='is_commanded', 
                                 aggfunc=lambda x: x.mean() * 100 if not x.empty else 0),
            'PitchCount': pd.NamedAgg(column='play_id', aggfunc='count'),
            'MinDev_inches': pd.NamedAgg(column='deviation_inches', aggfunc='min'),
            'MaxDev_inches': pd.NamedAgg(column='deviation_inches', aggfunc='max'),
            'MedianDev_inches': pd.NamedAgg(column='deviation_inches', aggfunc='median'),
        }
        
        # Add percentile calculations
        percentiles = [25, 75]
        for p in percentiles:
            agg_funcs[f'Dev_P{p}'] = pd.NamedAgg(
                column='deviation_inches', 
                aggfunc=lambda x: np.percentile(x, p)
            )
        
        # Group and aggregate
        agg_results = filtered_df.groupby(group_by, dropna=True).agg(**agg_funcs).reset_index()
        
        # Clean up NaN values
        if 'StdDev_inches' in agg_results.columns:
            agg_results['StdDev_inches'] = agg_results['StdDev_inches'].fillna(0)
            
        # Rename command percentage column
        agg_results.rename(columns={'CmdPct': f'Cmd%_<{cmd_threshold_inches}in'}, inplace=True)
        
        self.logger.info(f"Calculated aggregate metrics grouped by: {group_by}")
        return agg_results