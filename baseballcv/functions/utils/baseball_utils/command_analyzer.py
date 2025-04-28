# Proposed location: baseballcv/functions/utils/baseball_utils/command_analyzer.py

import pandas as pd
import numpy as np
import os
import glob
import math
import re
import shutil
import cv2 # Needed for video processing & drawing
from typing import List, Dict, Tuple, Optional

from baseballcv.utilities import BaseballCVLogger, ProgressBar
from baseballcv.functions.savant_scraper import BaseballSavVideoScraper
from baseballcv.functions.utils.savant_utils import GamePlayIDScraper

class CommandAnalyzer:
    """
    Analyzes pitcher command using GloveTracker CSVs as primary input.
    Fetches Statcast data internally. Optionally downloads videos and creates
    overlay visualizations showing target vs actual pitch location.

    Core analysis uses CSV-based intent frame finding and target coordinates.
    """
    PLAY_ID_REGEX = re.compile(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})')

    def __init__(
        self,
        csv_input_dir: str,
        logger: Optional[BaseballCVLogger] = None,
        verbose: bool = True,
        device: str = 'cpu' # Only used if internal models needed (not in this version)
    ):
        """
        Initialize the CommandAnalyzer.

        Args:
            csv_input_dir (str): Path to the directory containing input GloveTracker CSV files.
            logger (BaseballCVLogger, optional): Logger instance.
            verbose (bool): Whether to print verbose logs.
            device (str): Device setting (placeholder for potential future use).
        """
        self.csv_input_dir = csv_input_dir
        self.logger = logger if logger else BaseballCVLogger.get_logger(self.__class__.__name__)
        self.verbose = verbose
        self.device = device # Store if needed later
        self.statcast_cache: Dict[int, Optional[pd.DataFrame]] = {}
        self._video_downloader_instance = None

        if not os.path.isdir(self.csv_input_dir):
             raise FileNotFoundError(f"CSV input directory not found: {self.csv_input_dir}")

        self.logger.info(f"CommandAnalyzer initialized. Reading CSVs from: {self.csv_input_dir}")
        self.logger.info("Statcast data will be fetched internally as needed.")


    # --- Internal Helper Instantiation ---
    def _get_video_downloader(self, temp_dir: str) -> Optional[BaseballSavVideoScraper]:
        """Creates or returns a minimal BaseballSavVideoScraper instance for downloading."""
        # This scraper is used ONLY for its download method, needs dummy init values
        if self._video_downloader_instance is None:
             # Need dummy date that has games for GamePlayIDScraper init inside scraper
             dummy_start = "2024-04-01" # Use a known valid in-season date
             temp_dl_folder = os.path.join(temp_dir, "temp_scraper_init_video_dl") # Unique temp folder name
             try:
                 # Initialize scraper WITHOUT the logger argument
                 self._video_downloader_instance = BaseballSavVideoScraper(
                     start_dt=dummy_start,
                     download_folder=temp_dl_folder,
                     max_return_videos=1 # Minimize initial work
                     # NO logger=None argument here
                 )
                 self.logger.debug("Internal video downloader instance created.")
             except Exception as e:
                  self.logger.error(f"Failed to create internal video downloader: {e}", exc_info=self.verbose) # Show traceback if verbose
                  # Clean up temp dir even on failure
                  if os.path.exists(temp_dl_folder):
                      try: shutil.rmtree(temp_dl_folder)
                      except Exception: pass # Ignore cleanup error
                  return None
             finally:
                  # Clean up temp download folder created during init
                  if os.path.exists(temp_dl_folder):
                      try: shutil.rmtree(temp_dl_folder)
                      except Exception as e_clean:
                           self.logger.debug(f"Could not remove temp scraper init dir: {e_clean}")
        return self._video_downloader_instance

    # --- Helper Functions ---
    def _extract_ids_from_filename(self, filename: str) -> Tuple[Optional[int], Optional[str]]:
        """Extracts game_pk and play_id (UUID) from CSV filename."""
        # (Keep implementation from previous response)
        basename = os.path.basename(filename);
        match = re.search(r'tracked_(\d+)_([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', basename)
        if match:
            try: return int(match.group(1)), str(match.group(2))
            except (ValueError, IndexError): pass
        play_id_match = self.PLAY_ID_REGEX.search(basename)
        if play_id_match:
             play_id = str(play_id_match.group(1));
             try:
                 prefix = basename.split(play_id)[0]; game_pk_match = re.findall(r'\d+', prefix)
                 if game_pk_match: return int(game_pk_match[-1]), play_id
             except (ValueError, IndexError): pass
        self.logger.warning(f"Could not extract valid game_pk and play_id from filename: {basename}")
        return None, None

    def _fetch_statcast_for_game(self, game_pk: int) -> Optional[pd.DataFrame]:
        """Fetches Statcast pitch data for a specific game_pk."""
        # (Keep implementation from previous response)
        if game_pk in self.statcast_cache: return self.statcast_cache[game_pk]
        self.logger.info(f"Fetching Statcast data for game_pk: {game_pk}...")
        try:
            valid_dummy_start_date = "2024-05-01" # Use valid date
            temp_fetcher = GamePlayIDScraper(start_dt=valid_dummy_start_date, logger=self.logger)
            temp_fetcher.game_pks = [game_pk]; temp_fetcher.home_team = ["N/A"]; temp_fetcher.away_team = ["N/A"]
            pl_df = temp_fetcher._get_play_ids(game_pk, "N/A", "N/A"); del temp_fetcher
            if pl_df is not None and not pl_df.is_empty():
                pd_df = pl_df.to_pandas(); required_cols = ['play_id','game_pk','plate_x','plate_z']
                if not all(col in pd_df.columns for col in required_cols):
                     missing = [col for col in required_cols if col not in pd_df.columns]; self.logger.error(f"Fetched Statcast game {game_pk} missing cols: {missing}"); self.statcast_cache[game_pk] = None; return None
                pd_df['play_id'] = pd_df['play_id'].astype(str); pd_df['game_pk'] = pd_df['game_pk'].astype(int)
                self.statcast_cache[game_pk] = pd_df; self.logger.info(f"Cached {len(pd_df)} pitches for game {game_pk}.")
                return pd_df
            else: self.logger.warning(f"No pitch data returned for game_pk: {game_pk}"); self.statcast_cache[game_pk] = None; return None
        except Exception as e: self.logger.error(f"Error fetching Statcast for game {game_pk}: {e}", exc_info=self.verbose); self.statcast_cache[game_pk] = None; return None

    def _download_video_for_play(self, game_pk: int, play_id: str, video_output_dir: str) -> Optional[str]:
        """Downloads video for a specific play, returns path if successful."""
        # (Keep implementation from previous response)
        os.makedirs(video_output_dir, exist_ok=True)
        output_path = os.path.join(video_output_dir, f"{game_pk}_{play_id}.mp4")
        if os.path.exists(output_path): self.logger.debug(f"Video already exists: {output_path}"); return output_path
        self.logger.info(f"Downloading video for play {play_id} (Game {game_pk})...")
        try:
            downloader = self._get_video_downloader(video_output_dir)
            if downloader:
                 downloader.download_folder = video_output_dir # Set target dir
                 video_page_url = downloader.SAVANT_VIDEO_URL.format(play_id)
                 video_page_response = downloader.requests_with_retry(video_page_url)
                 if not video_page_response: raise ValueError("Failed video page")
                 soup = downloader.BeautifulSoup(video_page_response.content, 'html.parser')
                 video_container = soup.find('div', class_='video-box')
                 if not video_container: raise ValueError("No video container")
                 source = video_container.find('video').find('source', type='video/mp4')
                 if not source or not source.get('src'): raise ValueError("No MP4 source URL")
                 video_url = source['src']
                 video_response = downloader.requests_with_retry(video_url, stream=True)
                 if not video_response: raise ValueError("Failed video stream")
                 downloader._write_content(game_pk, play_id, video_response)
                 self.logger.info(f"Video downloaded: {output_path}")
                 return output_path
            else: self.logger.error(f"Video downloader NA for {play_id}."); return None
        except Exception as e: self.logger.error(f"Failed video download for {play_id}: {e}"); return None


    def _find_intent_frame_from_csv(self, df: pd.DataFrame, velocity_threshold: float = 5.0) -> Optional[int]:
        """ Identifies the 'intent frame' based on glove stability from CSV data. """
        # (Keep the exact same implementation as previous CSV-Focused V2 version)
        valid_glove_frames=df[df['glove_processed_x'].notna()&df['glove_processed_y'].notna()&(df['is_interpolated']==False)].copy();
        if len(valid_glove_frames)<2: return None
        valid_glove_frames=valid_glove_frames.sort_values(by='frame_idx').reset_index();valid_glove_frames['dt']=valid_glove_frames['frame_idx'].diff().fillna(1.0)
        valid_glove_frames.loc[valid_glove_frames['dt']<=0,'dt']=1.0;valid_glove_frames['dx']=valid_glove_frames['glove_processed_x'].diff().fillna(0)
        valid_glove_frames['dy']=valid_glove_frames['glove_processed_y'].diff().fillna(0);valid_glove_frames['velocity']=np.sqrt(valid_glove_frames['dx']**2+valid_glove_frames['dy']**2)/valid_glove_frames['dt']
        stable_frames=valid_glove_frames[valid_glove_frames['velocity']<velocity_threshold]
        if not stable_frames.empty: return int(stable_frames['frame_idx'].iloc[-1])
        else:
            if not valid_glove_frames.empty and 'velocity' in valid_glove_frames.columns and len(valid_glove_frames)>1:
                min_vel_idx=valid_glove_frames['velocity'].iloc[1:].idxmin();
                if pd.notna(min_vel_idx): return int(valid_glove_frames.loc[min_vel_idx,'frame_idx'])
            return None

    def _get_coords_for_frame(self, df: pd.DataFrame, frame_idx: int) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """ Helper to get plate center and PPI for a specific frame from CSV """
        frame_row = df[df['frame_idx'] == frame_idx]
        if frame_row.empty:
            # Try to find nearest frame with data if exact match missing
            available_frames = df['frame_idx'].unique()
            closest_frame_idx = min(available_frames, key=lambda x: abs(x - frame_idx))
            frame_row = df[df['frame_idx'] == closest_frame_idx]
            if frame_row.empty: return None, None, None, None # Give up if still nothing

        data = frame_row.iloc[0]
        plate_cx_px = data.get('homeplate_center_x')
        plate_cy_px = data.get('homeplate_center_y')
        ppi = data.get('pixels_per_inch')

        if pd.isna(plate_cx_px) or pd.isna(plate_cy_px) or pd.isna(ppi) or ppi <= 0:
            self.logger.debug(f"Missing plate center or PPI at frame {frame_idx} (or nearest {closest_frame_idx if 'closest_frame_idx' in locals() else 'N/A'})")
            return None, None, None, None

        return plate_cx_px, plate_cy_px, ppi, int(data['frame_idx']) # Return actual frame index used

    def _convert_inches_to_pixels(self, inches_x: float, inches_y: float,
                                plate_cx_px: float, plate_cy_px: float, ppi: float) -> Optional[Tuple[int, int]]:
        """ Converts inches relative to plate center back to pixel coordinates. """
        if pd.isna(inches_x) or pd.isna(inches_y) or pd.isna(plate_cx_px) or pd.isna(plate_cy_px) or pd.isna(ppi) or ppi <= 0:
            return None
        try:
            pixel_x = int(plate_cx_px + (inches_x * ppi))
            # Pixel Y increases downwards, Real Y increases upwards from center
            pixel_y = int(plate_cy_px - (inches_y * ppi))
            return pixel_x, pixel_y
        except (ValueError, TypeError):
             return None

    # --- End Helpers ---


    # --- Main Analysis Logic ---
    def calculate_command_metrics(
        self,
        csv_path: str,
        statcast_row: pd.Series
        ) -> Optional[Dict]:
        """
        Calculates command deviation using CSV data. Returns pixel coords for overlay.
        """
        try:
            game_pk = int(statcast_row['game_pk'])
            play_id = str(statcast_row['play_id'])
        except (KeyError, ValueError): return None # Essential IDs missing

        try:
            track_df = pd.read_csv(csv_path); required=['frame_idx','glove_processed_x','glove_processed_y','is_interpolated','pixels_per_inch','homeplate_center_x','homeplate_center_y']
            if not all(col in track_df.columns for col in required): self.logger.warning(f"CSV {os.path.basename(csv_path)} missing required cols."); return None
        except Exception as e: self.logger.error(f"Failed read CSV {csv_path}: {e}"); return None

        intent_frame_idx = self._find_intent_frame_from_csv(track_df, velocity_threshold=5.0)
        if intent_frame_idx is None: self.logger.warning(f"No intent frame for {play_id}."); return None

        intent_frame_data_rows = track_df[track_df['frame_idx'] == intent_frame_idx]
        if intent_frame_data_rows.empty: self.logger.warning(f"Intent frame {intent_frame_idx} not found CSV {play_id}."); return None
        intent_frame_data = intent_frame_data_rows.iloc[0]

        target_x_inches = intent_frame_data['glove_processed_x']; target_y_inches = intent_frame_data['glove_processed_y']
        # Get plate info *from the intent frame* for coordinate conversion
        plate_cx_intent_px = intent_frame_data['homeplate_center_x']
        plate_cy_intent_px = intent_frame_data['homeplate_center_y']
        ppi_intent = intent_frame_data['pixels_per_inch']

        if pd.isna(target_x_inches) or pd.isna(target_y_inches): self.logger.warning(f"Glove target coords missing CSV frame {intent_frame_idx}, play {play_id}."); return None
        if pd.isna(ppi_intent) or ppi_intent <= 0: self.logger.warning(f"Invalid PPI ({ppi_intent}) in CSV intent frame for {play_id}."); return None
        if pd.isna(plate_cx_intent_px) or pd.isna(plate_cy_intent_px): self.logger.warning(f"Plate center missing in CSV intent frame for {play_id}."); return None


        try: actual_pitch_x_ft = statcast_row['plate_x']; actual_pitch_z_ft = statcast_row['plate_z'];
        except KeyError: self.logger.warning(f"Missing plate_x/z in Statcast for {play_id}"); return None
        if pd.isna(actual_pitch_x_ft) or pd.isna(actual_pitch_z_ft): self.logger.warning(f"NaN Statcast loc for {play_id}."); return None
        actual_pitch_x_inches = actual_pitch_x_ft*12.0; actual_pitch_z_inches = actual_pitch_z_ft*12.0

        dev_x = actual_pitch_x_inches-target_x_inches; dev_y = actual_pitch_z_inches-target_y_inches
        deviation_inches = np.sqrt(dev_x**2 + dev_y**2)

        # --- Calculate Pixel Coordinates for Overlay ---
        target_px = self._convert_inches_to_pixels(
            target_x_inches, target_y_inches, plate_cx_intent_px, plate_cy_intent_px, ppi_intent
        )
        # For actual pitch, need plate center/ppi at *crossing* frame. Find it in CSV.
        # Simplification: Assume ball crosses near the *end* of CSV data or use intent frame data if end data missing
        crossing_frame_data_rows = track_df[track_df['baseball_real_x'].notna()].sort_values('frame_idx', ascending=False)
        plate_cx_cross_px, plate_cy_cross_px, ppi_cross, crossing_frame_idx = None, None, None, None
        if not crossing_frame_data_rows.empty:
            plate_cx_cross_px, plate_cy_cross_px, ppi_cross, crossing_frame_idx = self._get_coords_for_frame(track_df, crossing_frame_data_rows.iloc[0]['frame_idx'])
        else:
             # Fallback to intent frame if no ball crossing detected in CSV
             plate_cx_cross_px, plate_cy_cross_px, ppi_cross, crossing_frame_idx = plate_cx_intent_px, plate_cy_intent_px, ppi_intent, intent_frame_idx
             self.logger.debug(f"Using intent frame {crossing_frame_idx} plate data for actual pitch pixel conversion (no ball crossing found).")

        actual_px = self._convert_inches_to_pixels(
            actual_pitch_x_inches, actual_pitch_z_inches, plate_cx_cross_px, plate_cy_cross_px, ppi_cross
        )
        # Use intent frame index if crossing frame couldn't be determined from ball data
        if crossing_frame_idx is None: crossing_frame_idx = intent_frame_idx


        results={# Core Info
                 "game_pk":game_pk,"play_id":play_id,
                 # Analysis Results
                 "intent_frame_csv":intent_frame_idx, "target_x_inches":target_x_inches,"target_y_inches":target_y_inches,"actual_pitch_x":actual_pitch_x_inches,"actual_pitch_z":actual_pitch_z_inches,"deviation_inches":deviation_inches,"deviation_vector_x":dev_x,"deviation_vector_y":dev_y,
                 # Overlay Info
                 "target_px": target_px, # Tuple (x, y) or None
                 "actual_px": actual_px, # Tuple (x, y) or None
                 "crossing_frame_est": crossing_frame_idx, # Best guess frame for crossing
                 "plate_center_at_intent": (plate_cx_intent_px, plate_cy_intent_px),
                 "ppi_at_intent": ppi_intent
                 }
        for col in ['pitcher','pitch_type','p_throws','stand','balls','strikes','outs_when_up']:
            if col in statcast_row.index: results[col] = statcast_row[col]
        return results

    def _create_overlay_video(self, video_path: str, analysis_results: Dict, overlay_output_path: str):
        """ Creates video with target/actual overlays and side panel """
        self.logger.info(f"Creating overlay video for {analysis_results['play_id']} -> {os.path.basename(overlay_output_path)}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Overlay Error: Cannot open video {video_path}")
            return

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # --- Side Panel Setup ---
        panel_width = 300 # Width of the side panel
        panel_height = height
        total_width = width + panel_width
        # Panel background
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        # Define plot area within panel (relative coords, origin bottom-left for plot)
        plot_margin = 40
        plot_origin_x = plot_margin
        plot_origin_y = panel_height - plot_margin
        plot_width = panel_width - 2 * plot_margin
        plot_height = plot_width # Make it square
        plot_top_y = plot_origin_y - plot_height
        plot_scale = plot_width / 48.0 # Scale factor: e.g., +/- 24 inches fits in plot_width

        def inches_to_panel_pixels(x_in, y_in):
             # Convert inches relative to plate center to pixel coords in panel plot area
             # X: plate center x + (inches * scale)
             # Y: plot origin y - (inches * scale) -> Y is flipped in pixels
             px = int(plot_origin_x + (plot_width / 2) + (x_in * plot_scale))
             py = int(plot_origin_y - (plot_height / 2) - (y_in * plot_scale)) # Vertical center needs adjustment if needed
             # This simple centering assumes plot center = plate center.
             # Adjust plot_origin_y or py calculation if plate center != plot center
             return px, py

        # Draw static elements on panel
        cv2.rectangle(panel, (plot_origin_x, plot_top_y), (plot_origin_x + plot_width, plot_origin_y), (255, 255, 255), 1) # Plot border
        cv2.putText(panel, "Target (T) vs Actual (A)", (plot_margin, plot_top_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(panel, f"Dev: {analysis_results.get('deviation_inches', 0):.1f} in", (plot_margin, panel_height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw target and actual points on panel if available
        target_panel_px = inches_to_panel_pixels(analysis_results['target_x_inches'], analysis_results['target_y_inches'])
        actual_panel_px = inches_to_panel_pixels(analysis_results['actual_pitch_x'], analysis_results['actual_pitch_z'])

        if target_panel_px and actual_panel_px:
             # Draw vector line
             cv2.line(panel, target_panel_px, actual_panel_px, (0, 255, 255), 1) # Yellow line
             # Draw target marker (e.g., Blue circle)
             cv2.circle(panel, target_panel_px, 5, (255, 150, 0), -1)
             cv2.putText(panel, "T", (target_panel_px[0]+5, target_panel_px[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
             # Draw actual marker (e.g., Red 'X')
             cv2.drawMarker(panel, actual_panel_px, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
             cv2.putText(panel, "A", (actual_panel_px[0]+5, actual_panel_px[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        # --- End Side Panel Setup ---


        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(overlay_output_path, fourcc, fps, (total_width, height))

        target_visible = False
        target_pixel_coords = analysis_results.get('target_px') # Tuple (x,y) or None
        actual_pixel_coords = analysis_results.get('actual_px')
        intent_frame = analysis_results.get('intent_frame_csv')
        crossing_frame = analysis_results.get('crossing_frame_est')

        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret: break

            # Combine frame with static side panel
            combined_frame = np.zeros((height, total_width, 3), dtype=np.uint8)
            combined_frame[:, :width] = frame
            combined_frame[:, width:] = panel

            # Start showing target 'X' from intent frame onwards
            if target_pixel_coords and intent_frame is not None and frame_idx >= intent_frame:
                 target_visible = True

            if target_visible and target_pixel_coords:
                cv2.drawMarker(combined_frame, target_pixel_coords, (255, 150, 0), cv2.MARKER_TILTED_CROSS, 20, 2) # Blue target X

            # Show actual 'X' only on the estimated crossing frame
            if actual_pixel_coords and crossing_frame is not None and frame_idx == crossing_frame:
                 cv2.drawMarker(combined_frame, actual_pixel_coords, (0, 0, 255), cv2.MARKER_CROSS, 20, 2) # Red actual X
                 # Optionally draw line on main frame too
                 if target_pixel_coords:
                      cv2.line(combined_frame, target_pixel_coords, actual_pixel_coords, (0, 255, 255), 1)

            # Add frame number to main video part
            cv2.putText(combined_frame, f"F: {frame_idx}", (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            out.write(combined_frame)

        cap.release()
        out.release()
        self.logger.info(f"Overlay video saved: {overlay_output_path}")


    def analyze_folder(self, output_csv: str = "command_analysis_results.csv",
                       create_overlay: bool = False, video_output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Analyzes pitches based on CSV files in csv_input_dir.
        Fetches Statcast data internally. Optionally downloads videos AND creates overlays.

        Args:
            output_csv (str): Filename to save the results CSV within the csv_input_dir.
            create_overlay (bool): If True, creates annotated videos showing target vs actual.
                                   Requires videos to be available or downloadable.
            video_output_dir (str, optional): Directory to save downloaded videos and generated overlays.
                                              Required if create_overlay is True. Defaults to
                                              a 'command_videos' subdirectory within csv_input_dir.

        Returns:
            pd.DataFrame: DataFrame containing command metrics for processed pitches.
        """
        all_results = []
        download_videos_flag = create_overlay # Automatically download if overlay needed

        if create_overlay and video_output_dir is None:
             video_output_dir = os.path.join(self.csv_input_dir, "command_videos")
             self.logger.info(f"Video output directory not specified for overlays, defaulting to: {video_output_dir}")
        if create_overlay:
             os.makedirs(video_output_dir, exist_ok=True)

        self.logger.info(f"Starting analysis. Reading CSVs from: {self.csv_input_dir}")
        csv_files = glob.glob(os.path.join(self.csv_input_dir, "**", "tracked_*_tracking.csv"), recursive=True)
        csv_files = list(set(csv_files))

        if not csv_files:
            self.logger.error(f"No GloveTracker CSV files found in {self.csv_input_dir} or subdirs.")
            return pd.DataFrame()

        self.logger.info(f"Found {len(csv_files)} CSV files to process.")

        # --- Main Processing Loop ---
        for csv_path in ProgressBar(iterable=csv_files, desc="Analyzing Pitch Command"):
            game_pk, play_id = self._extract_ids_from_filename(csv_path)
            if game_pk is None or play_id is None: continue

            statcast_game_df = self._fetch_statcast_for_game(game_pk)
            if statcast_game_df is None: continue

            statcast_row_df = statcast_game_df[statcast_game_df['play_id'] == play_id]
            if statcast_row_df.empty: continue
            statcast_row = statcast_row_df.iloc[0]

            video_path = None
            if create_overlay:
                 video_path = self._download_video_for_play(game_pk, play_id, video_output_dir)
                 if not video_path:
                      self.logger.warning(f"Could not download/find video for {play_id}, skipping overlay.")
                      # Decide if you still want to calculate metrics without overlay
                      # continue # Option: skip metrics if overlay requested but video failed

            # --- Calculate Metrics ---
            pitch_metrics = self.calculate_command_metrics(csv_path, statcast_row)

            if pitch_metrics:
                 all_results.append(pitch_metrics)
                 # --- Create Overlay if requested and possible ---
                 if create_overlay and video_path and pitch_metrics.get("target_px") and pitch_metrics.get("actual_px"):
                      overlay_filename = f"cmd_overlay_{game_pk}_{play_id}.mp4"
                      overlay_output_path = os.path.join(video_output_dir, overlay_filename)
                      try:
                           self._create_overlay_video(video_path, pitch_metrics, overlay_output_path)
                      except Exception as e_overlay:
                           self.logger.error(f"Failed to create overlay for {play_id}: {e_overlay}", exc_info=self.verbose)


        # --- Finalize ---
        # (Same as previous version: create DF, merge, save)
        if not all_results: self.logger.warning("No pitches analyzed."); return pd.DataFrame()
        results_df = pd.DataFrame(all_results)
        if self.statcast_cache:
            full_statcast = pd.concat([df for df in self.statcast_cache.values() if df is not None], ignore_index=True)
            if not full_statcast.empty:
                 cols_to_merge=list(full_statcast.columns);cols_already_present=list(results_df.columns);cols_to_merge=[col for col in cols_to_merge if col not in results_df.columns and col!='play_id'];cols_to_merge.insert(0,'play_id')
                 if len(cols_to_merge)>1:
                     merge_statcast=full_statcast[cols_to_merge].drop_duplicates(subset=['play_id']);merge_statcast['play_id']=merge_statcast['play_id'].astype(str);results_df['play_id']=results_df['play_id'].astype(str)
                     results_df=pd.merge(results_df,merge_statcast,on='play_id',how='left',suffixes=('', '_statcast'));results_df=results_df[[col for col in results_df.columns if not col.endswith('_statcast')]]
        output_path = os.path.join(self.csv_input_dir, output_csv);
        try:os.makedirs(os.path.dirname(output_path),exist_ok=True);results_df.to_csv(output_path,index=False);self.logger.info(f"Analysis results saved to {output_path}")
        except Exception as e:self.logger.error(f"Failed to save results CSV: {e}")
        return results_df

    # calculate_aggregate_metrics remains the same
    def calculate_aggregate_metrics(self, results_df: pd.DataFrame, group_by: List[str] = ['pitcher'], cmd_threshold_inches: float = 6.0) -> pd.DataFrame:
        # (Keep the exact same implementation as previous versions)
        if results_df is None or results_df.empty: self.logger.error("Input DF empty"); return pd.DataFrame()
        if 'deviation_inches' not in results_df.columns: self.logger.error("Missing 'deviation_inches'"); return pd.DataFrame()
        valid_group_by=[col for col in group_by if col in results_df.columns];
        if len(valid_group_by)!=len(group_by):missing=[col for col in group_by if col not in valid_group_by];self.logger.error(f"Grouping cols not found: {missing}");group_by=valid_group_by
        if not group_by: return pd.DataFrame()
        df_filt=results_df.dropna(subset=['deviation_inches']+group_by);
        if df_filt.empty: self.logger.warning("No valid data for aggregation after dropping NaNs."); return pd.DataFrame()
        df_filt=df_filt.copy();df_filt['is_commanded']=df_filt['deviation_inches']<=cmd_threshold_inches
        agg_funcs={'AvgDev_inches': pd.NamedAgg(column='deviation_inches',aggfunc='mean'),'StdDev_inches': pd.NamedAgg(column='deviation_inches',aggfunc='std'),'CmdPct': pd.NamedAgg(column='is_commanded',aggfunc=lambda x: x.mean()*100 if not x.empty else 0),'Pitches': pd.NamedAgg(column='play_id',aggfunc='count')}
        agg_metrics=df_filt.groupby(group_by,dropna=True).agg(**agg_funcs).reset_index()
        if 'StdDev_inches' in agg_metrics.columns: agg_metrics['StdDev_inches']=agg_metrics['StdDev_inches'].fillna(0)
        agg_metrics.rename(columns={'CmdPct': f'Cmd%_<{cmd_threshold_inches}in'},inplace=True)
        self.logger.info(f"Calculated aggregate metrics grouped by: {group_by}")
        return agg_metrics