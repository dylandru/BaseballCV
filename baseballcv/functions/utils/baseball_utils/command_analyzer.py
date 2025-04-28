# Location: baseballcv/functions/utils/baseball_utils/command_analyzer.py

import pandas as pd
import numpy as np
import os
import glob
import math
import re # For regex parsing of play_id
import shutil
from typing import List, Dict, Tuple, Optional

from baseballcv.utilities import BaseballCVLogger, ProgressBar
# Need scrapers for internal fetching/downloading
from baseballcv.functions.savant_scraper import BaseballSavVideoScraper
from baseballcv.functions.utils.savant_utils import GamePlayIDScraper # To fetch play data
# Also need GloveTracker for the optional internal generation in local_video mode (if kept)
# Let's remove local_video mode for now as requested and focus on csv_input mode
# from baseballcv.functions.utils.baseball_utils.glove_tracker import GloveTracker
# from baseballcv.functions.load_tools import LoadTools # May be needed by internal GloveTracker if model paths aren't absolute


class CommandAnalyzer:
    """
    Analyzes pitcher command using GloveTracker CSVs as primary input.

    - Reads CSVs from `csv_input_dir`.
    - Extracts game_pk and play_id from CSV filenames using flexible parsing.
    - Fetches necessary Statcast data internally based on game_pk.
    - Optionally downloads corresponding videos to `video_output_dir`.
    - Performs command analysis using CSV data and fetched Statcast data.
    - Core analysis uses CSV-based intent frame finding and target coordinates.
    """
    # Regex to find UUID format (play_id)
    PLAY_ID_REGEX = re.compile(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})')

    def __init__(
        self,
        csv_input_dir: str, # Directory containing input GloveTracker CSVs
        logger: Optional[BaseballCVLogger] = None,
        verbose: bool = True,
        device: str = 'cpu' # Device setting (currently unused by core logic)
    ):
        """
        Initialize the CommandAnalyzer.

        Args:
            csv_input_dir (str): Path to the directory containing input GloveTracker CSV files
                                 (e.g., tracked_GAMEPK_PLAYID_tracking.csv or similar pattern
                                 containing play_id UUID).
            logger (BaseballCVLogger, optional): Logger instance.
            verbose (bool): Whether to print verbose logs.
            device (str): Device setting (currently unused by core logic).
        """
        self.csv_input_dir = csv_input_dir # Renamed from input_dir for clarity
        self.logger = logger if logger else BaseballCVLogger.get_logger(self.__class__.__name__)
        self.verbose = verbose
        self.device = device
        self.statcast_cache: Dict[int, Optional[pd.DataFrame]] = {} # Cache fetched game data
        self._video_downloader_instance = None # Cache for video downloader

        if not os.path.isdir(self.csv_input_dir):
             raise FileNotFoundError(f"CSV input directory not found: {self.csv_input_dir}")

        self.logger.info(f"CommandAnalyzer initialized. Reading CSVs from: {self.csv_input_dir}")
        self.logger.info("Statcast data will be fetched internally as needed.")
        # No Statcast DF needed at init anymore

    # --- Internal Helper Instantiation ---
    def _get_video_downloader(self, temp_dir: str) -> Optional[BaseballSavVideoScraper]:
        """Creates or returns a minimal BaseballSavVideoScraper instance for downloading."""
        # (Keep implementation from previous response)
        if self._video_downloader_instance is None:
             dummy_start = "2024-04-01"; temp_dl_folder = os.path.join(temp_dir, "temp_scraper_init")
             try:
                 # Suppress logs during this dummy init if possible, or use a null logger
                 # Providing logger=None might make it create its own default logger
                 self._video_downloader_instance = BaseballSavVideoScraper(start_dt=dummy_start, download_folder=temp_dl_folder, max_return_videos=1, logger=None)
                 self.logger.debug("Internal video downloader instance created.")
             except Exception as e: self.logger.error(f"Failed to create internal video downloader: {e}"); return None
             finally:
                  if os.path.exists(temp_dl_folder): shutil.rmtree(temp_dl_folder) # Clean up dummy dir
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
            temp_fetcher = GamePlayIDScraper(start_dt="2024-01-01", logger=self.logger) # Use shared logger
            temp_fetcher.game_pks = [game_pk]; temp_fetcher.home_team = ["N/A"]; temp_fetcher.away_team = ["N/A"]
            pl_df = temp_fetcher._get_play_ids(game_pk, "N/A", "N/A"); del temp_fetcher
            if pl_df is not None and not pl_df.is_empty():
                pd_df = pl_df.to_pandas(); pd_df['play_id'] = pd_df['play_id'].astype(str)
                self.statcast_cache[game_pk] = pd_df; self.logger.info(f"Cached {len(pd_df)} pitches for game {game_pk}.")
                return pd_df
            else: self.logger.warning(f"No pitch data returned for game_pk: {game_pk}"); self.statcast_cache[game_pk] = None; return None
        except Exception as e: self.logger.error(f"Error fetching Statcast for game {game_pk}: {e}"); self.statcast_cache[game_pk] = None; return None

    def _download_video_for_play(self, game_pk: int, play_id: str, video_output_dir: str):
        """Downloads video for a specific play using internal downloader."""
        # (Keep implementation from previous response)
        os.makedirs(video_output_dir, exist_ok=True)
        output_path = os.path.join(video_output_dir, f"{game_pk}_{play_id}.mp4")
        if os.path.exists(output_path): self.logger.debug(f"Video exists: {output_path}"); return
        self.logger.info(f"Downloading video for play {play_id} (Game {game_pk})...")
        try:
            downloader = self._get_video_downloader(video_output_dir)
            if downloader:
                 downloader.download_folder = video_output_dir
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
            else: self.logger.error(f"Video downloader NA for {play_id}.")
        except Exception as e: self.logger.error(f"Failed video download for {play_id}: {e}")


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
    #--- End Helpers ---


    # --- Main Analysis Logic ---
    def calculate_command_metrics(
        self,
        csv_path: str,
        statcast_row: pd.Series # Must contain game_pk, play_id, plate_x, plate_z etc.
        ) -> Optional[Dict]:
        """
        Calculates command deviation for a single pitch using CSV data only.
        (Keep the exact same core logic as CSV-Focused V2 version)
        """
        # (Keep implementation exactly the same as CSV-Focused V2 version)
        try:game_pk=int(statcast_row['game_pk']);play_id=str(statcast_row['play_id'])
        except(KeyError,ValueError):self.logger.error(f"Statcast row missing pk/id for {csv_path}");return None
        try: track_df = pd.read_csv(csv_path); required=['frame_idx','glove_processed_x','glove_processed_y','is_interpolated','pixels_per_inch'];
        except Exception as e: self.logger.error(f"Failed read CSV {csv_path}: {e}"); return None
        if not all(col in track_df.columns for col in required): self.logger.warning(f"CSV {os.path.basename(csv_path)} missing required columns."); return None
        intent_frame_idx = self._find_intent_frame_from_csv(track_df, velocity_threshold=5.0)
        if intent_frame_idx is None: self.logger.warning(f"No intent frame for {play_id}."); return None
        intent_frame_data_rows = track_df[track_df['frame_idx'] == intent_frame_idx]
        if intent_frame_data_rows.empty: self.logger.warning(f"Intent frame {intent_frame_idx} not found CSV {play_id}."); return None
        intent_frame_data = intent_frame_data_rows.iloc[0]
        target_x_inches=intent_frame_data['glove_processed_x'];target_y_inches=intent_frame_data['glove_processed_y'];pixels_per_inch=intent_frame_data['pixels_per_inch']
        if pd.isna(target_x_inches) or pd.isna(target_y_inches): self.logger.warning(f"Glove target coords missing CSV frame {intent_frame_idx}, play {play_id}."); return None
        if pd.isna(pixels_per_inch) or pixels_per_inch <= 0: self.logger.warning(f"Invalid PPI ({pixels_per_inch}) in CSV for {play_id}."); return None
        try: actual_pitch_x_ft=statcast_row['plate_x']; actual_pitch_z_ft=statcast_row['plate_z'];
        except KeyError: self.logger.warning(f"Missing plate_x/z in Statcast for {play_id}"); return None
        if pd.isna(actual_pitch_x_ft) or pd.isna(actual_pitch_z_ft): self.logger.warning(f"NaN Statcast loc for {play_id}."); return None
        actual_pitch_x_inches=actual_pitch_x_ft*12.0; actual_pitch_z_inches=actual_pitch_z_ft*12.0
        dev_x=actual_pitch_x_inches-target_x_inches; dev_y=actual_pitch_z_inches-target_y_inches
        deviation_inches=np.sqrt(dev_x**2 + dev_y**2)
        if deviation_inches > 36.0 and self.verbose: self.logger.debug(f"High Dev {play_id}: {deviation_inches:.2f}in. Intent:{intent_frame_idx}, Target(in):({target_x_inches:.2f},{target_y_inches:.2f}), Actual(in):({actual_pitch_x_inches:.2f},{actual_pitch_z_inches:.2f})")
        results={"game_pk":game_pk,"play_id":play_id,"intent_frame_csv":intent_frame_idx,"target_x_inches":target_x_inches,"target_y_inches":target_y_inches,"actual_pitch_x":actual_pitch_x_inches,"actual_pitch_z":actual_pitch_z_inches,"deviation_inches":deviation_inches,"deviation_vector_x":dev_x,"deviation_vector_y":dev_y}
        for col in ['pitcher','pitch_type','p_throws','stand','balls','strikes','outs_when_up']:
            if col in statcast_row.index: results[col] = statcast_row[col]
        return results


    def analyze_folder(self, output_csv: str = "command_analysis_results.csv",
                       download_videos: bool = False, video_output_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Analyzes pitches based on CSV files in csv_input_dir.
        Fetches Statcast data internally. Optionally downloads videos.

        Args:
            output_csv (str): Filename to save the results CSV within the csv_input_dir.
            download_videos (bool): If True, download corresponding videos from Savant.
            video_output_dir (str, optional): Directory to save downloaded videos.
                                              Required if download_videos is True. Defaults to
                                              a 'downloaded_videos' subdirectory within csv_input_dir.

        Returns:
            pd.DataFrame: DataFrame containing command metrics for processed pitches.
        """
        all_results = []

        if download_videos and video_output_dir is None:
             video_output_dir = os.path.join(self.csv_input_dir, "downloaded_videos") # Corrected variable name
             self.logger.info(f"Video output directory not specified, defaulting to: {video_output_dir}")
        if download_videos:
             os.makedirs(video_output_dir, exist_ok=True)

        self.logger.info(f"Starting analysis. Reading CSVs from: {self.csv_input_dir}") # Corrected variable name
        csv_files = glob.glob(os.path.join(self.csv_input_dir, "**", "tracked_*_tracking.csv"), recursive=True)
        csv_files = list(set(csv_files))

        if not csv_files:
            self.logger.error(f"No GloveTracker CSV files found in {self.csv_input_dir} or subdirs.") # Corrected variable name
            return pd.DataFrame()

        self.logger.info(f"Found {len(csv_files)} CSV files to process.")

        # --- Main Processing Loop ---
        for csv_path in ProgressBar(iterable=csv_files, desc="Analyzing Pitch Command"):
            game_pk, play_id = self._extract_ids_from_filename(csv_path) # Removed is_video flag, always CSV here
            if game_pk is None or play_id is None:
                self.logger.warning(f"Skipping CSV due to invalid name format: {csv_path}")
                continue

            # --- Statcast Fetching ---
            statcast_game_df = self._fetch_statcast_for_game(game_pk) # Fetch data for the game
            if statcast_game_df is None:
                 self.logger.warning(f"Skipping analysis for {play_id} - Failed to fetch Statcast data for game {game_pk}.")
                 continue # Skip this CSV if game data failed

            statcast_row_df = statcast_game_df[statcast_game_df['play_id'] == play_id]
            if statcast_row_df.empty:
                self.logger.warning(f"Skipping play {play_id} - Data not found in fetched Statcast data for game {game_pk}.")
                continue # Skip this CSV if specific play not found
            statcast_row = statcast_row_df.iloc[0]

            # --- Optional Video Download ---
            if download_videos:
                 # Ensure video_output_dir is set
                 if not video_output_dir:
                      self.logger.error("video_output_dir must be specified when download_videos is True.")
                      continue # Or raise error
                 self._download_video_for_play(game_pk, play_id, video_output_dir)

            # --- Calculate Metrics ---
            pitch_metrics = self.calculate_command_metrics(csv_path, statcast_row)
            if pitch_metrics:
                 all_results.append(pitch_metrics)

        # --- Finalize ---
        if not all_results:
             self.logger.warning("No pitches could be successfully analyzed.")
             return pd.DataFrame()

        results_df = pd.DataFrame(all_results)
        # (Merge and Save logic remains the same, saves relative to csv_input_dir)
        # Merge additional columns from cached statcast data if needed
        if self.statcast_cache:
            full_statcast = pd.concat([df for df in self.statcast_cache.values() if df is not None], ignore_index=True)
            if not full_statcast.empty:
                 cols_to_merge=list(full_statcast.columns);cols_already_present=list(results_df.columns);cols_to_merge=[col for col in cols_to_merge if col not in results_df.columns and col!='play_id'];cols_to_merge.insert(0,'play_id')
                 if len(cols_to_merge)>1:
                     merge_statcast=full_statcast[cols_to_merge].drop_duplicates(subset=['play_id']);merge_statcast['play_id']=merge_statcast['play_id'].astype(str);results_df['play_id']=results_df['play_id'].astype(str)
                     results_df=pd.merge(results_df,merge_statcast,on='play_id',how='left',suffixes=('', '_statcast'));results_df=results_df[[col for col in results_df.columns if not col.endswith('_statcast')]]

        output_path = os.path.join(self.csv_input_dir, output_csv) # Save results in the input_dir
        try:os.makedirs(os.path.dirname(output_path),exist_ok=True);results_df.to_csv(output_path,index=False);self.logger.info(f"Analysis results saved to {output_path}")
        except Exception as e:self.logger.error(f"Failed to save results CSV to {output_path}: {e}")
        return results_df


    # calculate_aggregate_metrics remains the same
    def calculate_aggregate_metrics(self, results_df: pd.DataFrame, group_by: List[str] = ['pitcher'], cmd_threshold_inches: float = 6.0) -> pd.DataFrame:
        """Calculates aggregate command metrics grouped by specified columns."""
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