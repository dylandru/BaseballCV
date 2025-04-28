# Proposed location: baseballcv/functions/utils/baseball_utils/command_analyzer.py

import pandas as pd
import numpy as np
import os
import glob
import math
import re # For regex parsing of play_id
import shutil # For moving downloaded videos if needed
from typing import List, Dict, Tuple, Optional

from baseballcv.utilities import BaseballCVLogger, ProgressBar
# Need scrapers for internal fetching/downloading
from baseballcv.functions.savant_scraper import BaseballSavVideoScraper
from baseballcv.functions.utils.savant_utils import GamePlayIDScraper # To fetch play data

class CommandAnalyzer:
    """
    Analyzes pitcher command using GloveTracker CSVs as primary input.

    - Reads CSVs from `csv_input_dir`.
    - Extracts game_pk and play_id from CSV filenames.
    - Fetches necessary Statcast data internally based on game_pk.
    - Optionally downloads corresponding videos to `video_output_dir`.
    - Performs command analysis using CSV data and fetched Statcast data.
    """
    # Regex to find UUID format (play_id) - less strict now
    PLAY_ID_REGEX = re.compile(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})')

    def __init__(
        self,
        csv_input_dir: str, # Directory containing input GloveTracker CSVs
        logger: Optional[BaseballCVLogger] = None,
        verbose: bool = True,
        device: str = 'cpu' # Device for potential future internal model use (not used now)
    ):
        """
        Initialize the CommandAnalyzer.

        Args:
            csv_input_dir (str): Path to the directory containing input GloveTracker CSV files
                                 (e.g., tracked_GAMEPK_PLAYID_tracking.csv).
            logger (BaseballCVLogger, optional): Logger instance.
            verbose (bool): Whether to print verbose logs.
            device (str): Device setting (currently unused by analyzer core logic).
        """
        self.csv_input_dir = csv_input_dir
        self.logger = logger if logger else BaseballCVLogger.get_logger(self.__class__.__name__)
        self.verbose = verbose
        self.device = device # Store device if needed later
        self.statcast_cache: Dict[int, Optional[pd.DataFrame]] = {} # Cache fetched game data
        self._video_downloader_instance = None # Cache for video downloader

        if not os.path.isdir(self.csv_input_dir):
             raise FileNotFoundError(f"CSV input directory not found: {csv_input_dir}")

        self.logger.info(f"CommandAnalyzer initialized. Reading CSVs from: {self.csv_input_dir}")
        self.logger.info("Statcast data will be fetched internally as needed.")

    # --- Internal Helper Instantiation ---
    def _get_video_downloader(self, temp_dir: str) -> BaseballSavVideoScraper:
        """Creates or returns a minimal BaseballSavVideoScraper instance for downloading."""
        # This scraper is used ONLY for its download method, needs dummy init values
        if self._video_downloader_instance is None:
             # Need dummy date that has games for GamePlayIDScraper init inside scraper
             dummy_start = "2024-04-01"
             try:
                 # We don't care about the initial data fetch here, just the object
                 self._video_downloader_instance = BaseballSavVideoScraper(
                     start_dt=dummy_start,
                     download_folder=temp_dir, # Will override later
                     max_return_videos=1 # Minimize initial work
                 )
                 self.logger.debug("Internal video downloader instance created.")
             except Exception as e:
                  self.logger.error(f"Failed to create internal video downloader: {e}")
                  return None
        return self._video_downloader_instance

    # --- Helper Functions ---
    def _extract_ids_from_filename(self, filename: str) -> Tuple[Optional[int], Optional[str]]:
        """Extracts game_pk and play_id (UUID) from CSV filename."""
        basename = os.path.basename(filename)
        # Regex for tracked_GAMEPK_PLAYID...csv
        match = re.search(r'tracked_(\d+)_([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', basename)
        if match:
            try:
                game_pk = int(match.group(1))
                play_id = str(match.group(2))
                return game_pk, play_id
            except (ValueError, IndexError):
                pass
        # Fallback: Try to extract UUID anywhere
        play_id_match = self.PLAY_ID_REGEX.search(basename)
        if play_id_match:
             play_id = str(play_id_match.group(1))
             # Attempt to find game_pk before the play_id if possible
             try:
                 prefix = basename.split(play_id)[0]
                 game_pk_match = re.findall(r'\d+', prefix) # Find numbers before play_id
                 if game_pk_match:
                     # Assume the last number sequence before play_id is game_pk
                     game_pk = int(game_pk_match[-1])
                     return game_pk, play_id
             except (ValueError, IndexError):
                 pass

        self.logger.warning(f"Could not extract valid game_pk and play_id from filename: {basename}")
        return None, None

    def _fetch_statcast_for_game(self, game_pk: int) -> Optional[pd.DataFrame]:
        """Fetches Statcast pitch data for a specific game_pk."""
        if game_pk in self.statcast_cache:
            return self.statcast_cache[game_pk] # Return cached data

        self.logger.info(f"Fetching Statcast data for game_pk: {game_pk}...")
        try:
            # Use GamePlayIDScraper logic directly for efficiency
            # Need dummy date for init
            temp_fetcher = GamePlayIDScraper(start_dt="2024-01-01")
            # Override game_pks list
            temp_fetcher.game_pks = [game_pk]
            temp_fetcher.home_team = ["N/A"] # Dummy values needed for internal call
            temp_fetcher.away_team = ["N/A"]
            # Call internal method to get data for this game
            pl_df = temp_fetcher._get_play_ids(game_pk, "N/A", "N/A")
            del temp_fetcher # Clean up

            if pl_df is not None and not pl_df.is_empty():
                pd_df = pl_df.to_pandas()
                pd_df['play_id'] = pd_df['play_id'].astype(str) # Ensure string ID
                self.statcast_cache[game_pk] = pd_df # Cache successful fetch
                self.logger.info(f"Successfully fetched and cached {len(pd_df)} pitches for game {game_pk}.")
                return pd_df
            else:
                self.logger.warning(f"No pitch data returned for game_pk: {game_pk}")
                self.statcast_cache[game_pk] = None # Cache failure
                return None
        except Exception as e:
            self.logger.error(f"Error fetching Statcast data for game_pk {game_pk}: {e}", exc_info=False)
            self.statcast_cache[game_pk] = None # Cache failure
            return None

    def _download_video_for_play(self, game_pk: int, play_id: str, video_output_dir: str):
        """Downloads video for a specific play using internal downloader."""
        os.makedirs(video_output_dir, exist_ok=True)
        output_path = os.path.join(video_output_dir, f"{game_pk}_{play_id}.mp4")

        if os.path.exists(output_path):
            self.logger.debug(f"Video already exists: {output_path}")
            return

        self.logger.info(f"Downloading video for play {play_id} (Game {game_pk})...")
        try:
            downloader = self._get_video_downloader(video_output_dir)
            if downloader:
                 # Temporarily set the download folder for the internal method
                 downloader.download_folder = video_output_dir
                 # The _download_videos method requires the response from _get_play_ids,
                 # which isn't how we want to use it here. We need direct download.
                 # Re-implementing simple download logic:
                 video_page_url = downloader.SAVANT_VIDEO_URL.format(play_id)
                 video_page_response = downloader.requests_with_retry(video_page_url)
                 if not video_page_response: raise ValueError("Failed to get video page")

                 soup = downloader.BeautifulSoup(video_page_response.content, 'html.parser')
                 video_container = soup.find('div', class_='video-box')
                 if not video_container: raise ValueError("Video container not found on page")

                 source = video_container.find('video').find('source', type='video/mp4')
                 if not source or not source.get('src'): raise ValueError("MP4 source URL not found")
                 video_url = source['src']

                 video_response = downloader.requests_with_retry(video_url, stream=True)
                 if not video_response: raise ValueError("Failed to get video stream")

                 downloader._write_content(game_pk, play_id, video_response) # Use existing write method
                 self.logger.info(f"Video downloaded successfully: {output_path}")

            else:
                 self.logger.error(f"Video downloader instance not available for {play_id}.")

        except Exception as e:
             self.logger.error(f"Failed to download video for play {play_id}: {e}", exc_info=False)


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
        statcast_row: pd.Series
        ) -> Optional[Dict]:
        """
        Calculates command deviation for a single pitch using CSV data only.
        (Keep the exact same core logic as CSV-Focused V2 version)
        """
        # (Keep implementation exactly the same as CSV-Focused V2 version)
        try:game_pk=int(statcast_row['game_pk']);play_id=str(statcast_row['play_id'])
        except(KeyError,ValueError):self.logger.error(f"Statcast row missing pk/id for {csv_path}");return None
        try: track_df=pd.read_csv(csv_path); required=['frame_idx','glove_processed_x','glove_processed_y','is_interpolated','pixels_per_inch'];
        except Exception as e: self.logger.error(f"Failed read CSV {csv_path}: {e}"); return None
        if not all(col in track_df.columns for col in required): self.logger.warning(f"CSV {os.path.basename(csv_path)} missing required columns."); return None
        intent_frame_idx=self._find_intent_frame_from_csv(track_df, velocity_threshold=5.0)
        if intent_frame_idx is None: self.logger.warning(f"No intent frame for {play_id}."); return None
        intent_frame_data_rows=track_df[track_df['frame_idx'] == intent_frame_idx]
        if intent_frame_data_rows.empty: self.logger.warning(f"Intent frame {intent_frame_idx} not found CSV {play_id}."); return None
        intent_frame_data=intent_frame_data_rows.iloc[0]
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
        Analyzes pitches based on CSV files in input_dir.
        Fetches Statcast data internally. Optionally downloads videos.

        Args:
            output_csv (str): Filename to save the results CSV within the input_dir.
            download_videos (bool): If True, download corresponding videos from Savant.
            video_output_dir (str, optional): Directory to save downloaded videos.
                                              Required if download_videos is True. Defaults to
                                              a 'videos' subdirectory within input_dir.

        Returns:
            pd.DataFrame: DataFrame containing command metrics for processed pitches.
        """
        all_results = []

        if download_videos and video_output_dir is None:
             video_output_dir = os.path.join(self.input_dir, "videos")
             self.logger.info(f"Video output directory not specified, defaulting to: {video_output_dir}")
        if download_videos:
             os.makedirs(video_output_dir, exist_ok=True)

        self.logger.info(f"Starting analysis. Reading CSVs from: {self.input_dir}")
        csv_files = glob.glob(os.path.join(self.input_dir, "**", "tracked_*_tracking.csv"), recursive=True)
        csv_files = list(set(csv_files))

        if not csv_files:
            self.logger.error(f"No GloveTracker CSV files found in {self.input_dir} or subdirs.")
            return pd.DataFrame()

        self.logger.info(f"Found {len(csv_files)} CSV files to process.")

        # --- Main Processing Loop ---
        for csv_path in ProgressBar(iterable=csv_files, desc="Analyzing Pitch Command"):
            game_pk, play_id = self._extract_ids_from_filename(csv_path)
            if game_pk is None or play_id is None:
                self.logger.warning(f"Skipping CSV due to invalid name format: {csv_path}")
                continue

            # --- Statcast Fetching ---
            statcast_game_df = self._fetch_statcast_for_game(game_pk)
            if statcast_game_df is None:
                 self.logger.warning(f"Skipping analysis for {play_id} - Failed to fetch Statcast data for game {game_pk}.")
                 continue

            statcast_row_df = statcast_game_df[statcast_game_df['play_id'] == play_id]
            if statcast_row_df.empty:
                self.logger.warning(f"Skipping play {play_id} - Data not found in fetched Statcast data for game {game_pk}.")
                continue
            statcast_row = statcast_row_df.iloc[0]

            # --- Optional Video Download ---
            if download_videos:
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
        # (Merge and Save logic remains the same, saves relative to input_dir)
        if self.statcast_cache: # Merge remaining cols from cached data
            full_statcast = pd.concat([df for df in self.statcast_cache.values() if df is not None], ignore_index=True)
            if not full_statcast.empty:
                 cols_to_merge=list(full_statcast.columns);cols_already_present=list(results_df.columns);cols_to_merge=[col for col in cols_to_merge if col not in results_df.columns and col!='play_id'];cols_to_merge.insert(0,'play_id')
                 if len(cols_to_merge)>1:
                     merge_statcast=full_statcast[cols_to_merge].drop_duplicates(subset=['play_id']);merge_statcast['play_id']=merge_statcast['play_id'].astype(str);results_df['play_id']=results_df['play_id'].astype(str)
                     results_df=pd.merge(results_df,merge_statcast,on='play_id',how='left',suffixes=('', '_statcast'));results_df=results_df[[col for col in results_df.columns if not col.endswith('_statcast')]]

        output_path = os.path.join(self.input_dir, output_csv)
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