# Proposed location: baseballcv/functions/utils/baseball_utils/command_analyzer.py

import pandas as pd
import numpy as np
import os
import glob
import math
import tempfile
from typing import List, Dict, Tuple, Optional

from baseballcv.utilities import BaseballCVLogger, ProgressBar
# Needs GloveTracker to generate missing CSVs in 'local_video' mode
from baseballcv.functions.utils.baseball_utils.glove_tracker import GloveTracker
# LoadTools might be needed by internal GloveTracker if model paths aren't absolute
from baseballcv.functions.load_tools import LoadTools

class CommandAnalyzer:
    """
    Analyzes pitcher command using GloveTracker CSV data and Statcast data.

    Supports two modes:
    1. 'scrape': Analyzes existing GloveTracker CSVs found in `glove_tracking_dir`.
                 Assumes CSVs were generated externally for the relevant plays.
    2. 'local_video': Analyzes video files from `video_base_dir`. For each video:
        - Checks `glove_tracking_dir` for a corresponding CSV.
        - If found, uses the existing CSV.
        - If not found, generates the CSV using an internal GloveTracker instance
          and saves it to `glove_tracking_dir`, then uses the generated CSV.

    Uses CSV data (existing or generated) for the core command calculation.
    Requires Statcast data covering all relevant plays to be provided during init.
    """

    def __init__(
        self,
        statcast_df: pd.DataFrame,
        processing_mode: str = 'scrape', # 'scrape' or 'local_video'
        glove_tracking_dir: str = 'command_analyzer_csvs', # Input for 'scrape', Input/Output/Cache for 'local_video'
        video_base_dir: Optional[str] = None, # REQUIRED for 'local_video' mode
        logger: Optional[BaseballCVLogger] = None,
        verbose: bool = True,
        # Parameters for internal GloveTracker (used ONLY in 'local_video' mode if CSV is missing)
        device: str = 'cpu',
        glove_tracker_confidence: float = 0.35,
        glove_tracker_filtering: bool = True,
        glove_tracker_max_velocity: float = 120.0,
        glove_tracker_suppress_warnings: bool = True
    ):
        """
        Initialize the CommandAnalyzer. Args are described in the class docstring.
        """
        self.logger = logger if logger else BaseballCVLogger.get_logger(self.__class__.__name__)
        self.verbose = verbose
        self.statcast_data = statcast_df
        self.processing_mode = processing_mode.lower()
        self.device = device
        self.glove_tracking_dir = glove_tracking_dir
        os.makedirs(self.glove_tracking_dir, exist_ok=True) # Ensure CSV dir exists

        self._internal_glove_tracker = None # Initialize tracker cache

        # Validate mode and directories
        if self.processing_mode == 'scrape':
            if not os.path.isdir(self.glove_tracking_dir):
                 self.logger.warning(f"CSV input directory '{self.glove_tracking_dir}' not found for 'scrape' mode.")
            self.video_base_dir = None
            self.logger.info(f"CommandAnalyzer initialized in SCRAPE mode. Reading CSVs from: {self.glove_tracking_dir}")
        elif self.processing_mode == 'local_video':
            if video_base_dir is None or not os.path.isdir(video_base_dir):
                raise ValueError("video_base_dir must be a valid directory for 'local_video' mode.")
            self.video_base_dir = video_base_dir
            self.logger.info(f"CommandAnalyzer initialized in LOCAL_VIDEO mode. Reading videos from: {self.video_base_dir}")
            self.logger.info(f"Existing/Generated CSVs read/stored in: {self.glove_tracking_dir}")
            # Store config for internal tracker
            self.gt_config = {
                "confidence_threshold": glove_tracker_confidence,
                "enable_filtering": glove_tracker_filtering,
                "max_velocity_inches_per_sec": glove_tracker_max_velocity,
                "suppress_detection_warnings": glove_tracker_suppress_warnings,
                "device": self.device
            }
        else:
            raise ValueError(f"Invalid processing_mode: {processing_mode}. Must be 'scrape' or 'local_video'.")

        # Validate Statcast data
        required_cols = ['play_id', 'game_pk', 'plate_x', 'plate_z']
        if self.statcast_data is None or not all(col in self.statcast_data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in (self.statcast_data.columns if self.statcast_data is not None else [])]
            raise ValueError(f"Provided Statcast DataFrame is missing required columns: {missing}")
        self.statcast_data['play_id'] = self.statcast_data['play_id'].astype(str)
        self.statcast_data['game_pk'] = self.statcast_data['game_pk'].astype(int)
        self.logger.info(f"Using provided Statcast data with {len(self.statcast_data)} rows.")

    # --- Internal GloveTracker Instantiation ---
    def _get_glove_tracker_instance(self) -> GloveTracker:
        """Creates or returns the internal GloveTracker instance."""
        # (Same as previous version)
        if self._internal_glove_tracker is None:
            self.logger.info("Initializing internal GloveTracker instance...")
            try:
                self._internal_glove_tracker = GloveTracker(
                    results_dir=self.glove_tracking_dir, device=self.gt_config['device'],
                    confidence_threshold=self.gt_config['confidence_threshold'],
                    enable_filtering=self.gt_config['enable_filtering'],
                    max_velocity_inches_per_sec=self.gt_config['max_velocity_inches_per_sec'],
                    logger=self.logger,
                    suppress_detection_warnings=self.gt_config['suppress_detection_warnings']
                )
                self.logger.info("Internal GloveTracker initialized.")
            except Exception as e:
                self.logger.error(f"Failed to initialize internal GloveTracker: {e}", exc_info=True)
                raise RuntimeError("Could not create internal GloveTracker instance") from e
        self._internal_glove_tracker.results_dir = self.glove_tracking_dir # Ensure output dir is correct
        return self._internal_glove_tracker

    # --- Helper Functions ---
    def _extract_ids_from_filename(self, filename: str, is_video: bool = False) -> Tuple[Optional[int], Optional[str]]:
        """Extracts game_pk and Statcast play_id from filename (CSV or Video)."""
        # (Same robust extraction logic as previous version)
        basename = os.path.basename(filename)
        if is_video:
            name_part = os.path.splitext(basename)[0]
            parts = name_part.split('_')
            if len(parts) >= 2:
                 game_pk_str = parts[0]; play_id = '_'.join(parts[1:])
                 try:
                      game_pk = int(game_pk_str)
                      if len(play_id.split('-')) == 5: return game_pk, str(play_id)
                 except ValueError: pass
        else: # CSV
            parts = basename.replace("tracked_", "").replace("_tracking.csv", "").split('_')
            if len(parts) >= 2:
                game_pk_str = parts[0]; play_id = '_'.join(parts[1:])
                try:
                    game_pk = int(game_pk_str)
                    if len(play_id.split('-')) == 5: return game_pk, str(play_id)
                except ValueError: pass
            sub_parts = basename.replace(".csv", "").split('_') # Fallback
            if len(sub_parts) >= 2:
                 game_pk_str = sub_parts[0]; play_id = '_'.join(sub_parts[1:])
                 try:
                      game_pk = int(game_pk_str)
                      if len(play_id.split('-')) == 5: return game_pk, str(play_id)
                 except ValueError: pass
        self.logger.warning(f"Could not extract valid game_pk and play_id from filename: {basename}")
        return None, None

    def _find_intent_frame_from_csv(self, df: pd.DataFrame, velocity_threshold: float = 5.0) -> Optional[int]:
        """ Identifies the 'intent frame' based on glove stability from CSV data. """
        # (Same logic as previous CSV-Focused V2 version)
        valid_glove_frames=df[df['glove_processed_x'].notna()&df['glove_processed_y'].notna()&(df['is_interpolated']==False)].copy()
        if len(valid_glove_frames)<2: return None
        valid_glove_frames=valid_glove_frames.sort_values(by='frame_idx').reset_index();valid_glove_frames['dt']=valid_glove_frames['frame_idx'].diff().fillna(1.0)
        valid_glove_frames.loc[valid_glove_frames['dt']<=0,'dt']=1.0;valid_glove_frames['dx']=valid_glove_frames['glove_processed_x'].diff().fillna(0)
        valid_glove_frames['dy']=valid_glove_frames['glove_processed_y'].diff().fillna(0);valid_glove_frames['velocity']=np.sqrt(valid_glove_frames['dx']**2+valid_glove_frames['dy']**2)/valid_glove_frames['dt']
        stable_frames=valid_glove_frames[valid_glove_frames['velocity']<velocity_threshold]
        if not stable_frames.empty: return int(stable_frames['frame_idx'].iloc[-1])
        else:
             if not valid_glove_frames.empty and 'velocity' in valid_glove_frames.columns and len(valid_glove_frames)>1:
                 min_vel_idx=valid_glove_frames['velocity'].iloc[1:].idxmin()
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
        (Same core logic as CSV-Focused V2 version)
        """
        # (Keep implementation exactly the same as CSV-Focused V2 version)
        game_pk = int(statcast_row['game_pk'])
        play_id = str(statcast_row['play_id'])
        try:
            track_df = pd.read_csv(csv_path); required_csv_cols = ['frame_idx','glove_processed_x','glove_processed_y','is_interpolated','pixels_per_inch']
            if not all(col in track_df.columns for col in required_csv_cols): self.logger.warning(f"CSV {os.path.basename(csv_path)} missing required columns."); return None
        except Exception as e: self.logger.error(f"Failed read CSV {csv_path}: {e}"); return None

        intent_frame_idx = self._find_intent_frame_from_csv(track_df, velocity_threshold=5.0)
        if intent_frame_idx is None: self.logger.warning(f"No intent frame for {play_id}."); return None

        intent_frame_data_rows = track_df[track_df['frame_idx'] == intent_frame_idx]
        if intent_frame_data_rows.empty: self.logger.warning(f"Intent frame {intent_frame_idx} not found in CSV for {play_id}."); return None
        intent_frame_data = intent_frame_data_rows.iloc[0]

        target_x_inches = intent_frame_data['glove_processed_x']; target_y_inches = intent_frame_data['glove_processed_y']; pixels_per_inch = intent_frame_data['pixels_per_inch']
        if pd.isna(target_x_inches) or pd.isna(target_y_inches): self.logger.warning(f"Glove target coords missing CSV frame {intent_frame_idx}, play {play_id}."); return None
        if pd.isna(pixels_per_inch) or pixels_per_inch <= 0: self.logger.warning(f"Invalid PPI ({pixels_per_inch}) in CSV for {play_id}."); return None

        try:
            actual_pitch_x_ft = statcast_row['plate_x']; actual_pitch_z_ft = statcast_row['plate_z']
            if pd.isna(actual_pitch_x_ft) or pd.isna(actual_pitch_z_ft): raise ValueError("NaN location")
            actual_pitch_x_inches = actual_pitch_x_ft * 12.0; actual_pitch_z_inches = actual_pitch_z_ft * 12.0
        except (KeyError, TypeError, ValueError) as e: self.logger.warning(f"Invalid Statcast loc for {play_id}: {e}."); return None

        dev_x = actual_pitch_x_inches - target_x_inches; dev_y = actual_pitch_z_inches - target_y_inches
        deviation_inches = np.sqrt(dev_x**2 + dev_y**2)

        if deviation_inches > 36.0 and self.verbose: self.logger.debug(f"High Dev {play_id}: {deviation_inches:.2f}in. IntentFrame:{intent_frame_idx}, Target(in):({target_x_inches:.2f},{target_y_inches:.2f}), Actual(in):({actual_pitch_x_inches:.2f},{actual_pitch_z_inches:.2f})")

        results = {"game_pk":game_pk,"play_id":play_id,"intent_frame_csv":intent_frame_idx,"target_x_inches":target_x_inches,"target_y_inches":target_y_inches,"actual_pitch_x":actual_pitch_x_inches,"actual_pitch_z":actual_pitch_z_inches,"deviation_inches":deviation_inches,"deviation_vector_x":dev_x,"deviation_vector_y":dev_y}
        for col in ['pitcher','pitch_type','p_throws','stand','balls','strikes','outs_when_up']:
             if col in statcast_row.index: results[col] = statcast_row[col]
        return results


    def analyze_folder(self, output_csv: str = "command_analysis_results.csv") -> pd.DataFrame:
        """
        Analyzes pitches based on the configured processing_mode ('scrape' or 'local_video').
        """
        all_results = []
        items_to_process = []
        lookup_dict = {} # Maps play_id -> statcast_row

        if self.statcast_data is None or self.statcast_data.empty:
             self.logger.error("Statcast DataFrame missing. Cannot analyze.")
             return pd.DataFrame()

        # Build lookup dictionary from Statcast data
        for index, row in self.statcast_data.iterrows():
            lookup_dict[str(row['play_id'])] = row # Ensure key is string

        if self.processing_mode == 'scrape':
            self.logger.info(f"Starting analysis in SCRAPE mode from: {self.glove_tracking_dir}")
            # In scrape mode, we ONLY process existing CSVs
            items_to_process = glob.glob(os.path.join(self.glove_tracking_dir, "**", "tracked_*_tracking.csv"), recursive=True)
            items_to_process = list(set(items_to_process))
            if not items_to_process:
                self.logger.error(f"No GloveTracker CSV files found in {self.glove_tracking_dir} or subdirs for 'scrape' mode.")
                return pd.DataFrame()
            desc = "Analyzing Pitch Command (scrape mode - using existing CSVs)"
            process_item = self._process_csv_item # Use the CSV processing function

        elif self.processing_mode == 'local_video':
            self.logger.info(f"Starting analysis in LOCAL_VIDEO mode from: {self.video_base_dir}")
            # In local_video mode, we iterate through VIDEO files
            video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.mts']
            for ext in video_extensions:
                 items_to_process.extend(glob.glob(os.path.join(self.video_base_dir, f"*{ext}")))
                 items_to_process.extend(glob.glob(os.path.join(self.video_base_dir, f"*{ext.upper()}")))
            items_to_process = list(set(items_to_process))
            if not items_to_process:
                 self.logger.error(f"No video files found in {self.video_base_dir} for 'local_video' mode.")
                 return pd.DataFrame()
            desc = "Analyzing Pitch Command (local_video mode)"
            process_item = self._process_video_item # Use the video processing function

        self.logger.info(f"Found {len(items_to_process)} items to process in {self.processing_mode} mode.")

        # --- Main Processing Loop ---
        for item_path in ProgressBar(iterable=items_to_process, desc=desc):
            # Process item returns metrics dict or None
            pitch_metrics = process_item(item_path, lookup_dict)
            if pitch_metrics:
                 all_results.append(pitch_metrics)

        # --- Finalize ---
        if not all_results:
             self.logger.warning("No pitches could be successfully analyzed.")
             return pd.DataFrame()

        results_df = pd.DataFrame(all_results)
        # (Merge and Save logic remains the same)
        if self.statcast_data is not None and not self.statcast_data.empty:
            cols_to_merge=list(self.statcast_data.columns);cols_already_present=list(results_df.columns);cols_to_merge=[col for col in cols_to_merge if col not in results_df.columns and col!='play_id'];cols_to_merge.insert(0,'play_id')
            if len(cols_to_merge)>1:
                merge_statcast=self.statcast_data[cols_to_merge].drop_duplicates(subset=['play_id']);merge_statcast['play_id']=merge_statcast['play_id'].astype(str);results_df['play_id']=results_df['play_id'].astype(str)
                results_df=pd.merge(results_df,merge_statcast,on='play_id',how='left',suffixes=('', '_statcast'));results_df=results_df[[col for col in results_df.columns if not col.endswith('_statcast')]]

        output_path = os.path.join(self.glove_tracking_dir, output_csv) # Save in the CSV dir
        try:os.makedirs(os.path.dirname(output_path),exist_ok=True);results_df.to_csv(output_path,index=False);self.logger.info(f"Results ({self.processing_mode} mode) saved to {output_path}")
        except Exception as e:self.logger.error(f"Failed to save results CSV to {output_path}: {e}")
        return results_df

    def _process_csv_item(self, csv_path: str, statcast_lookup: Dict) -> Optional[Dict]:
        """Processes a single CSV file in 'scrape' mode."""
        game_pk, play_id = self._extract_ids_from_filename(csv_path, is_video=False)
        if game_pk is None or play_id is None:
            self.logger.warning(f"Skipping CSV due to invalid name format: {csv_path}")
            return None

        statcast_row = statcast_lookup.get(play_id)
        if statcast_row is None:
            # Log this specific case clearly
            self.logger.warning(f"Statcast data for play_id '{play_id}' (from CSV: {os.path.basename(csv_path)}) not found in provided Statcast DataFrame. Skipping analysis for this CSV.")
            return None

        # Pass the found row to the calculator
        return self.calculate_command_metrics(csv_path, statcast_row)

    def _process_video_item(self, video_path: str, statcast_lookup: Dict) -> Optional[Dict]:
        """Processes a single video file in 'local_video' mode."""
        game_pk, play_id = self._extract_ids_from_filename(video_path, is_video=True)
        if game_pk is None or play_id is None:
            self.logger.warning(f"Skipping video due to invalid name format: {video_path}")
            return None

        statcast_row = statcast_lookup.get(play_id)
        if statcast_row is None:
            self.logger.warning(f"Skipping video {os.path.basename(video_path)} - Statcast data for play_id '{play_id}' not found in provided lookup.")
            return None

        # --- Determine CSV Path & Generate if Missing ---
        csv_filename = f"tracked_{game_pk}_{play_id}_tracking.csv"
        # Check primary location first
        csv_path_to_use = os.path.join(self.glove_tracking_dir, csv_filename)

        if not os.path.exists(csv_path_to_use):
             # Check common results subdir as fallback read location
             csv_path_results_subdir = os.path.join(self.glove_tracking_dir, "results", csv_filename)
             if os.path.exists(csv_path_results_subdir):
                 csv_path_to_use = csv_path_results_subdir
                 self.logger.debug(f"Found existing CSV in results subdir: {csv_path_to_use}")
             else:
                 # --- Generate Missing CSV ---
                 self.logger.info(f"CSV for play {play_id} not found. Generating from video: {os.path.basename(video_path)}...")
                 try:
                     tracker = self._get_glove_tracker_instance()
                     # Ensure tracker saves directly to self.glove_tracking_dir
                     tracker.results_dir = self.glove_tracking_dir
                     tracker.track_video(video_path=video_path, show_plot=False, create_video=False, generate_heatmap=False)

                     # Verify CSV was created in the primary location
                     if os.path.exists(csv_path_primary):
                          csv_path_to_use = csv_path_primary
                          self.logger.info(f"CSV generated successfully: {csv_path_to_use}")
                     elif os.path.exists(csv_path_results_subdir): # Check subdir just in case
                          csv_path_to_use = csv_path_results_subdir
                          self.logger.warning(f"CSV generated in unexpected subdir: {csv_path_to_use}")
                     else:
                          self.logger.error(f"GloveTracker ran for {play_id} but failed to create expected CSV at {csv_path_primary} or {csv_path_results_subdir}")
                          return None # Skip analysis if CSV generation failed
                 except Exception as e:
                      self.logger.error(f"Error running internal GloveTracker for {play_id}: {e}", exc_info=True)
                      return None
        else:
             self.logger.debug(f"Found existing CSV for play {play_id}: {csv_path_to_use}")

        # --- Calculate Metrics using the determined CSV path ---
        # Pass the specific statcast_row found earlier
        return self.calculate_command_metrics(csv_path_to_use, statcast_row)

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