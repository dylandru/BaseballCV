# Proposed location: baseballcv/functions/utils/baseball_utils/command_analyzer.py

import pandas as pd
import numpy as np
import os
import glob
from typing import List, Dict, Tuple, Optional

from baseballcv.utilities import BaseballCVLogger, ProgressBar

class CommandAnalyzer:
    """
    CSV-Focused Command Analyzer (V2):
    - Uses GloveTracker CSVs to find intent frame based on stability.
    - Uses processed glove coordinates (inches) from CSV at intent frame as target.
    - Compares target to Statcast pitch location (inches).
    - Adds checks and logging for diagnosing coordinate/deviation issues.
    - Does NOT require video files.
    """

    def __init__(
        self,
        glove_tracking_dir: str,
        statcast_df: pd.DataFrame,
        logger: Optional[BaseballCVLogger] = None,
        verbose: bool = True
    ):
        self.glove_tracking_dir = glove_tracking_dir
        self.logger = logger if logger else BaseballCVLogger.get_logger(self.__class__.__name__)
        self.verbose = verbose
        self.statcast_data = statcast_df

        if not os.path.isdir(glove_tracking_dir):
             raise FileNotFoundError(f"Glove tracking directory not found: {glove_tracking_dir}")

        required_cols = ['play_id', 'game_pk', 'plate_x', 'plate_z']
        if self.statcast_data is None or not all(col in self.statcast_data.columns for col in required_cols):
            missing = [col for col in required_cols if col not in (self.statcast_data.columns if self.statcast_data is not None else [])]
            raise ValueError(f"Provided Statcast DataFrame is missing required columns: {missing}")

        self.statcast_data['play_id'] = self.statcast_data['play_id'].astype(str)
        self.statcast_data['game_pk'] = self.statcast_data['game_pk'].astype(int)

        self.logger.info(f"CSV-Focused CommandAnalyzer V2 initialized. Reading CSVs from: {glove_tracking_dir}")
        self.logger.info(f"Using provided Statcast data with {len(self.statcast_data)} rows.")

    # --- Helper Functions ---
    def _extract_ids_from_filename(self, filename: str) -> Tuple[Optional[int], Optional[str]]:
        # (Same as previous version)
        basename = os.path.basename(filename)
        parts = basename.replace("tracked_", "").replace("_tracking.csv", "").split('_')
        if len(parts) == 2:
            game_pk_str, play_id = parts
            try: return int(game_pk_str), str(play_id)
            except ValueError: pass
        sub_parts = basename.replace(".csv", "").split('_')
        if len(sub_parts) == 2:
             game_pk_str, play_id = sub_parts
             try: return int(game_pk_str), str(play_id)
             except ValueError: pass
        self.logger.warning(f"Could not extract valid game_pk and play_id from filename: {basename}")
        return None, None

    def _find_intent_frame_from_csv(self, df: pd.DataFrame, velocity_threshold: float = 5.0) -> Optional[int]:
        """ Identifies the 'intent frame' based on glove stability from CSV data. """
        # (Same logic as previous version - finds last stable frame or min velocity frame)
        valid_glove_frames = df[
            df['glove_processed_x'].notna() &
            df['glove_processed_y'].notna() &
            (df['is_interpolated'] == False)
        ].copy()

        if len(valid_glove_frames) < 2: return None

        valid_glove_frames = valid_glove_frames.sort_values(by='frame_idx').reset_index()
        valid_glove_frames['dt'] = valid_glove_frames['frame_idx'].diff().fillna(1.0)
        valid_glove_frames.loc[valid_glove_frames['dt'] <= 0, 'dt'] = 1.0
        valid_glove_frames['dx'] = valid_glove_frames['glove_processed_x'].diff().fillna(0)
        valid_glove_frames['dy'] = valid_glove_frames['glove_processed_y'].diff().fillna(0)
        valid_glove_frames['velocity'] = np.sqrt(valid_glove_frames['dx']**2 + valid_glove_frames['dy']**2) / valid_glove_frames['dt']

        stable_frames = valid_glove_frames[valid_glove_frames['velocity'] < velocity_threshold]

        if not stable_frames.empty:
            intent_frame = int(stable_frames['frame_idx'].iloc[-1])
            return intent_frame
        else:
             if not valid_glove_frames.empty and 'velocity' in valid_glove_frames.columns and len(valid_glove_frames) > 1:
                 min_vel_frame_idx = valid_glove_frames['velocity'].iloc[1:].idxmin()
                 if pd.notna(min_vel_frame_idx):
                      min_vel_frame = int(valid_glove_frames.loc[min_vel_frame_idx, 'frame_idx'])
                      return min_vel_frame
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
        Includes checks for coordinate system assumptions.
        """
        game_pk = int(statcast_row['game_pk'])
        play_id = str(statcast_row['play_id'])

        try:
            track_df = pd.read_csv(csv_path)
            # Check if essential columns exist in the CSV
            required_csv_cols = ['frame_idx', 'glove_processed_x', 'glove_processed_y', 'is_interpolated', 'pixels_per_inch']
            if not all(col in track_df.columns for col in required_csv_cols):
                 self.logger.warning(f"CSV {csv_path} missing required columns for analysis. Skipping.")
                 return None
        except Exception as e:
            self.logger.error(f"Failed to read/validate tracking CSV {csv_path}: {e}")
            return None

        # --- Find Intent Frame using CSV data ---
        intent_frame_idx = self._find_intent_frame_from_csv(track_df, velocity_threshold=5.0) # Tune threshold
        if intent_frame_idx is None:
            self.logger.warning(f"Could not determine intent frame from CSV for play {play_id}. Skipping.")
            return None

        # --- Get Target Location from CSV at Intent Frame ---
        intent_frame_data_rows = track_df[track_df['frame_idx'] == intent_frame_idx]
        if intent_frame_data_rows.empty:
             self.logger.warning(f"Intent frame {intent_frame_idx} not found in data for play_id {play_id}. Skipping.")
             return None
        intent_frame_data = intent_frame_data_rows.iloc[0]

        target_x_inches = intent_frame_data['glove_processed_x']
        target_y_inches = intent_frame_data['glove_processed_y']
        pixels_per_inch = intent_frame_data['pixels_per_inch'] # Get calibration factor used by GloveTracker

        # Validate extracted target coordinates and calibration factor
        if pd.isna(target_x_inches) or pd.isna(target_y_inches):
             self.logger.warning(f"Glove target coords missing in CSV for intent frame {intent_frame_idx} in play {play_id}. Skipping.")
             return None
        if pd.isna(pixels_per_inch) or pixels_per_inch <= 0:
             self.logger.warning(f"Invalid pixels_per_inch ({pixels_per_inch}) in CSV for play {play_id}. Target coords might be unreliable. Skipping.")
             return None

        # --- Get Actual Pitch Location (Statcast) ---
        try:
            # Assume plate_x, plate_z are feet relative to plate center
            actual_pitch_x_ft = statcast_row['plate_x']
            actual_pitch_z_ft = statcast_row['plate_z']
            if pd.isna(actual_pitch_x_ft) or pd.isna(actual_pitch_z_ft): raise ValueError("NaN location")
            # Convert Statcast feet to inches
            actual_pitch_x_inches = actual_pitch_x_ft * 12.0
            actual_pitch_z_inches = actual_pitch_z_ft * 12.0
        except (KeyError, TypeError, ValueError) as e:
            self.logger.warning(f"Invalid Statcast location data for play {play_id}: {e}. Skipping.")
            return None

        # --- Calculate Deviation ---
        # Comparing inches from CSV (relative to plate center) with inches from Statcast (relative to plate center)
        dev_x = actual_pitch_x_inches - target_x_inches
        dev_y = actual_pitch_z_inches - target_y_inches
        deviation_inches = np.sqrt(dev_x**2 + dev_y**2)

        # --- Diagnostic Logging for High Deviations ---
        # Log details if deviation seems suspiciously large (e.g., > 36 inches)
        if deviation_inches > 36.0 and self.verbose:
             self.logger.warning(f"High Deviation Detected for play {play_id}: {deviation_inches:.2f} inches.")
             self.logger.debug(f"  Intent Frame (CSV): {intent_frame_idx}")
             self.logger.debug(f"  Target Coords (CSV, inches): x={target_x_inches:.2f}, y={target_y_inches:.2f}")
             self.logger.debug(f"  Actual Coords (Statcast, inches): x={actual_pitch_x_inches:.2f}, z={actual_pitch_z_inches:.2f}")
             self.logger.debug(f"  pixels_per_inch used by GloveTracker: {pixels_per_inch:.2f}")
             # You could add more info from intent_frame_data if needed

        # --- Compile Results ---
        results = {
            "game_pk": game_pk,
            "play_id": play_id,
            "intent_frame_csv": intent_frame_idx,
            "target_x_inches": target_x_inches,
            "target_y_inches": target_y_inches,
            "actual_pitch_x": actual_pitch_x_inches,
            "actual_pitch_z": actual_pitch_z_inches,
            "deviation_inches": deviation_inches,
            "deviation_vector_x": dev_x,
            "deviation_vector_y": dev_y
        }
        # Add other Statcast info
        for col in ['pitcher', 'pitch_type', 'p_throws', 'stand', 'balls', 'strikes', 'outs_when_up']:
             if col in statcast_row.index:
                  results[col] = statcast_row[col]
        return results

    # analyze_folder remains the same as the previous CSV-focused version
    def analyze_folder(self, output_csv: str = "command_analysis_results_csv_only_v2.csv") -> pd.DataFrame:
        """
        Analyzes pitches based SOLELY on GloveTracker CSVs and the Statcast DataFrame.

        Args:
            output_csv (str): Filename to save the results CSV within the glove_tracking_dir.

        Returns:
            pd.DataFrame: DataFrame containing command metrics for processed pitches.
        """
        csv_files = glob.glob(os.path.join(self.glove_tracking_dir, "**", "tracked_*_tracking.csv"), recursive=True)
        csv_files = list(set(csv_files))

        if not csv_files:
            self.logger.error(f"No GloveTracker CSV files found in {self.glove_tracking_dir} or subdirs.")
            return pd.DataFrame()
        if self.statcast_data is None or self.statcast_data.empty:
             self.logger.error("Statcast DataFrame missing. Cannot analyze.")
             return pd.DataFrame()

        all_results = []
        self.logger.info(f"Found {len(csv_files)} tracking files for CSV-based analysis (V2).")

        for csv_path in ProgressBar(iterable=csv_files, desc="Analyzing Pitch Command (CSV Only V2)"):
            game_pk, play_id = self._extract_ids_from_filename(csv_path)
            if game_pk is None or play_id is None:
                self.logger.warning(f"Skipping CSV due to invalid name format: {csv_path}")
                continue

            statcast_row_df = self.statcast_data[self.statcast_data['play_id'] == play_id]
            if statcast_row_df.empty:
                self.logger.warning(f"Skipping play {play_id} (Game {game_pk}) - Statcast data not found in provided DF.")
                continue
            statcast_row = statcast_row_df.iloc[0]

            # Calculate metrics using the CSV-based approach
            pitch_metrics = self.calculate_command_metrics(
                csv_path=csv_path,
                statcast_row=statcast_row
            )

            if pitch_metrics: # Appends even if only partial data (e.g., missing deviation)
                 all_results.append(pitch_metrics)
            # Skips logged within calculate_command_metrics

        if not all_results:
             self.logger.warning("No pitches could be successfully analyzed.")
             return pd.DataFrame()

        results_df = pd.DataFrame(all_results)

        # --- Merge additional Statcast data if needed ---
        if self.statcast_data is not None and not self.statcast_data.empty:
            cols_to_merge = list(self.statcast_data.columns)
            cols_already_present = list(results_df.columns)
            cols_to_merge = [col for col in cols_to_merge if col not in results_df.columns and col != 'play_id']
            cols_to_merge.insert(0, 'play_id')

            if len(cols_to_merge) > 1:
                merge_statcast = self.statcast_data[cols_to_merge].drop_duplicates(subset=['play_id'])
                merge_statcast['play_id'] = merge_statcast['play_id'].astype(str)
                results_df['play_id'] = results_df['play_id'].astype(str)
                results_df = pd.merge(results_df, merge_statcast, on='play_id', how='left', suffixes=('', '_statcast'))
                results_df = results_df[[col for col in results_df.columns if not col.endswith('_statcast')]]

        # --- Save results ---
        output_path = os.path.join(self.glove_tracking_dir, output_csv)
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            results_df.to_csv(output_path, index=False)
            self.logger.info(f"CSV-Only command analysis results (V2) saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results CSV to {output_path}: {e}")

        return results_df

    # calculate_aggregate_metrics remains the same
    def calculate_aggregate_metrics(self, results_df: pd.DataFrame, group_by: List[str] = ['pitcher'], cmd_threshold_inches: float = 6.0) -> pd.DataFrame:
        """Calculates aggregate command metrics grouped by specified columns."""
        # (Keep the exact same implementation as previous versions)
        if results_df is None or results_df.empty: self.logger.error("Input DF empty"); return pd.DataFrame()
        if 'deviation_inches' not in results_df.columns: self.logger.error("Missing 'deviation_inches'"); return pd.DataFrame()

        valid_group_by = [col for col in group_by if col in results_df.columns]
        if len(valid_group_by) != len(group_by):
             missing = [col for col in group_by if col not in valid_group_by]; self.logger.error(f"Grouping cols not found: {missing}")
             if not valid_group_by: return pd.DataFrame()
             group_by = valid_group_by

        results_df_filtered = results_df.dropna(subset=['deviation_inches'] + group_by)
        if results_df_filtered.empty: self.logger.warning("No valid data for aggregation after dropping NaNs."); return pd.DataFrame()

        df_filt = results_df_filtered.copy()
        df_filt['is_commanded'] = df_filt['deviation_inches'] <= cmd_threshold_inches

        agg_funcs = {
            'AvgDev_inches': pd.NamedAgg(column='deviation_inches', aggfunc='mean'),
            'StdDev_inches': pd.NamedAgg(column='deviation_inches', aggfunc='std'),
            'CmdPct': pd.NamedAgg(column='is_commanded', aggfunc=lambda x: x.mean() * 100 if not x.empty else 0),
            'Pitches': pd.NamedAgg(column='play_id', aggfunc='count')
        }
        agg_metrics = df_filt.groupby(group_by, dropna=False).agg(**agg_funcs).reset_index()

        if 'StdDev_inches' in agg_metrics.columns: agg_metrics['StdDev_inches'] = agg_metrics['StdDev_inches'].fillna(0)
        agg_metrics.rename(columns={'CmdPct': f'Cmd%_<{cmd_threshold_inches}in'}, inplace=True)
        self.logger.info(f"Calculated aggregate metrics grouped by: {group_by}")
        return agg_metrics