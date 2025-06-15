import requests
import pandas as pd
from io import StringIO

class SavantScraper:
    """
    Scrapes data from Baseball Savant. It uses the Statcast Search CSV endpoint
    and enriches the data with calls to the MLB Gumbo API to find video playIds.
    """
    def __init__(self):
        self.search_api_url = "https://baseballsavant.mlb.com/statcast_search/csv"
        self.gumbo_api_url = "https://statsapi.mlb.com/api/v1.1/game/{}/feed/live"
        self.gumbo_cache = {}

    def _fetch_gumbo_data(self, game_pk: int):
        """
        Fetches and caches the Gumbo live feed data for a given game_pk.
        """
        if game_pk in self.gumbo_cache:
            return self.gumbo_cache[game_pk]
        
        try:
            url = self.gumbo_api_url.format(game_pk)
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            self.gumbo_cache[game_pk] = data
            return data
        except requests.exceptions.RequestException as e:
            print(f"DEBUG: Failed to fetch Gumbo data for game_pk {game_pk}: {e}")
            self.gumbo_cache[game_pk] = None # Cache failure to avoid retries
            return None

    def _find_play_id_from_gumbo(self, statcast_row: pd.Series, all_gumbo_plays: list):
        """
        Matches a row from a Statcast search with its corresponding Gumbo play event
        to find the video playId UUID.
        """
        try:
            # Statcast at_bat_number is 1-indexed, Gumbo's atBatIndex is 0-indexed.
            target_at_bat_index = statcast_row['at_bat_number'] - 1
            target_pitch_number = statcast_row['pitch_number']

            for play in all_gumbo_plays:
                if play.get('about', {}).get('atBatIndex') == target_at_bat_index:
                    for event in play.get('playEvents', []):
                        # Match the specific pitch within the at-bat
                        if event.get('isPitch') and event.get('pitchNumber') == target_pitch_number:
                            # The 'playId' field in this event is the UUID we need.
                            play_id = event.get('playId')
                            if play_id:
                                return play_id
        except (KeyError, IndexError, TypeError) as e:
            print(f"DEBUG: Error processing Gumbo data for a row: {e}")
            return None
        return None

    def _construct_video_url(self, play_id: str) -> str:
        """Constructs the video URL from a playId."""
        if not play_id or pd.isna(play_id):
            return "NO_PLAY_ID_FOUND"
        return f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"

    def get_data_by_filters(self, search_params: dict, max_results: int = 50) -> pd.DataFrame:
        """
        Fetches and processes Statcast data for a set of search filters.
        """
        query_params = search_params.copy()
        query_params['hf_video'] = ['1']
        payload = {key: '|'.join(map(str, val)) for key, val in query_params.items() if val}
        payload.update({'h_max': max_results, 'all': 'true', 'type': 'details'})
        
        print(f"--- DEBUG: Sending Request to Statcast ---\nPayload: {payload}")
        try:
            response = requests.get(self.search_api_url, params=payload, timeout=90)
            response.raise_for_status()
            
            csv_data = StringIO(response.text)
            if not csv_data.getvalue().strip():
                print("DEBUG: Statcast search returned no data.")
                return pd.DataFrame()

            df = pd.read_csv(csv_data)
            print(f"DEBUG: Initial Statcast search returned {len(df)} rows.")
            if df.empty:
                return df

            # --- Gumbo Enrichment Step ---
            print("DEBUG: Enriching with Gumbo data to find playIds...")
            df['play_id'] = None # Initialize column
            
            for game_pk in df['game_pk'].unique():
                gumbo_data = self._fetch_gumbo_data(game_pk)
                if not gumbo_data:
                    continue
                
                all_gumbo_plays = gumbo_data.get('liveData', {}).get('plays', {}).get('allPlays', [])
                if not all_gumbo_plays:
                    continue
                
                # Get indices for all rows corresponding to the current game
                game_indices = df[df['game_pk'] == game_pk].index
                
                def find_id_for_row(row):
                    return self._find_play_id_from_gumbo(row, all_gumbo_plays)

                # Apply the finder function to this game's subset of the DataFrame
                found_ids = df.loc[game_indices].apply(find_id_for_row, axis=1)
                df.loc[game_indices, 'play_id'] = found_ids

            # --- Final Processing ---
            df.dropna(subset=['play_id'], inplace=True)
            print(f"DEBUG: Found {len(df)} rows with a valid 'play_id' from Gumbo.")
            
            if not df.empty:
                df['video_url'] = df['play_id'].apply(self._construct_video_url)

            return df
            
        except requests.exceptions.RequestException as e:
            print(f"--- DEBUG: Request Failed ---\nError: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"--- DEBUG: An unexpected error occurred ---\nError: {e}")
            return pd.DataFrame()

    def get_data_by_play_id(self, game_pk: int, at_bat_number: int, pitch_number: int) -> pd.DataFrame:
        # This function can be simplified as get_data_by_filters now handles enrichment
        params = {'game_pk': [game_pk]}
        df = self.get_data_by_filters(params, max_results=500)
        
        if not df.empty:
            df['at_bat_number'] = pd.to_numeric(df['at_bat_number'], errors='coerce')
            df['pitch_number'] = pd.to_numeric(df['pitch_number'], errors='coerce')
            play_df = df[
                (df['at_bat_number'] == at_bat_number) &
                (df['pitch_number'] == pitch_number)
            ].copy()
            return play_df
        return pd.DataFrame()