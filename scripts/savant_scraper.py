import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from bs4 import BeautifulSoup
import time
import shutil
import statcast_pitches

'''Class BaseballSavVideoScraper based on code from BSav_Scraper_Vid Repo, which can be found at https://github.com/dylandru/BSav_Scraper_Vid'''

class BaseballSavVideoScraper:
    def __init__(self):
        self.session = requests.Session()

    def run_statcast_pull_scraper(self,
                          start_date: pd.Timestamp = pd.Timestamp('2024-05-01'),
                          end_date: pd.Timestamp = pd.Timestamp('2024-06-01'),
                          download_folder: str = 'savant_videos',
                          max_workers: int = 5,
                          team: str = None,
                          pitch_call: str = None,
                          max_videos: int = None,
                          max_videos_per_game: int = None) -> pd.DataFrame:
        """
        Run scraper from Statcast Pull of Play IDs. Retrieves data and processes each row in parallel.

        Args:
            start_date (pd.Timestamp): Timestamp of start date for pull. Defaults to 2024-05-01.
            end_date (pd.Timestamp): Timestamp of end date for pull. Defaults to 2024-06-01.
            download_folder (str): Folder path where videos are downloaded. Defaults to 'savant_videos'.
            max_workers (int, optional): Max number of concurrent workers. Defaults to 5.
            team (str, optional): Team filter for which videos are scraped. Defaults to None.
            pitch_call (str, optional): Pitch call filter for which videos are scraped. Defaults to None.
            max_videos (int, optional): Max number of videos to pull. Defaults to None.
            max_videos_per_game (int, optional): Max number of videos to pull for single game. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing Play IDs and relavent information on the videos that were scraped.

        Raises:
            Exception: Any error in downloading a video for a given play. 
        
        """
        try:
            print("Retrieving Play IDs to scrape...")
            df = self.playids_for_date_range(start_date=start_date, end_date=end_date, team=team, pitch_call=pitch_call) #retrieves Play IDs to scrape

            if not df.empty and 'play_id' in df.columns:
                os.makedirs(download_folder, exist_ok=True)

                if max_videos_per_game is not None:
                    df = df.groupby('game_pk').head(max_videos_per_game).reset_index(drop=True)

                if max_videos is not None:
                    df = df.head(max_videos)
                    
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_play_id = {executor.submit(self.get_video_for_play_id, row['play_id'], row['game_pk'], download_folder): row for _, row in df.iterrows()} #sets futures to download videos for given Play IDs 
                    for future in as_completed(future_to_play_id):
                        play_id = future_to_play_id[future]
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Error processing Play ID {play_id['play_id']}: {str(e)}")
                return df
            else:
                print("Play ID column not in Statcast pull or DataFrame is empty")
                return pd.DataFrame()

        except KeyboardInterrupt:
            print("Ctrl+C detected. Shutting down.")
            return pd.DataFrame()
        
    def download_video(self, video_url, save_path, max_retries=5):
        attempt = 0
        while attempt < max_retries:
            try:
                with self.session.get(video_url, stream=True) as r:
                    r.raise_for_status()
                    with open(save_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"Video downloaded to {save_path}")
                return
            except Exception as e:
                print(f"Error downloading video {video_url}: {e}. Attempt {attempt + 1} of {max_retries}")
                attempt += 1
                time.sleep(2)

    def get_video_url(self, page_url, max_retries=5):
        attempt = 0
        while attempt < max_retries:
            try:
                response = self.session.get(page_url)
                response.raise_for_status() 
                soup = BeautifulSoup(response.content, 'html.parser')
                video_container = soup.find('div', class_='video-box')
                if video_container:
                    return video_container.find('video').find('source', type='video/mp4')['src']
                return None
            except Exception as e:
                print(f"Error fetching video URL from {page_url}: {e}. Attempt {attempt + 1} of {max_retries}")
                attempt += 1
                time.sleep(2) 

    def fetch_game_data(self, game_pk, max_retries=5):
        """Fetch game data for a single game_pk using the global session."""
        attempt = 0
        while attempt < max_retries:
            try:
                url = f'https://baseballsavant.mlb.com/gf?game_pk={game_pk}'
                response = self.session.get(url)
                response.raise_for_status() 
                return response.json()
            except Exception as e:
                print(f"Error fetching game data for game_pk {game_pk}: {e}. Attempt {attempt + 1} of {max_retries}")
                attempt += 1
                time.sleep(2)  # Wait for 2 seconds before retrying

    def process_game_data(self, game_data, pitch_call=None):
        """Process game data and filter by pitch_call if provided."""
        team_home_data = game_data.get('team_home', [])
        df = pd.json_normalize(team_home_data)
        df['game_pk'] = game_data.get('game_pk')
        if pitch_call:
            df = df.loc[df['pitch_call'] == pitch_call]
        return df

    def playids_for_date_range(self, start_date, end_date, team: str = None, pitch_call: str = None):
        """
        Retrieves PlayIDs for games played within date range. Can filter by team or pitch call.
        """

        # this could be changed to only select needed columns
        statcast_query = f"""
            SELECT *
            FROM pitches
            WHERE 
                game_date >= '{start_date}'
                AND game_date <= '{end_date}'
                AND (home_team = '{team}' OR away_team = '{team}');
            """

        # statcast_pitches repository: https://github.com/Jensen-holm/statcast-era-pitches
        statcast_df = statcast_pitches.load(query=statcast_query, pandas=True) 

        game_pks = statcast_df['game_pk'].unique()

        dfs = [self.process_game_data(self.fetch_game_data(game_pk), pitch_call=pitch_call) for game_pk in game_pks]

        play_id_df = pd.concat(dfs, ignore_index=True)
        return play_id_df
        
    def get_video_for_play_id(self, play_id, game_pk, download_folder):
        """Process single play ID to download corresponding video."""
        page_url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
        try:
            video_url = self.get_video_url(page_url)
            if video_url:
                save_path = os.path.join(download_folder, f"{game_pk}_{play_id}.mp4") #Video currently named for Play ID
                self.download_video(video_url, save_path)
            else:
                print(f"No video found for playId {play_id}. Please check that the playId is correct or that the video exists at baseballsavant.mlb.com/sporty-videos?playId={play_id}.")
        except Exception as e:
            print(f"Unable to complete request. Error: {e}")

    def cleanup_savant_videos(self, folder_path: str) -> None:
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted {folder_path}")
            except Exception as e:
                print(f"Error deleting {folder_path}: {e}")
                
            return None
