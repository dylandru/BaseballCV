from baseballcv.functions.utils.savant_utils import GamePlayIDScraper, Crawler
import concurrent.futures
import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
import os
import shutil
import polars as pl
import pandas as pd


class NewBaseballSavVideoScraper(Crawler):
    def __init__(self, start_dt: str, end_dt: str = None, 
                 player: int = None, team_abbr: str = None, pitch_type: str = None,
                 download_folder: str = 'savant_videos', 
                 max_return_videos: int = 10, 
                 max_videos_per_game: int = None):

        super().__init__(start_dt, end_dt)

        self.play_ids_df = GamePlayIDScraper(start_dt, end_dt, team_abbr,
                                          player, pitch_type=pitch_type, 
                                          max_return_videos=max_return_videos, 
                                          max_videos_per_game=max_videos_per_game).run_executor()
        
        self.play_ids = pl.Series(self.play_ids_df.select("play_id")).to_list()
        self.game_pks = pl.Series(self.play_ids_df.select("game_pk")).to_list()
        self.VIDEO_URL = 'https://baseballsavant.mlb.com/sporty-videos?playId={}'
        self.download_folder = download_folder
        self.max_return_videos = max_return_videos
        self.max_videos_per_game = max_videos_per_game
        os.makedirs(self.download_folder, exist_ok=True)

    def run_statcast_pull_scraper(self):
        """
        Legacy method name for backward compatibility.
        """
        return self.run_executor()

    def run_executor(self) -> pd.DataFrame:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            pairs = zip(self.game_pks, self.play_ids) # Ensures these are the same order
            executor.map(lambda x: self._download_videos(*x), pairs)
        
        return self.play_ids_df.to_pandas()


    def _download_videos(self, game_pk, play_id):
        self.rate_limiter()
        video_response = requests.get(self.VIDEO_URL.format(play_id))

        if video_response.status_code == 200:

            soup = BeautifulSoup(video_response.content, 'html.parser')

            video_container = soup.find('div', class_='video-box')
            if video_container:
                video_url = video_container.find('video').find('source', type='video/mp4')['src']
                if not video_url:
                    print("This is where logging.warning will go: Warning: No video url was returned")

                video_container_response = self._scrape_video_retry(video_url)

                self._write_content(game_pk, play_id, video_container_response)
                print('Successfully downloaded video', play_id)
    
    def _write_content(self, game_pk, play_id, response):
        content_file = os.path.join(self.download_folder, f'{game_pk}_{play_id}.mp4')
        with open(content_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size = 8192):
                f.write(chunk)

    def _scrape_video_retry(self, video_url: str) -> requests.Response:

        retry = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504]) # 5 retries with 2 second wait
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount('https://', adapter)
        response = session.get(video_url, stream=True)

        return response


    def cleanup_savant_videos(self, folder_path: str) -> None:
        """Delete folder of downloaded BaseballSavant videos."""
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted {folder_path}")
            except Exception as e:
                print(f"Error deleting {folder_path}: {e}")
