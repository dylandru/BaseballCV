from baseballcv.functions.utils.savant_utils import GamePlayIDScraper, Crawler
import concurrent.futures
import requests
from bs4 import BeautifulSoup
import os
import shutil

class BaseballSavVideoScraper(Crawler):
    def __init__(self, start_dt: str, end_dt: str = None, 
                 player: int = None, team_abbr: str = None, pitch_type: str = None,
                 download_folder: str = 'savant_videos', 
                 max_return_videos: int = 10, 
                 max_videos_per_game: int = None):

        super().__init__(start_dt, end_dt)

        self.play_ids = GamePlayIDScraper(start_dt, end_dt, team_abbr,
                                          player=player, pitch_type=pitch_type, 
                                          max_return_videos=max_return_videos, 
                                          max_videos_per_game=max_videos_per_game).run_executor()
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

    def run_executor(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self._download_videos, self.play_ids)
                

    def _download_videos(self, play_id):
        self.rate_limiter()
        response = requests.get(self.VIDEO_URL.format(play_id))

        if response.status_code == 200:

            soup = BeautifulSoup(response.content, 'html.parser')

            video_container = soup.find('div', class_='video-box')
            if video_container:
                    video_url = video_container.find('video').find('source', type='video/mp4')['src']
                    with requests.Session().get(video_url, stream=True) as r:
                        self._write_content(play_id, r)
                    print('Successfully downloaded video', play_id)
    
    def _write_content(self, play_id, response):
        content_file = os.path.join(self.download_folder, f'{play_id}.mp4')
        with open(content_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size = 8192):
                f.write(chunk)

    def cleanup_savant_videos(self, folder_path: str) -> None:
        """Delete folder of downloaded BaseballSavant videos."""
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"Deleted {folder_path}")
            except Exception as e:
                print(f"Error deleting {folder_path}: {e}")
