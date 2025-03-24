import requests
import concurrent.futures
from bs4 import BeautifulSoup
import os
from baseballcv.functions.utils.savant_scraper import GamedayScraper
    
class BaseballSavScraper:
    def __init__(self, start_dt: str, end_dt: str = None, 
                 player: int = None, team: str = None, pitch_type: str = None,
                 download_folder: str = 'savant_videos', 
                 max_return_videos: int = 10, 
                 max_videos_per_game: int = None):
        
        self.play_ids = GamedayScraper(start_dt, end_dt, player, team, pitch_type, max_return_videos, max_videos_per_game).process_games()
        self.VIDEO_URL = 'https://baseballsavant.mlb.com/sporty-videos?playId={}'
        self.download_folder = download_folder
        self.max_return_videos = max_return_videos
        self.max_videos_per_game = max_videos_per_game
        os.makedirs(self.download_folder, exist_ok=True)

    def load_and_write_content(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self._download_videos, self.play_ids)
                

    def _download_videos(self, play_id):
        response = requests.get(self.VIDEO_URL.format(play_id))

        if response.status_code == 200:

            soup = BeautifulSoup(response.content, 'html.parser')

            video_container = soup.find('div', class_='video-box')
            if video_container:
                    video_url = video_container.find('video').find('source', type='video/mp4')['src']
                    with requests.Session().get(video_url, stream=True) as r:
                        self._write_content(play_id, r)
                    print('Successfully downloaded video', play_id)
        return None
    
    def _write_content(self, play_id, response):
        content_file = os.path.join(self.download_folder, f'{play_id}.mp4')
        with open(content_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size = 8192):
                f.write(chunk)



if __name__ == '__main__':
    d = BaseballSavScraper('2024-06-02', '2024-06-08', max_return_videos=10, max_videos_per_game=10, team='BOS').load_and_write_content()
