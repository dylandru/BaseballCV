from baseballcv.functions.utils.savant_utils import GamePlayIDScraper, Crawler
import concurrent.futures
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import os
import shutil
import polars as pl
import pandas as pd
from baseballcv.utilities import BaseballCVLogger

class BaseballSavVideoScraper(Crawler):
    """
    Class that scrapes Video Data off Baseball Savant. Inherits from the `Crawler` class to perform request with retry 
    and rate limiting.
    """
    def __init__(self, start_dt: str, end_dt: str = None, 
                 player: int = None, team_abbr: str = None, pitch_type: str = None,
                 download_folder: str = 'savant_videos', 
                 max_return_videos: int = 10, 
                 max_videos_per_game: int = None) -> None:

        self.logger = BaseballCVLogger().get_logger(self.__class__.__name__)
        super().__init__(start_dt, end_dt, self.logger)

        self.play_ids_df = GamePlayIDScraper(start_dt, end_dt, team_abbr,
                                          player, pitch_type=pitch_type, 
                                          max_return_videos=max_return_videos, 
                                          max_videos_per_game=max_videos_per_game,
                                          logger=self.logger).run_executor()
        
        self.play_ids = pl.Series(self.play_ids_df.select("play_id")).to_list()
        self.game_pks = pl.Series(self.play_ids_df.select("game_pk")).to_list()
        self.SAVANT_VIDEO_URL = 'https://baseballsavant.mlb.com/sporty-videos?playId={}'
        self.download_folder = download_folder
        self.max_return_videos = max_return_videos
        self.max_videos_per_game = max_videos_per_game
        os.makedirs(self.download_folder, exist_ok=True)

    def run_executor(self) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            pairs = zip(self.game_pks, self.play_ids) # Ensures these are the same order
            for _ in tqdm(executor.map(lambda x: self._download_videos(*x), pairs), desc="Downloading Videos", total=len(self.play_ids)):
                pass
    
    def get_play_ids_df(self) -> pd.DataFrame:
        """
        Function that returns a queried `DataFrame`. 

        Returns:
            DataFrame: The play ids df query. Output is similar to statcast csv ouptut.
        """
        return self.play_ids_df.to_pandas()


    def _download_videos(self, game_pk: int, play_id: str) -> None:
        """
        Function that downloads each video query and writes it to the `download_folder`
        using the `_write_content` function.

        Args:
            game_pk (int): The game id of the game. Used as the video file name.
            play_id (str): The play id of the game. Used to query the url and part of the video file name.

        Returns:
            None
        """
        self.rate_limiter()
        video_response = self.requests_with_retry(self.SAVANT_VIDEO_URL.format(play_id))

        soup = BeautifulSoup(video_response.content, 'html.parser')

        video_container = soup.find('div', class_='video-box')
        if video_container:
            video_url = video_container.find('video').find('source', type='video/mp4')['src']

            if video_url:
                video_container_response = self.requests_with_retry(video_url, stream=True)
                self._write_content(game_pk, play_id, video_container_response)
                self.logger.info('Successfully downloaded video %s', play_id)
    
    def _write_content(self, game_pk: int, play_id: str, response: requests.Response) -> None:
        """
        Function that writes the requested video content to the `download_folder`.

        Args:
            game_pk (int): The game id of the game. Used as the video file name.
            play_id (str): The play id of the game. Used to query the url and part of the video file name.
            response (Response): The successful response connection that was used on the url. 

        Returns:
            None
        """
        content_file = os.path.join(self.download_folder, f'{game_pk}_{play_id}.mp4')
        with open(content_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size = 8192):
                f.write(chunk)

    def cleanup_savant_videos(self) -> None:
        """
        Function that deletes the `download_folder` directory.

        Returns:
            None
        """
        if os.path.exists(self.download_folder):
            try:
                shutil.rmtree(self.download_folder)
                self.logger.info("Deleted %s", self.download_folder)
            except Exception as e:
                self.logger.error("Error deleting %s: %s", self.download_folder, e)
