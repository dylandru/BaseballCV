import pytest
from baseballcv.functions.new_scraper import NewBaseballSavVideoScraper
from baseballcv.functions.utils.savant_utils.crawler import Crawler
from baseballcv.functions.utils.savant_utils.gameday import GamePlayIDScraper
import polars as pl
import pandas as pd
from unittest.mock import patch, Mock
import requests
import os
import time

# TODO: Here are the tests to write
# Test with teams and players
# The various exceptions that are thrown
# Test with a response error in the url.
# Maybe a long query
# Test on old dates like 2015

# TODO: Fix these tests so they don't require a link
class TestCrawler(Crawler):
        def __init__(self, start_dt, end_dt = None):
                super().__init__(start_dt, end_dt)
        def run_executor(self):
            return super().run_executor()
        
@pytest.fixture
def test_crawler():
    return TestCrawler('2024-02-01')

class TestNewSavantScraper:

    def test_init(self):
        # Don't want to be running queries a bunch so set patch on the url
        # This tests the dates and directory output
        # THis test can use the data
        with patch("baseballcv.functions.utils.savant_utils.gameday.GamePlayIDScraper.run_executor", 
                   return_value=pl.DataFrame({'play_id': 'eedwfe-saewqw',
                                              'game_pk': '12345'})):

            start_dt, end_dt = '2024-05-01', '2024-04-28'
            scraper = NewBaseballSavVideoScraper(start_dt, end_dt)
            
            assert issubclass(NewBaseballSavVideoScraper, Crawler), "Savant Scraper is a subclass of Crawler"

            test_start_dt, test_end_dt = scraper.start_dt, scraper.end_dt

            assert end_dt == test_start_dt, "The dates should be swapped"
            assert start_dt == test_end_dt, "The dates should be swapped"
            assert os.path.exists(scraper.download_folder), "A Download Folder should be created"
    
    def test_video_names(self):
        """
        This can't be replaced, too many links to add so just leaving it as it.
        """
        scraper = NewBaseballSavVideoScraper('2024-04-12', max_return_videos=20, max_videos_per_game=2)

        assert len(scraper.game_pks) == len(scraper.play_ids), "Game PKs and Play Ids should be the same length"
        assert os.path.exists(scraper.download_folder), "A Download Folder should be created"

        scraper.run_executor()
        df = scraper.get_play_ids_df()

        game_pk, play_id = str(df['game_pk'].iloc[0]), str(df['play_id'].iloc[0])

        assert not df.empty, "There should be a returned DataFrame"
        assert isinstance(df, pd.DataFrame), "This better be a pandas DataFrame"

        download_folder = scraper.download_folder
        dir = os.listdir(download_folder)
        assert len(dir) <= 20, "There should be at most 20 returned videos"

        # Because threadpool is random with the executions, I have to manually search for the game pk, yikes
        found = False
        for file in dir:
            assert file.endswith('.mp4'), "File should be in .mp4 format"
            assert "_" in file, "There should be a _ seperator for game pk and play id"

            test_game_pk, test_play_id = file.split('_')[0], file.split('_')[1].split('.')[0]
            if game_pk == test_game_pk and play_id == test_play_id:
                found = True
            else:
                continue

        assert found, "Wrong Naming Convention. There are no instances where the Game Pk and Play ID are in the directory."
            
    def test_rate_limiter(self, test_crawler):

        crawler = test_crawler
        rate = 10
        last_called = 0

        crawler.rate_limiter(rate)

        assert last_called < crawler.last_called, "Should be a call on the limiter"

        with patch('random.uniform', return_value=0), patch("time.time", return_value = 0.05): # Should be less than the rate of 0.1
            expected_wait = 0.05 + 0

            with patch('time.sleep') as mock_sleep:
                crawler.last_called = 0
                crawler.rate_limiter(rate)
                assert mock_sleep.call_args[0][0] == expected_wait, "Wait time should be the same as expected wait time"

    def test_network_error(self, test_crawler):
        with patch('requests.get', side_effect=[requests.exceptions.RequestException("Temporary network error"), 
                                                requests.exceptions.RequestException("Temporary network error"),
                                                Mock(status_code=200)]) as mock_get:
            response = test_crawler.requests_with_retry('https://example.com/video_url')
            assert response.status_code == 200, "The 3rd request should be successful."
            assert mock_get.call_count == 3, "Mock get should be called 3 times."

    def test_teams_players_pitch_types(self):
        # Spencer Strider Game, Braves were away
        # This test can use the data
        with patch('baseballcv.functions.utils.savant_utils.gameday.Crawler', return_value = [745604]):
            team_abbr = ['ATL', None]
            for abbr in team_abbr:
                play_ids_df = GamePlayIDScraper('2024-03-29', team_abbr=abbr, player=675911, pitch_type='FF').run_executor()
                assert not play_ids_df.is_empty()
                assert play_ids_df.select(pl.col("game_pk").n_unique()).item() == 1, "Should only be one game returned"
                assert play_ids_df.select(pl.col("game_pk").unique()).item() == 745604, "This is the game that should be returned."
                assert play_ids_df.select(pl.col("pitcher").n_unique()).item() == 1, "Should only be one pitcher returned"
                assert play_ids_df.select(pl.col("pitcher").unique()).item() == 675911, "Should return Spencer Strider ID"
                assert play_ids_df.select(pl.col("pitch_type").n_unique()).item() == 1, "Should only be one pitch type returned"
                assert play_ids_df.select(pl.col("pitch_type").unique()).item() == 'FF'
                assert play_ids_df.select(pl.col("away_team").n_unique()).item() == 1, "Should only be one team returned"
                assert play_ids_df.select(pl.col("away_team").unique()).item() == 'ATL', "Should return the Braves"
                time.sleep(2) # Gives the API a little buffer

    def test_scraper_exceptions(self):
        pass

    def test_old_dates(self):
        pass


        

