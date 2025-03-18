import os
import tempfile
import shutil
import pytest
from unittest.mock import patch
import pandas as pd

# I didn't test the play id and video downlaods. May not need testing
class TestSavantScraper:
    """ Tests the functionality of the Baseball Savant Scraper Class """

    @pytest.fixture(scope="class")
    def setup(self):
        """ Sets up the environment for Savant Scraper"""
    
        temp_dir = tempfile.mkdtemp()
        return {'temp_dir': temp_dir}
    
    @pytest.fixture(scope="class")
    def clean(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, dict) and 'temp_dir' in attr:
                if os.path.exists(attr['temp_dir']):
                    shutil.rmtree(attr['temp_dir'])
    
    def test_run_statcast_pull_scraper(self, scraper, setup):
        """
        Tests the main video scraping functionality.
        Uses yesterday's date to ensure game data exists and limits to 2 videos 
        total to keep the test quick while still verifying functionality.
        """

        temp_dir = setup['temp_dir']

        #Run Scraper for Cubs on 4/01/2024 w/ minimal videos
        df = scraper.run_statcast_pull_scraper(
            start_date="2024-04-01",
            end_date="2024-04-01",
            download_folder=temp_dir,
            max_workers=2,
            max_videos=2,
            max_videos_per_game=1,
            team="CHC"
        )
        
        assert not df.empty, "Should return data"
        assert os.path.exists(temp_dir), "Download folder should exist"
        videos = os.listdir(temp_dir)
        assert len(videos) > 0 and videos[0].endswith(".mp4"), "Should have downloaded videos"
        scraper.cleanup_savant_videos(temp_dir)
        assert not os.path.exists(temp_dir), "Should remove the videos"

    def test_scraper_no_play_ids(self, scraper):
        """
        Tests on offseason dates to make sure a Value Error is returned when scraping
        invalid dates.
        """
        with pytest.raises(ValueError):
            scraper.run_statcast_pull_scraper(start_date = "2024-01-01", 
                                               end_date = "2024-01-01")
    
    # TODO: Potentially refactor savant scraper to raise exception after too many tries.
    @pytest.mark.skip()
    def test_fetch_data_fail(self, scraper):
        """
        Tests fetching game data, making sure an exception is thrown for invalid game pk.
        """
        with patch('requests.Session.get', side_effect = Exception("Simulated Network Failure")) as mock_get:
            with pytest.raises(Exception):
                scraper.fetch_game_data(game_pk=123, max_retries = 1)
            assert mock_get.call_count == 2

    def test_play_ids_blank_df(self, scraper):
        with patch('baseballcv.functions.savant_scraper.BaseballSavVideoScraper.playids_for_date_range', return_value=pd.DataFrame()) as empty_df:
            df = scraper.run_statcast_pull_scraper()

        assert df.empty
        empty_df.assert_called_once()
