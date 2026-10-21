import pytest
import os
import json
import requests
import pandas as pd
import polars as pl
from unittest.mock import patch, MagicMock, Mock
from baseballcv.functions import BaseballSavVideoScraper
from baseballcv.functions.utils import requests_with_retry
from baseballcv.functions.utils.savant_utils.gameday import _get_team

class TestSavantScraper:
    """
    Test suite for the `BaseballSavVideoScraper`.

    This suite tests for the various capabilities for the scraper, making sure 
    proper video files are written and dataframes are returned. It also tests for
    appropriate filtering for teams/players as well as API calls.
    """
    
    def validate_video_names(self, scraper: BaseballSavVideoScraper) -> None:
        """
        Function that helps validate the video names created in the 
        destination folder.

        Args:
            scraper (BaseballSavVideoScraper): An instance of the `BaseballSavVideoScraper` class
        """
        df = scraper.play_ids_df

        random_row = df.sample(1).iloc[0]
        game_pk, play_id = str(random_row['game_pk']), str(random_row['play_id'])

        videos = os.listdir(scraper.download_folder)

        assert len(videos) > 0, 'There should be videos for this test'

        # Because threadpool is random with the executions, I have to manually search for the game pk, yikes
        found = False
        for video in videos:
            assert video.endswith('.mp4'), f'Video {video} should be .mp4 format'
            assert "_" in video, f"There should be a _ seperator for game pk and play id in video {video}"

            base_name = video.rsplit('.', 1)[0]
            test_game_pk, test_play_id = base_name.split('_', 1)
            if game_pk == test_game_pk and play_id == test_play_id:
                found = True
                break

        assert found, "Wrong Naming Convention. There are no instances where the Game Pk and Play ID are in the directory."

    # Network test
    def test_date_range(self, tmp_path_factory):
        """
        Tests the `from_date_range` class alternate constructor.

        Args:
            tmp_path_factory: Fixture for a temp directory to load videos

        This tests for the following:
        - A folder was created
        - A pandas DataFrame is created and not empty
        - Custom filtering logic works as expected
        - Validates the naming convention for scraped videos
        - Removing videos logic works as expected
        """
        scraper = BaseballSavVideoScraper.from_date_range('2024-05-01', '2024-05-02', 
                                                          max_return_videos=10, 
                                                          download_folder=str(tmp_path_factory.mktemp('savant_videos')))

        assert os.path.exists(scraper.download_folder), 'A download folder should be created'
        assert isinstance(scraper.play_ids_df, pd.DataFrame), 'The play IDs should be a pandas DataFrame'
        assert not scraper.play_ids_df.empty, 'The DataFrame shouldn\'t be empty'

        df = scraper.play_ids_df
        first_len = len(df)

        df = df[df['release_speed'] >= 90] # Do a test filter
        second_len = len(df)

        assert first_len > second_len, 'A filter should decrease the size of the DataFrame'

        scraper.play_ids_df = df

        scraper.run_executor()

        self.validate_video_names(scraper)

        scraper.cleanup_savant_videos()

        assert not os.path.exists(scraper.download_folder), '`cleanup_savant_videos` should remove the folder'

    # Network test
    def test_game_pks(self, tmp_path_factory):
        """
        Tests the `from_game_pks` class alternate constructor.

        Args:
            tmp_path_factory: Fixture for a temp directory to load videos

        This tests for the following:
        - A folder was created
        - A pandas DataFrame is created and not empty
        - The number of returned videos is the same as the input
        """
        dummy_game = [{776990: {'home_team': 'KC', 'away_team': 'CLE'}}]

        scraper = BaseballSavVideoScraper.from_game_pks(dummy_game, 
                                                        download_folder=str(tmp_path_factory.mktemp('savant_videos')), 
                                                        max_return_videos=5)

        assert os.path.exists(scraper.download_folder), 'A download folder should be created'
        assert isinstance(scraper.play_ids_df, pd.DataFrame), 'The play IDs should be a pandas DataFrame'
        assert not scraper.play_ids_df.empty, 'The DataFrame shouldn\'t be empty'

        scraper.run_executor()

        assert len(os.listdir(scraper.download_folder)) == 5, '5 videos should be written'

        self.validate_video_names(scraper)
        scraper.cleanup_savant_videos()


    def test_network_error(self):
        """
        Tests the `request_with_retry` function.

        Args:
            test_crawler: Fixture of the instantiated `TestCrawler` class.

        The following are tested for in this test case:
        - The function is retrying the connection 3 times and is successful on the third attempt.
        """
        with patch('requests.get', side_effect=[requests.exceptions.RequestException("Temporary network error"), 
                                                requests.exceptions.RequestException("Temporary network error"),
                                                Mock(status_code=200)]) as mock_get:
            response = requests_with_retry('https://example.com/video_url')
            assert response.status_code == 200, "The 3rd request should be successful."
            assert mock_get.call_count == 3, "Mock get should be called 3 times."

    # Mini network test
    def test_team_player_filter(self):
        """
        Tests the `_get_team` function.

        This tests the following (assume same season):
        - The correct team is returned for a given player
        - A no data exception is thrown for a given player that's not on a given team
        - ValueError is raised for not recognized team and player
        """
        with open('tests/data/test_functions/savant_json_ex/success_game.json', 'r') as f:
            mock_json_data = json.load(f)

        assert _get_team(team=None, player=608070, season=2025) == 'CLE', 'The team returned for given player should be Cleveland'

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_json_data

        with patch('requests.get', return_value=mock_response), \
            patch('baseballcv.functions.utils.savant_utils.gameday.thread_game_dates', return_value=[{12345: {'home_team': '', 'away_team': ''}}]):
            
            with pytest.raises(pl.exceptions.NoDataError):
                # Should raise since Lane Thomas didn't play for the Guardians until after the trade deadline
                BaseballSavVideoScraper.from_date_range('2024-04-08', '2024-04-10', team_abbr='CLE', player=657041, pitch_type='FF')

        with pytest.raises(ValueError):
            _get_team(team='MAN', player=None, season=2025)
            _get_team(player=12345, team=None, season=2025)
    def test_from_play_ids(self, tmp_path):
        """
        Tests the `from_play_ids` class alternate constructor.

        This tests the following:
        - Play IDs are passed through to the DataFrame untouched
        - A mismatched `game_pks` list raises a ValueError
        """
        scraper = BaseballSavVideoScraper.from_play_ids(['abc-123', 'def-456'],
                                                        download_folder=str(tmp_path))

        assert list(scraper.play_ids_df['play_id']) == ['abc-123', 'def-456'], 'The play IDs should match the input'
        assert scraper.play_ids_df['game_pk'].isna().all(), 'game_pk should be empty when not provided'

        scraper = BaseballSavVideoScraper.from_play_ids(['abc-123'], game_pks=[776990],
                                                        download_folder=str(tmp_path))
        assert list(scraper.play_ids_df['game_pk']) == [776990], 'The game pks should match the input'

        with pytest.raises(ValueError):
            BaseballSavVideoScraper.from_play_ids(['abc-123'], game_pks=[1, 2], download_folder=str(tmp_path))

    def test_download_video_session(self, tmp_path):
        """
        Regression test for the sporty-clips hotlink protection (#180).

        The video page and the mp4 must be fetched through the same session (cookies +
        Referer), otherwise the mp4 host rejects the request with a 403.
        """
        page_html = ('<html><body><div class="video-box"><video>'
                     '<source src="https://sporty-clips.mlb.com/fake.mp4" type="video/mp4">'
                     '</video></div></body></html>')
        page_response = Mock(status_code=200, content=page_html)
        clip_response = Mock(status_code=200)
        clip_response.iter_content.return_value = [b'fake', b'bytes']

        session = MagicMock()
        session.get.side_effect = [page_response, clip_response]

        # The scraper's session is created in `__init__`, so the patch must wrap construction
        with patch('requests.Session', return_value=session):
            scraper = BaseballSavVideoScraper.from_play_ids(['abc-123'], game_pks=[776990],
                                                            download_folder=str(tmp_path))
            scraper._download_video(776990, 'abc-123')

        assert session.get.call_count == 2, 'Both the page and the mp4 should be requested through the session'

        page_url = session.get.call_args_list[0].args[0]
        clip_url = session.get.call_args_list[1].args[0]
        assert 'sporty-videos?playId=abc-123' in page_url, 'The first request should be the video page'
        assert clip_url == 'https://sporty-clips.mlb.com/fake.mp4', 'The second request should be the mp4 URL'

        assert os.path.exists(os.path.join(str(tmp_path), '776990_abc-123.mp4')), 'The video should be written'

    def test_download_video_no_game_pk(self, tmp_path):
        """
        Videos scraped without a `game_pk` (via `from_play_ids`) should be named by play ID only.
        """
        page_html = '<html><body><video><source src="https://sporty-clips.mlb.com/fake.mp4" type="video/mp4"></video></body></html>'
        page_response = Mock(status_code=200, content=page_html)
        clip_response = Mock(status_code=200)
        clip_response.iter_content.return_value = [b'fake']

        session = MagicMock()
        session.get.side_effect = [page_response, clip_response]

        with patch('requests.Session', return_value=session):
            scraper = BaseballSavVideoScraper.from_play_ids(['abc-123'], download_folder=str(tmp_path))
            scraper._download_video(None, 'abc-123')

        # Fallback path: no video-box container, source found directly
        assert os.path.exists(os.path.join(str(tmp_path), 'abc-123.mp4')), 'The video should be named by play ID only'
