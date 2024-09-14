import pytest
from scripts.savant_scraper import BaseballSavVideoScraper
from unittest.mock import patch

@patch("scripts.savant_scraper.BaseballSavVideoScraper.get_video_url")
@patch("scripts.savant_scraper.BaseballSavVideoScraper.download_video")
def test_get_video_for_play_id(mock_download_video, mock_get_video_url):
    """
    Test that the `get_video_for_play_id` function correctly retrieves and downloads a video.
    
    Args:
        mock_download_video (MagicMock): Mock object for the download_video method.
        mock_get_video_url (MagicMock): Mock object for the get_video_url method.
    """
    scraper = BaseballSavVideoScraper()
    
    mock_get_video_url.return_value = "http://fakeurl.com/video.mp4"
    scraper.get_video_for_play_id("12345", "67890", "/fake/folder")

    mock_download_video.assert_called_once()  # Ensure download_video was called
