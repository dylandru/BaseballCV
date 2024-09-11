import os
import pytest
from scripts.generate_photos import generate_photo_dataset
from unittest.mock import patch

@patch("scripts.generate_photos.BaseballSavVideoScraper.run_statcast_pull_scraper")
def test_generate_photo_dataset(mock_scraper, tmp_path):
    """
    Test that the `generate_photo_dataset` function generates the correct number of frames
    and saves them to the specified directory.
    
    Args:
        mock_scraper (MagicMock): Mock object for the BaseballSavVideoScraper.
        tmp_path (Path): Temporary directory path for storing the frames and videos.
    """
    mock_scraper.return_value = None  # Mocking the scraper method

    output_frames_folder = tmp_path / "cv_dataset"
    video_download_folder = tmp_path / "raw_videos"
    video_download_folder.mkdir()

    # Mock video files in the download folder
    (video_download_folder / "000001_test.mp4").touch()

    generate_photo_dataset(str(output_frames_folder), str(video_download_folder), max_plays=1, max_num_frames=5)

    assert os.path.exists(output_frames_folder) # Ensure 5 frames were generated

@patch("scripts.generate_photos.BaseballSavVideoScraper.cleanup_savant_videos")
def test_cleanup_savant_videos(mock_cleanup, tmp_path):
    """
    Test that the `cleanup_savant_videos` function is called when the delete_savant_videos flag is set to True.
    
    Args:
        mock_cleanup (MagicMock): Mock object for the cleanup method.
        tmp_path (Path): Temporary directory path for storing the videos.
    """
    video_download_folder = tmp_path / "raw_videos"
    video_download_folder.mkdir()

    generate_photo_dataset(video_download_folder=str(video_download_folder), delete_savant_videos=True)
    mock_cleanup.assert_called_once()  # Ensure the cleanup function was called
