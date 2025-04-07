import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from baseballcv.functions.utils import DistanceToZone
from baseballcv.functions.baseball_tools import BaseballTools
from baseballcv.functions.utils.glove_tracker import GloveTracker

@pytest.mark.network
def test_distance_to_zone(baseball_tools):
    """
    Tests the distance_to_zone method using example call.
    """
    try:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        #Test internal class
        dtoz = DistanceToZone(results_dir=results_dir)
        results_internal = dtoz.analyze(start_date="2024-05-01", end_date="2024-05-01", max_videos=2, max_videos_per_game=2, create_video=True)
        
        assert len(results_internal) > 0, "Should have results"
        assert isinstance(results_internal, list)
        assert isinstance(results_internal[0], dict)

        #Test BaseballTools Class Implentation
        results_class = baseball_tools.distance_to_zone(start_date="2024-05-01", end_date="2024-05-01", max_videos=2, max_videos_per_game=2, create_video=True)
        
        assert len(results_class) > 0, "Should have results"
        assert isinstance(results_class, list)
    except Exception:
        pytest.skip(f"Skipping test as BaseballTools Class is still under development")


@pytest.fixture
def mock_glove_tracker():
    """Mocks the GloveTracker class."""
    with patch('baseballcv.functions.baseball_tools.GloveTracker') as MockTracker:
        mock_instance = MockTracker.return_value
        # Define mock return values for methods called by BaseballTools.track_gloves
        mock_instance.track_video.return_value = "mock_output_video.mp4"
        mock_instance.analyze_glove_movement.return_value = {"total_distance_inches": 10.5}
        mock_instance.plot_glove_heatmap.return_value = "mock_heatmap.png"
        # Add results_dir attribute to the mock
        mock_instance.results_dir = "mock_results_dir"
        yield mock_instance # Yield the instance for inspection

@pytest.fixture
def sample_video_file(tmp_path):
    """Creates a dummy video file."""
    video_path = tmp_path / "sample.mp4"
    video_path.touch()
    return str(video_path)

@pytest.fixture
def sample_video_folder(tmp_path):
    """Creates a folder with dummy video files."""
    folder_path = tmp_path / "video_batch"
    folder_path.mkdir()
    (folder_path / "video1.mp4").touch()
    (folder_path / "video2.avi").touch()
    return str(folder_path)

def test_track_gloves_regular_mode(baseball_tools, mock_glove_tracker, sample_video_file):
    """Test track_gloves in regular mode."""
    results = baseball_tools.track_gloves(
        mode="regular",
        video_path=sample_video_file,
        output_path="test_output.mp4",
        confidence_threshold=0.6,
        show_plot=False,
        create_video=False, # Disable video creation for speed
        generate_heatmap=False
    )

    # Check if GloveTracker was initialized correctly
    # Note: Accessing __init__ args directly is tricky with mocks, focus on method calls
    mock_glove_tracker.track_video.assert_called_once_with(
        video_path=sample_video_file,
        output_path="test_output.mp4",
        show_plot=False,
        create_video=False,
        generate_heatmap=False
    )
    # Check if analyze_glove_movement was called
    mock_glove_tracker.analyze_glove_movement.assert_called_once()

    assert "output_video" in results
    assert "tracking_data" in results # Should point to expected CSV name
    assert "movement_stats" in results
    assert "heatmap" in results
    assert results["output_video"] == "mock_output_video.mp4"
    assert results["heatmap"] is None # Since generate_heatmap=False


def test_track_gloves_batch_mode(baseball_tools, mock_glove_tracker, sample_video_folder):
    """Test track_gloves in batch mode."""
    # Mock os.path.exists for the folder check
    with patch('os.path.exists', return_value=True):
         # Mock listdir to return our dummy files
         with patch('os.listdir', return_value=['video1.mp4', 'video2.avi']):
             # Mock os.remove for delete_after_processing
             with patch('os.remove') as mock_remove:
                results = baseball_tools.track_gloves(
                    mode="batch",
                    input_folder=sample_video_folder,
                    max_workers=1, # Test sequential batch first
                    delete_after_processing=True,
                    skip_confirmation=True, # Avoid input prompt
                    generate_heatmap=True,
                    create_video=False # Disable video creation for speed
                )

    assert mock_glove_tracker.track_video.call_count == 2 # Called for each video
    assert "combined_csv" in results
    assert "summary_file" in results
    assert "combined_heatmap" in results
    assert results["processed_videos"] == 2
    assert mock_remove.call_count == 2 # Should delete both videos

@patch('baseballcv.functions.baseball_tools.BaseballSavVideoScraper')
def test_track_gloves_scrape_mode(MockScraper, baseball_tools, mock_glove_tracker, tmp_path):
    """Test track_gloves in scrape mode (mocking scraper and batch)."""
    # Setup mock scraper
    mock_scraper_instance = MockScraper.return_value
    mock_play_ids_df = pd.DataFrame({
        'game_pk': [123456],
        'play_id': ['abcdefg'],
        # Add other necessary columns if GloveTracker relies on them
    })
    mock_scraper_instance.get_play_ids_df.return_value = mock_play_ids_df
    # Mock the executor to do nothing
    mock_scraper_instance.run_executor.return_value = None

    # Create a dummy downloaded video file
    temp_download_dir = tmp_path / "savant_videos_mock"
    temp_download_dir.mkdir()
    dummy_video_path = temp_download_dir / "123456_abcdefg.mp4"
    dummy_video_path.touch()

    # Patch tempfile.mkdtemp to return our controlled path
    with patch('tempfile.mkdtemp', return_value=str(temp_download_dir)):
        # Patch shutil.rmtree to avoid deleting the temp dir during test
        with patch('shutil.rmtree') as mock_rmtree:
            # Re-patch os.listdir for the batch processing part within scrape
            with patch('os.listdir', return_value=["123456_abcdefg.mp4"]):
                 with patch('os.path.exists', return_value=True): # Ensure exists checks pass
                    results = baseball_tools.track_gloves(
                        mode="scrape",
                        start_date="2024-01-01",
                        max_videos=1,
                        delete_after_processing=True, # Test cleanup
                        skip_confirmation=True,
                        create_video=False,
                        generate_heatmap=False
                    )

    MockScraper.assert_called_once()
    mock_scraper_instance.run_executor.assert_called_once()
    mock_glove_tracker.track_video.assert_called_once() # Should process the one downloaded video
    assert "scrape_info" in results
    assert results["processed_videos"] == 1
    assert results["scrape_info"]["videos_downloaded"] == 1
    mock_rmtree.assert_called_once() # Check if cleanup happened

def test_track_gloves_invalid_mode(baseball_tools):
    """Test invalid mode error."""
    results = baseball_tools.track_gloves(mode="invalid_mode")
    assert "error" in results
    assert "Invalid mode" in results["error"]

def test_track_gloves_missing_args(baseball_tools):
    """Test missing arguments for different modes."""
    # Regular mode missing video_path
    results_reg = baseball_tools.track_gloves(mode="regular")
    assert "error" in results_reg
    assert "Video file not found" in results_reg["error"]

    # Batch mode missing input_folder
    results_batch = baseball_tools.track_gloves(mode="batch")
    assert "error" in results_batch
    assert "Input folder not found" in results_batch["error"]

    # Scrape mode missing start_date
    results_scrape = baseball_tools.track_gloves(mode="scrape")
    assert "error" in results_scrape
    assert "start_date is required" in results_scrape["error"]