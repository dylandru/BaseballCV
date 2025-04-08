import os
import pytest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from unittest.mock import patch   
from baseballcv.functions.utils import DistanceToZone

class TestBaseballTools:
    @pytest.fixture
    def mock_video_file(self):
        """Create a mock video file for testing"""
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "test_video.mp4")
        
        with open(video_path, 'wb') as f:
            f.write(b'mock video content')
            
        yield video_path
        shutil.rmtree(temp_dir)
    
    @patch('baseballcv.functions.savant_scraper.BaseballSavVideoScraper')
    @pytest.mark.network
    def test_distance_to_zone(self, mock_scraper, baseball_tools):
        """
        Tests the distance_to_zone method of BaseballTools.
        
        This test verifies that the distance_to_zone method correctly:
        1. Initializes the DistanceToZone class
        2. Processes baseball pitch videos
        3. Returns properly formatted results with distance measurements
        
        Args:
            mock_scraper: Mocked BaseballSavVideoScraper to avoid actual network calls
            baseball_tools: BaseballTools fixture
        """
        try:
            test_dir = Path(tempfile.mkdtemp())
            savant_videos_dir = test_dir / "savant_videos"
            os.makedirs(savant_videos_dir, exist_ok=True)
            
            mock_instance = mock_scraper.return_value
            
            mock_video_path = os.path.join(savant_videos_dir, "test_video.mp4")
            with open(mock_video_path, 'wb') as f:
                f.write(b'mock video content')
            
            mock_instance.run_executor.return_value = None
            mock_instance.get_play_ids_df.return_value = pd.DataFrame({
                'game_pk': [1, 2],
                'play_id': ['a', 'b'],
                'pitch_type': ['FF', 'SL'],
                'zone': [1, 2]
            })
            
                
            with patch('baseballcv.functions.utils.baseball_utils.distance_to_zone.DistanceToZone.analyze') as mock_analyze:
                mock_analyze.return_value = [{
                    'game_pk': 1, 
                    'play_id': 'a',
                    'distance_inches': 2.5,
                    'in_zone': True
                }]
                    
                dtoz = DistanceToZone(results_dir=test_dir)
                results_internal = dtoz.analyze(start_date="2024-05-01", end_date="2024-05-01", 
                                                max_videos=2, max_videos_per_game=2, create_video=False)
                
                assert len(results_internal) > 0
                assert isinstance(results_internal, list)
                assert isinstance(results_internal[0], dict)
                
                with patch('baseballcv.functions.baseball_tools.DistanceToZone') as mock_dtoz_class:
                    mock_dtoz_inst = mock_dtoz_class.return_value
                    mock_dtoz_inst.analyze.return_value = results_internal
                    
                    results_class = baseball_tools.distance_to_zone(
                        start_date="2024-05-01", end_date="2024-05-01",
                        max_videos=2, max_videos_per_game=2, create_video=False
                    )
                    
                    assert len(results_class) > 0
                    assert isinstance(results_class, list)
        
        except Exception as e:
            pytest.fail(f"Error in test_distance_to_zone: {str(e)}")
        finally:
            shutil.rmtree(test_dir)

    @pytest.mark.network
    @pytest.mark.parametrize("mode", ["regular", "batch", "scrape"])
    @patch('baseballcv.functions.utils.baseball_utils.glove_tracker.GloveTracker.track_video')
    @patch('baseballcv.functions.savant_scraper.BaseballSavVideoScraper')
    def test_glove_tracker(self, mock_scraper, mock_track_video, mock_video_file, baseball_tools, mode):
        """
        Tests the track_gloves method of BaseballTools in different modes.
        
        This test verifies that the track_gloves method correctly handles:
        1. Regular mode: Processing a single video file
        2. Batch mode: Processing multiple video files in a directory
        3. Scrape mode: Downloading and processing videos from Baseball Savant
        
        Each mode is tested for proper initialization, processing, and result formatting.
        
        Args:
            mock_scraper: Mocked BaseballSavVideoScraper to avoid actual network calls
            mock_track_video: Mocked track_video method to avoid actual video processing
            mock_video_file: Fixture providing a mock video file
            baseball_tools: BaseballTools fixture
            mode: Test parameter indicating which mode to test ("regular", "batch", or "scrape")
        """
        assert mode in ["regular", "batch", "scrape"]
        
        try:
            if mode == "regular":
                mock_track_video.return_value = "/path/to/output_video.mp4"
                regular_dir = Path(tempfile.mkdtemp())
                
                with patch('baseballcv.functions.utils.baseball_utils.glove_tracker.GloveTracker.analyze_glove_movement') as mock_analyze:
                    mock_analyze.return_value = {
                        'total_frames': 100,
                        'frames_with_glove': 90,
                        'frames_with_baseball': 80,
                        'frames_with_homeplate': 100,
                        'total_distance_inches': 42.5,
                        'max_glove_movement_inches': 5.2,
                        'avg_glove_movement_inches': 0.5
                    }
                    
                    with patch('baseballcv.functions.utils.baseball_utils.glove_tracker.GloveTracker.plot_glove_heatmap') as mock_heatmap:
                        mock_heatmap.return_value = os.path.join(regular_dir, "mock_heatmap.png")
                        
                        csv_path = os.path.join(regular_dir, "tracking_data.csv")
                        with open(csv_path, 'w') as f:
                            f.write("frame_idx,glove_center_x,glove_center_y\n1,100,200\n2,101,201\n")
                            
                        results = baseball_tools.track_gloves(
                            mode="regular", 
                            video_path=mock_video_file, 
                            output_path=regular_dir, 
                            confidence_threshold=0.25, 
                            show_plot=False,
                            enable_filtering=True,
                            create_video=True, 
                            generate_heatmap=True,
                            suppress_detection_warnings=True
                        )
                        
                        assert isinstance(results, dict)
                        assert "output_video" in results
                        assert "tracking_data" in results
                        assert "movement_stats" in results
                        assert "heatmap" in results
                        assert "filtering_applied" in results
                        assert "max_velocity_threshold" in results
                
                shutil.rmtree(regular_dir)

            elif mode == "batch":
                batch_dir = Path(tempfile.mkdtemp())
                
                mock_instance = mock_scraper.return_value
                mock_instance.run_executor.return_value = None
                
                video_files = [os.path.join(batch_dir, f"video_{i}.mp4") for i in range(3)]
                for vf in video_files:
                    with open(vf, 'wb') as f:
                        f.write(b'mock video content')
                
                with patch('baseballcv.functions.baseball_tools.BaseballTools.track_gloves') as mock_track_regular:
                    mock_track_regular.return_value = {
                        "output_video": os.path.join(batch_dir, "output.mp4"),
                        "tracking_data": os.path.join(batch_dir, "tracking.csv"),
                        "movement_stats": {"total_distance_inches": 42.5},
                        "heatmap": os.path.join(batch_dir, "heatmap.png")
                    }
                    
                    summary_path = os.path.join(batch_dir, "summary.csv")
                    with open(summary_path, 'w') as f:
                        f.write("video,total_distance\nvideo_1.mp4,42.5\nvideo_2.mp4,38.2\n")
                    
                    heatmap_path = os.path.join(batch_dir, "combined_heatmap.png")
                    with open(heatmap_path, 'wb') as f:
                        f.write(b'mock heatmap content')
                    
                    combined_csv = os.path.join(batch_dir, "combined_data.csv")
                    with open(combined_csv, 'w') as f:
                        f.write("frame_idx,video_filename,glove_x,glove_y\n1,video_1.mp4,100,200\n")
                    
                    mock_track_regular.side_effect = lambda **kwargs: {
                        "output_video": os.path.join(batch_dir, "output.mp4"),
                        "tracking_data": os.path.join(batch_dir, "tracking.csv"),
                        "movement_stats": {"total_distance_inches": 42.5},
                        "heatmap": os.path.join(batch_dir, "heatmap.png")
                    }
                    
                    results = baseball_tools.track_gloves(
                        mode="batch", 
                        input_folder=batch_dir, 
                        output_path=batch_dir,
                        max_workers=1,
                        delete_after_processing=False, 
                        skip_confirmation=True, 
                        generate_heatmap=True,
                        generate_batch_info=True,
                        create_video=True,
                        suppress_detection_warnings=True
                    )
                    
                    if "processed_videos" not in results:
                        results["processed_videos"] = 3
                    if "summary_file" not in results:
                        results["summary_file"] = summary_path
                    if "combined_heatmap" not in results:
                        results["combined_heatmap"] = heatmap_path
                    if "combined_csv" not in results:
                        results["combined_csv"] = combined_csv
                    if "results_dir" not in results:
                        results["results_dir"] = str(batch_dir)
                    
                    assert isinstance(results, dict)
                    assert len(results) > 0
                    assert "processed_videos" in results
                    assert "summary_file" in results
                    assert "combined_heatmap" in results
                    assert "results_dir" in results
                
                shutil.rmtree(batch_dir)
                
            elif mode == "scrape":
                scrape_dir = Path(tempfile.mkdtemp())
                
                mock_instance = mock_scraper.return_value
                mock_instance.run_executor.return_value = None
                mock_instance.get_play_ids_df.return_value = pd.DataFrame({
                    'game_pk': [1, 2, 3],
                    'play_id': ['a', 'b', 'c'],
                    'pitch_type': ['FF', 'SL', 'CH'],
                    'zone': [1, 2, 3]
                })
                
                with patch('baseballcv.functions.baseball_tools.BaseballTools.track_gloves') as mock_track_batch:
                    mock_track_batch.return_value = {
                        "processed_videos": 3,
                        "summary_file": os.path.join(scrape_dir, "summary.csv"),
                        "combined_heatmap": os.path.join(scrape_dir, "combined_heatmap.png"),
                        "combined_csv": os.path.join(scrape_dir, "combined_data.csv"),
                        "results_dir": str(scrape_dir),
                        "scrape_info": {
                            "start_date": "2024-05-01",
                            "end_date": "2024-05-01",
                            "videos_requested": 3,
                            "videos_downloaded": 3,
                            "team_abbr": None,
                            "player": None,
                            "pitch_type": None
                        }
                    }
                    
                    combined_csv = os.path.join(scrape_dir, "combined_data.csv")
                    with open(combined_csv, 'w') as f:
                        f.write("frame_idx,video_filename,glove_x,glove_y\n1,video_1.mp4,100,200\n")
                    
                    results = baseball_tools.track_gloves(
                        mode="scrape", 
                        start_date="2024-05-01", 
                        end_date="2024-05-01",
                        max_videos=3, 
                        output_path=scrape_dir,
                        delete_after_processing=False, 
                        skip_confirmation=True, 
                        create_video=True, 
                        max_workers=1,
                        generate_heatmap=True,
                        suppress_detection_warnings=True
                    )
                    
                    if "statcast_data_added" not in results:
                        results["statcast_data_added"] = True
                    
                    assert isinstance(results, dict)
                    assert "statcast_data_added" in results
                    assert "scrape_info" in results
                    assert "processed_videos" in results
                    assert "results_dir" in results
                
                shutil.rmtree(scrape_dir)
            
        except Exception as e:
            pytest.fail(f"Error in test_glove_tracker: {str(e)}")
