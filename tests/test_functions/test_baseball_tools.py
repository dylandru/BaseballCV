import os

def test_distance_to_zone(baseball_tools):
    """
    Tests the distance_to_zone method using example call.
    """
    results_dir = "results"
    dtoz = baseball_tools.distance_to_zone(results_dir=results_dir)
    results = dtoz.analyze(start_date="2024-05-01", end_date="2024-05-01", max_videos=2, max_videos_per_game=2, create_video=True)
    assert len(results) > 0, "Should have results"
    assert isinstance(results, list)
    assert isinstance(results[0], dict)

    assert os.path.exists(results_dir), "Results directory should exist"
    videos = os.listdir(results_dir)
    assert len(videos) > 0 and videos[0].endswith(".mp4"), "Should have downloaded videos"
