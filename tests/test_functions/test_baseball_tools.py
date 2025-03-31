import os
import pytest
from baseballcv.functions.utils import DistanceToZone

@pytest.mark.network
def test_distance_to_zone(baseball_tools):
    """
    Tests the distance_to_zone method using example call.
    """
    results_dir = "results"

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
