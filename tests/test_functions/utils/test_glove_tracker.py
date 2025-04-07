import pytest
import os
import shutil
import tempfile
import cv2
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from baseballcv.functions.utils.glove_tracker import GloveTracker
from baseballcv.utilities import BaseballCVLogger

# Mock YOLO results for testing _process_detections
class MockYOLOResult:
    def __init__(self, boxes_data):
        self.boxes = MockBoxes(boxes_data)

class MockBoxes:
    def __init__(self, data):
        self.data = data
        self.cls = [torch.tensor([d[4]]) for d in data]
        self.conf = [torch.tensor([d[5]]) for d in data]
        self.xyxy = [torch.tensor([d[:4]]) for d in data]

# Mock torch tensor for compatibility
class MockTensor:
    def __init__(self, data):
        self._data = data

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self._data)

    def item(self):
        return self._data

    def tolist(self):
         return self._data

# Minimal mocking for torch tensors if torch is not available/needed for simple tests
try:
    import torch
except ImportError:
    torch = MagicMock()
    torch.tensor = MockTensor


@pytest.fixture(scope="module")
def glove_tracker_instance(tmp_path_factory):
    """Provides a GloveTracker instance for testing."""
    # Mock LoadTools to prevent actual model loading
    with patch('baseballcv.functions.load_tools.LoadTools') as MockLoadTools:
        mock_load_tools = MockLoadTools.return_value
        mock_load_tools.load_model.return_value = "mock_model_path.pt"

        # Mock YOLO model
        with patch('ultralytics.YOLO') as MockYOLO:
            mock_yolo_instance = MockYOLO.return_value
            # Define mock model names (adjust class IDs based on actual model if needed)
            mock_yolo_instance.names = {0: 'glove', 1: 'homeplate', 2: 'baseball'}

            results_dir = tmp_path_factory.mktemp("glove_results")
            tracker = GloveTracker(
                results_dir=str(results_dir),
                logger=BaseballCVLogger.get_logger("TestGloveTracker"),
                device='cpu' # Force CPU for testing without GPU
            )
            # Manually set class IDs based on mock names
            tracker.glove_class_id = 0
            tracker.homeplate_class_id = 1
            tracker.baseball_class_id = 2
            return tracker

@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    """Creates a short dummy video file for testing."""
    video_dir = tmp_path_factory.mktemp("test_videos")
    video_path = video_dir / "test_video.mp4"
    frame_width, frame_height = 640, 480
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_width, frame_height))

    for i in range(60): # 2 seconds of video
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    out.release()
    return str(video_path)

@pytest.fixture
def sample_tracking_data():
    """Provides sample tracking data for testing analysis functions."""
    data = {
        'frame_idx': [0, 1, 2, 3, 4, 5],
        'homeplate_center_x': [320, 320, 320, None, 320, 320],
        'homeplate_center_y': [400, 400, 400, None, 400, 400],
        'homeplate_width': [100, 100, 100, None, 100, 100],
        'glove_center_x': [300, 305, 310, 315, 320, 325],
        'glove_center_y': [350, 348, 345, 342, 340, 338],
        'glove_real_x': [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5], # Assuming pixels_per_inch = 10
        'glove_real_y': [5.0, 5.2, 5.5, 5.8, 6.0, 6.2],   # Assuming pixels_per_inch = 10
        'pixels_per_inch': [10.0] * 6
    }
    return pd.DataFrame(data)

# --- Test Cases ---

def test_glove_tracker_initialization(glove_tracker_instance):
    """Test if the GloveTracker initializes correctly."""
    assert glove_tracker_instance is not None
    assert glove_tracker_instance.glove_class_id == 0
    assert glove_tracker_instance.homeplate_class_id == 1
    assert glove_tracker_instance.baseball_class_id == 2
    assert os.path.exists(glove_tracker_instance.results_dir)

def test_process_detections_basic(glove_tracker_instance):
    """Test basic processing of mock YOLO results."""
    # Mock results for a single frame
    mock_boxes_data = [
        ([100, 100, 150, 150, 0, 0.9]), # glove
        ([300, 380, 400, 420, 1, 0.8]), # homeplate
        ([200, 200, 220, 220, 2, 0.7])  # baseball
    ]
    mock_yolo_results = [MockYOLOResult(mock_boxes_data)]
    frame = np.zeros((480, 640, 3), dtype=np.uint8) # Dummy frame
    frame_idx = 10
    fps = 30

    detections = glove_tracker_instance._process_detections(mock_yolo_results, frame, frame_idx, fps)

    assert detections['frame_idx'] == frame_idx
    assert detections['glove'] is not None
    assert detections['glove']['confidence'] == 0.9
    assert detections['homeplate'] is not None
    assert detections['homeplate']['width'] == 100
    assert detections['baseball'] is not None

    # Test real-world coordinates (should be calculated now that home plate is set)
    assert 'glove' in detections['real_world_coords']
    assert 'baseball' in detections['real_world_coords']
    assert glove_tracker_instance.pixels_per_inch is not None

def test_process_detections_no_homeplate(glove_tracker_instance):
    """Test processing when home plate is not detected initially."""
    glove_tracker_instance.home_plate_reference = None # Reset
    glove_tracker_instance.pixels_per_inch = None
    mock_boxes_data = [
        ([100, 100, 150, 150, 0, 0.9]) # glove only
    ]
    mock_yolo_results = [MockYOLOResult(mock_boxes_data)]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_idx = 0

    detections = glove_tracker_instance._process_detections(mock_yolo_results, frame, frame_idx)

    assert detections['glove'] is not None
    assert detections['homeplate'] is None
    assert 'glove' not in detections['real_world_coords'] # No real coords without reference
    assert glove_tracker_instance.pixels_per_inch is None

def test_filter_glove_detection(glove_tracker_instance):
    """Test the outlier filtering logic."""
    glove_tracker_instance.pixels_per_inch = 10.0 # Set pixels per inch for calculation
    fps = 30.0
    max_vel = 120.0 # inches per second

    prev_detection = {
        'frame_idx': 0,
        'real_world_coords': {'glove': (0.0, 0.0)}
    }
    # Plausible movement (e.g., 5 inches in 1/30 sec = 150 in/sec - slightly above threshold)
    current_detection_implausible = {
        'frame_idx': 1,
        'real_world_coords': {'glove': (5.0, 0.0)} # 5 inches away
    }
    # Plausible movement (e.g., 2 inches in 1/30 sec = 60 in/sec)
    current_detection_plausible = {
        'frame_idx': 1,
        'real_world_coords': {'glove': (2.0, 0.0)} # 2 inches away
    }

    # Test implausible movement (should be filtered out)
    assert not glove_tracker_instance._filter_glove_detection(
        prev_detection, current_detection_implausible, fps, max_vel
    )

    # Test plausible movement (should pass)
    assert glove_tracker_instance._filter_glove_detection(
        prev_detection, current_detection_plausible, fps, max_vel
    )

    # Test first frame (no previous)
    assert glove_tracker_instance._filter_glove_detection(
        None, current_detection_plausible, fps, max_vel
    )

def test_fill_missing_detections(glove_tracker_instance):
    """Test filling missing glove detections."""
    glove_tracker_instance.tracking_data = [
        {'frame_idx': 0, 'real_world_coords': {'glove': (1.0, 1.0)}, 'glove': True},
        {'frame_idx': 1, 'real_world_coords': {}, 'glove': None}, # Missing
        {'frame_idx': 2, 'real_world_coords': {'glove': (3.0, 3.0)}, 'glove': True},
        {'frame_idx': 3, 'real_world_coords': {}, 'glove': None}, # Missing
        {'frame_idx': 4, 'real_world_coords': {}, 'glove': None}, # Missing
        {'frame_idx': 5, 'real_world_coords': {'glove': (6.0, 6.0)}, 'glove': True}
    ]
    # Expected: Fill frame 1 with (1,1), frame 3 & 4 with (3,3)
    filled_x, filled_y = glove_tracker_instance._fill_missing_detections()

    expected_x = [1.0, 3.0, 6.0]  # Should only return points from valid sequences
    expected_y = [1.0, 3.0, 6.0]

    assert filled_x == pytest.approx(expected_x)
    assert filled_y == pytest.approx(expected_y)

def test_handle_missing_detections(glove_tracker_instance):
    """Test enhanced handling of missing detections with interpolation."""
    glove_tracker_instance.tracking_data = [
        {'frame_idx': 0, 'real_world_coords': {'glove': (0, 0)}, 'glove': True},
        {'frame_idx': 1, 'real_world_coords': {}, 'glove': None}, # Gap of 1
        {'frame_idx': 2, 'real_world_coords': {'glove': (2, 2)}, 'glove': True},
        {'frame_idx': 3, 'real_world_coords': {}, 'glove': None}, # Gap of 2
        {'frame_idx': 4, 'real_world_coords': {}, 'glove': None},
        {'frame_idx': 5, 'real_world_coords': {'glove': (5, 5)}, 'glove': True},
        {'frame_idx': 10, 'real_world_coords': {'glove': (10, 10)}, 'glove': True}, # Large gap, new sequence
    ]

    filled_x, filled_y = glove_tracker_instance._handle_missing_detections()

    # Sequence 1: Frames 0-5 (includes interpolated frames 1, 3, 4)
    # Frame 1: Interpolated between (0,0) at frame 0 and (2,2) at frame 2 -> (1,1)
    # Frame 3: Interpolated between (2,2) at frame 2 and (5,5) at frame 5 -> (3,3)
    # Frame 4: Interpolated between (2,2) at frame 2 and (5,5) at frame 5 -> (4,4)
    expected_x_seq1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    expected_y_seq1 = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    # Sequence 2: Frame 10 (only one point, not enough for sequence, should be ignored by plotting)

    # The _handle_missing_detections focuses on returning valid sequences.
    # Since sequence 2 only has one point, it will return only seq 1.
    assert filled_x == pytest.approx(expected_x_seq1)
    assert filled_y == pytest.approx(expected_y_seq1)


@patch('ultralytics.YOLO.predict')
def test_track_video(mock_predict, glove_tracker_instance, sample_video, tmp_path):
    """Test the main video tracking function."""
    # Mock the predict method to return consistent results
    # For simplicity, let's mock a static glove and homeplate
    glove_box = [100, 100, 150, 150, 0, 0.9] # class 0 (glove)
    hp_box = [300, 380, 400, 420, 1, 0.8]    # class 1 (homeplate)
    mock_results = [MockYOLOResult([glove_box, hp_box])]
    mock_predict.return_value = mock_results

    output_video_path = tmp_path / "tracked_output.mp4"

    result_path = glove_tracker_instance.track_video(
        video_path=sample_video,
        output_path=str(output_video_path),
        show_plot=True,
        create_video=True,
        generate_heatmap=True
    )

    assert result_path == str(output_video_path)
    assert os.path.exists(result_path)
    # Check CSV creation
    csv_filename = os.path.splitext(os.path.basename(output_video_path))[0] + "_tracking.csv"
    csv_path = os.path.join(glove_tracker_instance.results_dir, csv_filename)
    assert os.path.exists(csv_path)
    # Check heatmap creation
    heatmap_filename = f"glove_heatmap_{os.path.splitext(os.path.basename(sample_video))[0]}.png"
    heatmap_path = os.path.join(glove_tracker_instance.results_dir, heatmap_filename)
    assert os.path.exists(heatmap_path)

def test_plot_glove_heatmap(glove_tracker_instance, sample_tracking_data, tmp_path):
    """Test heatmap generation from CSV."""
    csv_path = tmp_path / "sample_data.csv"
    sample_tracking_data.to_csv(csv_path, index=False)
    heatmap_output_path = tmp_path / "heatmap.png"

    result_path = glove_tracker_instance.plot_glove_heatmap(
        csv_path=str(csv_path),
        output_path=str(heatmap_output_path),
        generate_heatmap=True
    )

    assert result_path == str(heatmap_output_path)
    assert os.path.exists(result_path)

def test_analyze_glove_movement(glove_tracker_instance, sample_tracking_data, tmp_path):
    """Test movement analysis from CSV."""
    csv_path = tmp_path / "sample_data.csv"
    sample_tracking_data.to_csv(csv_path, index=False)

    stats = glove_tracker_instance.analyze_glove_movement(str(csv_path))

    assert stats is not None
    assert 'total_distance_inches' in stats
    assert 'max_distance_between_frames_inches' in stats
    assert stats['total_frames'] == 6
    assert stats['frames_with_glove'] == 6