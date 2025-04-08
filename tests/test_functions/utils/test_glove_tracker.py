import pytest
import os
import shutil
import tempfile
import cv2
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from baseballcv.functions.utils import GloveTracker
from baseballcv.utilities import BaseballCVLogger

class MockTensor:
    """Mock PyTorch tensor for testing."""
    def __init__(self, value):
        self.value = value
    
    def item(self):
        return self.value
    
    def cpu(self):
        return self
    
    def numpy(self):
        return np.array(self.value)

class MockBox:
    """Mock YOLO box with proper tensor attributes."""
    def __init__(self, x1, y1, x2, y2, cls_id, conf):
        self.xyxy = [MockTensor([x1, y1, x2, y2])]
        self.cls = [MockTensor(cls_id)]
        self.conf = [MockTensor(conf)]

class MockBoxes:
    """Mock YOLO boxes that is properly iterable."""
    def __init__(self, box_data):
        self.boxes = []
        for data in box_data:
            x1, y1, x2, y2, cls_id, conf = data
            self.boxes.append(MockBox(x1, y1, x2, y2, cls_id, conf))
    
    def __iter__(self):
        return iter(self.boxes)

class MockResult:
    """Mock YOLO detection result."""
    def __init__(self, box_data):
        self.boxes = MockBoxes(box_data)

class TestGloveTracker:
    @pytest.fixture(scope="class")
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def glove_tracker(self, temp_dir):
        """Create a GloveTracker instance with mocked model loading."""
        with patch('baseballcv.functions.load_tools.LoadTools') as mock_load_tools:
            mock_instance = mock_load_tools.return_value
            mock_instance.load_model.return_value = "mock_model_path.pt"
            
            with patch('ultralytics.YOLO') as mock_yolo:
                mock_model = mock_yolo.return_value
                mock_model.names = {0: 'glove', 1: 'homeplate', 2: 'baseball'}
                
                tracker = GloveTracker(
                    results_dir=temp_dir,
                    device='cpu',
                    suppress_detection_warnings=True
                )
                
                # Manually set properties since we're mocking the model loading
                tracker.glove_class_id = 0
                tracker.homeplate_class_id = 1
                tracker.baseball_class_id = 2
                
                yield tracker
    
    @pytest.fixture
    def sample_video(self, temp_dir):
        """Create a small sample video for testing."""
        video_path = os.path.join(temp_dir, "test_video.mp4")
        
        # Create a simple video with a few frames
        width, height = 640, 480
        fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Create 10 frames
        for i in range(10):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            out.write(frame)
        
        out.release()
        yield video_path
    
    @pytest.fixture
    def sample_tracking_data(self, temp_dir):
        """Create sample tracking data for testing."""
        csv_path = os.path.join(temp_dir, "sample_tracking.csv")
        
        # Create DataFrame with all required columns
        data = {
            'frame_idx': range(6),
            'homeplate_center_x': [320, 320, 320, None, 320, 320],
            'homeplate_center_y': [400, 400, 400, None, 400, 400],
            'homeplate_width': [100, 100, 100, None, 100, 100],
            'glove_center_x': [300, 305, 310, 315, 320, 325],
            'glove_center_y': [350, 348, 345, 342, 340, 338],
            'glove_real_x': [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5],
            'glove_real_y': [5.0, 5.2, 5.5, 5.8, 6.0, 6.2],
            'baseball_center_x': [250, 255, None, None, 270, 275],
            'baseball_center_y': [300, 305, None, None, 320, 325],
            'baseball_real_x': [-7.0, -6.5, None, None, -5.0, -4.5],
            'baseball_real_y': [10.0, 10.5, None, None, 12.0, 12.5],
            'pixels_per_inch': [10.0] * 6
        }
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        yield csv_path
    
    def test_initialization(self, glove_tracker):
        """Test if the GloveTracker initializes correctly."""
        assert glove_tracker is not None
        assert glove_tracker.glove_class_id == 0
        assert glove_tracker.homeplate_class_id == 1
        assert glove_tracker.baseball_class_id == 2
        assert os.path.exists(glove_tracker.results_dir)
    
    def test_process_detections(self, glove_tracker):
        """Test the detection processing logic."""
        # Create mock detection results
        box_data = [
            [100, 100, 150, 150, 0, 0.9],  # glove
            [300, 380, 400, 420, 1, 0.8],  # homeplate
            [200, 200, 220, 220, 2, 0.7]   # baseball
        ]
        mock_results = [MockResult(box_data)]
        
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Process the detections
        detections = glove_tracker._process_detections(mock_results, frame, 0)
        
        # Check results
        assert detections['glove'] is not None
        assert detections['homeplate'] is not None
        assert detections['baseball'] is not None
        assert detections['glove']['confidence'] == 0.9
        assert detections['homeplate']['confidence'] == 0.8
        assert detections['baseball']['confidence'] == 0.7
    
    @patch('ultralytics.YOLO.predict')
    @patch('baseballcv.functions.utils.baseball_utils.glove_tracker.GloveTracker.plot_glove_heatmap')
    def test_track_video(self, mock_plot_heatmap, mock_predict, glove_tracker, sample_video, temp_dir):
        """Test the video tracking functionality with mock predictions."""
        # Setup mock prediction results
        box_data = [
            [100, 100, 150, 150, 0, 0.9],  # glove
            [300, 380, 400, 420, 1, 0.8]   # homeplate
        ]
        mock_predict.return_value = [MockResult(box_data)]
        
        # Mock the heatmap generation to avoid CSV file issues
        mock_plot_heatmap.return_value = os.path.join(temp_dir, "mock_heatmap.png")
        
        # Create a sample tracking data file to satisfy dependencies
        sample_data = {
            'frame_idx': range(10),
            'glove_center_x': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'glove_center_y': [200, 201, 202, 203, 204, 205, 206, 207, 208, 209],
            'homeplate_center_x': [300, 300, 300, 300, 300, 300, 300, 300, 300, 300],
            'homeplate_center_y': [400, 400, 400, 400, 400, 400, 400, 400, 400, 400],
            'baseball_center_x': [250, 251, 252, 253, 254, 255, 256, 257, 258, 259],
            'baseball_center_y': [350, 351, 352, 353, 354, 355, 356, 357, 358, 359]
        }
        csv_path = os.path.join(temp_dir, "output_video_tracking.csv")
        pd.DataFrame(sample_data).to_csv(csv_path, index=False)
        
        # Run tracking
        output_path = os.path.join(temp_dir, "output_video.mp4")
        result = glove_tracker.track_video(
            video_path=sample_video,
            output_path=output_path,
            show_plot=False,  # Disable plot to avoid figure rendering issues
            create_video=True,
            generate_heatmap=True
        )
        
        # Check results
        assert os.path.exists(result)
        assert mock_plot_heatmap.called
    
    def test_analyze_glove_movement(self, glove_tracker, sample_tracking_data):
        """Test the movement analysis functionality."""
        # Analyze the sample data
        stats = glove_tracker.analyze_glove_movement(sample_tracking_data)
        
        # Check results
        assert stats is not None
        assert 'total_frames' in stats
        assert 'frames_with_glove' in stats
        assert 'frames_with_baseball' in stats
        assert 'total_distance_inches' in stats
        assert stats['total_frames'] == 6
    
    def test_filter_detection(self, glove_tracker):
        """Test the detection filtering logic."""
        # Setup test data
        prev_detection = {
            'frame_idx': 0,
            'real_world_coords': {'glove': (0.0, 0.0)}
        }
        
        # Test with reasonable velocity
        current_valid = {
            'frame_idx': 1,
            'real_world_coords': {'glove': (1.0, 1.0)}  # ~42 in/sec at 30fps
        }
        
        # Test with unreasonable velocity
        current_invalid = {
            'frame_idx': 1,
            'real_world_coords': {'glove': (10.0, 10.0)}  # ~420 in/sec at 30fps
        }
        
        # Check filtering
        fps = 30.0
        assert glove_tracker._filter_glove_detection(prev_detection, current_valid, fps, 120.0)
        assert not glove_tracker._filter_glove_detection(prev_detection, current_invalid, fps, 120.0)
    
    def test_handle_missing_detections(self, glove_tracker):
        """Test the handling of missing detections."""
        # Setup test data with gaps
        glove_tracker.tracking_data = [
            {'frame_idx': 0, 'real_world_coords': {'glove': (0.0, 0.0)}, 'glove': {}},
            {'frame_idx': 1, 'real_world_coords': {}, 'glove': None},  # Missing
            {'frame_idx': 2, 'real_world_coords': {'glove': (2.0, 2.0)}, 'glove': {}},
            {'frame_idx': 5, 'real_world_coords': {'glove': (5.0, 5.0)}, 'glove': {}}  # Gap of 2 frames
        ]
        
        # Handle missing detections
        x_coords, y_coords = glove_tracker._handle_missing_detections()
        
        # Should interpolate the missing values
        assert len(x_coords) > 0
        assert len(y_coords) > 0