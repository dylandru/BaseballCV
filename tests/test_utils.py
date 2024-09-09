import os
import pytest
from scripts.function_utils.utils import extract_frames_from_video, model_aliases

@pytest.fixture
def video_path(tmp_path):
    """
    Fixture to create a mock video file path using a temporary directory.
    
    Returns:
        str: The path to the mock video file.
    """
    # Mock a video file creation here using OpenCV or a temporary file
    video_file = tmp_path / "test_video.mp4"
    # Create a sample video file for testing
    # (This would involve creating a short video with cv2.VideoWriter, omitted here for brevity)
    return str(video_file)

def test_extract_frames_from_video(video_path, tmp_path):
    """
    Test that the `extract_frames_from_video` function extracts the correct number of frames
    and saves them to the specified directory.
    
    Args:
        video_path (str): Path to the mock video file created by the fixture.
        tmp_path (Path): Temporary directory path for storing the frames.
    """
    output_frames_folder = tmp_path / "frames"
    output_frames_folder.mkdir()

    frames = extract_frames_from_video(video_path, "test_game", str(output_frames_folder), 5)
    
    assert len(frames) == 5  # Ensure 5 frames were extracted
    assert all(os.path.exists(frame) for frame in frames)  # Ensure the frames were saved

def test_model_aliases():
    """
    Test that the `model_aliases` dictionary contains the correct keys and values.
    """
    assert "phc_detector" in model_aliases
    assert model_aliases["phc_detector"].endswith(".txt")
