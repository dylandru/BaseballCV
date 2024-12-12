import shutil
import os

def test_generate_photo_dataset(data_tools):
    """
    Tests the generate_photo_dataset method using example call.
    Keeps part of output for the following test_automated_annotation method.
    """
    
    # Use small dates and limited plays for testing
    try:
        data_tools.generate_photo_dataset(
            output_frames_folder="test_dataset",
            video_download_folder="test_videos",
            max_plays=2,
            max_num_frames=10,
            max_videos_per_game=1,
            start_date="2024-03-28",
            end_date="2024-03-28",
            delete_savant_videos=True
        )
        
        assert os.path.exists("test_dataset")
        frames = os.listdir("test_dataset")
        assert len(frames) > 0, "Should have generated at least some frames"
        
        for frame in frames:
            assert frame.endswith(('.jpg', '.png')), f"Invalid frame format: {frame}"
    
    except Exception as e:
        print(f"An error occurred: {e}")
        raise
            

def test_automated_annotation(load_tools, data_tools):
    """
    Tests the automated_annotation method using example call.
    """
    test_generate_photo_dataset(data_tools)  # Reuse dataset generation
    
    try:
        data_tools.automated_annotation(
            model_alias="phc_detector",  
            model_type="detection",
            image_dir="test_dataset",
            output_dir="test_annotated",
            conf=0.5
        )
        
        assert os.path.exists("test_annotated")
        assert os.path.exists(os.path.join("test_annotated", "annotations"))
        
        images = os.listdir("test_annotated")
        annotations = os.listdir(os.path.join("test_annotated", "annotations"))
        assert len(images) > 0, "Should have copied some images"
        assert len(annotations) > 0, "Should have generated some annotations"
        
        for ann_file in annotations:
            assert ann_file.endswith('.txt'), f"Invalid annotation format: {ann_file}"
        
        os.remove(load_tools.yolo_model_aliases['phc_detector'].replace('.txt', '.pt'))
        shutil.rmtree("test_annotated")
        shutil.rmtree("test_dataset")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise
