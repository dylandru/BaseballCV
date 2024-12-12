import os
import shutil

def test_load_model(load_tools):
    """
    Tests loading a model using LoadTools with example call.
    Verifies that the model is downloaded correctly and the file is in the expected location.
    """
    
    # Test loading PHC Detector model
    model_path = load_tools.load_model(
        model_alias="phc_detector",
        model_type="YOLO",
        use_bdl_api=True
    )
    
    assert os.path.exists(model_path), "Model file should exist after download"
    assert model_path.endswith('.pt'), "YOLO model should have .pt"
    
    if os.path.exists(model_path):
        os.remove(model_path)

def test_load_dataset(load_tools):
    """
    Tests loading a dataset using LoadTools with example call.
    Verifies that the dataset is downloaded and extracted correctly.
    """ 
    
    # Test loading baseball dataset
    dataset_path = load_tools.load_dataset(
        dataset_alias="baseball",
        use_bdl_api=True
    )
    
    assert os.path.exists(dataset_path), "Dataset directory should exist"
    assert os.path.isdir(dataset_path), "Dataset path should be a directory"
    files = os.listdir(dataset_path)
    assert len(files) > 0, "Dataset should contain files"
    
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)