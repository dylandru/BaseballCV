import os
import shutil
import pytest
from unittest.mock import patch

def test_load_model_yolo(load_tools):
    """
    Tests loading a model using LoadTools with example call.
    Verifies that the model is downloaded correctly and the file is in the expected location.
    It also checks that the path ends with pt (pytorch) since this is an explicit call with a YOLO model.
    """
    #os.makedirs('models/YOLO/pitcher_hitter_catcher_detector/model_weights', exist_ok=True) Don't think this part is necessary
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

def test_load_fake_txt(load_tools, capsys):
    """
    Creates a Fake Txt File with a dummy link that opens and should return the txt file name.
    """

    test_dir = 'test-load-download/'
    os.makedirs(test_dir, exist_ok=True)

    with open(f'{test_dir}/data.txt', 'w') as f:
        f.write('https://example-url.com/download.zip')

    dataset_path = load_tools.load_dataset(dataset_alias="data",
                                        use_bdl_api=False,
                                        file_txt_path=f'{test_dir}/data.txt')
    
    assert os.path.exists(test_dir), "Should be a file"
    assert dataset_path == "data", "Dataset txt File should be the same name as the test"

    cap = capsys.readouterr()

    assert cap.out == "Download failed. STATUS: 404\nDataset download failed.\n", "Should fail to unzip since it's a fake url"

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

@pytest.mark.parametrize("alias, model", [
    ('phc_detector', 'YOLO'), 
    ('ball_tracking', 'FLORENCE2'),
    ('paligemma2_ball_tracking', 'PALIGEMMA2'),
    ]
)
def test_load_model_types(load_tools, alias, model):
    """
    Test for the proper alias and model is returning true in the model type if statements.
    Note: This only focus on if statements, doesn't return the unzipped model.
    """

    with patch.object(load_tools, 'load_model') as mocklm:
        mocklm.return_value = "Success"

        model_path = load_tools.load_model(model_alias=alias, model_type=model)

        assert model_path == "Success", "Should return Success"
        mocklm.assert_called_once()

test_param_data = [
    ("data", {'dataset_alias': 'gogoguardians'}, ValueError),
    ("model", {'model_alias': 'gogoguardians'}, ValueError),
    ("model", {'model_alias': 'phc_detector', 'model_type': 'XGBoost'}, ValueError)
]
@pytest.mark.parametrize("load_type, params, expectation", test_param_data)
def test_load_tools_fail_params(load_type, params, expectation, load_tools):
    """
    Tests to make sure inputting the wrong parameter results in an error.
    This is iteratively checking for both the model and dataset parameters.
    """
    if load_type == "data":
        with pytest.raises(expectation):
            load_tools.load_dataset(**params)
    
    else:
        with pytest.raises(expectation):
            load_tools.load_model(**params)