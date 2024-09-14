import os
import pytest
from scripts.load_tools import load_model, load_dataset
from unittest.mock import patch, mock_open

@patch("requests.get")
def test_load_model(mock_get, tmp_path):
    """
    Test that the `load_model` function downloads and saves the model file correctly.
    
    Args:
        mock_get (MagicMock): Mock object for the requests.get method.
        tmp_path (Path): Temporary directory path for storing the downloaded model.
    """
    mock_get.return_value.status_code = 200
    mock_get.return_value.iter_content = lambda chunk_size: [b'fake_model_weights']

    model_alias = "phc_detector"
    model_weights_path = load_model(model_alias)

    assert os.path.exists(model_weights_path)  # Ensure the model file was downloaded

@patch("requests.get")
@patch("zipfile.ZipFile.extractall")
def test_load_dataset(mock_extractall, mock_get, tmp_path):
    """
    Test that the `load_dataset` function downloads and extracts the dataset correctly.
    
    Args:
        mock_extractall (MagicMock): Mock object for the ZipFile.extractall method.
        mock_get (MagicMock): Mock object for the requests.get method.
        tmp_path (Path): Temporary directory path for storing the dataset.
    """
    mock_get.return_value.status_code = 200
    mock_get.return_value.iter_content = lambda chunk_size: [b'fake_zip_content']

    txt_file_path = tmp_path / "dataset.txt"
    with open(txt_file_path, "w") as f:
        f.write("http://fakeurl.com/dataset.zip")

    load_dataset(str(txt_file_path))

    mock_extractall.assert_called_once()  # Ensure the dataset was extracted
