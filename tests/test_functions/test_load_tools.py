import os
import shutil
import pytest
import tempfile
from pathlib import Path

# TODO: Test for Hugging Face dataset.
class TestLoadTools:
    """ Test Class that affirms Load Tools works properly """

    @pytest.fixture(scope="class", autouse=True)
    def setup(self):
        """ Sets up the environment for Load Tools"""
    
        temp_dir = tempfile.mkdtemp()

        with open(f'{temp_dir}/data.txt', 'w') as f:
            f.write('https://example-url.com/download.zip')

        yield {
            'temp_dir': temp_dir,
            }
    
    @pytest.fixture(scope="class")
    def clean(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, dict) and 'temp_dir' in attr:
                if os.path.exists(attr['temp_dir']):
                    shutil.rmtree(attr['temp_dir'])


    def test_load_model(self, load_tools):
        """
        Tests loading a model using LoadTools with example call.
        Verifies that the model is downloaded correctly and the file is in the expected location.
        It also checks that the path ends with pt (pytorch) since this is an explicit call with a YOLO model.
        """

        model_path = load_tools.load_model(model_alias = "phc_detector", model_type="YOLO", use_bdl_api=True)
        assert os.path.exists(model_path), "Model file should exist after download"
        assert model_path.endswith('.pt'), "YOLO model should have .pt"
        
        if os.path.exists(model_path):
            os.remove(model_path)

    def test_load_dataset(self, load_tools):
        """
        Tests loading a dataset using LoadTools with example call.
        Verifies that the dataset is downloaded and extracted correctly.
        """
        # Test loading baseball dataset
        dataset_path = load_tools.load_dataset(dataset_alias="baseball", use_bdl_api=True)
        
        assert os.path.exists(dataset_path), "Dataset directory should exist"
        assert os.path.isdir(dataset_path), "Dataset path should be a directory"
        files = os.listdir(dataset_path)
        assert len(files) > 0, "Dataset should contain files"
        
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)

    def test_load_fake_txt(self, load_tools, setup):
        """
        Creates a Fake Txt File with a dummy link that opens and should return the txt file name.
        """
        temp_dir = setup['temp_dir']

        dataset_path = load_tools.load_dataset(dataset_alias="data",
                                                use_bdl_api=False,
                                                file_txt_path=f'{temp_dir}/data.txt')

        assert os.path.exists(temp_dir), "Should be a file"
        assert dataset_path == Path("data"), "Dataset txt File should be the same name as the test"

    fail_params = [
                ("data", {'dataset_alias': 'gogoguardians'}, ValueError),
                ("model", {'model_alias': 'gogoguardians'}, ValueError),
                ("model", {'model_alias': 'phc_detector', 'model_type': 'XGBoost'}, ValueError)
            ]
    @pytest.mark.parametrize("load_type, params, expectation", fail_params)
    def test_load_tools_fail_params(self, load_type, params, expectation, load_tools):
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