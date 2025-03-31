import shutil
import os
import tempfile
import pytest
from unittest.mock import patch
import pandas as pd
from baseballcv.functions.dataset_tools import DataTools
from baseballcv.functions.load_tools import LoadTools

class TestDatasetTools:
    """ Test class for various Dataset Generation Tools """

    @pytest.fixture(scope="class")
    def setup(self):
        """ Sets up the environment for Dataset Tools"""
    
        temp_dir = tempfile.mkdtemp()
        temp_video_dir = tempfile.mkdtemp()

        return {'temp_dir': temp_dir, 'temp_video_dir': temp_video_dir}

    @pytest.fixture(scope="class")
    def clean(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, dict) and 'temp_dir' in attr:
                if os.path.exists(attr['temp_dir']):
                    shutil.rmtree(attr['temp_dir'])

    def test_data_tools_init(self):
        """
        Tests the instance generation of data tools, affirming the defaults of itself
        don't change.
        """

        tools = DataTools()

        assert isinstance(tools.LoadTools, LoadTools)
        assert tools.output_folder == ''

    @pytest.mark.network
    @pytest.mark.parametrize("use_supervision, value", [("use_supervision", False), ("use_supervision", True)])
    def test_generate_photo_dataset(self, data_tools, setup, use_supervision, value):
        """
        Tests the generate_photo_dataset method using example call.
        Keeps part of output for the following test_automated_annotation method.
        """
        temp_dir = setup['temp_dir']
        temp_video_dir = setup['temp_video_dir']

        data_tools.generate_photo_dataset(
            output_frames_folder=temp_dir,
            video_download_folder=temp_video_dir,
            max_plays=2,
            max_num_frames=10,
            max_videos_per_game=1,
            start_date="2024-04-01",
            end_date="2024-04-01",
            delete_savant_videos=True,
            use_supervision=value
        )

        assert os.path.exists(temp_dir)
        frames = os.listdir(temp_dir)
        assert len(frames) > 0, "Should generate some frames"

        for frame in frames:
            assert frame.endswith(('.jpg', '.png')), f'Invalid frame format {frame}'

    @pytest.mark.network
    @pytest.mark.parametrize("mode, value", [("mode", "autodistill"), ("mode", "legacy")])
    def test_automated_annotation(self, setup, data_tools, load_tools, mode, value):
        """
        Tests the annotation tools to make sure the proper file systems are loaded 
        and manipulated.
        """
        temp_dir = setup['temp_dir']
        temp_video_dir = setup['temp_video_dir']

        data_tools.generate_photo_dataset(
            output_frames_folder=temp_dir,
            video_download_folder=temp_video_dir,
            max_plays=2,
            max_num_frames=10,
            max_videos_per_game=1,
            start_date="2024-04-01",
            end_date="2024-04-01",
            delete_savant_videos=True
        )

        ontology = { "a mitt worn by a baseball player for catching a baseball": "glove" } if value == "autodistill" else None

        legacy_annotation_dir = tempfile.mkdtemp()
        autodistill_annotation_dir = tempfile.mkdtemp()
        annotation_dir = legacy_annotation_dir if value != "autodistill" else autodistill_annotation_dir

        with tempfile.TemporaryDirectory() as annotation_dir:
            data_tools.automated_annotation(
                model_alias="glove_tracking",
                model_type="detection",
                image_dir=temp_dir,
                output_dir=annotation_dir if value != "autodistill" else f"{annotation_dir}_autodistill",
                conf=0.5,
                mode=value,
                ontology=ontology
            )
            assert os.path.exists(annotation_dir)

            if value != "autodistill":
                assert os.path.exists(os.path.join(annotation_dir, "annotations"))
                images = os.listdir(annotation_dir)
                annotations = os.listdir(os.path.join(annotation_dir, "annotations"))
                assert len(images) > 0, "Should have copied some images"
                assert len(annotations) > 0, "Should have generated some annotations"
                
                for ann_file in annotations:
                    assert ann_file.endswith('.txt'), f"Invalid annotation format: {ann_file}"
                
                os.remove(load_tools.yolo_model_aliases['glove_tracking'].replace('.txt', '.pt')) # Remove the loaded pytorch file

            else:
                assert os.path.exists(os.path.join(f"{annotation_dir}_autodistill", "train", "labels")), "Should have labels"
                assert os.path.exists(os.path.join(f"{annotation_dir}_autodistill", "train", "images")), "Should have images"
                assert os.path.exists(os.path.join(f"{annotation_dir}_autodistill", "data.yaml")), "Should have data.yaml"
                images = os.listdir(os.path.join(f"{annotation_dir}_autodistill", "train", "images"))
                annotations = os.listdir(os.path.join(f"{annotation_dir}_autodistill", "train", "labels"))
                assert len(images) > 0, "Should have copied some images"
                assert len(annotations) > 0, "Should have generated some annotations"
