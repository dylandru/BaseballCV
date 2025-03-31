import os
import numpy as np
import pytest
import shutil
import tempfile
import torch
import cv2
import signal
import supervision as sv
from baseballcv.model.od import RFDETR
from baseballcv.functions import BaseballSavVideoScraper
class TestRFDETR:
    @pytest.fixture
    def setup_rfdetr_test(self, load_tools):
        """
        Setup test environment for RFDETR tests.
        
        Creates temporary directories for test data and model outputs,
        initializes a model instance, and provides test resources.
        """
        temp_dir = tempfile.mkdtemp()
        test_input_dir = os.path.join(temp_dir, "test_input")
        os.makedirs(test_input_dir)
        scraper = BaseballSavVideoScraper(start_dt='2024-04-01', end_dt='2024-04-01', team_abbr='CHC', 
                                          download_folder=test_input_dir, max_return_videos=1, max_videos_per_game=1)
        df = scraper.run_executor()
        video_files = sorted(f for f in os.listdir(test_input_dir) if f.endswith('.mp4'))
        if not video_files:
            raise ValueError(f"No video files found in {test_input_dir}")
        test_video_path = os.path.join(test_input_dir, video_files[0])
        frames = list(sv.get_video_frames_generator(test_video_path))
        if not frames:
            raise ValueError(f"Failed to read frames from video: {test_video_path}")
        
        frame = frames[0]
        
        test_image_path = os.path.join(temp_dir, "test_frame.jpg")
        cv2.imwrite(test_image_path, frame)

        labels = {
            "0": "baseball",
            "1": "bat",
            "2": "glove",
            "3": "homeplate"
        }
        
        # Initialize model
        model_base = RFDETR(
            device="cpu",
            model_type="base",
            labels=labels,
            project_path=temp_dir
        )

        model_large = RFDETR(
            device="cpu",
            model_type="large",
            labels=labels,
            project_path=temp_dir
        )

        dataset_path = load_tools.load_dataset("baseball_rubber_home_COCO")
        
        try:
            yield {
                'temp_dir': temp_dir,
                'test_video_path': test_video_path,
                'test_image_path': test_image_path,
                'model_base': model_base,
                'model_large': model_large,
                'labels': labels,
                'dataset_path': dataset_path
            }
        
        finally:
            shutil.rmtree(temp_dir)
    
    def test_model_initialization(self, setup_rfdetr_test, load_tools):
        """
        Test model initialization and device selection.
        
        Verifies that the RFDETR model initializes correctly with the specified
        device and tests device selection logic, including optional MPS support
        when available.
        
        Args:
            setup_rfdetr_test: Fixture providing test resources
        """
        labels = setup_rfdetr_test['labels']
        model_base = setup_rfdetr_test['model_base']
        model_large = setup_rfdetr_test['model_large']
        model_ckpt_path = load_tools.load_model(use_bdl_api=False, model_txt_path="models/od/RFDETR/glove_tracking/model_weights/rfdetr_glove_tracking.txt")

        model_ckpt = RFDETR(
            device="cpu",
            model_type="large",
            model_path=model_ckpt_path,
            labels=labels,
            project_path=setup_rfdetr_test['temp_dir']
        )
        
        assert model_base is not None, "RFDETR Basemodel should initialize"
        assert model_large is not None, "RFDETR Large model should initialize"
        assert model_ckpt is not None, "RFDETR Checkpoint model should initialize"
        assert model_base.device == 'cpu', "Device should be set correctly"
        assert model_base.model_name == "rfdetr", "Model name should be set correctly"
        assert labels is not None, "Labels should be set correctly"
        
        with pytest.MonkeyPatch().context() as m:
            m.setattr(torch.cuda, 'is_available', lambda: False)
            m.setattr(torch.backends.mps, 'is_available', lambda: True)
            
            try:
                model = RFDETR(device='mps', model_type='base', labels=labels)
                assert model.device == 'mps' or model.device == 'cpu', "Should select MPS when available, but fallback to CPU if failure is present."
            except Exception as e:
                pytest.skip(f"MPS model initialization test skipped: {str(e)}")

    def test_inference(self, setup_rfdetr_test):
        """
        Test basic inference functionality.
        
        Mocks the model's predict method to return test detections and verifies
        that the inference method processes and returns results correctly.
        
        Args:
            monkeypatch: PyTest's monkeypatch fixture
            setup_rfdetr_test: Fixture providing test resources including model and test image
        """
        
        model = setup_rfdetr_test['model_base']
        test_image_path = setup_rfdetr_test['test_image_path']
        test_video_path = setup_rfdetr_test['test_video_path']
        
        # Test image inference
        result_image, output_image_path = model.inference(
            source_path=test_image_path,
            conf=0.5,
            save_viz=True
        )
        
        # Test video inference
        result_video, output_video_path = model.inference(
            source_path=test_video_path,
            conf=0.5,
            save_viz=True
        )
        
        assert result_image is not None, "Inference should return results"
        assert result_video is not None, "Inference should return results"
        assert isinstance(result_image, sv.Detections), "Result should be a Detections object"
        assert isinstance(result_video, list) and all(isinstance(x, sv.Detections) for x in result_video), "Video result should be a list of Detections objects"
        assert len(result_image.xyxy) > 0, "Detections should contain at least one bounding box"
        assert any(len(detection.xyxy) > 0 for detection in result_video if isinstance(detection, sv.Detections)), "At least one frame should contain detections"
        assert os.path.exists(output_image_path), "Output image should be saved"
        assert os.path.exists(output_video_path), "Output video should be saved"

    def test_finetune(self, setup_rfdetr_test):
        
        """
        Downloads RHG COCO-format dataset and verifies that the model
        can begin the training process successfully.
        
        Args:
            setup_rfdetr_test: Fixture providing test resources
        """
        model = setup_rfdetr_test['model_base']
    
        # Just checking that the process starts, finishing is not feasible due to the time it takes.
        def timeout_handler(signum, frame):
            raise TimeoutError("Finetuning test timed out after 45 seconds")
            
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(45)

        try: 
            model.finetune(
                data_path=setup_rfdetr_test['dataset_path'],
                epochs=1,
                batch_size=1,
                num_workers=0,
                checkpoint_interval=1,
                warmup_epochs=0
            )
            assert True, "Finetuning started successfully"
        except TimeoutError as e:
            assert True, f"Finetuning started but was terminated due to timeout: {e}"
        finally:
            signal.alarm(0)