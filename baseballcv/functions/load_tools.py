from pathlib import Path
import requests
import os
from tqdm import tqdm
import zipfile
import io
from typing import Optional, Union
import shutil
from huggingface_hub import snapshot_download
from baseballcv.utilities import BaseballCVLogger
from datasets import load_dataset

class LoadTools:
    """
    Class dedicated to downloading / loading models and datasets from either the BallDataLab API or specified text files.

    Attributes:
        session (requests.Session): Session object for making requests.
        chunk_size (int): Size of chunks to use when downloading files.
        BDL_MODEL_API (str): Base URL for the BallDataLab model API.
        BDL_DATASET_API (str): Base URL for the BallDataLab dataset API.

    Methods:
        load_model(model_alias: str, model_type: str = 'YOLO', use_bdl_api: Optional[bool] = True) -> str:
            Loads a given baseball computer vision model into the repository.
        load_dataset(dataset_alias: str, use_bdl_api: Optional[bool] = True) -> str:
            Loads a zipped dataset and extracts it to a folder.
        _download_files(url: str, dest: Union[str, os.PathLike], is_folder: bool = False, is_labeled: bool = False) -> None:
            Protected method to handle model and dataset downloads.
        _get_url(alias: str, txt_path: str, use_bdl_api: bool, api_endpoint: str) -> str:
            Protected method to obtain the download URL from the BDL API or a text file.
    """

    def __init__(self):
        self.session = requests.Session()
        self.chunk_size = 1024
        self.BDL_MODEL_API = "https://balldatalab.com/api/models/"
        self.BDL_DATASET_API = "https://balldatalab.com/api/datasets/"
        self.yolo_model_aliases = {
            'phc_detector': 'models/od/YOLO/pitcher_hitter_catcher_detector/model_weights/pitcher_hitter_catcher_detector_v4.txt',
            'bat_tracking': 'models/od/YOLO/bat_tracking/model_weights/bat_tracking.txt',
            'ball_tracking': 'models/od/YOLO/ball_tracking/model_weights/ball_tracking.txt',
            'glove_tracking': 'models/od/YOLO/glove_tracking/model_weights/glove_tracking.txt',
            'ball_trackingv4': 'models/od/YOLO/ball_tracking/model_weights/ball_trackingv4.txt',
            'amateur_pitcher_hitter': 'models/od/YOLOv9/amateur_pitcher_hitter/model_weights/amateur_pitcher_hitter.txt',
            'homeplate_tracking': 'models/od/YOLOv9/homeplate_tracking/model_weights/homeplate_tracking.txt'
        }
        self.florence_model_aliases = {
            'ball_tracking': 'models/vlm/FLORENCE2/ball_tracking/model_weights/florence_ball_tracking.txt',
            'florence_ball_tracking': 'models/vlm/FLORENCE2/ball_tracking/model_weights/florence_ball_tracking.txt'
        }
        self.paligemma2_model_aliases = {
            'paligemma2_ball_tracking': 'models/vlm/paligemma2/ball_tracking/model_weights/paligemma2_ball_tracking.txt'
        }
        self.detr_model_aliases = {
            'detr_baseball_v2': 'hf:dyland222/detr-coco-baseball_v2'
        }
        self.rfdetr_model_aliases = {
            'rfdetr_glove_tracking': 'models/od/RFDETR/glove_tracking/model_weights/rfdetr_glove_tracking.txt'
        }
        self.dataset_aliases = {
            'okd_nokd': 'datasets/yolo/OKD_NOKD.txt',
            'baseball_rubber_home_glove': 'datasets/yolo/baseball_rubber_home_glove.txt',
            'baseball_rubber_home': 'datasets/yolo/baseball_rubber_home.txt',
            'broadcast_10k_frames': 'datasets/raw_photos/broadcast_10k_frames.txt',
            'broadcast_15k_frames': 'datasets/raw_photos/broadcast_15k_frames.txt',
            'baseball_rubber_home_COCO': 'datasets/COCO/baseball_rubber_home_COCO.txt',
            'baseball_rubber_home_glove_COCO': 'datasets/COCO/baseball_rubber_home_glove_COCO.txt',
            'baseball': 'datasets/yolo/baseball.txt',
            'phc': 'datasets/yolo/phc.txt',
            'amateur_pitcher_hitter': 'datasets/yolo/amateur_pitcher_hitter.txt',
            'amateur_hitter_pitcher_jsonl': 'datasets/JSONL/amateur_hitter_pitcher_jsonl.txt',
            'international_amateur_baseball_catcher_photos': 'hf:dyland222/international_amateur_baseball_catcher_photos_dataset',
            'international_amateur_baseball_catcher_video': 'hf:dyland222/international_amateur_baseball_catcher_video_dataset',
            'international_amateur_baseball_photos': 'hf:dyland222/international_amateur_baseball_photos_dataset',
            'international_amateur_baseball_game_video': 'hf:dyland222/international_amateur_baseball_game_videos',
            'international_amateur_baseball_bp_video': 'hf:dyland222/international_amateur_baseball_bp_videos',
            'international_amateur_pitcher_photo': 'hf:dyland222/international_amateur_pitcher_photo_dataset'
        }
        self.logger = BaseballCVLogger.get_logger(self.__class__.__name__)


    def _download_files(self, url: str, dest: Union[str, os.PathLike], is_folder: bool = False, is_labeled: bool = False) -> None:
        response = self.session.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(dest)}")
            
            if is_folder: 
                content = io.BytesIO()
                for data in response.iter_content(chunk_size=self.chunk_size):
                    size = content.write(data)
                    progress_bar.update(size)
                
                progress_bar.close()

                os.makedirs(dest, exist_ok=True)
                
                with zipfile.ZipFile(content) as zip_ref:
                    temp_dir = os.path.join(dest, "_temp_extract")
                    os.makedirs(temp_dir, exist_ok=True)
                    zip_ref.extractall(temp_dir)
                    
                    # Error Handling for unzipping files
                    dest_name = os.path.basename(os.path.normpath(dest))
                    same_name_dir = os.path.join(temp_dir, dest_name)
                    
                    if os.path.isdir(same_name_dir):
                        for item in os.listdir(same_name_dir):
                            src_path = os.path.join(same_name_dir, item)
                            dst_path = os.path.join(dest, item)
                            
                            if os.path.exists(dst_path):
                                if os.path.isdir(dst_path):
                                    shutil.rmtree(dst_path)
                                else:
                                    os.remove(dst_path)
                            
                            shutil.move(src_path, dst_path)
                    else:
                        for item in os.listdir(temp_dir):
                            src_path = os.path.join(temp_dir, item)
                            dst_path = os.path.join(dest, item)
                            
                            if os.path.exists(dst_path):
                                if os.path.isdir(dst_path):
                                    shutil.rmtree(dst_path)
                                else:
                                    os.remove(dst_path)
                            
                            shutil.move(src_path, dst_path)
                    
                    shutil.rmtree(temp_dir)
                
                self.logger.info(f"Dataset downloaded and extracted to {dest}")
            else:
                with open(dest, 'wb') as file:
                    for data in response.iter_content(chunk_size=self.chunk_size):
                        size = file.write(data)
                        progress_bar.update(size)
                
                progress_bar.close()
                self.logger.info(f"Model downloaded to {dest}")
        else:
            self.logger.error(f"Download failed. STATUS: {response.status_code}")

    def _get_url(self, alias: str, txt_path: str, use_bdl_api: bool, api_endpoint: str) -> str:
        if use_bdl_api:
            return f"{api_endpoint}{alias}"
        else:
            with open(txt_path, 'r') as file:
                return file.read().strip()

    def load_model(self, model_alias: str = None, model_type: Optional[str] = 'YOLO', use_bdl_api: Optional[bool] = True, model_txt_path: Optional[str] = None) -> str:
        '''
        Loads a given baseball computer vision model into the repository.

        Args:
            model_alias (str): Alias of the model to load.
            model_type (str): The type of the model to utilize. Defaults to YOLO.
            use_bdl_api (Optional[bool]): Whether to use the BallDataLab API.
            model_txt_path (Optional[str]): Path to .txt file containing download link to model weights. 
                                            Only used if use_bdl_api is specified as False.

        Returns:
            model_weights_path (str):  Path to where the model weights are saved within the repo.
        '''

        if model_alias is None and use_bdl_api:
            self.logger.error("model_alias must be provided if use_bdl_api is True")
            return None
        
        if model_type == 'YOLO':
            model_txt_path = self.yolo_model_aliases.get(model_alias) if use_bdl_api else model_txt_path
        elif model_type == 'FLORENCE2':
            model_txt_path = self.florence_model_aliases.get(model_alias) if use_bdl_api else model_txt_path
        elif model_type == 'PALIGEMMA2':
            model_txt_path = self.paligemma2_model_aliases.get(model_alias) if use_bdl_api else model_txt_path
        elif model_type == 'DETR':
            model_txt_path = self.detr_model_aliases.get(model_alias) if use_bdl_api else model_txt_path
        elif model_type == 'RFDETR':
            model_txt_path = self.rfdetr_model_aliases.get(model_alias) if use_bdl_api else model_txt_path
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        if not model_txt_path:
            raise ValueError(f"Invalid alias: {model_alias}")

        base_dir = os.path.dirname(model_txt_path)
        base_name = os.path.splitext(os.path.basename(model_txt_path))[0]
        os.makedirs(base_dir, exist_ok=True)

        is_hf_model = model_txt_path.startswith("hf:")
        if model_type == 'YOLO':
            model_weights_path = f"{base_dir}/{base_name}.pt"
        else:
            model_weights_path = f"{base_name}" if is_hf_model else f"{base_dir}/{base_name}"
            os.makedirs(model_weights_path, exist_ok=True)

        if os.path.exists(model_weights_path) and not is_hf_model:
            self.logger.info(f"Model found at {model_weights_path}")
            return model_weights_path

        if is_hf_model:
            try:
                snapshot_download(
                    repo_id=model_txt_path[3:],
                    local_dir=model_weights_path,
                    repo_type="model",
                    ignore_patterns=["*.md", "*.gitattributes", "*.gitignore"],
                )
                self.logger.info(f"Successfully downloaded model from HF to {model_weights_path}")
                return model_weights_path
            except Exception as e:
                self.logger.error(f"Error downloading from Hugging Face: {e}")
                raise
            
        else: 
            url = self._get_url(model_alias, model_txt_path, use_bdl_api, self.BDL_MODEL_API)
            self._download_files(url, model_weights_path, is_folder=(model_type=='FLORENCE2' or model_type=='PALIGEMMA2' or model_type=='DETR'))
        
        return model_weights_path

    def load_dataset(self, dataset_alias: str, use_bdl_api: Optional[bool] = True, file_txt_path: Optional[str] = None) -> str:
        '''
        Loads a zipped dataset and extracts it to a folder.

        Args:
            dataset_alias (str): Alias of the dataset to load that corresponds to a dataset folder to download
            use_bdl_api (Optional[bool]): Whether to use the BallDataLab API. Defaults to True.
            file_txt_path (Optional[str]): Path to .txt file containing download link to zip file containing dataset. 
                                           Only used if use_bdl_api is specified as False.

        Returns:
            dir_name (str): Path to the folder containing the dataset.
        '''
        
        txt_path = self.dataset_aliases.get(dataset_alias) if use_bdl_api else file_txt_path
        if not txt_path:
            raise ValueError(f"Invalid alias or missing path: {dataset_alias}")
        
        is_hf_dataset = txt_path.startswith("hf:")  

        if is_hf_dataset: #HF datasets
            repo_id = txt_path[3:]  
            
            if os.path.exists(dataset_alias):
                self.logger.info(f"Dataset found at {dataset_alias}")
                return Path(dataset_alias)
                
            self.logger.info(f"Processing dataset from HF w/ alias: {dataset_alias}...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=dataset_alias,
                    repo_type="dataset",
                    token=None,
                    ignore_patterns=["*.md", "*.gitattributes", "*.gitignore"]
                )

                dataset = load_dataset(repo_id, split="train")

                for i, example in tqdm(enumerate(dataset["train"]), total=len(dataset["train"])):
                    image, filename = example["image"], example["filename"]
                    base_path = f"{dataset_alias}/train/"
                    image.save(f"{base_path}/{filename}")

                [os.remove(os.path.join(root, file)) for root, dirs, files in os.walk(dataset_alias) for file in files if file.endswith('.parquet')]

                self.logger.info(f"Successfully downloaded dataset from Hugging Face to {base_path}")
                return Path(base_path)
             
            except Exception as e:
                self.logger.error(f"Error downloading from Hugging Face: {e}")
                raise

        else: #Non-HF datasets
            base = os.path.splitext(os.path.basename(txt_path))[0]
            dir_name = "unlabeled_" + base if 'raw_photos' in base or 'frames' in base or 'frames' in dataset_alias else base

            if os.path.exists(dir_name):
                self.logger.info(f"Dataset found at {dir_name}")
                return Path(dir_name)

            url = self._get_url(dataset_alias, txt_path, use_bdl_api, self.BDL_DATASET_API)
            self._download_files(url, dir_name, is_folder=True)

            redundant_dir = os.path.join(dir_name, dataset_alias)
            if os.path.exists(redundant_dir):
                self.logger.info(f"Processing dataset {dataset_alias}...")
                for item in os.listdir(redundant_dir):
                    source = os.path.join(redundant_dir, item)
                    destination = os.path.join(dir_name, item)
                    
                    if os.path.isdir(source):
                        if not os.path.exists(destination):
                            shutil.move(source, destination)
                        else:
                            for subitem in os.listdir(source):
                                shutil.move(os.path.join(source, subitem), destination)
                            os.rmdir(source)
                    else: 
                        if not os.path.exists(destination):
                            shutil.move(source, destination)
                
                if os.path.exists(redundant_dir):
                    os.rmdir(redundant_dir)
                self.logger.info(f"Successfully processed dataset {dataset_alias}.")

            return Path(dir_name)
