import requests
import os
from tqdm import tqdm
import zipfile
import io
from typing import Optional, Union
import shutil

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
        _download_files(url: str, dest: Union[str, os.PathLike], is_dataset: bool = False) -> None:
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
            'phc_detector': 'models/YOLO/pitcher_hitter_catcher_detector/model_weights/pitcher_hitter_catcher_detector_v4.txt',
            'bat_tracking': 'models/YOLO/bat_tracking/model_weights/bat_tracking.txt',
            'ball_tracking': 'models/YOLO/ball_tracking/model_weights/ball_tracking.txt',
            'glove_tracking': 'models/YOLO/glove_tracking/model_weights/glove_tracking.txt',
            'ball_trackingv4': 'models/YOLO/ball_tracking/model_weights/ball_trackingv4.txt'
        }
        self.florence_model_aliases = {
            'ball_tracking': 'models/FLORENCE2/ball_tracking/model_weights/florence_ball_tracking.txt',
            'florence_ball_tracking': 'models/vlm/FLORENCE2/ball_tracking/model_weights/florence_ball_tracking.txt'
        }
        self.paligemma2_model_aliases = {
            'paligemma2_ball_tracking': 'models/vlm/paligemma2/ball_tracking/model_weights/paligemma2_ball_tracking.txt'
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
            'phc': 'datasets/yolo/phc.txt'
        }

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

                with zipfile.ZipFile(content) as zip_ref:
                    for file in zip_ref.namelist():
                        if not file.startswith('__MACOSX') and not file.startswith('._'):
                            if is_labeled:
                                zip_ref.extract(file, dest)
                            else:
                                if '/' in file:
                                    filename = file.split('/')[-1]
                                    if filename:
                                        with zip_ref.open(file) as source, open(os.path.join(dest, filename), 'wb') as target:
                                            shutil.copyfileobj(source, target)
                                else:
                                    zip_ref.extract(file, dest)
                
                if not is_labeled:
                    for root, dirs, files in os.walk(dest, topdown=False):
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            if not os.listdir(dir_path):
                                os.rmdir(dir_path)
                
                print(f"Dataset downloaded and extracted to {dest}")
            else:
                with open(dest, 'wb') as file:
                    for data in response.iter_content(chunk_size=self.chunk_size):
                        size = file.write(data)
                        progress_bar.update(size)
                
                progress_bar.close()
                print(f"Model downloaded to {dest}")
        else:
            print(f"Download failed. STATUS: {response.status_code}")

    def _get_url(self, alias: str, txt_path: str, use_bdl_api: bool, api_endpoint: str) -> str:
        if use_bdl_api:
            return f"{api_endpoint}{alias}"
        else:
            with open(txt_path, 'r') as file:
                return file.read().strip()

    def load_model(self, model_alias: str, model_type: str = 'YOLO', use_bdl_api: Optional[bool] = True, model_txt_path: Optional[str] = None) -> str:
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
        if model_type == 'YOLO':
            model_txt_path = self.yolo_model_aliases.get(model_alias) if use_bdl_api else model_txt_path
        elif model_type == 'FLORENCE2':
            model_txt_path = self.florence_model_aliases.get(model_alias) if use_bdl_api else model_txt_path
        elif model_type == 'PALIGEMMA2':
            model_txt_path = self.paligemma2_model_aliases.get(model_alias) if use_bdl_api else model_txt_path
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        if not model_txt_path:
            raise ValueError(f"Invalid alias: {model_alias}")

        base_dir = os.path.dirname(model_txt_path)
        base_name = os.path.splitext(os.path.basename(model_txt_path))[0]
        os.makedirs(base_dir, exist_ok=True)

        if model_type == 'YOLO':
            model_weights_path = f"{base_dir}/{base_name}.pt"
        else:
            model_weights_path = f"{base_dir}/{base_name}"
            os.makedirs(model_weights_path, exist_ok=True)

        if os.path.exists(model_weights_path):
            print(f"Model found at {model_weights_path}")
            return model_weights_path

        url = self._get_url(model_alias, model_txt_path, use_bdl_api, self.BDL_MODEL_API)
        self._download_files(url, model_weights_path, is_folder=(model_type=='FLORENCE2'))
        
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

        base = os.path.splitext(os.path.basename(txt_path))[0]
        dir_name = "unlabeled_" + base if 'raw_photos' in base or 'frames' in base or 'frames' in dataset_alias else base

        if os.path.exists(dir_name):
            print(f"Dataset found at {dir_name}")
            return dir_name

        url = self._get_url(dataset_alias, txt_path, use_bdl_api, self.BDL_DATASET_API)
        os.makedirs(dir_name, exist_ok=True)
        self._download_files(url, dir_name, is_folder=True)

        return dir_name
