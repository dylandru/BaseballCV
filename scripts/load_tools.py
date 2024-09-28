import requests
import os
from tqdm import tqdm
import zipfile
import io
from scripts.function_utils import model_aliases, dataset_aliases
from typing import Optional

def load_model(model_alias: str, model_type = 'YOLO') -> str:
    '''
    Loads a given baseball computer vision model into the repository.

    Args:
        model_alias (str): Alias of the model to load that corresponds to a model file to download

    Returns:
        model_weights_path (str): Path to where the model weights are saved within the repo.
    '''

    if model_type == 'YOLO':

        model_txt_path = model_aliases.get(model_alias)
        if not model_txt_path:
            raise ValueError(f"This is not a model alias: {model_alias}")

        with open(model_txt_path, 'r') as file:
            link = file.read().strip()

        model_weights_path = f"{os.path.dirname(model_txt_path)}/{os.path.splitext(os.path.basename(model_txt_path))[0]}.pt"

        if os.path.exists(model_weights_path):
            print(f"Model found at {model_weights_path}")
            return model_weights_path

        response = requests.get(link, stream=True)

        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
    
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {model_weights_path}")
            
            with open(model_weights_path, 'wb') as file:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)
            
            progress_bar.close()
            print(f"Model downloaded successfully: {model_weights_path}")
        else:
            print(f"Model download failed. Status code: {response.status_code}")

    else:
        raise ValueError(f"Invalid Model Type: {model_type}")
    
    return model_weights_path

def load_dataset(dataset_alias: str, file_txt_path: Optional[str] = None) -> str:
    '''
    Loads a zipped dataset from a .txt file containing a link to the dataset and extracts it to a folder.

    Args:
        dataset_alias (str): Alias of the dataset to load that corresponds to a dataset folder to download
        file_txt_path (Optional[str]): Path to .txt file containing download link to zip file containing dataset. 

    Returns:
        dir_name (str): Path to the folder containing the dataset.
    '''

    if file_txt_path is None:
        file_txt_path = dataset_aliases.get(dataset_alias)
        if file_txt_path is None:
            raise ValueError(f"This is not a dataset alias.")

    with open(file_txt_path, 'r') as file:
        link = file.read().strip()

    base = os.path.splitext(os.path.basename(file_txt_path))[0]
    dir_name = f"unlabeled_{base}" if 'raw_photos' in base else base #add unlabeled_ to raw_photos datasets

    if os.path.exists(dir_name):
        print(f"Dataset found at {dir_name}")
        return dir_name

    response = requests.get(link, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {dir_name}")
        
        content = io.BytesIO()
        for data in response.iter_content(1024):
            size = content.write(data)
            progress_bar.update(size)
        progress_bar.close()

        os.makedirs(dir_name, exist_ok=True)

        with zipfile.ZipFile(content) as zip_ref:
            for file in zip_ref.namelist():
                if not file.startswith('__MACOSX') and not file.startswith('._'): #prevents extracting MACOSX files from zip
                    zip_ref.extract(file, dir_name)
        
        print(f"Dataset downloaded and extracted to {dir_name}.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

    return dir_name
