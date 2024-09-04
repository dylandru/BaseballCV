import requests
import os
from tqdm import tqdm
from function_utils.utils import model_aliases

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


