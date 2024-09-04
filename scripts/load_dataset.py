import requests
import zipfile
import io
import os
import tqdm

def load_dataset(txt_file_path: str) -> None:
    '''
    Loads dataset from a .txt file containing a link to the dataset.

    Args:
        txt_file_path (str): Path to .txt file containing download link to zip file containing dataset. 

    Returns:
        None: Dataset is downloaded to the name of the .txt file.
    '''
    with open(txt_file_path, 'r') as file:
        link = file.readline().strip()

    response = requests.get(link, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm.tqdm(total=total_size, unit='iB', unit_scale=True, desc="Downloading Dataset") #shows progress of .zip download
        
        content = io.BytesIO()
        for data in response.iter_content(1024):
            size = content.write(data)
            progress_bar.update(size)
        progress_bar.close()

        with zipfile.ZipFile(content) as zip_ref:
            dir = os.path.splitext(os.path.basename(txt_file_path))[0] #names folder after .txt file
            os.makedirs(dir, exist_ok=True)
            zip_ref.extractall(dir)
        
        print(f"Dataset downloaded and extracted to {dir}.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

