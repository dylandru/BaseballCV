# Functions

This directory contains the functions used in the project.

## Main Scripts

- `load_tools.py`: Contains a class LoadTools containing functions for both loading the raw and annotated datasets as well as the available models.
- `dataset_tools.py`: Contains a class DataTools to generate and manipulate datasets.
- `savant_scraper.py`: Contains a class BaseballSavVideoScraper to scrape baseball videos from Savant.
- `function_utils/utils.py`: Contains utility functions used across the project.

### load_tools.py

#### `LoadTools` class

Key function(s):
- `load_dataset(dataset_alias: str, use_bdl_api: Optional[bool] = True, file_txt_path: Optional[str] = None) -> str`: 
  Loads a zipped dataset and extracts it to a folder. It can use either a dataset alias or a file path to a text file containing the download link.

- `load_model(model_alias: str, model_type: str = 'YOLO', use_bdl_api: Optional[bool] = True, model_txt_path: Optional[str] = None) -> str`: 
  Loads a given baseball computer vision model into the repository. It can use either a model alias or a file path to a text file containing the download link.

### dataset_tools.py

#### `DataTools` class

Key function(s):
- `generate_photo_dataset(max_plays=5000, max_num_frames=10000, max_videos_per_game=10, start_date="2024-05-01", end_date="2024-07-31", delete_savant_videos=True)`: Generates a photo dataset from a diverse set of baseball videos from Savant.

- `automated_annotation(model_alias: str, dataset_path: str, conf: float = 0.6)`: Automatically annotates a given dataset using a pre-trained YOLOmodel.

### savant_scraper.py

#### `BaseballSavVideoScraper` class

Key function(s):
- `run_statcast_pull_scraper(start_date="2024-05-01", end_date="2024-07-31", max_videos_per_game=10, delete_savant_videos=True)`: Scrapes baseball videos from Savant with customization on date, max number of videos per game, whether to delete the downloaded videos after use, among other parameters.


## Usage

To use these functions, please consult the main README.md, the individual files docstrings, or the notebooks in the notebooks directory. These references should allow the user to understand the use-cases for each of these scripts.
