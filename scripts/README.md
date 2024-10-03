# Scripts

This directory contains the scripts used in the project.

## Main Scripts

- `load_tools.py`: Contains a class LoadTools containing functions for both loading the raw and annotated datasets as well as the available models.
- `generate_photos.py`: Script to utilize to generate one's own raw photos for the creation of a dataset.
- `function_utils/utils.py`: Contains utility functions used across the project.

### load_tools.py

Key class:

- Main Scripts in `LoadTools` class

-- `load_dataset(dataset_alias: str, use_bdl_api: Optional[bool] = True, file_txt_path: Optional[str] = None) -> str`: 
  Loads a zipped dataset and extracts it to a folder. It can use either a dataset alias or a file path to a text file containing the download link.

-- `load_model(model_alias: str, model_type: str = 'YOLO', use_bdl_api: Optional[bool] = True, model_txt_path: Optional[str] = None) -> str`: 
  Loads a given baseball computer vision model into the repository. It can use either a model alias or a file path to a text file containing the download link.

### generate_photos.py

Key function:
- `generate_photo_dataset(max_plays=5000, max_num_frames=10000, max_videos_per_game=10, start_date="2024-05-01", end_date="2024-07-31", delete_savant_videos=True)`: Generates a photo dataset from a diverse set of baseball videos from Savant.


### savant_scraper.py

Key functions:
- `scrape_savant_videos(start_date="2024-05-01", end_date="2024-07-31", max_videos_per_game=10, delete_savant_videos=True)`: Scrapes baseball videos from Savant with customization on date, max number of videos per game, whether to delete the downloaded videos after use, among other parameters.

## Usage

To use these scripts, please consult the main README.md, the individual files docstrings, or the notebooks in the notebooks directory. These references should allow the user to understand the use-cases for each of these scripts.
