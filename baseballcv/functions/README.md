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

##### `__init__` Args:
- `start_dt` : A string representing the starting date in query (i.e. '2024-08-10') **<ins>REQUIRED</ins>**
- `end_dt` : A string representing the ending date in query (i.e. '2024-08-10')
- `player` : A integer representing the player ID you want to filter for (i.e. 608070 for Jose Ràmirez)
- `team_abbr` : A string representing the team abbreviation you want to filter for (i.e 'CHC' for Chicago Cubs)
- `pitch_type` : A string representing the kind of pitch you want to filter for (i.e. 'FF' for 4-Seam Fastballs)
- `download_folder` : A string representing the name of the output folder you want the videos being saved to
- `max_return_videos` : A integer representing the maximum videos to return in a query. Defaults to 10.
- `max_videos_per_game` : A integer representing the maximum videos returned for each game. Defaults to None.

Example Use Cases:
```python
BaseballSavVideoScraper('2024-05-10', max_return_videos = None) # Return ALL plays in query
BaseballSavVideoScraper('2024-10-10', '2024-05-10') # Valid query, our function swaps the dates
BaseballSavVideoScraper('2024-04-12', player = 60870) # Filters for Jose Ramirez plays
BaseballSavVideoScraper('2024-04-12', player = 60870, team_abbr = 'CLE') # Specifies team you want with Jose Ramirez plays, makes filtering time faster and queries more reliable
```

Key function(s):
- `run_executor()` : Runs a multi-threading channel that efficiently extracts savant video.
- `get_play_ids_df() -> pd.DataFrame` : Returns a pandas DataFrame of the extracted play ids and pitch-level metrics associated with the play. Similar to the savant csv loaded in from pybaseball.
- `cleanup_savant_videos()` : Removes the download folder directory.


## Usage

To use these functions, please consult the main README.md, the individual files docstrings, or the notebooks in the notebooks directory. These references should allow the user to understand the use-cases for each of these scripts.
