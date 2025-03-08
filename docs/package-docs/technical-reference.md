---
layout: default
title: API Reference
nav_order: 2
has_children: true
permalink: /api-reference
---

# API Reference
{: .no_toc }

Complete API documentation for the BaseballCV package.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Functions Module
{: .d-inline-block }

baseballcv.functions
{: .label .label-green }

The functions module provides the core functionality for baseball video analysis, data loading, and dataset generation.

### BaseballTools
{: .d-inline-block }

Main Class
{: .label .label-purple }

```python
from baseballcv.functions import BaseballTools
```

The main class for analyzing baseball videos with Computer Vision.

#### Constructor

```python
BaseballTools(device: str = 'cpu', verbose: bool = True)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `device` | str | Device to use for the analysis | 'cpu' |
| `verbose` | bool | Whether to print verbose output | True |

#### Methods

##### distance_to_zone

```python
distance_to_zone(
    start_date: str,
    end_date: str,
    team: str = None,
    pitch_call: str = None,
    max_videos: int = None,
    max_videos_per_game: int = None,
    create_video: bool = True,
    catcher_model: str = 'phc_detector',
    glove_model: str = 'glove_tracking',
    ball_model: str = 'ball_trackingv4'
) → List[Dict]
```

Calculates the distance of a pitch to the strike zone in a video, as well as other information about the Play ID.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `start_date` | str | Start date of the analysis (YYYY-MM-DD) | Required |
| `end_date` | str | End date of the analysis (YYYY-MM-DD) | Required |
| `team` | str | Team to analyze | None |
| `pitch_call` | str | Pitch call to analyze | None |
| `max_videos` | int | Maximum number of videos to analyze | None |
| `max_videos_per_game` | int | Maximum videos per game | None |
| `create_video` | bool | Whether to create analysis video | True |
| `catcher_model` | str | PHCDetector model name | 'phc_detector' |
| `glove_model` | str | GloveTracking model name | 'glove_tracking' |
| `ball_model` | str | BallTracking model name | 'ball_trackingv4' |

**Returns**

List[Dict] containing for each video:
- `frame_number`: Frame where ball crosses plate
- `distance`: Distance from ball to strike zone center
- `position`: Position relative to strike zone ("high", "low", "inside", "outside")

**Example**

```python
tools = BaseballTools(device='cuda')
results = tools.distance_to_zone(
    start_date="2024-05-01",
    end_date="2024-05-02",
    team="NYY",
    create_video=True
)
```

### BaseballSavVideoScraper
{: .d-inline-block }

Video Scraping
{: .label .label-blue }

```python
from baseballcv.functions import BaseballSavVideoScraper
```

Class for scraping baseball videos from Baseball Savant based on various criteria.

#### Methods

##### run_statcast_pull_scraper

```python
run_statcast_pull_scraper(
    start_date: str | pd.Timestamp = '2024-05-01',
    end_date: str | pd.Timestamp = '2024-06-01',
    download_folder: str = 'savant_videos',
    max_workers: int = 5,
    team: str = None,
    pitch_call: str = None,
    max_videos: int = None,
    max_videos_per_game: int = None
) → pd.DataFrame
```

Run scraper from Statcast Pull of Play IDs. Retrieves data and processes each row in parallel.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `start_date` | str \| pd.Timestamp | Start date for pull | '2024-05-01' |
| `end_date` | str \| pd.Timestamp | End date for pull | '2024-06-01' |
| `download_folder` | str | Folder for downloaded videos | 'savant_videos' |
| `max_workers` | int | Max concurrent workers | 5 |
| `team` | str | Team filter | None |
| `pitch_call` | str | Pitch call filter | None |
| `max_videos` | int | Max videos to pull | None |
| `max_videos_per_game` | int | Max videos per game | None |

**Returns**

pd.DataFrame containing Play IDs and video metadata

**Raises**

Exception: Any error in downloading a video

##### playids_for_date_range

```python
playids_for_date_range(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    team: str = None,
    pitch_call: str = None
) → pd.DataFrame
```

Retrieve Play IDs for a given date range from Baseball Savant.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `start_date` | str \| pd.Timestamp | Start date | Required |
| `end_date` | str \| pd.Timestamp | End date | Required |
| `team` | str | Team filter | None |
| `pitch_call` | str | Pitch call filter | None |

**Returns**

pd.DataFrame containing Play IDs and metadata

### LoadTools
{: .d-inline-block }

Model & Dataset Loading
{: .label .label-yellow }

```python
from baseballcv.functions import LoadTools
```

Class for downloading and loading models and datasets from either the BallDataLab API or specified text files.

#### Attributes

| Name | Type | Description |
|:-----|:-----|:------------|
| `session` | requests.Session | Session for making requests |
| `chunk_size` | int | Size of download chunks |
| `BDL_MODEL_API` | str | BallDataLab model API URL |
| `BDL_DATASET_API` | str | BallDataLab dataset API URL |
| `yolo_model_aliases` | Dict[str, str] | YOLO model path mappings |
| `florence_model_aliases` | Dict[str, str] | Florence2 model path mappings |
| `paligemma2_model_aliases` | Dict[str, str] | PaliGemma2 model path mappings |
| `dataset_aliases` | Dict[str, str] | Dataset path mappings |

#### Methods

##### load_model

```python
load_model(
    model_alias: str,
    model_type: str = 'YOLO',
    use_bdl_api: Optional[bool] = True,
    model_txt_path: Optional[str] = None
) → str
```

Loads a baseball computer vision model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `model_alias` | str | Model alias to load | Required |
| `model_type` | str | Model type ('YOLO', 'FLORENCE2', 'PALIGEMMA2') | 'YOLO' |
| `use_bdl_api` | bool | Use BallDataLab API | True |
| `model_txt_path` | str | Path to download link file | None |

Available model aliases:
- YOLO: 'phc_detector', 'bat_tracking', 'ball_tracking', etc.
- Florence2: 'florence_ball_tracking'
- PaliGemma2: 'paligemma2_ball_tracking'

**Returns**

str: Path to saved model weights

##### load_dataset

```python
load_dataset(
    dataset_alias: str,
    use_bdl_api: Optional[bool] = True,
    file_txt_path: Optional[str] = None
) → str
```

Loads and extracts a dataset.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `dataset_alias` | str | Dataset alias to load | Required |
| `use_bdl_api` | bool | Use BallDataLab API | True |
| `file_txt_path` | str | Path to download link file | None |

Available dataset aliases:
- Labeled: 'baseball_rubber_home_glove', 'baseball', 'phc'
- Raw photos: 'broadcast_10k_frames', 'broadcast_15k_frames'

**Returns**

str: Path to extracted dataset

## Models Module
{: .d-inline-block }

baseballcv.model
{: .label .label-green }

The models module provides implementations of various computer vision models optimized for baseball analysis.

### YOLOv9
{: .d-inline-block }

Object Detection
{: .label .label-blue }

```python
from baseballcv.model import YOLOv9
```

Class for using YOLOv9 models for object detection in baseball contexts.

#### Constructor

```python
YOLOv9(
    device: str | int = "cuda",
    model_path: str = '',
    cfg_path: str = 'models/detect/yolov9-c.yaml',
    name: str = 'yolov9-c'
)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `device` | str \| int | Device for inference ('cpu', 'cuda', or GPU index) | 'cuda' |
| `model_path` | str | Path to initial weights | '' |
| `cfg_path` | str | Path to model config | 'models/detect/yolov9-c.yaml' |
| `name` | str | Name of the model | 'yolov9-c' |

#### Methods

##### finetune

```python
finetune(
    data_path: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    **kwargs
) → Dict
```

Finetune a YOLOv9 model on baseball data.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `data_path` | str | Path to data config file | Required |
| `epochs` | int | Number of training epochs | 100 |
| `imgsz` | int | Image size | 640 |
| `batch_size` | int | Batch size | 16 |

Additional kwargs:
- `rect` (bool): Rectangular training. Default: False
- `resume` (bool): Resume training. Default: False
- `nosave` (bool): Only save final checkpoint. Default: False
- `noval` (bool): Skip validation. Default: False
- `optimizer` (str): Optimizer to use. Default: 'SGD'
- See [source code](#) for full list of parameters

**Returns**

Dict containing training results and metrics:
- `metrics`: Dictionary of training metrics
- `best_fitness`: Best model fitness score
- `final_epoch`: Final training epoch
- `training_time`: Total training time

**Example**

```python
model = YOLOv9(device='cuda')
results = model.finetune(
    data_path='data/baseball.yaml',
    epochs=50,
    batch_size=32,
    imgsz=640
)
```

### Florence2
{: .d-inline-block }

Vision Language Model
{: .label .label-purple }

```python
from baseballcv.model.vlm import Florence2
```

Class for using Florence-2 vision language models for baseball analysis.

#### Constructor

```python
Florence2(
    device: str = None,
    model_id: str = 'microsoft/florence-2-large',
    model_run_path: str = None,
    batch_size: int = 8,
    torch_dtype: torch.dtype = torch.float32,
    use_pretrained_lora: bool = False
)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `device` | str | Device for inference (auto-select if None) | None |
| `model_id` | str | HuggingFace model ID | 'microsoft/florence-2-large' |
| `model_run_path` | str | Path to save model run information | None |
| `batch_size` | int | Batch size for training/inference | 8 |
| `torch_dtype` | torch.dtype | Torch data type | torch.float32 |
| `use_pretrained_lora` | bool | Use pretrained LoRA weights | False |

#### Methods

##### finetune

```python
finetune(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    epochs: int = 10,
    learning_rate: float = 2e-4,
    **kwargs
) → None
```

Finetune the Florence-2 model on baseball data.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `train_data` | Dict[str, Any] | Training data dictionary | Required |
| `val_data` | Dict[str, Any] | Validation data dictionary | Required |
| `epochs` | int | Number of training epochs | 10 |
| `learning_rate` | float | Learning rate | 2e-4 |

Additional kwargs:
- `weight_decay` (float): Weight decay. Default: 0.01
- `warmup_steps` (int): Number of warmup steps. Default: 500
- `mixed_precision` (bool): Use mixed precision training. Default: False
- See [source code](#) for full list of parameters

**Example**

```python
model = Florence2(device='cuda')
model.finetune(
    train_data=train_dataset,
    val_data=val_dataset,
    epochs=20,
    learning_rate=1e-4
)
```

### PaliGemma2
{: .d-inline-block }

Vision Language Model
{: .label .label-purple }

```python
from baseballcv.model.vlm import PaliGemma2
```

Class for using PaliGemma2 vision language models for baseball analysis.

#### Constructor

```python
PaliGemma2(
    device: str = None,
    model_id: str = 'google/paligemma2-3b-pt-224',
    model_run_path: str = None,
    batch_size: int = 8,
    torch_dtype: torch.dtype = torch.float32,
    use_pretrained_lora: bool = False
)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `device` | str | Device for inference (auto-select if None) | None |
| `model_id` | str | HuggingFace model ID | 'google/paligemma2-3b-pt-224' |
| `model_run_path` | str | Path to save model run information | None |
| `batch_size` | int | Batch size for training/inference | 8 |
| `torch_dtype` | torch.dtype | Torch data type | torch.float32 |
| `use_pretrained_lora` | bool | Use pretrained LoRA weights | False |

#### Methods

Similar to Florence2, see [Florence2 Methods](#florence2) for details.

## Datasets Module
{: .d-inline-block }

baseballcv.datasets
{: .label .label-green }

The datasets module provides tools for processing and managing baseball computer vision datasets.

### DataProcessor
{: .d-inline-block }

Dataset Processing
{: .label .label-yellow }

```python
from baseballcv.datasets.processing import DataProcessor
```

Class for processing and converting between different dataset formats.

#### Methods

##### convert_yolo_to_coco

```python
convert_yolo_to_coco(
    yolo_dir: str,
    output_path: str,
    class_mapping: Dict[int, str]
) → None
```

Convert YOLO format annotations to COCO format.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `yolo_dir` | str | Directory containing YOLO dataset | Required |
| `output_path` | str | Path to save COCO dataset | Required |
| `class_mapping` | Dict[int, str] | Mapping from class IDs to names | Required |

**Example**

```python
processor = DataProcessor()
class_mapping = {0: 'baseball', 1: 'glove', 2: 'bat'}
processor.convert_yolo_to_coco(
    yolo_dir='baseball_dataset',
    output_path='baseball_coco.json',
    class_mapping=class_mapping
)
```

##### convert_coco_to_yolo

```python
convert_coco_to_yolo(
    coco_path: str,
    output_dir: str
) → None
```

Convert COCO format annotations to YOLO format.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `coco_path` | str | Path to COCO dataset | Required |
| `output_dir` | str | Directory to save YOLO dataset | Required |

### CocoDetectionDataset
{: .d-inline-block }

Dataset Format
{: .label .label-blue }

```python
from baseballcv.datasets.formats.coco import CocoDetectionDataset
```

Dataset class for loading COCO format detection datasets.

#### Constructor

```python
CocoDetectionDataset(
    annotations_path: str,
    image_dir: str
)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `annotations_path` | str | Path to COCO annotations file | Required |
| `image_dir` | str | Directory containing images | Required |

### JSONLDetection
{: .d-inline-block }

Dataset Format
{: .label .label-blue }

```python
from baseballcv.datasets.formats.jsonl import JSONLDetection
```

Dataset class for loading JSONL format detection datasets.

#### Constructor

```python
JSONLDetection(
    jsonl_path: str,
    image_dir: str
)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `jsonl_path` | str | Path to JSONL annotations file | Required |
| `image_dir` | str | Directory containing images | Required |

## Version Information
{: .d-inline-block }

Package Info
{: .label .label-green }

Current package version: 0.1.11 