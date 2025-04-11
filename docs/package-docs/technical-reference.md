---
layout: default
title: API Reference
nav_order: 2
has_children: true
permalink: /technical-reference
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
    team_abbr: str = None,
    pitch_type: str = None,
    player: int = None,
    max_videos: int = None,
    max_videos_per_game: int = None,
    create_video: bool = True,
    catcher_model: str = 'phc_detector',
    glove_model: str = 'glove_tracking',
    ball_model: str = 'ball_trackingv4',
    zone_vertical_adjustment: float = 0.5,
    save_csv: bool = True,
    csv_path: str = None
) → List[Dict]
```

Calculates the distance of a pitch to the strike zone in a video, as well as other information about the Play ID.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `start_date` | str | Start date of the analysis (YYYY-MM-DD) | Required |
| `end_date` | str | End date of the analysis (YYYY-MM-DD) | Required |
| `team_abbr` | str | Team abbreviation to filter by | None |
| `pitch_type` | str | Pitch type to filter by (e.g., "FF" for fastball) | None |
| `player` | int | Player ID to filter by | None |
| `max_videos` | int | Maximum number of videos to analyze | None |
| `max_videos_per_game` | int | Maximum videos per game to analyze | None |
| `create_video` | bool | Whether to create analysis video | True |
| `catcher_model` | str | PHCDetector model name | 'phc_detector' |
| `glove_model` | str | GloveTracking model name | 'glove_tracking' |
| `ball_model` | str | BallTracking model name | 'ball_trackingv4' |
| `zone_vertical_adjustment` | float | Factor to adjust strike zone vertically | 0.5 |
| `save_csv` | bool | Whether to save analysis results to CSV | True |
| `csv_path` | str | Custom path for CSV file | None |

**Returns**

List[Dict] containing for each video:
- `video_name`: Name of the video file
- `play_id`: ID of the play from Baseball Savant
- `game_pk`: Game ID from Baseball Savant
- `ball_glove_frame`: Frame where ball reaches glove
- `ball_center`: Coordinates of ball center
- `strike_zone`: Strike zone coordinates (left, top, right, bottom)
- `distance_to_zone`: Distance from ball to strike zone
- `position`: Position relative to strike zone ("high", "low", "inside", "outside")
- `annotated_video`: Path to the annotated video if create_video is True
- `in_zone`: Whether the pitch is in the strike zone

**Example**

```python
tools = BaseballTools(device='cuda')
results = tools.distance_to_zone(
    start_date="2024-05-01",
    end_date="2024-05-02",
    team_abbr="NYY",
    create_video=True
)
```

##### track_gloves

```python
track_gloves(
    mode: str = "regular",
    device: str = None,
    confidence_threshold: float = 0.5,
    enable_filtering: bool = True,
    max_velocity_inches_per_sec: float = 120.0,
    show_plot: bool = True,
    generate_heatmap: bool = True,
    create_video: bool = True,
    video_path: str = None,
    output_path: str = None,
    input_folder: str = None,
    delete_after_processing: bool = False,
    skip_confirmation: bool = False,
    max_workers: int = 1,
    generate_batch_info: bool = True,
    start_date: str = None,
    end_date: str = None,
    team_abbr: str = None,
    player: int = None,
    pitch_type: str = None,
    max_videos: int = 10,
    suppress_detection_warnings: bool = False,
    max_videos_per_game: int = None
) → Dict
```

Track the catcher's glove, home plate, and baseball in videos using one of three modes.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `mode` | str | Processing mode - "regular", "batch", or "scrape" | "regular" |
| `device` | str | Device to run the model on | None |
| `confidence_threshold` | float | Confidence threshold for detections | 0.5 |
| `enable_filtering` | bool | Enable filtering of outlier detections | True |
| `max_velocity_inches_per_sec` | float | Maximum plausible velocity for filtering | 120.0 |
| `show_plot` | bool | Show 2D tracking plot in output video | True |
| `generate_heatmap` | bool | Generate heatmap of glove positions | True |
| `create_video` | bool | Create an output video file | True |
| `video_path` | str | Path to input video (for regular mode) | None |
| `output_path` | str | Path for output video (for regular mode) | None |
| `input_folder` | str | Folder with videos (for batch mode) | None |
| `delete_after_processing` | bool | Delete videos after processing | False |
| `skip_confirmation` | bool | Skip deletion confirmation dialog | False |
| `max_workers` | int | Maximum parallel workers for batch mode | 1 |
| `generate_batch_info` | bool | Generate batch summary info | True |
| `start_date` | str | Start date for scrape mode (YYYY-MM-DD) | None |
| `end_date` | str | End date for scrape mode (YYYY-MM-DD) | None |
| `team_abbr` | str | Team abbreviation for scrape mode | None |
| `player` | int | Player ID for scrape mode | None |
| `pitch_type` | str | Pitch type for scrape mode | None |
| `max_videos` | int | Maximum videos to download in scrape mode | 10 |
| `suppress_detection_warnings` | bool | Suppress detection warning messages | False |
| `max_videos_per_game` | int | Maximum videos per game in scrape mode | None |

**Returns**

Dict containing:
- For regular mode:
  - `output_video`: Path to output video
  - `tracking_data`: Path to CSV with tracking data
  - `movement_stats`: Statistics about glove movement
  - `heatmap`: Path to glove position heatmap
- For batch mode:
  - `processed_videos`: Number of processed videos
  - `individual_results`: List of results for each video
  - `combined_csv`: Path to combined tracking data
  - `summary_file`: Path to summary statistics file
  - `combined_heatmap`: Path to combined heatmap
- For scrape mode:
  - Same as batch mode plus scrape information

**Example**

```python
# Regular mode - process a single video
tools = BaseballTools(device='cuda')
result = tools.track_gloves(
    mode="regular",
    video_path="catcher_video.mp4"
)

# Batch mode - process multiple videos
result = tools.track_gloves(
    mode="batch",
    input_folder="catcher_videos",
    max_workers=4
)

# Scrape mode - download and process videos
result = tools.track_gloves(
    mode="scrape",
    start_date="2024-05-01",
    end_date="2024-05-02",
    team_abbr="NYY",
    max_videos=5
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

#### Constructor

```python
BaseballSavVideoScraper(
    start_dt: str,
    end_dt: str = None,
    player: int = None,
    team_abbr: str = None,
    pitch_type: str = None,
    download_folder: str = 'savant_videos',
    max_return_videos: int = 10,
    max_videos_per_game: int = None
)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `start_dt` | str | Start date (YYYY-MM-DD) | Required |
| `end_dt` | str | End date (YYYY-MM-DD) | None |
| `player` | int | Player ID to filter by | None |
| `team_abbr` | str | Team abbreviation to filter by | None |
| `pitch_type` | str | Pitch type to filter by | None |
| `download_folder` | str | Folder for downloaded videos | 'savant_videos' |
| `max_return_videos` | int | Maximum videos to return | 10 |
| `max_videos_per_game` | int | Maximum videos per game | None |

#### Methods

##### run_executor

```python
run_executor() → None
```

Run multithreaded video downloading process.

##### get_play_ids_df

```python
get_play_ids_df() → pd.DataFrame
```

Returns a pandas DataFrame of the extracted play IDs and associated data.

##### cleanup_savant_videos

```python
cleanup_savant_videos() → None
```

Deletes the download folder directory.

### DataTools
{: .d-inline-block }

Dataset Creation
{: .label .label-blue }

```python
from baseballcv.functions import DataTools
```

Class for generating and processing datasets for computer vision tasks in baseball.

#### Constructor

```python
DataTools()
```

#### Methods

##### generate_photo_dataset

```python
generate_photo_dataset(
    output_frames_folder: str = "cv_dataset",
    video_download_folder: str = "raw_videos",
    max_plays: int = 10,
    max_num_frames: int = 6000,
    max_videos_per_game: int = 10,
    start_date: str = "2024-05-22",
    end_date: str = "2024-07-25",
    delete_savant_videos: bool = True,
    use_savant_scraper: bool = True,
    input_video_folder: str = None,
    use_supervision: bool = False,
    frame_stride: int = 30
) → str
```

Extracts random frames from baseball videos to create a photo dataset.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `output_frames_folder` | str | Folder to save photos | "cv_dataset" |
| `video_download_folder` | str | Folder for videos | "raw_videos" |
| `max_plays` | int | Maximum plays to download | 10 |
| `max_num_frames` | int | Maximum frames to extract | 6000 |
| `max_videos_per_game` | int | Max videos per game | 10 |
| `start_date` | str | Start date (YYYY-MM-DD) | "2024-05-22" |
| `end_date` | str | End date (YYYY-MM-DD) | "2024-07-25" |
| `delete_savant_videos` | bool | Delete videos after extraction | True |
| `use_savant_scraper` | bool | Use savant scraper | True |
| `input_video_folder` | str | Custom video folder | None |
| `use_supervision` | bool | Use supervision library | False |
| `frame_stride` | int | Frame skip for supervision | 30 |

**Returns**

str: Path to the folder containing the extracted frames

##### automated_annotation

```python
automated_annotation(
    model_alias: str = None,
    model_type: str = 'detection',
    image_dir: str = "cv_dataset",
    output_dir: str = "labeled_dataset",
    conf: float = 0.80,
    device: str = 'cpu',
    mode: str = 'autodistill',
    ontology: dict = None,
    extension: str = '.jpg'
) → str
```

Automatically annotates images using pre-trained models.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `model_alias` | str | Alias of model to use | None |
| `model_type` | str | Type of CV model | 'detection' |
| `image_dir` | str | Directory with images | "cv_dataset" |
| `output_dir` | str | Directory for output | "labeled_dataset" |
| `conf` | float | Confidence threshold | 0.80 |
| `device` | str | Device to run model on | 'cpu' |
| `mode` | str | Annotation mode | 'autodistill' |
| `ontology` | dict | Ontology for autodistill | None |
| `extension` | str | Image file extension | '.jpg' |

**Returns**

str: Path to the output directory with annotated data

### LoadTools
{: .d-inline-block }

Model & Dataset Loading
{: .label .label-yellow }

```python
from baseballcv.functions import LoadTools
```

Class for downloading and loading models and datasets.

#### Constructor

```python
LoadTools()
```

#### Methods

##### load_model

```python
load_model(
    model_alias: str = None,
    model_type: str = 'YOLO',
    use_bdl_api: bool = True,
    model_txt_path: str = None
) → str
```

Loads a baseball computer vision model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `model_alias` | str | Model alias to load | None |
| `model_type` | str | Model type ('YOLO', 'FLORENCE2', 'PALIGEMMA2', 'DETR', 'RFDETR') | 'YOLO' |
| `use_bdl_api` | bool | Use BallDataLab API | True |
| `model_txt_path` | str | Path to download link file | None |

**Available model aliases:**
- YOLO: 'phc_detector', 'bat_tracking', 'ball_tracking', 'glove_tracking', 'ball_trackingv4', 'amateur_pitcher_hitter', 'homeplate_tracking'
- Florence2: 'ball_tracking', 'florence_ball_tracking'
- PaliGemma2: 'paligemma2_ball_tracking'
- DETR: 'detr_baseball_v2'
- RFDETR: 'rfdetr_glove_tracking'

**Returns**

str: Path to saved model weights

##### load_dataset

```python
load_dataset(
    dataset_alias: str,
    use_bdl_api: bool = True,
    file_txt_path: str = None
) → str
```

Loads and extracts a dataset.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `dataset_alias` | str | Dataset alias to load | Required |
| `use_bdl_api` | bool | Use BallDataLab API | True |
| `file_txt_path` | str | Path to download link file | None |

**Available dataset aliases:**
- YOLO format: 'okd_nokd', 'baseball_rubber_home_glove', 'baseball_rubber_home', 'baseball', 'phc', 'amateur_pitcher_hitter'
- COCO format: 'baseball_rubber_home_COCO', 'baseball_rubber_home_glove_COCO'
- JSONL format: 'amateur_hitter_pitcher_jsonl'
- Raw photos: 'broadcast_10k_frames', 'broadcast_15k_frames'
- HuggingFace datasets: 'international_amateur_baseball_catcher_photos', 'international_amateur_baseball_catcher_video', 'international_amateur_baseball_photos', 'international_amateur_baseball_game_video', 'international_amateur_baseball_bp_video', 'international_amateur_pitcher_photo'

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

##### inference

```python
inference(
    source: str | List[str],
    imgsz: tuple = (640, 640),
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 1000,
    **kwargs
) → List[Dict]
```

Run inference with YOLOv9 model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `source` | str \| List[str] | Path(s) to image or video | Required |
| `imgsz` | tuple | Input image size | (640, 640) |
| `conf_thres` | float | Confidence threshold | 0.25 |
| `iou_thres` | float | IoU threshold | 0.45 |
| `max_det` | int | Maximum detections per image | 1000 |

Additional kwargs:
- `view_img` (bool): Show results. Default: False
- `save_txt` (bool): Save results to *.txt. Default: False
- `save_conf` (bool): Save confidences in --save-txt labels. Default: False
- `save_crop` (bool): Save cropped prediction boxes. Default: False
- `hide_labels` (bool): Hide labels. Default: False
- `hide_conf` (bool): Hide confidences. Default: False
- `vid_stride` (int): Video frame-rate stride. Default: 1

**Returns**

List[Dict]: List of dictionaries containing detection results

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

Finetune a YOLOv9 model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `data_path` | str | Path to data config file | Required |
| `epochs` | int | Number of training epochs | 100 |
| `imgsz` | int | Image size | 640 |
| `batch_size` | int | Batch size | 16 |

**Returns**

Dict: Training results

##### evaluate

```python
evaluate(
    data_path: str,
    batch_size: int = 32,
    imgsz: int = 640,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    max_det: int = 300,
    **kwargs
) → Tuple
```

Evaluate YOLOv9 model performance.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `data_path` | str | Path to data config file | Required |
| `batch_size` | int | Batch size | 32 |
| `imgsz` | int | Image size | 640 |
| `conf_thres` | float | Confidence threshold | 0.001 |
| `iou_thres` | float | IoU threshold | 0.7 |
| `max_det` | int | Maximum detections | 300 |

**Returns**

Tuple: Evaluation metrics including mAP values

### RFDETR
{: .d-inline-block }

Object Detection
{: .label .label-blue }

```python
from baseballcv.model import RFDETR
```

RF DETR implementation for object detection.

#### Constructor

```python
RFDETR(
    device: str = "cpu",
    model_path: str = None,
    imgsz: int = 560,
    model_type: str = "base",
    labels: List[str] = None,
    project_path: str = "rfdetr_runs"
)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `device` | str | Device for inference | "cpu" |
| `model_path` | str | Path to model weights | None |
| `imgsz` | int | Image size | 560 |
| `model_type` | str | Model type ("base" or "large") | "base" |
| `labels` | List[str] | Class labels | None |
| `project_path` | str | Output directory | "rfdetr_runs" |

#### Methods

##### inference

```python
inference(
    source_path: str,
    conf: float = 0.2,
    save_viz: bool = True
) → Tuple[List, str]
```

Run inference with RF DETR model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `source_path` | str | Path to image or video | Required |
| `conf` | float | Confidence threshold | 0.2 |
| `save_viz` | bool | Save visualization | True |

**Returns**

Tuple: List of detections and path to output visualization

##### finetune

```python
finetune(
    data_path: str,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 0.0001,
    lr_encoder: float = 0.00015,
    **kwargs
) → RFDETR
```

Finetune RF DETR model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `data_path` | str | Path to dataset | Required |
| `epochs` | int | Number of epochs | 50 |
| `batch_size` | int | Batch size | 4 |
| `lr` | float | Learning rate | 0.0001 |
| `lr_encoder` | float | Encoder learning rate | 0.00015 |

**Returns**

RFDETR: Trained model instance

### Florence2
{: .d-inline-block }

Vision Language Model
{: .label .label-purple }

```python
from baseballcv.model import Florence2
```

Florence2 vision language model for baseball analysis.

#### Constructor

```python
Florence2(
    model_id: str = 'microsoft/Florence-2-large',
    model_run_path: str = f'florence2_run_{datetime.now().strftime("%Y%m%d")}',
    batch_size: int = 1
)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `model_id` | str | HuggingFace model ID | 'microsoft/Florence-2-large' |
| `model_run_path` | str | Output directory | Timestamped |
| `batch_size` | int | Batch size | 1 |

#### Methods

##### inference

```python
inference(
    image_path: str,
    task: str = "<OD>",
    text_input: str = None,
    question: str = None
) → Union[str, Dict]
```

Run inference with Florence2 model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `image_path` | str | Path to image | Required |
| `task` | str | Task type | "<OD>" |
| `text_input` | str | Text input | None |
| `question` | str | Question for VQA | None |

**Available tasks:**
- `<OD>`: Object detection
- `<CAPTION>`: Image captioning
- `<DETAILED_CAPTION>`: Detailed image captioning
- `<MORE_DETAILED_CAPTION>`: Very detailed image captioning
- `<VQA>`: Visual question answering
- `<OPEN_VOCABULARY_DETECTION>`: Open vocabulary detection

**Returns**

Union[str, Dict]: Inference results based on the task

##### finetune

```python
finetune(
    dataset: str,
    classes: Dict[int, str],
    train_test_split: Tuple[int, int, int] = (80, 10, 10),
    epochs: int = 20,
    lr: float = 4e-6,
    **kwargs
) → Dict
```

Finetune Florence2 model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `dataset` | str | Path to dataset | Required |
| `classes` | Dict[int, str] | Class ID to name mapping | Required |
| `train_test_split` | Tuple[int, int, int] | Training/Test/Validation split | (80, 10, 10) |
| `epochs` | int | Number of epochs | 20 |
| `lr` | float | Learning rate | 4e-6 |

Additional kwargs:
- `save_dir` (str): Directory to save model. Default: "model_checkpoints"
- `lora_r` (int): LoRA rank. Default: 8
- `lora_scaling` (int): LoRA scaling factor. Default: 8
- `patience` (int): Early stopping patience. Default: 5
- `lora_dropout` (float): LoRA dropout. Default: 0.05
- `warmup_epochs` (int): Warmup epochs. Default: 1
- `gradient_accumulation_steps` (int): Gradient accumulation steps. Default: 2
- `create_peft_config` (bool): Create PEFT config. Default: True

**Returns**

Dict: Training metrics and model information

### PaliGemma2
{: .d-inline-block }

Vision Language Model
{: .label .label-purple }

```python
from baseballcv.model import PaliGemma2
```

PaliGemma2 vision language model for baseball analysis.

#### Constructor

```python
PaliGemma2(
    device: str = None,
    model_id: str = 'google/paligemma2-3b-pt-224',
    model_run_path: str = f'paligemma2_run_{datetime.now().strftime("%Y%m%d")}',
    batch_size: int = 8,
    torch_dtype: torch.dtype = torch.float32,
    use_pretrained_lora: bool = False
)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `device` | str | Device for inference | None (auto) |
| `model_id` | str | HuggingFace model ID | 'google/paligemma2-3b-pt-224' |
| `model_run_path` | str | Output directory | Timestamped |
| `batch_size` | int | Batch size | 8 |
| `torch_dtype` | torch.dtype | Torch data type | torch.float32 |
| `use_pretrained_lora` | bool | Use pretrained LoRA weights | False |

#### Methods

##### inference

```python
inference(
    image_path: str,
    text_input: str,
    task: str = "<TEXT_TO_TEXT>",
    classes: List[str] = None
) → Tuple[str, str]
```

Run inference with PaliGemma2 model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `image_path` | str | Path to image | Required |
| `text_input` | str | Text input | Required |
| `task` | str | Task type | "<TEXT_TO_TEXT>" |
| `classes` | List[str] | Class names for detection | None |

**Available tasks:**
- `<TEXT_TO_TEXT>`: General text generation
- `<TEXT_TO_OD>`: Object detection

**Returns**

Tuple[str, str]: Generated text and path to visualization (for detection)

##### finetune

```python
finetune(
    dataset: str,
    classes: Dict[int, str],
    train_test_split: Tuple[int, int, int] = (80, 10, 10),
    freeze_vision_encoders: bool = False,
    epochs: int = 20,
    lr: float = 4e-6,
    **kwargs
) → Dict
```

Finetune PaliGemma2 model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `dataset` | str | Path to dataset | Required |
| `classes` | Dict[int, str] | Class ID to name mapping | Required |
| `train_test_split` | Tuple[int, int, int] | Training/Test/Validation split | (80, 10, 10) |
| `freeze_vision_encoders` | bool | Freeze vision encoders | False |
| `epochs` | int | Number of epochs | 20 |
| `lr` | float | Learning rate | 4e-6 |

Additional kwargs: Similar to Florence2.finetune

**Returns**

Dict: Training metrics and model information

### DETR
{: .d-inline-block }

Object Detection
{: .label .label-blue }

```python
from baseballcv.model import DETR
```

DETR (DEtection TRansformer) for object detection tasks.

#### Constructor

```python
DETR(
    num_labels: int,
    device: str = None,
    model_id: str = "facebook/detr-resnet-50",
    model_run_path: str = f'detr_run_{datetime.now().strftime("%Y%m%d")}',
    batch_size: int = 8,
    image_size: Tuple[int, int] = (800, 800)
)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `num_labels` | int | Number of object classes | Required |
| `device` | str | Device for inference | None (auto) |
| `model_id` | str | HuggingFace model ID | "facebook/detr-resnet-50" |
| `model_run_path` | str | Output directory | Timestamped |
| `batch_size` | int | Batch size | 8 |
| `image_size` | Tuple[int, int] | Input image size | (800, 800) |

#### Methods

##### finetune

```python
finetune(
    dataset_dir: str,
    classes: dict,
    save_dir: str = "finetuned_detr",
    batch_size: int = 4,
    epochs: int = 10,
    lr: float = 1e-4,
    lr_backbone: float = 1e-5,
    **kwargs
) → Dict
```

Finetune DETR model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `dataset_dir` | str | Path to dataset | Required |
| `classes` | dict | Class ID to name mapping | Required |
| `save_dir` | str | Directory to save model | "finetuned_detr" |
| `batch_size` | int | Batch size | 4 |
| `epochs` | int | Number of epochs | 10 |
| `lr` | float | Learning rate | 1e-4 |
| `lr_backbone` | float | Backbone learning rate | 1e-5 |

**Returns**

Dict: Training results and model paths

##### inference

```python
inference(
    file_path: str,
    classes: dict = None,
    conf: float = 0.2,
    save: bool = False,
    save_viz_dir: str = 'visualizations',
    show_video: bool = False
) → List[Dict]
```

Run inference with DETR model.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `file_path` | str | Path to image or video | Required |
| `classes` | dict | Class ID to name mapping | None |
| `conf` | float | Confidence threshold | 0.2 |
| `save` | bool | Save visualization | False |
| `save_viz_dir` | str | Directory for visualizations | 'visualizations' |
| `show_video` | bool | Display video during inference | False |

**Returns**

List[Dict]: Detection results

##### evaluate

```python
evaluate(
    dataset_dir: str,
    conf: float = 0.2
) → Dict
```

Evaluate DETR model performance.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `dataset_dir` | str | Path to dataset | Required |
| `conf` | float | Confidence threshold | 0.2 |

**Returns**

Dict: Evaluation metrics

## Datasets Module
{: .d-inline-block }

baseballcv.datasets
{: .label .label-green }

The datasets module provides classes for processing and working with baseball datasets.

### DataProcessor
{: .d-inline-block }

Dataset Processing
{: .label .label-yellow }

```python
from baseballcv.datasets import DataProcessor
```

Class for processing and converting between different dataset formats.

#### Constructor

```python
DataProcessor(logger=None)
```

#### Methods

##### prepare_dataset

```python
prepare_dataset(
    base_path: str,
    dict_classes: Dict[int, str],
    train_test_split: Tuple[int, int, int] = (80, 10, 10),
    dataset_type: str = "yolo"
) → Tuple[str, str, str, str, str]
```

Prepare dataset for use with models.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `base_path` | str | Path to dataset | Required |
| `dict_classes` | Dict[int, str] | Class ID to name mapping | Required |
| `train_test_split` | Tuple[int, int, int] | Training/Test/Validation split | (80, 10, 10) |
| `dataset_type` | str | Dataset type ("yolo", "coco", etc.) | "yolo" |

**Returns**

Tuple[str, str, str, str, str]: Paths to train images, valid images, train annotations, test annotations, valid annotations

## Utilities Module
{: .d-inline-block }

baseballcv.utilities
{: .label .label-green }

Utility classes and functions for BaseballCV.

### BaseballCVLogger
{: .d-inline-block }

Logging
{: .label .label-yellow }

```python
from baseballcv.utilities import BaseballCVLogger
```

Logger for BaseballCV applications.

#### Methods

##### get_logger

```python
get_logger(name: str = None, **kwargs) → BaseballCVLogger
```

Get or create a logger instance.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `name` | str | Logger name | None |
| `**kwargs` | dict | Additional arguments | |

**Returns**

BaseballCVLogger: Logger instance

### ProgressBar
{: .d-inline-block }

Progress Display
{: .label .label-yellow }

```python
from baseballcv.utilities import ProgressBar
```

Custom progress bar for BaseballCV operations.

#### Constructor

```python
ProgressBar(
    iterable=None,
    total: int = None,
    desc: str = "Processing",
    unit: str = "it",
    color: str = "green",
    disable: bool = False,
    bar_format: str = None,
    postfix: dict = None,
    initial: int = 0
)
```

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `iterable` | Any | Iterable to track | None |
| `total` | int | Total items to process | None |
| `desc` | str | Description of the task | "Processing" |
| `unit` | str | Unit of items | "it" |
| `color` | str | Color of the progress bar | "green" |
| `disable` | bool | Whether to disable display | False |
| `bar_format` | str | Custom format string | None |
| `postfix` | dict | Additional info to display | None |
| `initial` | int | Initial counter value | 0 |

#### Methods

##### update

```python
update(n: int = 1, postfix: dict = None) → None
```

Update progress bar.

| Parameter | Type | Description | Default |
|:----------|:-----|:------------|:--------|
| `n` | int | Increment amount | 1 |
| `postfix` | dict | Additional display info | None |

## Version Information
{: .d-inline-block }

Package Info
{: .label .label-green }

Current package version: 0.1.21
