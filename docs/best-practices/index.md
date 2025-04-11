---
layout: default
title: Best Practices
nav_order: 10
has_children: false
permalink: /best-practices
---

# Best Practices for BaseballCV

Understanding and following best practices ensures you get the most out of BaseballCV while maintaining efficient and reliable analysis pipelines. This guide covers essential guidelines and tips for working with the framework effectively.

## Model Selection

### Choosing the Right Model

BaseballCV offers several models optimized for different baseball analysis tasks. Selecting the appropriate model depends on your specific requirements:

```python
from baseballcv.functions import LoadTools
from baseballcv.model import Florence2, PaliGemma2, YOLOv9, DETR, RFDETR
from ultralytics import YOLO

def select_appropriate_model(use_case: str):
    """
    Choose the appropriate model based on use case requirements.
    """
    load_tools = LoadTools()
    
    if use_case == "real_time_ball_tracking":
        # YOLO for fast ball tracking
        return YOLO(load_tools.load_model("ball_trackingv4"))
    
    elif use_case == "advanced_ball_tracking":
        # YOLOv9 for improved accuracy and performance
        return YOLOv9(device="cuda", name="yolov9-c")
    
    elif use_case == "player_detection":
        # DETR for precise player detection
        return DETR(num_labels=4, device="cuda")
    
    elif use_case == "glove_tracking":
        # RFDETR for specialized glove tracking
        return RFDETR(device="cuda", labels=["glove", "homeplate", "baseball"])
    
    elif use_case == "scene_understanding":
        # Florence2 for contextual understanding
        return Florence2()
    
    elif use_case == "natural_language_queries":
        # PaliGemma2 for complex queries
        return PaliGemma2()
```

| Model Type | Strengths | Best For | Memory Requirements |
|:-----------|:----------|:---------|:--------------------|
| YOLO | Speed, good accuracy | Real-time tracking | Low-medium |
| YOLOv9 | Better accuracy, still fast | Enhanced object detection | Medium |
| DETR | Precise bounding boxes | Detailed player detection | Medium-high |
| RFDETR | Specialized detection | Precise glove tracking | Medium-high |
| Florence2 | Visual understanding | Scene analysis, context | High |
| PaliGemma2 | Language + vision | Natural language queries | High |

## Data Collection & Processing

### Video Quality Considerations

For optimal ball and player tracking:

```python
from baseballcv.functions import BaseballSavVideoScraper
from baseballcv.functions import DataTools

def prepare_quality_dataset():
    """
    Collect and prepare high-quality video data using best practices.
    """
    # 1. Collect diverse, high-quality footage
    scraper = BaseballSavVideoScraper(
        start_dt="2024-05-01",
        end_dt="2024-05-10",
        # Gather diverse samples across games and teams
        max_videos_per_game=3,
        max_return_videos=30
    )
    scraper.run_executor()
    
    # 2. Generate balanced dataset with optimal settings
    data_tools = DataTools()
    frames = data_tools.generate_photo_dataset(
        output_frames_folder="high_quality_dataset",
        video_download_folder="savant_videos",
        max_num_frames=3000,
        # Use random sampling to avoid temporal bias
        frame_stride=30
    )
    
    return frames
```

**Video Quality Recommendations:**
- **Resolution:** 1280x720 pixels or higher
- **Frame Rate:** 30fps or higher
- **Camera Angle:** Consistent center field or side view
- **Lighting:** Well-lit scenes with minimal shadows
- **Focus:** Clear, sharp footage for accurate ball tracking

### Efficient Frame Extraction

When extracting frames for analysis:

```python
def optimize_frame_extraction(video_path, analysis_type):
    """
    Extract frames optimized for specific analysis types.
    """
    data_tools = DataTools()
    
    if analysis_type == "pitch_trajectory":
        # For pitch trajectory analysis, extract every frame during pitch
        frames = data_tools.generate_photo_dataset(
            output_frames_folder="trajectory_frames",
            input_video_folder=video_path,
            use_savant_scraper=False,
            # Dense sampling for smooth trajectories
            frame_stride=1
        )
    
    elif analysis_type == "player_positioning":
        # For player positioning, we need fewer frames
        frames = data_tools.generate_photo_dataset(
            output_frames_folder="player_position_frames",
            input_video_folder=video_path,
            use_savant_scraper=False,
            # Sparser sampling for positioning
            frame_stride=10
        )
        
    return frames
```

## Analysis Pipeline Design

### GloveTracker Optimization

When using the GloveTracker functionality, consider these best practices:

```python
from baseballcv.functions import BaseballTools

def optimize_glove_tracking(video_path):
    """
    Optimize glove tracking for accuracy and performance.
    """
    baseball_tools = BaseballTools(device="cuda")
    
    # Use regular mode for single video analysis
    results = baseball_tools.track_gloves(
        mode="regular",
        video_path=video_path,
        # Balance between performance and accuracy
        confidence_threshold=0.5,
        # Filter out physically impossible movements
        enable_filtering=True,
        max_velocity_inches_per_sec=120.0,
        # Generate visualization and analysis artifacts
        show_plot=True,
        generate_heatmap=True
    )
    
    # Analyze movement statistics
    stats = results['movement_stats']
    
    # Check key metrics
    glove_stability = stats['avg_distance_between_frames_inches']
    total_range = (stats['x_range_inches']**2 + stats['y_range_inches']**2)**0.5
    
    # Report findings
    metrics = {
        'stability': 'high' if glove_stability < 0.5 else 'medium' if glove_stability < 1.0 else 'low',
        'range_of_motion': total_range,
        'tracking_quality': 'good' if stats['frames_with_glove'] / stats['total_frames'] > 0.7 else 'fair'
    }
    
    return metrics
```

**GloveTracker Tips:**
- Use `enable_filtering=True` to remove unrealistic glove jumps
- Adjust `max_velocity_inches_per_sec` based on the level of play (lower for amateurs)
- Use batch mode with `max_workers` set to CPU cores - 1 for parallel processing
- Generate heatmaps for positional analysis
- Set appropriate `confidence_threshold` (0.4-0.6 recommended)

### DistanceToZone Best Practices

For strike zone analysis:

```python
from baseballcv.functions import BaseballTools

def optimize_zone_analysis(pitcher_id=None, team_abbr=None):
    """
    Optimize distance to zone analysis for accurate strike zone assessment.
    """
    baseball_tools = BaseballTools(device="cuda")
    
    # Filter by specific pitcher or team
    results = baseball_tools.distance_to_zone(
        start_date="2024-05-01",
        end_date="2024-05-02",
        team_abbr=team_abbr,
        player=pitcher_id,
        # Use ball tracking v4 for improved accuracy
        ball_model="ball_trackingv4",
        # Adjust zone vertical position based on camera angle
        zone_vertical_adjustment=0.5,
        # Save results for further analysis
        save_csv=True,
        csv_path="pitch_analysis.csv"
    )
    
    # Analyze results
    in_zone_count = sum(1 for r in results if r.get('in_zone') == True)
    total_pitches = len(results)
    
    # Calculate zone accuracy
    zone_percentage = (in_zone_count / total_pitches) * 100 if total_pitches > 0 else 0
    
    return {
        'total_pitches': total_pitches,
        'in_zone_pitches': in_zone_count,
        'zone_percentage': zone_percentage,
        'detailed_results': results
    }
```

**DistanceToZone Tips:**
- Use a recent PHC model for accurate player detection
- Tune `zone_vertical_adjustment` between 0.3-0.7 based on camera angle
- Filter by specific pitchers for consistent zone calibration
- Always enable CSV output for further analysis
- Use the same models consistently for comparative analysis

## Model Training

### Efficient Fine-tuning

When fine-tuning models on custom baseball data:

```python
from baseballcv.model import DETR, YOLOv9
from baseballcv.functions import LoadTools

def setup_efficient_training(model_type, dataset_path):
    """
    Set up efficient model training based on model type.
    """
    # Define baseball-specific classes
    classes = {
        0: "baseball",
        1: "glove",
        2: "homeplate",
        3: "pitcher",
        4: "batter",
        5: "catcher"
    }
    
    if model_type == "yolov9":
        model = YOLOv9(device="cuda")
        
        # YOLOv9 specific training configuration
        results = model.finetune(
            data_path=dataset_path,
            epochs=100,
            batch_size=16,
            imgsz=640,
            # Optimizer settings
            optimizer='AdamW',
            # Learning rate settings
            lr=0.001,
            # Early stopping
            patience=10,
            # Save settings
            save_period=10
        )
        
    elif model_type == "detr":
        model = DETR(
            num_labels=len(classes),
            device="cuda"
        )
        
        # DETR specific training configuration
        results = model.finetune(
            dataset_dir=dataset_path,
            classes=classes,
            batch_size=4,
            epochs=50,
            lr=1e-4,
            lr_backbone=1e-5,
            weight_decay=0.0001,
            patience=5,
            # Freeze backbone for transfer learning
            freeze_backbone=True,
            # Use mixed precision for faster training
            precision='16-mixed'
        )
    
    return results
```

**Training Best Practices:**
- Start with pre-trained weights when available
- Use smaller batch sizes with gradient accumulation for large models
- Implement early stopping with reasonable patience values
- For object detection, ensure balanced class distribution
- Use validation dataset from similar camera angles as intended target
- Log metrics and visualize predictions during training

## Resource Management

### GPU Memory Optimization

When working with large models on limited GPU resources:

```python
import torch
from baseballcv.model import Florence2, PaliGemma2

def optimize_vlm_inference(image_path, query):
    """
    Optimize vision-language model inference for memory efficiency.
    """
    # Clear GPU cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Choose appropriate model for the query type
    if "what" in query.lower() or "why" in query.lower() or "how" in query.lower():
        # Use Florence2 for factual or technical questions
        model = Florence2(batch_size=1)  # Minimum batch size
        
        result = model.inference(
            image_path=image_path,
            task="<VQA>",
            text_input=query
        )
    else:
        # Use specialized model config for detection tasks
        model = PaliGemma2(
            device="cuda",
            # Use lower precision for memory efficiency
            torch_dtype=torch.float16,
            batch_size=1
        )
        
        result = model.inference(
            image_path=image_path,
            task="<TEXT_TO_OD>", 
            text_input=query,
            classes=["baseball", "glove", "catcher", "pitcher"]
        )
    
    # Clear cache after inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result
```

**Memory Optimization Tips:**
- Use the most efficient model for each task
- Process in batches appropriate to your hardware
- Use 16-bit precision when possible
- Clear GPU cache between operations
- Monitor memory usage with `nvidia-smi` or equivalent
- Consider using CPU for inference if GPU memory is limited

### Multi-processing with Video Analysis

When processing multiple videos:

```python
from baseballcv.functions import BaseballTools
import multiprocessing as mp

def parallel_video_processing(video_folder):
    """
    Process multiple videos in parallel with appropriate resource allocation.
    """
    # Determine optimal worker count based on system
    cpu_count = mp.cpu_count()
    gpu_available = torch.cuda.is_available()
    
    # Conservative worker allocation
    if gpu_available:
        # Limit workers when using GPU to avoid memory issues
        optimal_workers = min(4, max(1, cpu_count // 2))
    else:
        # More workers possible on CPU-only
        optimal_workers = max(1, cpu_count - 1)
    
    baseball_tools = BaseballTools(device="cuda" if gpu_available else "cpu")
    
    # Process videos in batch mode with optimal workers
    results = baseball_tools.track_gloves(
        mode="batch",
        input_folder=video_folder,
        max_workers=optimal_workers,
        # Generate combined analysis
        generate_batch_info=True
    )
    
    return results
```

## Quality Assurance

### Validation of Results

Always validate the results of computer vision analysis before drawing conclusions:

```python
import pandas as pd
import numpy as np
from baseballcv.functions import BaseballTools

def validate_pitch_analysis(team_abbr, date):
    """
    Validate pitch analysis results against expected distributions.
    """
    # Run zone analysis
    baseball_tools = BaseballTools()
    results = baseball_tools.distance_to_zone(
        start_date=date,
        end_date=date,
        team_abbr=team_abbr,
        save_csv=True,
        csv_path="validation_analysis.csv"
    )
    
    # Load detailed results
    df = pd.read_csv("validation_analysis.csv")
    
    # Validation checks
    validation = {
        "total_pitches": len(df),
        "detection_rate": df["ball_center_x"].notna().mean(),
        "zone_rate": df["in_zone"].mean() if "in_zone" in df.columns else None,
        "avg_distance": df["distance_to_zone_inches"].mean() if "distance_to_zone_inches" in df.columns else None
    }
    
    # Sanity checks
    warnings = []
    if validation["detection_rate"] < 0.7:
        warnings.append("Low ball detection rate - check video quality")
    
    if validation["zone_rate"] is not None and (validation["zone_rate"] < 0.3 or validation["zone_rate"] > 0.7):
        warnings.append("Unusual strike zone rate - check zone calibration")
    
    if validation["avg_distance"] is not None and validation["avg_distance"] > 15:
        warnings.append("High average distance - check tracking accuracy")
    
    return {
        "metrics": validation,
        "warnings": warnings,
        "pass": len(warnings) == 0
    }
```

**Validation Best Practices:**
- Check detection rates across models (expect >70% for balls)
- Validate spatial relationships (e.g., glove should be near catcher)
- Compare with known statistics (e.g., strike zone rates)
- Visually inspect sample frames with detections
- Use multiple models when possible for cross-validation

## Integration and Deployment

### Packaging Analysis Pipelines

For production use, package your analysis pipelines into reusable components:

```python
import os
import json
from datetime import datetime
from baseballcv.functions import BaseballTools, DataTools, LoadTools

class BaseballCVPipeline:
    """
    Production-ready baseball analysis pipeline.
    """
    def __init__(self, output_dir="analysis_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load tools
        self.baseball_tools = BaseballTools(device="cuda")
        self.data_tools = DataTools()
        self.load_tools = LoadTools()
        
        # Configure logging
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f"analysis_log_{self.session_id}.json")
        self.log = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "analyses": []
        }
    
    def analyze_team_performance(self, team_abbr, start_date, end_date):
        """
        Comprehensive team performance analysis.
        """
        analysis_id = f"{team_abbr}_{start_date}_to_{end_date}"
        team_dir = os.path.join(self.output_dir, analysis_id)
        os.makedirs(team_dir, exist_ok=True)
        
        # Track analysis
        analysis_record = {
            "id": analysis_id,
            "type": "team_performance",
            "team": team_abbr,
            "date_range": [start_date, end_date],
            "start_time": datetime.now().isoformat(),
            "artifacts": {}
        }
        
        try:
            # 1. Distance to zone analysis
            zone_results = self.baseball_tools.distance_to_zone(
                start_date=start_date,
                end_date=end_date,
                team_abbr=team_abbr,
                save_csv=True,
                csv_path=os.path.join(team_dir, "zone_analysis.csv")
            )
            analysis_record["artifacts"]["zone_analysis"] = os.path.join(team_dir, "zone_analysis.csv")
            
            # 2. Glove tracking analysis
            glove_results = self.baseball_tools.track_gloves(
                mode="scrape",
                start_date=start_date,
                end_date=end_date,
                team_abbr=team_abbr,
                generate_batch_info=True
            )
            
            if "combined_csv" in glove_results:
                analysis_record["artifacts"]["glove_tracking"] = glove_results["combined_csv"]
            
            # 3. Save summary
            analysis_record["completion_time"] = datetime.now().isoformat()
            analysis_record["status"] = "success"
            
        except Exception as e:
            analysis_record["status"] = "error"
            analysis_record["error"] = str(e)
            raise
        finally:
            # Record analysis attempt
            self.log["analyses"].append(analysis_record)
            with open(self.log_file, "w") as f:
                json.dump(self.log, f, indent=2)
        
        return analysis_record
```

By following these best practices, you'll get the most out of BaseballCV while ensuring reliable, efficient, and accurate analysis of baseball footage.
