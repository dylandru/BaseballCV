---
layout: default
title: FAQ
nav_order: 12
has_children: false
permalink: /faq
---

# Frequently Asked Questions

This section addresses common questions about BaseballCV. If you don't find your question here, please join our [Discord community](https://discord.com/channels/1295073049087053874/1295073049518932042) for additional support.

## Installation and Setup

### What are the minimum hardware requirements for BaseballCV?

BaseballCV can run on various hardware configurations, though requirements depend on your specific use case. For basic usage and development, we recommend:
- CPU: 4+ cores
- RAM: 16GB minimum
- Storage: 50GB free space
- GPU: While not required, a CUDA-capable GPU significantly improves performance

For production environments processing large amounts of video, we recommend:
- CPU: 8+ cores
- RAM: 32GB minimum
- Storage: 200GB+ free space
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)

### Why am I getting CUDA errors when installing BaseballCV?

CUDA errors typically occur due to version mismatches between PyTorch and your CUDA installation. To resolve this:

1. First, check your CUDA version:
```bash
nvidia-smi
```

2. Install PyTorch with the matching CUDA version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

3. Then install BaseballCV:
```bash
pip install git+https://github.com/dylandru/BaseballCV.git
```

### Can I use BaseballCV without a GPU?

Yes, BaseballCV will automatically run on CPU if no GPU is available. However, processing will be significantly slower, especially for video analysis. For development and testing, CPU-only operation is feasible, but for production use, we strongly recommend GPU acceleration.

## Models and Usage

### Which model should I use for ball tracking?

BaseballCV offers several models for ball tracking, each with its strengths:

1. `ball_trackingv4` (YOLO): Best for real-time tracking and general use
2. `florence_ball_tracking`: Better for challenging scenarios where context is important
3. `paligemma2_ball_tracking`: Excellent for detailed analysis where speed is less critical

For most applications, we recommend starting with `ball_trackingv4` and only moving to the more complex models if you need additional capabilities.

### How can I improve ball detection accuracy?

To improve ball detection accuracy:

1. Optimize your confidence threshold:

```python
from baseballcv.functions import LoadTools
from ultralytics import YOLO

load_tools = LoadTools()
model = YOLO(load_tools.load_model("ball_trackingv4"))
results = model.predict(
    source=video_path,
    conf=0.25  # Adjust this value based on your needs
)
```

2. Use multiple frames for validation:

```python
from baseballcv.functions import DataTools

data_tools = DataTools()
frames = data_tools.generate_photo_dataset(
    video_path,
    max_num_frames=90  # Capture full pitch sequence
)
```

3. Consider combining multiple models for verification.

### What image resolution should I use?

BaseballCV works best with HD resolution (1280x720) videos, which matches our training datasets. While higher resolutions are supported, they may not improve accuracy and will increase processing time. Lower resolutions might work but could reduce detection accuracy, especially for fast-moving balls.

## Data and Datasets

### How do I create my own training dataset?

BaseballCV provides tools for creating custom datasets:

1. First, collect video footage:
```python
from baseballcv.scripts import BaseballSavVideoScraper

scraper = BaseballSavVideoScraper()
scraper.run_statcast_pull_scraper(
    start_date="2024-05-01",
    end_date="2024-05-31",
    max_videos=100
)
```

2. Then generate frames and annotations:

```python
from baseballcv.functions import DataTools

data_tools = DataTools()
data_tools.generate_photo_dataset(
    output_frames_folder="custom_dataset",
    max_num_frames=1000
)
```

### Why are some frames missing ball detections?

Ball detection can fail for several reasons:
- Motion blur in fast pitches
- Ball partially obscured
- Poor lighting conditions
- Ball too small in the frame

To mitigate these issues, BaseballCV uses temporal information from multiple frames to maintain tracking consistency.

## Performance and Optimization

### How can I speed up video processing?

To optimize processing speed:

1. Use GPU acceleration when available
2. Process frames in batches
3. Adjust frame extraction rate based on your needs
4. Use appropriate model for your use case (YOLO for speed, VLMs for accuracy)

### What's the maximum video length I can process?

There's no hard limit on video length, but practical limitations come from:
- Available memory
- Processing time requirements
- Storage space for extracted frames

For long videos, consider processing in segments using BaseballCV's data tools.

## Integration and Development

### Can I use BaseballCV with existing baseball analytics systems?

Yes, BaseballCV is designed to integrate with other systems. The framework provides:
- Standard data formats (YOLO, COCO)
- Integration with Baseball Savant
- Easy export of analysis results

### How do I contribute new features?

We welcome contributions! Start by:

1. Reading our [Contributing Guide](../contributing/)
2. Joining our Discord community
3. Opening an issue to discuss your proposed changes
4. Following our code style and testing guidelines

## Troubleshooting

### Why is my model downloading failing?

Common reasons for download failures:
- Network connectivity issues
- Insufficient disk space
- Invalid model alias

Try using the direct download URL from our documentation if the API download fails.

### What should I do if I find a bug?

If you encounter a bug:

1. Check if it's a known issue on our GitHub
2. Gather relevant information (error messages, system info)
3. Create a minimal reproducible example
4. Open an issue on GitHub with these details
5. Join our Discord for quick support

## Updates and Versioning

### How often is BaseballCV updated?

We maintain regular updates focusing on:
- Bug fixes
- Performance improvements
- New model releases
- Dataset expansions

Check our GitHub repository for the latest releases and changes.

### Will my existing code break with updates?

We strive to maintain backward compatibility. Major version changes (e.g., 1.0 to 2.0) may include breaking changes, but these are clearly documented in release notes. We recommend:
- Pinning dependency versions in production
- Testing upgrades in development first
- Following our changelog for breaking changes