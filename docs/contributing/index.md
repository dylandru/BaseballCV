---
layout: default
title: Contributing
nav_order: 9
has_children: false
permalink: /contributing
---

# Contributing to BaseballCV

We welcome contributions from the community! This guide explains how to contribute to BaseballCV effectively, whether you're improving code, adding datasets, or helping with documentation.

## Ways to Contribute

There are several ways to contribute to BaseballCV:

1. Code Development
   - Implementing new features
   - Bug fixes
   - Performance improvements
   - Model optimizations

2. Dataset Contributions
   - Adding new annotated datasets
   - Improving existing annotations
   - Validating dataset quality

3. Documentation
   - Improving existing documentation
   - Adding new examples
   - Writing tutorials
   - Fixing typos or unclear explanations

4. Testing
   - Writing unit tests
   - Integration testing
   - Model validation
   - Dataset validation

## Getting Started

### 1. Setting Up Your Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/BaseballCV.git
cd BaseballCV

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
```

### 2. Code Standards

We follow these coding standards:

```python
# Example of proper code style
def process_video(video_path: str, 
                 output_dir: str,
                 max_frames: int = 1000) -> List[str]:
    """
    Process video frames with proper documentation.

    Args:
        video_path (str): Path to input video
        output_dir (str): Directory for output frames
        max_frames (int, optional): Maximum frames to process. Defaults to 1000.

    Returns:
        List[str]: Paths to processed frames
    """
    # Your implementation here
    pass
```

- Use type hints for function arguments and returns
- Provide clear docstrings for all functions
- Follow PEP 8 style guidelines
- Add comments for complex logic
- Keep functions focused and single-purpose

## Development Workflow

### 1. Creating a New Feature

```bash
# Create a new branch
git checkout -b feature/your-feature-name

# Make your changes
# Add tests for new functionality
# Update documentation if needed

# Run tests
pytest tests/

# Submit your pull request
```

### 2. Testing Requirements

All contributions must include appropriate tests:

```python
# Example test file: tests/test_video_processing.py
import pytest
from baseballcv.scripts import extract_frames_from_video

def test_frame_extraction():
    """Test video frame extraction functionality"""
    video_path = "tests/data/test_video.mp4"
    output_dir = "tests/output"
    frames = extract_frames_from_video(
        video_path=video_path,
        output_dir=output_dir,
        max_frames=10
    )
    assert len(frames) == 10
    assert all(frame.endswith('.jpg') for frame in frames)
```

## Contributing Datasets

When contributing new datasets:

1. **Data Organization**
   ```
   dataset_name/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

2. **Annotation Requirements**
   - Follow existing format conventions (YOLO/COCO)
   - Provide class mappings
   - Include dataset statistics
   - Document any special considerations

3. **Dataset Documentation**
   ```python
   # Example dataset documentation
   dataset_info = {
      'name': 'new_baseball_dataset',
      'version': '1.0',
      'classes': {
          0: 'baseball',
          1: 'glove',
          # ... other classes
      },
      'statistics': {
          'total_images': 1000,
          'train_split': 800,
          'valid_split': 100,
          'test_split': 100
      }
   }
   ```

## Documentation Contributions

When contributing to documentation:

1. Write clear, concise explanations
2. Include practical examples
3. Follow Markdown formatting guidelines
4. Test all code examples
5. Update navigation and links as needed

## Pull Request Guidelines

When submitting a pull request:

1. **Title and Description**
   - Clear, descriptive title
   - Detailed description of changes
   - Reference any related issues

2. **Code Review Checklist**
   - [ ] Code follows style guidelines
   - [ ] All tests pass
   - [ ] Documentation is updated
   - [ ] Changes are tested
   - [ ] Branch is up to date with main

3. **Pull Request Template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Dataset contribution

   ## Test Coverage
   Describe testing approach

   ## Additional Notes
   Any extra information
   ```

## Using the Annotation App

For dataset contributions, you can use our Annotation App:

1. Visit [Annotation App](https://balldatalab.com/streamlit/baseballcv_annotation_app/)
2. Enter credentials
3. Select or create a project
4. Follow annotation guidelines
5. Submit contributions

## Getting Help

- Join our community discussions
- Check existing issues
- Review documentation
- Contact maintainers

{: .note }
Before starting work on a major contribution, please open an issue to discuss your proposed changes with the maintainers.

## Code of Conduct

We expect all contributors to:

1. Be respectful and professional
2. Welcome newcomers
3. Provide constructive feedback
4. Follow project guidelines

## Recognition

Contributors are recognized through:

1. GitHub contribution graph
2. Release notes
3. Documentation credits
4. Community acknowledgments

By contributing to BaseballCV, you agree to license your contributions under the project's MIT license.