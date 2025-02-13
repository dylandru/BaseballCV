---
layout: default
title: Building Custom Pipelines
parent: Using the Repository
nav_order: 2
---

# Building Custom Analysis Pipelines

BaseballCV's true power emerges when we combine its various components into comprehensive analysis pipelines. Let's explore how to create a custom pipeline that focuses on ball tracking and game context analysis, two fundamental aspects of baseball analysis.

## Understanding Pipeline Components

A baseball analysis pipeline typically involves several key steps working in harmony:

1. Data Acquisition: Preparing and loading video footage
2. Frame Analysis: Processing individual frames for specific elements
3. Sequential Analysis: Understanding how elements change over time
4. Context Integration: Adding game situation understanding
5. Results Compilation: Organizing findings into useful formats

Let's build a complete pipeline that demonstrates these concepts:

```python
from baseballcv.functions import LoadTools, DataTools
from baseballcv.models import Florence2
import cv2
import numpy as np

class BaseballAnalysisPipeline:
    """
    A comprehensive pipeline for analyzing baseball plays through ball tracking
    and game context understanding
    """
    def __init__(self):
        # Initialize our core tools
        self.load_tools = LoadTools()
        self.data_tools = DataTools()
        
        # Load our specialized models
        self.ball_model = self.load_model_with_retry("ball_tracking")
        self.context_model = Florence2()
        
        # Set up logging and monitoring
        self.setup_logging()
    
    def load_model_with_retry(self, model_name, max_attempts=3):
        """
        Robust model loading with retry logic
        """
        for attempt in range(max_attempts):
            try:
                model_path = self.load_tools.load_model(model_name)
                return YOLO(model_path)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise Exception(f"Failed to load model after {max_attempts} attempts")
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
    
    def process_sequence(self, video_path):
        """
        Process a complete baseball play sequence
        """
        # Extract frames from the video
        self.logger.info("Extracting frames from video sequence")
        frames = self.data_tools.generate_photo_dataset(
            video_path,
            max_num_frames=90,  # Capture full pitch sequence
            output_frames_folder="analysis_frames"
        )
        
        # Initialize our analysis containers
        sequence_data = []
        game_context = None
        
        # Process each frame in the sequence
        self.logger.info("Beginning frame analysis")
        for idx, frame in enumerate(frames):
            # Track the baseball
            ball_results = self.ball_model.predict(
                frame,
                conf=0.25,  # Confidence threshold
                verbose=False
            )
            
            # Get game context from the first frame
            if idx == 0:
                self.logger.info("Analyzing game context")
                game_context = self.context_model.inference(
                    frame,
                    task="<DETAILED_CAPTION>"
                )
            
            # Combine results for this frame
            frame_data = self.process_frame_results(ball_results)
            sequence_data.append(frame_data)
        
        # Analyze the complete sequence
        return self.analyze_sequence(sequence_data, game_context)
    
    def process_frame_results(self, ball_results):
        """
        Extract and organize ball detection results from a single frame
        """
        frame_data = {
            'timestamp': ball_results.timestamp,
            'ball_detections': []
        }
        
        # Process each detection in the frame
        for detection in ball_results.boxes:
            if detection.conf[0] > 0.25:  # Confidence threshold
                frame_data['ball_detections'].append({
                    'position': detection.xyxy[0].tolist(),
                    'confidence': detection.conf[0].item()
                })
        
        return frame_data
    
    def analyze_sequence(self, sequence_data, game_context):
        """
        Analyze the complete sequence of frames to understand the play
        """
        # Filter and clean ball positions
        ball_positions = self.extract_clean_trajectories(sequence_data)
        
        # Organize our findings
        analysis_results = {
            'sequence_length': len(sequence_data),
            'ball_trajectory': ball_positions,
            'game_situation': game_context,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return analysis_results
    
    def extract_clean_trajectories(self, sequence_data):
        """
        Clean and organize ball position data across frames
        """
        positions = []
        
        for frame_data in sequence_data:
            # Take the highest confidence detection if multiple exist
            if frame_data['ball_detections']:
                best_detection = max(
                    frame_data['ball_detections'],
                    key=lambda x: x['confidence']
                )
                positions.append(best_detection['position'])
            else:
                positions.append(None)
        
        # Interpolate missing positions
        return self.interpolate_missing_positions(positions)
    
    def interpolate_missing_positions(self, positions):
        """
        Fill in missing positions using linear interpolation
        """
        positions = np.array(positions)
        mask = positions != None
        
        if not np.any(mask):
            return positions
        
        # Create interpolation function for each coordinate
        interp_positions = []
        for i in range(4):  # For each coordinate (x1, y1, x2, y2)
            valid_pos = positions[mask, i]
            valid_idx = np.where(mask)[0]
            interp_func = np.interp(
                np.arange(len(positions)),
                valid_idx,
                valid_pos
            )
            interp_positions.append(interp_func)
        
        return np.array(interp_positions).T.tolist()

# Using the pipeline
def analyze_baseball_play(video_path):
    """
    Analyze a baseball play using our custom pipeline
    """
    pipeline = BaseballAnalysisPipeline()
    
    try:
        results = pipeline.process_sequence(video_path)
        print(f"Successfully analyzed sequence of {results['sequence_length']} frames")
        return results
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        return None

# Example usage
results = analyze_baseball_play("pitch_sequence.mp4")
```

Let's break down why this pipeline is effective:

### Robust Data Handling

The pipeline includes several features that make it reliable and robust:
- Retry logic for model loading
- Confidence thresholds for detections
- Interpolation for missing data points
- Proper error handling throughout

### Efficient Processing

We've organized the processing to minimize computational overhead:
- Context analysis runs only once per sequence
- Frame processing focuses only on essential elements
- Data structures are optimized for quick access

### Clear Data Organization

The pipeline maintains well-structured data throughout:
- Each frame's results are consistently formatted
- Temporal relationships are preserved
- Results are easy to analyze further

### Quality Control

Several quality control measures are built in:
- Confidence thresholds filter out weak detections
- Missing data is handled gracefully
- Processing steps are logged for monitoring

This pipeline structure provides a foundation for baseball video analysis that you can extend based on your specific needs. Whether you're analyzing pitches, tracking game situations, or building a complete game analysis system, these principles of organization and robustness will serve you well.