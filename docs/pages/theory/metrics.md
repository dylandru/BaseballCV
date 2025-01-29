# docs/pages/theory/metrics.md
---
layout: default
title: Evaluation Metrics
parent: Theory
nav_order: 4
---

# Evaluation Metrics in Baseball Computer Vision

Understanding how to evaluate computer vision models in baseball contexts is crucial for developing reliable systems.

## Implementation of Key Metrics

```python
import numpy as np
from sklearn.metrics import average_precision_score

class BaseballDetectionMetrics:
    """
    Evaluation metrics for baseball object detection
    """
    def __init__(self):
        self.iou_threshold = 0.5
        
    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union for two bounding boxes
        """
        # Convert boxes to [x1, y1, x2, y2] format
        box1 = np.array(box1)
        box2 = np.array(box2)
        
        # Calculate intersection coordinates
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate areas
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0
    
    def calculate_map(self, predictions, ground_truth):
        """
        Calculate Mean Average Precision for baseball objects
        """
        aps = []
        for class_id in range(len(predictions)):
            class_preds = predictions[class_id]
            class_gt = ground_truth[class_id]
            
            if len(class_gt) == 0:
                continue
                
            # Calculate precision and recall
            precision = []
            recall = []
            
            # Implementation of mAP calculation
            # ...
            
        return np.mean(aps)