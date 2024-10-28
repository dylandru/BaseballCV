import os
import json
from datetime import datetime

__all__ = ['DefaultTools']

class DefaultTools:
    @staticmethod
    def init_baseball_categories():
        return {
            "detection": [
                # Players
                {"id": 1, "name": "pitcher", "color": "#FF6B6B"},
                {"id": 2, "name": "batter", "color": "#4ECDC4"},
                {"id": 3, "name": "catcher", "color": "#45B7D1"},
                {"id": 4, "name": "umpire", "color": "#96CEB4"},
                {"id": 5, "name": "fielder", "color": "#FFEEAD"},
                # Equipment
                {"id": 6, "name": "bat", "color": "#D4A017"},
                {"id": 7, "name": "ball", "color": "#FF9999"},
                {"id": 8, "name": "glove", "color": "#8B4513"},
                # Field
                {"id": 9, "name": "home_plate", "color": "#FFFFFF"},
                {"id": 10, "name": "pitching_rubber", "color": "#CCCCCC"},
                {"id": 11, "name": "base", "color": "#FFFFFF"}
            ],
            "keypoints": {
                "batter": {
                    "keypoints": [
                        "head", "shoulder", "elbow_front", "elbow_back",
                        "hands", "hip", "knee_front", "knee_back",
                        "foot_front", "foot_back"
                    ],
                    "skeleton": [
                        [0, 1], [1, 2], [1, 3], [2, 4], [3, 4],
                        [1, 5], [5, 6], [5, 7], [6, 8], [7, 9]
                    ]
                },
                "pitch": {
                    "keypoints": ["release_point", "plate_crossing"],
                    "skeleton": [[0, 1]]
                }
            }
        }

    @staticmethod
    def init_project_structure():
        base_dir = os.path.join("streamlit", "annotation_app", "projects")
        config_path = os.path.join(base_dir, "project_config.json")
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        # Create project directories and files based on config
        for project_type in ["Detection", "Keypoint"]:
            type_dir = os.path.join(base_dir, project_type)
            if not os.path.exists(type_dir):
                os.makedirs(type_dir)
