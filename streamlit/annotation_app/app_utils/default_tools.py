import os
import json
from datetime import datetime

__all__ = ['DefaultTools']

class DefaultTools:
    
    PROJECTS_CONFIG = {
        "Detection": {
            "Bat_Detection": {
                "info": {
                    "name": "Bat Detection",
                    "description": "Detect baseball bats in images",
                    "type": "detection",
                    "date_created": datetime.now().isoformat()
                },
                "categories": [
                    {"id": 1, "name": "bat", "color": "#D4A017"}
                ]
            },
            "Player_Detection": {
                "info": {
                    "name": "Player Detection",
                    "description": "Detect players by role (pitcher, batter, catcher)",
                    "type": "detection",
                    "date_created": datetime.now().isoformat(),
                    "model_config": {
                        "model_alias": "phc_detector",
                        "confidence_threshold": 0.5
                    }
                },
                "categories": [
                    {"id": 1, "name": "pitcher", "color": "#FF6B6B"},
                    {"id": 2, "name": "batter", "color": "#4ECDC4"},
                    {"id": 3, "name": "catcher", "color": "#45B7D1"}
                ]
            }
        },
        "Keypoint": {
            "Batter_Pose": {
                "info": {
                    "name": "Batter Pose",
                    "description": "Annotate batter pose keypoints",
                    "type": "keypoint",
                    "date_created": datetime.now().isoformat()
                },
                "categories": [{
                    "id": 1,
                    "name": "batter",
                    "keypoints": [
                        "head", "shoulder", "elbow_front", "elbow_back",  #NEEDS CORRECTED
                        "hands", "hip", "knee_front", "knee_back",
                        "foot_front", "foot_back"
                    ],
                    "skeleton": [
                        [0, 1], [1, 2], [1, 3], [2, 4], [3, 4], #NEEDS CORRECTED
                        [1, 5], [5, 6], [5, 7], [6, 8], [7, 9]
                    ]
                }]
            }
        }
    }

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
                {"id": 6, "name": "bat", "color": "#D4A017"},
                {"id": 7, "name": "ball", "color": "#FF9999"},
                {"id": 8, "name": "glove", "color": "#8B4513"},
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
        
        
        os.makedirs(base_dir, exist_ok=True)
        
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                json.dump(DefaultTools.PROJECTS_CONFIG, f, indent=4)

        for project_type, projects in DefaultTools.PROJECTS_CONFIG.items():
            type_dir = os.path.join(base_dir, project_type)
            if not os.path.exists(type_dir):
                os.makedirs(type_dir)
                
            for project_id, project_config in projects.items():
                project_dir = os.path.join(type_dir, project_id)
                if not os.path.exists(project_dir):
                    os.makedirs(project_dir)
                    os.makedirs(os.path.join(project_dir, "images"))
                    
                    annotations = {
                        "info": project_config["info"],
                        "categories": project_config["categories"],
                        "images": [],
                        "annotations": []
                    }
                    with open(os.path.join(project_dir, "annotations.json"), "w") as f:
                        json.dump(annotations, f, indent=4)
                    
                    task_queue = {
                        "available_images": [],
                        "in_progress": {},
                        "completed": {},
                        "users": {}
                    }
                    with open(os.path.join(project_dir, "task_queue.json"), "w") as f:
                        json.dump(task_queue, f, indent=4)
