import os
from datetime import datetime
from .file_tools import FileTools
from typing import Optional, List, Dict, Any, Union

class TaskManager:
    def __init__(self, project_dir: str) -> None:
        self.project_dir = project_dir
        self.annotations_file = os.path.join(project_dir, "annotations.json")
        self.tasks_file = os.path.join(project_dir, "task_queue.json")
        self.file_tools = FileTools()
        self.init_tasks_file()
        
    def init_tasks_file(self) -> None:
        if not os.path.exists(self.tasks_file):
            tasks_data = {
                "available_images": [],
                "in_progress": {},
                "completed": {},
                "users": {}
            }
            self.file_tools.save_json(tasks_data, self.tasks_file)
    
    def add_task_batch(self, image_paths: List[str], batch_info: Optional[Dict] = None) -> int:
        tasks_data = self.file_tools.load_json(self.tasks_file)
        existing_images = (tasks_data["available_images"] + 
                         list(tasks_data["in_progress"].keys()) + 
                         list(tasks_data["completed"].keys()))
        new_images = [img for img in image_paths if img not in existing_images]
        tasks_data["available_images"].extend(new_images)
        self.file_tools.save_json(tasks_data, self.tasks_file)
        return len(new_images)
    
    def get_next_task(self, user_id: str) -> Optional[str]:
        tasks_data = self.file_tools.load_json(self.tasks_file)
        if not tasks_data["available_images"]:
            return None
        task = tasks_data["available_images"].pop(0)
        tasks_data["in_progress"][task] = {
            "user_id": user_id,
            "start_time": datetime.now().isoformat()
        }
        self.file_tools.save_json(tasks_data, self.tasks_file)
        return task
    
    def complete_task(self, image_path: str, user_id: str, annotations: List[Dict]) -> bool:
        try:
            tasks_data = self.file_tools.load_json(self.tasks_file)
            annotations_data = self.file_tools.load_json(self.annotations_file)
            tasks_data["completed"][image_path] = {
                "user_id": user_id,
                "completion_time": datetime.now().isoformat(),
                "annotation_count": len(annotations)
            }
            if image_path in tasks_data["in_progress"]:
                del tasks_data["in_progress"][image_path]
            if image_path in tasks_data["available_images"]:
                tasks_data["available_images"].remove(image_path)
            self.file_tools.save_json(tasks_data, self.tasks_file)
            image_filename = os.path.basename(image_path)
            matching_images = [img for img in annotations_data["images"] 
                             if img["file_name"] == image_filename]
            if matching_images:
                image_id = matching_images[0]["id"]
                for ann in annotations:
                    annotation_id = len(annotations_data["annotations"]) + 1
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": ann["category_id"],
                        "bbox": ann["bbox"],
                        "date_created": datetime.now().isoformat(),
                        "user_id": user_id
                    }
                    annotations_data["annotations"].append(annotation_info)
                self.file_tools.save_json(annotations_data, self.annotations_file)
                return True
            return False
        except Exception:
            return False
    
    def start_task(self, image_path: str, user_id: str) -> bool:
        try:
            tasks_data = self.file_tools.load_json(self.tasks_file)
            if image_path in tasks_data["available_images"]:
                tasks_data["available_images"].remove(image_path)
            tasks_data["in_progress"][image_path] = {
                "user_id": user_id,
                "start_time": datetime.now().isoformat()
            }
            self.file_tools.save_json(tasks_data, self.tasks_file)
            return True
        except Exception:
            return False
    
    def get_next_available_task(self, user_id: str) -> Optional[str]:
        try:
            tasks_data = self.file_tools.load_json(self.tasks_file)
            if not tasks_data["available_images"]:
                return None
            next_image = tasks_data["available_images"].pop(0)
            tasks_data["in_progress"][next_image] = {
                "user_id": user_id,
                "start_time": datetime.now().isoformat()
            }
            self.file_tools.save_json(tasks_data, self.tasks_file)
            return next_image
        except Exception:
            return None
