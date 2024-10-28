import os
from datetime import datetime
import json
from .file_tools import FileTools

class TaskManager:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.annotations_file = os.path.join(project_dir, "annotations.json")
        self.tasks_file = os.path.join(project_dir, "task_queue.json")
        self.init_tasks_file()
        
    def init_tasks_file(self):
        if not os.path.exists(self.tasks_file):
            tasks_data = {
                "available_images": [],
                "in_progress": {},
                "completed": {},
                "users": {}
            }
            FileTools.save_json(tasks_data, self.tasks_file)
    
    def add_task_batch(self, image_paths, batch_info=None):
        tasks_data = FileTools.load_json(self.tasks_file)
        
        existing_images = (tasks_data["available_images"] + 
                         list(tasks_data["in_progress"].keys()) + 
                         list(tasks_data["completed"].keys()))
        
        new_images = [img for img in image_paths if img not in existing_images]
        tasks_data["available_images"].extend(new_images)
        
        FileTools.save_json(tasks_data, self.tasks_file)
        return len(new_images)
    
    def get_next_task(self, user_id):
        tasks_data = FileTools.load_json(self.tasks_file)
        
        if not tasks_data["available_images"]:
            return None
            
        task = tasks_data["available_images"].pop(0)
        tasks_data["in_progress"][task] = {
            "user_id": user_id,
            "start_time": datetime.now().isoformat()
        }
        
        FileTools.save_json(tasks_data, self.tasks_file)
        return task
    
    def complete_task(self, image_path, user_id, annotations):
        try:
            with open(self.tasks_file, "r") as f:
                tasks_data = json.load(f)
            with open(self.annotations_file, "r") as f:
                annotations_data = json.load(f)
            
            print(f"Processing image: {image_path}")
            print(f"Current annotations: {annotations}")
            
            if image_path in tasks_data["in_progress"]:
                image_filename = os.path.basename(image_path)
                matching_images = [img for img in annotations_data["images"] 
                                 if img["file_name"] == image_filename]
                
                if not matching_images:
                    print(f"Error: Image {image_filename} not found in annotations.json")
                    return False
                    
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
                
                tasks_data["completed"][image_path] = {
                    "user_id": user_id,
                    "completion_time": datetime.now().isoformat(),
                    "annotation_count": len(annotations)
                }
                del tasks_data["in_progress"][image_path]
                
                FileTools.save_json(annotations_data, self.annotations_file)
                FileTools.save_json(tasks_data, self.tasks_file)
                
                print(f"Successfully saved {len(annotations)} annotations")
                return True
                
            print(f"Error: Image {image_path} not found in in_progress tasks")
            return False
                
        except Exception as e:
            print(f"Error saving annotations: {str(e)}")
            return False
