import os
from datetime import datetime
import json
from .file_tools import FileTools  # Assuming FileTools also contains s3_file_manager or similar functions

class TaskManager:
    def __init__(self, project_dir, s3_file_manager):
        self.project_dir = project_dir
        self.s3_file_manager = s3_file_manager  # S3 manager for file operations
        self.annotations_file = os.path.join(project_dir, "annotations.json")
        self.tasks_file = os.path.join(project_dir, "task_queue.json")
        self.init_tasks_file()

    def init_tasks_file(self):
        tasks_data = {
            "available_images": [],
            "in_progress": {},
            "completed": {},
            "users": {}
        }
        # Initialize tasks file in S3 if it does not exist
        if not self.s3_file_manager.file_exists(self.project_dir, "task_queue.json"):
            self.s3_file_manager.save_json(tasks_data, "task_queue.json", self.project_dir)
    
    def add_task_batch(self, image_paths, batch_info=None):
        tasks_data = self.s3_file_manager.load_json("task_queue.json", self.project_dir)
        
        existing_images = (
            tasks_data["available_images"] + 
            list(tasks_data["in_progress"].keys()) + 
            list(tasks_data["completed"].keys())
        )
        
        new_images = [img for img in image_paths if img not in existing_images]
        tasks_data["available_images"].extend(new_images)
        
        self.s3_file_manager.save_json(tasks_data, "task_queue.json", self.project_dir)
        return len(new_images)
    
    def get_next_task(self, user_id):
        tasks_data = self.s3_file_manager.load_json("task_queue.json", self.project_dir)
        
        if not tasks_data["available_images"]:
            return None
            
        task = tasks_data["available_images"].pop(0)
        tasks_data["in_progress"][task] = {
            "user_id": user_id,
            "start_time": datetime.now().isoformat()
        }
        
        self.s3_file_manager.save_json(tasks_data, "task_queue.json", self.project_dir)
        return task
    
    def complete_task(self, image_path, user_id, annotations):
        try:
            tasks_data = self.s3_file_manager.download_task_queue(self.project_dir)
            annotations_data = self.s3_file_manager.download_annotations(self.project_dir, user_id)

            if image_path in tasks_data["in_progress"]:
                # Find the image in the annotations data
                image_filename = os.path.basename(image_path)
                matching_images = [
                    img for img in annotations_data["images"]
                    if img["file_name"] == image_filename
                ]
                
                if not matching_images:
                    print(f"Error: Image {image_filename} not found in annotations.json.")
                    return False
                    
                image_id = matching_images[0]["id"]
                
                # Add annotations for the image
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
                
                # Mark task as completed and remove from in-progress
                tasks_data["completed"][image_path] = {
                    "user_id": user_id,
                    "completion_time": datetime.now().isoformat(),
                    "annotation_count": len(annotations)
                }
                del tasks_data["in_progress"][image_path]

                # Save updated annotations and task queue back to S3
                self.s3_file_manager.upload_annotations(self.project_dir, annotations_data, user_id, "Detection")
                self.s3_file_manager.update_task_queue(self.project_dir, tasks_data)
                
                print(f"Successfully saved {len(annotations)} annotations to S3.")
                return True

            print(f"Error: Image {image_path} not found in in_progress tasks.")
            return False

        except Exception as e:
            print(f"Error saving annotations: {str(e)}")
            return False
