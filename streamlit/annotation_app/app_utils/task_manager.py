import os
from datetime import datetime
from .file_tools import FileTools
from typing import Optional, List, Dict
from .s3 import S3Manager
import streamlit as st

class TaskManager:
    """
    A class to manage task processing and annotation data.

    Args:
        project_dir (str): Path to the project directory
        bucket_name (str): Name of the S3 bucket. Defaults to "baseballcv-annotations" bucket.
    """
    def __init__(self, project_dir: str, bucket_name: str = "baseballcv-annotations") -> None:
        self.project_dir = project_dir
        self.annotations_file = os.path.join(project_dir, "annotations.json")
        self.tasks_file = os.path.join(project_dir, "task_queue.json")
        self.file_tools = FileTools()
        self.s3 = S3Manager(bucket_name)
        self.init_tasks_file()
        
    def init_tasks_file(self) -> None:
        """Initialize the tasks file with default structure if it doesn't exist"""
        if not os.path.exists(self.tasks_file):
            tasks_data = {
                "available_images": [],
                "in_progress": {},
                "completed": {},
                "users": {},
                "ledger": {
                    "loaned_images": {},
                    "completed_images": {}
                }
            }
            self.file_tools.save_json(tasks_data, self.tasks_file)
        else:
            tasks_data = self.file_tools.load_json(self.tasks_file)
            if "ledger" not in tasks_data:
                tasks_data["ledger"] = {
                    "loaned_images": {},
                    "completed_images": {}
                }
                self.file_tools.save_json(tasks_data, self.tasks_file)

        return None

    def add_task_batch(self, image_paths: List[str], batch_info: Optional[Dict] = None) -> int:
        """
        Add a batch of images to the task queue

        Args:
            image_paths (List[str]): List of paths to the images to add
            batch_info (Optional[Dict]): Additional information about the batch. Defaults to None.

        Returns:
            int: Number of new images added
        """
        tasks_data = self.file_tools.load_json(self.tasks_file)
        existing_images = (tasks_data["available_images"] + 
                         list(tasks_data["in_progress"].keys()) + 
                         list(tasks_data["completed"].keys()))
        new_images = [img for img in image_paths if img not in existing_images]
        tasks_data["available_images"].extend(new_images)
        self.file_tools.save_json(tasks_data, self.tasks_file)
        return len(new_images)
    
    def get_next_task(self, user_id: str) -> Optional[str]:
        """
        Get the next available task for a user

        Args:
            user_id (str): The ID of the user requesting the task

        Returns:
            Optional[str]: Path to the next task or None if no tasks are available
        """
        tasks_data = self.file_tools.load_json(self.tasks_file)
        if not tasks_data["available_images"]:
            return None
            
        task = tasks_data["available_images"].pop(0)
        timestamp = datetime.now().isoformat()
        
        if "ledger" not in tasks_data:
            tasks_data["ledger"] = {"loaned_images": {}, "completed_images": {}}
        if user_id not in tasks_data["ledger"]["loaned_images"]:
            tasks_data["ledger"]["loaned_images"][user_id] = {}
            
        tasks_data["ledger"]["loaned_images"][user_id][task] = timestamp
        tasks_data["in_progress"][task] = {
            "user_id": user_id,
            "start_time": timestamp
        }
        
        self.file_tools.save_json(tasks_data, self.tasks_file)
        return task
    
    def get_previous_task(self, current_task: str) -> Optional[str]:
        """
        Get the previous task in the sequence

        Args:
            current_task (str): The path to the current task

        Returns:
            Optional[str]: Path to the previous task or None if no previous task exists
        """
        tasks_data = self.file_tools.load_json(self.tasks_file)
        all_tasks = (tasks_data["available_images"] + 
                    list(tasks_data["in_progress"].keys()) + 
                    list(tasks_data["completed"].keys()))
        try:
            current_index = all_tasks.index(current_task)
            if current_index > 0:
                return all_tasks[current_index - 1]
        except ValueError:
            pass
        return None
    
    def get_next_available_task(self, user_id: str) -> Optional[str]:
        """
        Get the next available task for a user

        Args:
            user_id (str): The ID of the user requesting the task

        Returns:
            Optional[str]: Path to the next task or None if no tasks are available
        """
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
        except Exception as e:
            print(f"Error getting next task: {e}")
            return None
    
    def start_task(self, image_path: str, user_id: str) -> bool:
        """
        Start a task for a user

        Args:
            image_path (str): The path to the image to start
            user_id (str): The ID of the user starting the task

        Returns:
            bool: True if the task was started successfully, False otherwise
        """
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
        except Exception as e:
            print(f"Error starting task: {e}")
            return False
    
    def _move_to_completed(self, image_path: str, user_id: str, annotations: List[Dict]) -> None:
        """
        Move a completed image and its annotations to the user's completed folder in S3.
        Organizes files into project-specific folders with images and annotations separated.

        Args:
            image_path (str): The path to the image to move
            user_id (str): The ID of the user completing the task
            annotations (List[Dict]): The annotations for the image
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_folder = f"{st.session_state.project_type}/{st.session_state.selected_project}/completed/com_{timestamp}"
        
        image_filename = os.path.basename(image_path)
        original_s3_path = f"{st.session_state.project_type}/{st.session_state.selected_project}/{image_filename}"
        completed_s3_path = f"{user_folder}/images/{image_filename}"
        
        try:
            self.s3.move_file(original_s3_path, completed_s3_path)
            print(f"Moved image from {original_s3_path} to {completed_s3_path}")
            
            annotations_data = {
                "image_file": image_filename,
                "annotations": annotations,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat(),
                "project_type": st.session_state.project_type,
                "project_name": st.session_state.selected_project,
                "original_path": original_s3_path,
                "completed_path": completed_s3_path
            }
            
            self.s3.upload_json_data(f"{user_folder}/annotations.json", annotations_data)
            
        except Exception as e:
            raise RuntimeError(f"Error moving file in S3: {e}")
    
    def complete_task(self, image_path: str, user_id: str, annotations: List[Dict]) -> bool:
        """
        Complete a task with annotations

        Args:
            image_path (str): The path to the image to move
            user_id (str): The ID of the user completing the task
            annotations (List[Dict]): The annotations for the image

        Returns:
            bool: True if the task was completed successfully, False otherwise
        """
        try:
            tasks_data = self.file_tools.load_json(self.tasks_file)
            annotations_data = self.file_tools.load_json(self.annotations_file)
            timestamp = datetime.now().isoformat()
            
            if image_path in tasks_data["ledger"]["loaned_images"].get(user_id, {}):
                del tasks_data["ledger"]["loaned_images"][user_id][image_path]
            
            if user_id not in tasks_data["ledger"]["completed_images"]:
                tasks_data["ledger"]["completed_images"][user_id] = {}
            tasks_data["ledger"]["completed_images"][user_id][image_path] = timestamp
            
            self._move_to_completed(image_path, user_id, annotations)
            
            tasks_data["completed"][image_path] = {
                "user_id": user_id,
                "completion_time": timestamp,
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
                        "date_created": timestamp,
                        "user_id": user_id
                    }
                    annotations_data["annotations"].append(annotation_info)
                self.file_tools.save_json(annotations_data, self.annotations_file)
                return True
            return False
        except Exception as e:
            print(f"Error completing task: {e}")
            return False
    
    def cleanup_incomplete_tasks(self) -> None:
        """Return incomplete tasks to available_images pool"""
        tasks_data = self.file_tools.load_json(self.tasks_file)
        
        if "ledger" not in tasks_data:
            tasks_data["ledger"] = {
                "loaned_images": {},
                "completed_images": {}
            }
   
        for image_path, task_info in tasks_data["in_progress"].items():
            if image_path not in tasks_data["available_images"]:
                tasks_data["available_images"].append(image_path)
                
        tasks_data["in_progress"] = {}
        tasks_data["ledger"]["loaned_images"] = {}
        
        self.file_tools.save_json(tasks_data, self.tasks_file)
        return None
    
    def _create_user_completed_folder(self, user_id: str, project_type: str, project_name: str) -> str:
        """
        Create a user-specific completed folder in S3 with timestamp within the project folder.

        Args:
            user_id (str): The ID of the user completing the task
            project_type (str): The type of project
            project_name (str): The name of the project

        Returns:
            str: The path to the user's completed folder
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_user_id = ''.join(c for c in user_id if c.isalnum() or c == '_')
        folder_name = f"{project_type}/{project_name}/completed/{clean_user_id}_{timestamp}"
        self.s3.create_folder(folder_name)
        self.s3.create_folder(f"{folder_name}/images")
        return folder_name
        