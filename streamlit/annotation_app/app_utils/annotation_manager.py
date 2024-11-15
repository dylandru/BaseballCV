import os
import json
from datetime import datetime
from PIL import Image
import cv2
from .task_manager import TaskManager
from .file_tools import FileTools

class AnnotationManager:
    """Manages annotation data and project configuration."""
    
    def __init__(self):
        """Initialize AnnotationManager."""
        self.base_dir = os.path.join('streamlit', 'annotation_app', 'projects')
        self.file_tools = FileTools()
        
    def create_project(self, project_name: str, config: dict) -> None:
        """Create new project with configuration.
        
        Args:
            project_name: Name of the project
            config: Project configuration dictionary
        """
        project_dir = os.path.join(self.base_dir, config['info']['type'], project_name)
        os.makedirs(project_dir, exist_ok=True)
        os.makedirs(os.path.join(project_dir, 'images'), exist_ok=True)
        
        config['info']['date_created'] = datetime.now().isoformat()
        config['images'] = []
        config['annotations'] = []
        
        self.file_tools.save_json(config, os.path.join(project_dir, 'annotations.json'))

    def create_project_structure(self, project_name, config):
        project_type = "Detection" if config["info"]["type"] == "detection" else "Keypoint"
        project_dir = os.path.join(self.base_project_dir, project_type, project_name)
        
        if not os.path.exists(project_dir):
            os.makedirs(project_dir, exist_ok=True)
            os.makedirs(os.path.join(project_dir, "images"), exist_ok=True)
            
            coco_data = {
                "info": config["info"],
                "images": [],
                "annotations": [],
                "categories": config["categories"]
            }
            
            with open(os.path.join(project_dir, "annotations.json"), "w") as f:
                json.dump(coco_data, f, indent=4)
                
            task_queue = {
                "available_images": [],
                "in_progress": {},
                "completed": {},
                "users": {}
            }
            with open(os.path.join(project_dir, "task_queue.json"), "w") as f:
                json.dump(task_queue, f, indent=4)
            
            return True
        return False
        
    def get_task_manager(self, project_name: str, project_type: str) -> TaskManager:
        """Get TaskManager instance for project."""
        project_dir = os.path.join(self.base_dir, project_type, project_name)
        if os.path.exists(project_dir):
            return TaskManager(project_dir)
                
        raise ValueError(f"Project {project_name} not found")
        
    def add_images_to_task_queue(self, project_name: str, project_type: str, images: list) -> int:
        """Add images to project and task queue.
        
        Args:
            project_name: Name of the project
            images: List of image files or paths
            
        Returns:
            int: Number of images added
        """
        project_dir = os.path.join(self.base_dir, project_type, project_name)
        image_dir = os.path.join(project_dir, 'images')
        
        image_paths = []
        for img in images:
            if isinstance(img, str):
                image_path = img
            else:
                image_path = os.path.join(image_dir, img.name)
                with open(image_path, 'wb') as f:
                    f.write(img.getbuffer())
            
            image_paths.append(image_path)
            self._add_image_to_annotations(project_name, project_type, image_path)
            
        task_manager = self.get_task_manager(project_name, project_type)
        return task_manager.add_task_batch(image_paths)
        
    def _add_image_to_annotations(self, project_name: str, project_type: str, image_path: str) -> None:
        """Add image entry to annotations file.
        
        Args:
            project_name: Name of the project
            image_path: Path to image file
        """
        annotations_file = os.path.join(self.base_dir, project_type, project_name, 'annotations.json')
        data = self.file_tools.load_json(annotations_file)
        
        image = Image.open(image_path)
        width, height = image.size
        
        image_info = {
            "id": len(data['images']) + 1,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height,
            "date_captured": datetime.now().isoformat()
        }
        
        data['images'].append(image_info)
        self.file_tools.save_json(data, annotations_file)
