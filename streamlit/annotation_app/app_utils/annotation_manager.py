import os
from datetime import datetime
from .task_manager import TaskManager
from .file_tools import FileTools 
import json
from PIL import ImageDraw

class AnnotationManager:
    def __init__(self):
        self.init_projects()
        
    def init_projects(self):
        for project_type in ["Detection", "Keypoint"]:
            base_dir = os.path.join("streamlit", "projects", project_type)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir)
                
    def normalize_coordinates(self, x, y, width, height):
        return x / width, y / height

    def normalize_bbox(self, bbox, width, height):
        x1, y1, x2, y2 = bbox
        return [x1/width, y1/height, x2/width, y2/height]

    def draw_bbox_preview(self, draw, start_point, current_point, category=None):
        if start_point:
            x1, y1 = start_point
            
            if draw._image.mode != 'RGB':
                new_image = draw._image.convert('RGB')
                draw = ImageDraw.Draw(new_image)
            
            orange_color = (255, 102, 0)
            try:
                draw.ellipse([int(x1-5), int(y1-5), int(x1+5), int(y1+5)], fill=orange_color)
                
                if current_point:
                    x2, y2 = current_point
                    
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=orange_color, width=2)
                    
                    if category:
                        draw.text((int(x1), int(y1-20)), category, fill=orange_color)
            except Exception as e:
                print(f"Error drawing preview: {str(e)}")

    def draw_annotations(self, draw, annotations, config, scale_factor):
        project_type = config.get("info", {}).get("type", "unknown")
        
        scale_w = scale_factor[0] if isinstance(scale_factor, tuple) else scale_factor
        scale_h = scale_factor[1] if isinstance(scale_factor, tuple) else scale_factor
        
        if draw._image.mode != 'RGB':
            new_image = draw._image.convert('RGB')
            draw = ImageDraw.Draw(new_image)
        
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
        if project_type == "detection":
            for ann in annotations:
                category = next(cat for cat in config.get("categories", []) 
                              if cat["id"] == ann["category_id"])
                color = category.get("color", "#FF6B6B")
                rgb_color = hex_to_rgb(color)
                
                x1, y1, w, h = ann["bbox"]
                x2, y2 = x1 + w, y1 + h
                
                scaled_coords = [
                    int(x1 * scale_w),
                    int(y1 * scale_h),
                    int(x2 * scale_w),
                    int(y2 * scale_h)
                ]
                
                draw.rectangle(scaled_coords, outline=rgb_color, width=2)
                label = f"{category['name']}"
                if "score" in ann:
                    label += f" ({ann['score']:.2f})"
                draw.text((scaled_coords[0], scaled_coords[1]-20), label, 
                         fill=rgb_color)
    
    def _create_project_structure(self, project_name, config):
        project_type = "Detection" if config["info"]["type"] == "detection" else "Keypoint"
        project_dir = os.path.join("streamlit2", "projects", project_type, project_name)
        
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
            os.makedirs(os.path.join(project_dir, "images"))
            
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

    def add_images_to_task_queue(self, project_name, image_files):
        try:
            project_path = self._get_project_path(project_name)
            if not project_path:
                raise ValueError(f"Project {project_name} not found")

            images_dir = os.path.join(project_path, "images")
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            saved_paths = []
            
            annotations_file = os.path.join(project_path, "annotations.json")
            with open(annotations_file, "r") as f:
                annotations_data = json.load(f)

            for uploaded_file in image_files:
                save_path = os.path.join(images_dir, uploaded_file.name)
                
                if hasattr(uploaded_file, 'read'):
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.read())
                else:
                    with open(uploaded_file, "rb") as src, open(save_path, "wb") as dst:
                        dst.write(src.read())
                        
                saved_paths.append(save_path)
                
                image_info = {
                    "id": len(annotations_data["images"]) + 1,
                    "file_name": os.path.basename(save_path),
                    "date_added": datetime.now().isoformat()
                }
                annotations_data["images"].append(image_info)

            with open(annotations_file, "w") as f:
                json.dump(annotations_data, f, indent=4)

            task_queue_file = os.path.join(project_path, "task_queue.json")
            with open(task_queue_file, "r") as f:
                task_queue = json.load(f)

            task_queue["available_images"].extend(saved_paths)

            with open(task_queue_file, "w") as f:
                json.dump(task_queue, f, indent=4)

            return len(saved_paths)

        except Exception as e:
            raise Exception(f"Error adding images: {str(e)}")
    
    def handle_video_upload(self, project_name, video_file, frame_interval=1):
        project_path = self._get_project_path(project_name)
        if not project_path:
            raise ValueError(f"Project {project_name} not found")
            
        frames_dir = os.path.join(project_path, "images")
        
        frames = FileTools.extract_frames(video_file, frames_dir, frame_interval)
        
        annotations_file = os.path.join(project_path, "annotations.json")
        coco_data = FileTools.load_json(annotations_file)
        
        for frame_path in frames:
            frame_name = os.path.basename(frame_path)
            image_info = {
                "id": len(coco_data["images"]) + 1,
                "file_name": frame_name,
                "frame_number": int(frame_name.split("_")[1].split(".")[0]),
                "date_added": datetime.now().isoformat()
            }
            coco_data["images"].append(image_info)
        
        FileTools.save_json(coco_data, annotations_file)
        
        task_queue_file = os.path.join(project_path, "task_queue.json")
        task_queue = FileTools.load_json(task_queue_file)
        task_queue["available_images"].extend(frames)
        FileTools.save_json(task_queue, task_queue_file)
        
        return frames
    
    def get_task_manager(self, project_name):
        project_path = self._get_project_path(project_name)
        if not project_path:
            raise ValueError(f"Project {project_name} not found")
        return TaskManager(project_path)
    
    def _get_project_path(self, project_name):
        for project_type in ["Detection", "Keypoint"]:
            path = os.path.join("streamlit2", "projects", project_type, project_name)
            if os.path.exists(path):
                return path
        return None
