from PIL import Image
from typing import List, Dict, Any, Optional

__all__ = ['ImageManager']

class ImageManager:
    """
    A class to manage image processing and scaling for annotation tasks.
    
    Args:
        project_dir (str): Path to the project directory
        max_width (int): Maximum width for display. Defaults to 1280.

    """
    def __init__(self, project_dir: str, max_width: int = 1280) -> None:
        self.project_dir = project_dir
        self.max_width = max_width
        self.current_image = None
        self.original_size = None
        self.display_size = None
        self.scale_factor = 1.0
        
    def load_image(self, image_path: str, zoom_level: float = 1.0) -> Image.Image:
        """
        Load an image and scale it to fit within the maximum width and height.

        Args:
            image_path (str): Path to the image file
            zoom_level (float): Zoom level for the image. Defaults to 1.0.

        Returns:
            Image.Image: The scaled image
        """
        self.current_image = Image.open(image_path)
        self.original_size = self.current_image.size
        
        orig_w, orig_h = self.original_size
        max_width = 1200
        max_height = 1000
        
        width_scale = max_width / orig_w
        height_scale = max_height / orig_h
        scale = min(width_scale, height_scale) * 0.8 * zoom_level
        
        self.scale_factor = scale
        self.display_size = (int(orig_w * scale), int(orig_h * scale))
        
        return self.current_image.resize(self.display_size, Image.Resampling.LANCZOS)
        
    def get_scale_factor(self) -> float:
        """
        Get the current scale factor for the image.

        Returns:
            float: The current scale factor
        """
        return self.scale_factor
        
    def scale_coordinates(self, coords: List[float], inverse: bool = False) -> List[float]:
        """
        Scale coordinates based on the current scale factor.

        Args:
            coords (List[float]): The coordinates to scale
            inverse (bool): Whether to invert the scaling. Defaults to False.

        Returns:
            List[float]: The scaled coordinates
        """
        try:
            if inverse:
                return [float(c) / self.scale_factor for c in coords]
            return [float(c) * self.scale_factor for c in coords]
        except (TypeError, ValueError):
            return coords
            
    def scale_object(self, obj: Dict[str, Any], scale: Optional[float] = None) -> Dict[str, Any]:
        """
        Scale an annotation object based on the current scale factor.

        Args:
            obj (Dict[str, Any]): The annotation object to scale
            scale (Optional[float]): The scale factor to use. Defaults to None.

        Returns:
            Dict[str, Any]: The scaled annotation object
        """
        if scale is None:
            scale = self.scale_factor
            
        try:
            if obj.get("type") == "group":
                scaled_group = {
                    "type": "group",
                    "left": float(obj["left"]) * scale,
                    "top": float(obj["top"]) * scale,
                    "objects": []
                }
                
                for sub_obj in obj["objects"]:
                    if sub_obj["type"] == "rect":
                        scaled_group["objects"].append({
                            "type": "rect",
                            "left": float(sub_obj["left"]),
                            "top": float(sub_obj["top"]),
                            "width": float(sub_obj["width"]) * scale,
                            "height": float(sub_obj["height"]) * scale,
                            "stroke": sub_obj["stroke"],
                            "fill": sub_obj["fill"],
                            "strokeWidth": sub_obj["strokeWidth"]
                        })
                    elif sub_obj["type"] == "text":
                        scaled_group["objects"].append({
                            "type": "text",
                            "left": float(sub_obj["left"]),
                            "top": float(sub_obj["top"]),
                            "text": sub_obj["text"],
                            "fill": sub_obj["fill"],
                            "fontSize": int(14 * scale),
                            "backgroundColor": sub_obj.get("backgroundColor", "rgba(0,0,0,0.5)"),
                            "padding": sub_obj.get("padding", 2)
                        })
                return scaled_group
                
            elif obj.get("type") == "rect":
                return {
                    "type": "rect",
                    "left": float(obj["left"]) * scale,
                    "top": float(obj["top"]) * scale,
                    "width": float(obj["width"]) * scale,
                    "height": float(obj["height"]) * scale,
                    "stroke": obj.get("stroke", "#FF6B00"),
                    "fill": obj.get("fill", "rgba(255, 107, 0, 0.3)"),
                    "strokeWidth": obj.get("strokeWidth", 2)
                }
            return obj
        except (KeyError, TypeError):
            return obj
