from PIL import Image
from typing import Tuple, List, Dict, Any, Optional, Union

__all__ = ['ImageManager']

class ImageManager:
    def __init__(self, project_dir: str, max_width: int = 1280) -> None:
        self.project_dir = project_dir
        self.max_width = max_width
        self.current_image = None
        self.original_size = None
        self.display_size = None
        self.scale_factor = 1.0
        
    def load_image(self, image_path: str, zoom_level: float = 1.0) -> Image.Image:
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
        return self.scale_factor
        
    def scale_coordinates(self, coords: List[float], inverse: bool = False) -> List[float]:
        try:
            if inverse:
                return [float(c) / self.scale_factor for c in coords]
            return [float(c) * self.scale_factor for c in coords]
        except (TypeError, ValueError):
            return coords
            
    def scale_object(self, obj: Dict[str, Any], scale: Optional[float] = None) -> Dict[str, Any]:
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
