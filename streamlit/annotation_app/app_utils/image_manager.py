from PIL import Image

__all__ = ['ImageManager']

class ImageManager:
    def __init__(self, project_dir, max_width=1280):
        self.project_dir = project_dir
        self.max_width = max_width
        self.current_image = None
        self.original_size = None
        self.display_size = None
        self.scale_factor = 1.0
        
    def load_image(self, image_path, zoom_level=1.0):
        self.current_image = Image.open(image_path)
        self.original_size = self.current_image.size
        
        # Calculate display size
        orig_w, orig_h = self.original_size
        scale = min(self.max_width/orig_w, 720/orig_h) * zoom_level
        self.scale_factor = scale
        self.display_size = (int(orig_w * scale), int(orig_h * scale))
        
        return self.current_image.resize(self.display_size)
        
    def get_scale_factor(self):
        return self.scale_factor
        
    def scale_coordinates(self, coords, inverse=False):
        if inverse:
            return [c / self.scale_factor for c in coords]
        return [c * self.scale_factor for c in coords]
