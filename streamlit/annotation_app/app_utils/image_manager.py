from PIL import Image

__all__ = ['ImageManager']

class ImageManager:
    def __init__(self, project_dir, max_width=1200):
        self.project_dir = project_dir
        self.max_width = max_width
        self.current_image = None
        self.original_size = None
        self.resized_size = None
        self.resized_ratio_w = 1.0
        self.resized_ratio_h = 1.0
        
    def load_image(self, image_path):
        self.current_image = Image.open(image_path)
        self.original_size = self.current_image.size
        return self.current_image
        
    def resize_for_display(self):
        if not self.current_image:
            return None
            
        w, h = self.original_size
        if w > self.max_width:
            ratio = self.max_width / w
            new_size = (int(w * ratio), int(h * ratio))
            resized_image = self.current_image.resize(new_size)
            self.resized_size = new_size
            self.resized_ratio_w = w / new_size[0]
            self.resized_ratio_h = h / new_size[1]
            return resized_image
        
        self.resized_size = self.original_size
        return self.current_image
        
    def display_to_original_coords(self, x, y):
        return x * self.resized_ratio_w, y * self.resized_ratio_h
