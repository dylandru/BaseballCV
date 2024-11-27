import os
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from .annotation_manager import AnnotationManager
from .image_manager import ImageManager
from .task_manager import TaskManager
from .default_tools import DefaultTools
from .file_tools import FileTools
from datetime import datetime
from .s3 import S3Manager
from .model_manager import ModelManager
import math

class AppPages:
    def __init__(self):
        self.base_project_dir = os.path.join('streamlit', 'annotation_app', 'projects')
        self.file_tools = FileTools()
        
        os.makedirs(self.base_project_dir, exist_ok=True)
        
        for project_type in ["Detection", "Keypoint"]:
            type_dir = os.path.join(self.base_project_dir, project_type)
            os.makedirs(type_dir, exist_ok=True)
        
        task_queue_file = os.path.join(self.base_project_dir, "task_queue.json")
        if not os.path.exists(task_queue_file):
            task_queue_data = {
                "available_images": [],
                "in_progress": {},
                "completed": {},
                "users": {}
            }
            self.file_tools.save_json(task_queue_data, task_queue_file)
        
        self.image_manager = ImageManager(project_dir=self.base_project_dir)
        self.manager = AnnotationManager()
        self.task_manager = TaskManager(project_dir=self.base_project_dir)
        self.s3_manager = S3Manager("baseballcv-annotations")
        self.project_data = None
        self.model_manager = ModelManager()
        self.default_tools = DefaultTools()
        self.model = None

    def show_welcome_page(self) -> None:
        st.markdown("""
            <style>
            .block-container {
                padding-top: 1rem !important;
            }
            .stApp header {
                display: none !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='text-align: center; padding: 0.5rem;'>
                <h1 style='color: white; margin-top: 0;'>BaseballCV Annotation Tool</h1>
                <p style='font-size: 1.2rem; color: #FF6B00;'>
                    Simple and efficient annotation tool for baseball images and videos
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style='background-color: #252525; padding: 1.5rem; border-radius: 10px; height: 200px;'>
                    <h3 style='color: #FF6B00;'>Start New Project</h3>
                    <p>Create a new annotation project from scratch</p>
                    <ul>
                        <li>Detection (players, equipment, field elements)</li>
                        <li>Keypoints (player poses, pitch tracking)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Create New Project", key="new_project"):
                st.session_state.page = "create_project"
                return st.rerun()

        with col2:
            st.markdown("""
                <div style='background-color: #252525; padding: 1.5rem; border-radius: 10px; height: 200px;'>
                    <h3 style='color: #FF6B00;'>Continue Existing Project</h3>
                    <p>Continue working on an existing annotation project</p>
                    <ul>
                        <li>View and edit annotations</li>
                        <li>Add new images or videos</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Open Existing Project", key="existing_project"):
                st.session_state.page = "select_project"
                return st.rerun()

    def create_project_screen(self) -> None:
        st.markdown("<h1 style='text-align: center; color: white'>Create New Project</h1>", unsafe_allow_html=True)
        
        project_type = st.radio(":orange[Project Type]", ["Detection", "Keypoints"])
        project_name = st.text_input(":orange[Project Name]")
        project_description = st.text_area(":orange[Project Description]")
        
        if project_type == "Detection":
            st.markdown("<h3 style='text-align: center; color: white'>Available Categories</h3>", unsafe_allow_html=True)
            categories = self.default_tools.init_baseball_categories()["detection"]
            selected_categories = st.multiselect(
                ":orange[Select categories to include]",
                options=[cat["name"] for cat in categories],
                default=[cat["name"] for cat in categories]
            )
        else:
            st.markdown("<h3 style='text-align: center; color: white'>Keypoint Presets</h3>", unsafe_allow_html=True)
            keypoint_presets = self.default_tools.init_baseball_categories()["keypoints"]
            selected_preset = st.selectbox(
                ":orange[Select keypoint configuration]",
                options=list(keypoint_presets.keys())
            )
        
        if st.button("Create Project"):
            if not project_name or not project_description:
                st.error("Please fill in both project name and description")
                return
                
            project_dir = os.path.join(self.base_project_dir, project_type, project_name)
            st.write(f"Creating project in: {project_dir}")
            
            config = {
                "info": {
                    "description": project_description,
                    "type": "detection" if project_type == "Detection" else "keypoint",
                    "date_created": datetime.now().isoformat(),
                    "model_config": {
                        "model_alias": "yolov8n"
                    }
                },
                "categories": [cat for cat in categories if cat["name"] in selected_categories] if project_type == "Detection" else [{
                    "id": 1,
                    "name": selected_preset,
                    "keypoints": keypoint_presets[selected_preset]["keypoints"],
                    "skeleton": keypoint_presets[selected_preset]["skeleton"]
                }]
            }

            try:
                self.manager.create_project_structure(project_name, config)
                st.success(f"Project created successfully in {project_dir}")
                st.session_state.selected_project = project_name
                st.session_state.project_type = project_type
                st.session_state.page = "add_media"
                return st.rerun()
            except Exception as e:
                st.error(f"Error creating project: {str(e)}")

    def show_project_selection(self) -> None:
        st.markdown("""
            <style>
            .block-container {
                padding-top: 0rem !important;
                padding-bottom: 3rem !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.title(":orange[Select Project]")
        
        project_types = ["Detection", "Keypoint"]
        for project_type in project_types:
            st.header(f":gray[{project_type} Projects]", divider="gray")
            
            projects_path = os.path.join(self.base_project_dir, project_type)
            
            if not os.path.exists(projects_path):
                continue
                
            projects = [p for p in os.listdir(projects_path) 
                       if os.path.isdir(os.path.join(projects_path, p))]
            
            if not projects:
                continue
                
            for project in projects:
                config_path = os.path.join(projects_path, project, "annotations.json")
                if os.path.exists(config_path):
                    config = self.file_tools.load_json(config_path)
                    
                    st.markdown(f"""
                        <div class='project-card'>
                            <h4 style='color: #FF6B00;'>{project}</h4>
                            <p>{config.get('info', {}).get('description', 'No description')}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Select", key=f"select_{project}"):
                            st.session_state.selected_project = project
                            st.session_state.project_type = project_type
                            st.session_state.page = "project_dashboard"
                            return st.rerun()
                    with col2:
                        if st.button("Details", key=f"details_{project}"):
                            st.session_state.project_details = project
                            return st.rerun()

    def show_project_dashboard(self) -> None:
        if not st.session_state.selected_project:
            st.error("No project selected")
            return
            
        st.markdown(f"<h1 style='color: orange; text-align: center; padding: 3rem; font-size: 3.5rem'>Project: {st.session_state.project_type}</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='action-card' style='display: flex; flex-direction: column; align-items: center; justify-content: center; margin: auto; width: 80%;'>
                    <h3 style='color: orange; text-align: center;'>Add Content</h3>
                    <p style='text-align: center;'>Upload new images or videos to annotate</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Upload Media", use_container_width=True):
                st.session_state.page = "add_media"
                return st.rerun()

        with col2:
            st.markdown("""
                <div class='action-card' style='display: flex; flex-direction: column; align-items: center; justify-content: center; margin: auto; width: 80%;'>
                    <h3 style='color: orange; text-align: center;'>Start Annotating</h3>
                    <p style='text-align: center;'>Begin or continue annotation work</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Open Annotator", use_container_width=True):
                st.session_state.page = "annotate"
                return st.rerun()

    def show_add_media_page(self) -> None:
        if not st.session_state.selected_project:
            st.error("No project selected")
            return
            
        st.title(":orange[Add Media]")
        
        tab1, tab2, tab3 = st.tabs(["Upload Images", "Upload Video", "Use our Images"])
        
        with tab1:
            uploaded_files = st.file_uploader(
                ":orange[Upload Images]",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )
            print(uploaded_files)
            
            if uploaded_files:
                if st.button("Add Images"):
                    try:
                        num_added = self.manager.add_images_to_task_queue(
                            st.session_state.selected_project,
                            st.session_state.project_type,
                            uploaded_files
                        )
                        st.success(f"Added {num_added} images successfully")
                        return st.rerun()
                    except Exception as e:
                        st.error(f"Error adding images: {str(e)}")

        with tab2:
            video_file = st.file_uploader(":orange[Upload Video]", type=["mp4", "avi", "mov"])
            if video_file:
                frame_interval = st.slider("Extract every nth frame", 1, 60, 5)
                if st.button("Process Video"):
                    try:
                        frames = self.manager.handle_video_upload(
                            st.session_state.selected_project,
                            st.session_state.project_type,
                            video_file,
                            frame_interval
                        )
                        st.success(f"Extracted {len(frames)} frames from video")
                        st.session_state.page = "project_dashboard"
                        return st.rerun()
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
        with tab3:
            st.markdown(":orange[Use our Images]")
            num_images = st.number_input("How many images would you like to download?", 
                                       min_value=1, 
                                       max_value=1000,
                                       value=10,
                                       help="Choose between 1-1000 images to download for annotation")
            
            if st.button("Download Images"):
                try:
                    project_path = os.path.join(self.base_project_dir, 
                                              st.session_state.project_type, 
                                              st.session_state.selected_project)
                    
                    #s3_folder = os.path.join(st.session_state.project_type, 
                    #                        st.session_state.selected_project)
                    
                    s3_folder = 'Detection/Player_Detection/'
                    photos = self.s3_manager.retrieve_raw_photos(
                        s3_folder_name=s3_folder,
                        local_path=os.path.join(project_path, "images"),
                        max_images=num_images
                    )
                    
                    if len(photos) > 0:
                        num_added = self.manager.add_images_to_task_queue(
                            st.session_state.selected_project,
                            st.session_state.project_type,
                            photos
                        )
                        st.success(f"Successfully downloaded {num_added} images!")
                        st.session_state.page = "project_dashboard"
                        st.rerun()
                    else:
                        st.info("No new images found")
                except Exception as e:
                    st.error(f"Error downloading images: {str(e)}")

    def show_progress_page(self) -> None:
        st.markdown("<h1 style='color: white;'>Project Progress</h1>", unsafe_allow_html=True)
        
        if not st.session_state.selected_project:
            st.error("No project selected")
            return
            
        task_manager = self.manager.get_task_manager(st.session_state.selected_project, st.session_state.project_type)
        

        tasks_data = self.file_tools.load_json(task_manager.tasks_file)
        total_images = (len(tasks_data["available_images"]) + 
                       len(tasks_data["in_progress"]) + 
                       len(tasks_data["completed"]))
        
        completed = len(tasks_data["completed"])
        in_progress = len(tasks_data["in_progress"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(":orange[Total Images]", total_images)
        with col2:
            st.metric(":orange[Completed]", completed) 
        with col3:
            completion_rate = (completed / total_images * 100) if total_images > 0 else 0
            st.metric(":orange[Completion Rate]", f"{completion_rate:.1f}%")

    def show_annotation_interface(self) -> None:
        if not st.session_state.selected_project or not st.session_state.project_type:
            st.error("No project selected")
            return
                
        self.project_dir = os.path.join(self.base_project_dir, 
                                        st.session_state.project_type,
                                        st.session_state.selected_project)
        
        try:
            self.project_data = self.file_tools.load_json(os.path.join(self.project_dir, "annotations.json"))
        except Exception as e:
            st.error(f"Error loading project data: {str(e)}")
            return

        # Initialize session state variables
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        if 'annotations' not in st.session_state:
            st.session_state.annotations = []
        if 'zoom_level' not in st.session_state:
            st.session_state.zoom_level = 1.0
        if 'rotation' not in st.session_state:
            st.session_state.rotation = 0

        # Add styling
        st.markdown("""
            <style>
            .block-container {
                padding: 0 !important;
                margin: 0 !important;
                padding-bottom: 200px !important;
            }
            
            div[data-testid="stVerticalBlock"] > div {
                padding: 0 !important;
                margin: 0 !important;
            }
            
            .top-controls {
                background: #1E1E1E;
                border-bottom: 1px solid #333;
                padding: 5px;
                margin: 0;
                display: flex;
                align-items: center;
            }
            
            .image-workspace {
                background: #2D2D2D;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 5px;
                margin: 0;
                min-height: calc(100vh - 120px);
            }
            
            .stImage {
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
            }
            
            .image-info {
                position: absolute;
                top: 10px;
                right: 10px;
                background: rgba(0,0,0,0.7);
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
                z-index: 100;
            }
            
            .annotation-label {
                background: #FF6B00;
                color: white;
                padding: 2px 6px;
                border-radius: 2px;
                font-size: 12px;
            }
            
            .stButton button {
                padding: 2px 8px !important;
                height: 30px !important;
                min-height: 30px !important;
            }
            
            .stSelectbox label {
                display: none !important;
            }
            
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            .legend {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: rgba(30, 30, 30, 0.95);
                padding: 15px 20px;
                border-top: 1px solid #333;
                display: flex;
                justify-content: center;
                gap: 20px;
                z-index: 1000;
            }
            
            .legend-item {
                display: flex;
                align-items: center;
                margin: 3px 0;
            }
            
            .legend-color {
                width: 12px;
                height: 12px;
                margin-right: 6px;
                border-radius: 2px;
            }
            
            .legend-label {
                color: white;
                font-size: 11px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .bottom-legend {
                position: fixed;
                bottom: 60px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(30, 30, 30, 0.95);
                padding: 15px 30px;
                border-radius: 8px;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 30px;
                z-index: 1000;
                width: auto;
                max-width: 80%;
                border: 1px solid #333; 
            }
            
            .legend-item {
                display: flex;
                align-items: center;
                gap: 10px;
                padding: 5px 10px;
            }
            
            .color-box {
                width: 18px;
                height: 18px;
                border-radius: 4px;
            }
            
            .legend-text {
                color: white;
                font-size: 14px;
                white-space: nowrap;
            }
            
            .canvas-container {
                margin-bottom: 150px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # Control buttons
        cols = st.columns([1, 1, 1, 1, 2])
        
        with cols[0]:
            if st.button("Rotate", use_container_width=True):
                st.session_state.rotation = (st.session_state.rotation - 90) % 360
        with cols[1]:
            if st.button("Zoom In", use_container_width=True):
                st.session_state.zoom_level *= 1.2
                return st.rerun()
        with cols[2]:
            if st.button("Zoom Out", use_container_width=True):
                st.session_state.zoom_level = max(0.1, st.session_state.zoom_level / 1.2)
                return st.rerun()
        with cols[3]:
            if st.button("Fit", use_container_width=True):
                st.session_state.zoom_level = 1.0
                return st.rerun()
        with cols[4]:
            st.selectbox(
                "Category",
                options=[cat['name'] for cat in self.project_data.get("categories", [])],
                key="current_category",
                label_visibility="collapsed"
            )

        # Get model configuration and cached instance
        model_config = self.project_data.get("info", {}).get("model_config", {})
        model_alias = model_config.get("model_alias")
        model = None
        if model_alias:
            model = ModelManager.get_model_instance(model_alias)

        # Rest of your existing code for image loading and annotation
        if not st.session_state.current_image:
            task_manager = self.manager.get_task_manager(st.session_state.selected_project, st.session_state.project_type)
            next_task = task_manager.get_next_task(st.session_state.user_id)
            
            if next_task:
                task_manager.start_task(next_task, st.session_state.user_id)
                st.session_state.current_image = next_task
                
                if model:
                    try:
                        predictions = self.model_manager.predict_image(
                            st.session_state.current_image,
                            model
                        )
                        
                        canvas_objects = []
                        for i, pred in enumerate(predictions):
                            bbox = pred.get("bbox")
                            category_id = pred.get("category_id")
                            
                            if bbox and category_id:
                                category = next((cat for cat in self.project_data.get("categories", [])
                                              if cat["id"] == category_id), None)
                                
                                if category:
                                    color = category.get("color", "#FF6B00")
                                    
                                    x_offset = 90
                                    y_offset = 110
                                    
                                    canvas_obj = {
                                        "type": "rect",
                                        "left": float(bbox[0]) - x_offset,
                                        "top": float(bbox[1]) - y_offset,
                                        "width": float(bbox[2]),
                                        "height": float(bbox[3]),
                                        "stroke": color,
                                        "fill": f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)",
                                        "strokeWidth": 2,
                                        "id": str(i)
                                    }
                                    canvas_objects.append(canvas_obj)
                
                        st.session_state.annotations = predictions
                        st.session_state.canvas_objects = canvas_objects
                        
                    except Exception as e:
                        st.warning(f"Model prediction failed: {str(e)}")
                        st.session_state.annotations = []
                        st.session_state.canvas_objects = []

        if st.session_state.current_image:
            image = self.image_manager.load_image(
                st.session_state.current_image,
                zoom_level=st.session_state.zoom_level
            )
            orig_w, orig_h = self.image_manager.original_size
            scale = self.image_manager.get_scale_factor()
            
            if st.session_state.rotation:
                angle_rad = math.radians(st.session_state.rotation)
                cos_a = abs(math.cos(angle_rad))
                sin_a = abs(math.sin(angle_rad))
                
                rot_w = int(image.width * cos_a + image.height * sin_a)
                rot_h = int(image.width * sin_a + image.height * cos_a)
                
                rotated_image = Image.new('RGBA', (rot_w, rot_h), (0, 0, 0, 0))
                image = image.rotate(st.session_state.rotation, expand=True, resample=Image.Resampling.LANCZOS)
                paste_x = (rot_w - image.width) // 2
                paste_y = (rot_h - image.height) // 2
                rotated_image.paste(image, (paste_x, paste_y))
                image = rotated_image

            initial_objects = []
            if hasattr(st.session_state, 'canvas_objects') and st.session_state.canvas_objects:
                for obj in st.session_state.canvas_objects:
                    try:
                        scaled_obj = {
                            "type": "rect",
                            "left": float(obj["left"]) * scale,
                            "top": float(obj["top"]) * scale,
                            "width": float(obj["width"]) * scale,
                            "height": float(obj["height"]) * scale,
                            "stroke": obj.get("stroke", "#FF6B00"),
                            "fill": obj.get("fill", "rgba(255, 107, 0, 0.3)"),
                            "strokeWidth": 2,
                            "id": obj.get("id", "")
                        }
                        initial_objects.append(scaled_obj)
                    except Exception as e:
                        continue

            if not st.session_state.current_category:
                st.warning("⚠️ Please select a category before drawing")
                drawing_mode = "transform"
            else:
                drawing_mode = "rect"

            if st.session_state.current_category:
                category = next((cat for cat in self.project_data.get("categories", [])
                                if cat["name"] == st.session_state.current_category), None)
                if category and "color" in category:
                    stroke_color = category["color"]
                    r = int(stroke_color[1:3], 16)
                    g = int(stroke_color[3:5], 16)
                    b = int(stroke_color[5:7], 16)
                    fill_color = f"rgba({r}, {g}, {b}, 0.3)"
                else:
                    stroke_color = "#FF6B00"
                    fill_color = "rgba(255, 107, 0, 0.3)"
            else:
                stroke_color = "#FF6B00"
                fill_color = "rgba(255, 107, 0, 0.3)"

            canvas_result = st_canvas(
                fill_color=fill_color,
                stroke_width=2,
                stroke_color=stroke_color,
                background_image=image,
                drawing_mode=drawing_mode,
                key="annotation_canvas",
                update_streamlit=True,
                height=image.height,
                width=image.width,
                background_color="#000000",
                initial_drawing={
                    "version": "5.3.0",
                    "objects": initial_objects
                } if initial_objects else None
            )

            if canvas_result.json_data is not None:
                objects = canvas_result.json_data.get("objects", [])
                
                if st.session_state.current_category and len(objects) > len(st.session_state.annotations):
                    last_object = objects[-1]
                    
                    x = last_object["left"] / scale
                    y = last_object["top"] / scale
                    w = last_object["width"] / scale
                    h = last_object["height"] / scale
                    
                    category_id = next((cat["id"] for cat in self.project_data.get("categories", [])
                                        if cat["name"] == st.session_state.current_category), None)
                    
                    if category_id is not None:
                        st.session_state.annotations.append({
                            "category_id": category_id,
                            "bbox": [x, y, w, h],
                            "stroke_color": stroke_color,
                            "fill_color": fill_color
                        })

            st.markdown(f"""
                <div class="image-info">
                    {os.path.basename(st.session_state.current_image)}<br/>
                    {orig_w} × {orig_h}px
                </div>
            """, unsafe_allow_html=True)

        if st.session_state.current_image:
            st.markdown("""
                <div style='
                    position: fixed;
                    bottom: 80px;
                    left: 60%;
                    transform: translateX(-50%);
                    background: rgba(30, 30, 30, 0.95);
                    padding: 8px 20px;
                    border-radius: 4px;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    gap: 20px;
                    z-index: 1000;
                    border: 1px solid #333;
                '>
            """ + "".join([
                f"""
                <div style='display: flex; align-items: center; gap: 8px;'>
                    <div style='width: 12px; height: 12px; border-radius: 2px; background-color: {category.get("color", "#FF6B00")};'></div>
                    <div style='color: white; font-size: 12px;'>{category.get("name", "Unknown")}</div>
                </div>
                """ for category in self.project_data.get("categories", [])
            ]) + "</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Previous", use_container_width=True):
                if self._save_current_annotations():
                    task_manager = self.manager.get_task_manager(st.session_state.selected_project, st.session_state.project_type)
                    prev_task = task_manager.get_previous_task(st.session_state.current_image)
                    if prev_task:
                        st.session_state.current_image = prev_task
                        st.session_state.annotations = []
                        st.session_state.canvas_objects = []
                        return st.rerun()
                    else:
                        st.warning("No previous image available")

        with col2:
            if st.button("➡️ Next", use_container_width=True):
                if self._save_current_annotations():
                    task_manager = self.manager.get_task_manager(st.session_state.selected_project, st.session_state.project_type)
                    next_task = task_manager.get_next_available_task(st.session_state.user_id)
                    
                    if next_task:
                        st.session_state.current_image = next_task
                        
                        model_config = self.project_data.get("info", {}).get("model_config", {})
                        model_alias = model_config.get("model_alias")
                        
                        if model_alias:
                            model = self.model_manager.load_model(model_alias)
                            try:
                                predictions = self.model_manager.predict_image(next_task, model)
                                
                                canvas_objects = []
                                for i, pred in enumerate(predictions):
                                    bbox = pred.get("bbox")
                                    category_id = pred.get("category_id")
                                    
                                    if bbox and category_id:
                                        category = next((cat for cat in self.project_data.get("categories", [])
                                                      if cat["id"] == category_id), None)
                                        
                                        if category:
                                            color = category.get("color", "#FF6B00")
                                            canvas_obj = {
                                                "type": "rect",
                                                "left": float(bbox[0]) - 90,
                                                "top": float(bbox[1]) - 110,
                                                "width": float(bbox[2]),
                                                "height": float(bbox[3]),
                                                "stroke": color,
                                                "fill": f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)",
                                                "strokeWidth": 2,
                                                "id": str(i)
                                            }
                                            canvas_objects.append(canvas_obj)
                            
                                st.session_state.annotations = predictions
                                st.session_state.canvas_objects = canvas_objects
                            except Exception as e:
                                st.warning(f"Model prediction failed: {str(e)}")
                                st.session_state.annotations = []
                                st.session_state.canvas_objects = []
                    
                        return st.rerun()
                    else:
                        st.warning("No more images available")

        with col3:
            if st.button("💾 Save", use_container_width=True, type="primary"):
                with st.spinner("Saving annotations..."):
                    if self._save_current_annotations():
                        st.success("Annotations saved successfully!")
                        
                        task_manager = self.manager.get_task_manager(st.session_state.selected_project, st.session_state.project_type)
                        next_task = task_manager.get_next_task(st.session_state.user_id)
                        
                        if next_task:
                            st.session_state.current_image = next_task
                            
                            model_config = self.project_data.get("info", {}).get("model_config", {})
                            model_alias = model_config.get("model_alias")
                            
                            if model_alias:
                                model = ModelManager.get_model_instance(model_alias)
                                try:
                                    predictions = self.model_manager.predict_image(next_task, model)
                                    
                                    canvas_objects = []
                                    for i, pred in enumerate(predictions):
                                        bbox = pred.get("bbox")
                                        category_id = pred.get("category_id")
                                        
                                        if bbox and category_id:
                                            category = next((cat for cat in self.project_data.get("categories", [])
                                                          if cat["id"] == category_id), None)
                                            
                                            if category:
                                                color = category.get("color", "#FF6B00")
                                                canvas_obj = {
                                                    "type": "rect",
                                                    "left": float(bbox[0]) - 90,
                                                    "top": float(bbox[1]) - 110,
                                                    "width": float(bbox[2]),
                                                    "height": float(bbox[3]),
                                                    "stroke": color,
                                                    "fill": f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)",
                                                    "strokeWidth": 2,
                                                    "id": str(i)
                                                }
                                                canvas_objects.append(canvas_obj)
                                
                                    st.session_state.annotations = predictions
                                    st.session_state.canvas_objects = canvas_objects
                                except Exception as e:
                                    st.warning(f"Model prediction failed: {str(e)}")
                                    st.session_state.annotations = []
                                    st.session_state.canvas_objects = []
                        
                            return st.rerun()
                        else:
                            st.warning("No more images available")
                    else:
                        st.error("Failed to save annotations. Please try again.")

    def _save_current_annotations(self) -> None:
        try:
            task_manager = self.manager.get_task_manager(st.session_state.selected_project, st.session_state.project_type)
            annotations_file = os.path.join(self.project_dir, "annotations.json")
            
            data = self.file_tools.load_json(annotations_file)
            
            image_filename = os.path.basename(st.session_state.current_image)
            
            matching_images = [img for img in data["images"] 
                             if img["file_name"] == image_filename]
            
            if not matching_images:
                image_id = len(data["images"]) + 1
                data["images"].append({
                    "id": image_id,
                    "file_name": image_filename,
                    "width": self.image_manager.original_size[0],
                    "height": self.image_manager.original_size[1]
                })
            else:
                image_id = matching_images[0]["id"]
            
            data["annotations"] = [ann for ann in data["annotations"] 
                                 if ann["image_id"] != image_id]
            
            for ann in st.session_state.annotations:
                ann_id = len(data["annotations"]) + 1
                ann_data = {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "date_created": datetime.now().isoformat(),
                    "user_id": st.session_state.user_id
                }
                data["annotations"].append(ann_data)
            
            self.file_tools.save_json(data, annotations_file)
            
            success = task_manager.complete_task(
                st.session_state.current_image,
                st.session_state.user_id,
                st.session_state.annotations
            )
            
            return success
            
        except Exception as e:
            st.error(f"Error saving annotations: {str(e)}")
            return False

    def app_style(self) -> None:
        st.markdown("""
            <style>
            .block-container {
                padding-top: 0rem !important;
                padding-bottom: 0rem !important;
            }
            
            header {display: none !important;}
            footer {display: none !important;}
            
            h1, h2, h3, h4, h5, h6 {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
            
            .stApp {
                background-color: #1E1E1E;
                color: #E0E0E0;
            }
            
            .main > div {
                padding: 0;
                max-width: 100%;
            }
            
            .css-1d391kg {
                background-color: #252525;
            }
            
            .stButton button {
                background-color: #FF6B00;
                color: white;
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 4px;
                transition: background-color 0.3s;
            }
            
            .tool-panel {
                background-color: #252525;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                border: 1px solid #333;
            }
            
            .workspace {
                display: flex;
                gap: 1rem;
                margin-top: 1rem;
            }
            
            .canvas-container {
                flex-grow: 1;
                background-color: #2A2A2A;
                border-radius: 8px;
                padding: 1rem;
                border: 1px solid #333;
                margin-bottom: 150px !important;
            }
            
            .annotation-panel {
                width: 300px;
                background-color: #252525;
                border-radius: 8px;
                padding: 1rem;
                border: 1px solid #333;
            }
            
            .stMarkdown {
                margin-top: 0 !important;
                margin-bottom: 0 !important;
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }
            </style>
        """, unsafe_allow_html=True)

        st.components.v1.html("""
            <script>
            document.addEventListener('keydown', function(e) {
                if (e.ctrlKey && e.key === 's') {
                    e.preventDefault();
                    document.querySelector('button:contains("Save Annotations")').click();
                }
                if (e.key === 'ArrowRight') {
                    document.querySelector('button:contains("Next")').click();
                }
                if (e.key === 'ArrowLeft') {
                    document.querySelector('button:contains("Previous")').click();
                }
            });
            </script>
        """)
