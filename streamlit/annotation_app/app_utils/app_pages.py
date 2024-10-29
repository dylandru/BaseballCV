import os
import streamlit as st
from PIL import ImageDraw, Image
import json
from streamlit_image_coordinates import streamlit_image_coordinates
from .annotation_manager import AnnotationManager
from .image_manager import ImageManager
from .task_manager import TaskManager
from .default_tools import DefaultTools
from .file_tools import FileTools
from datetime import datetime

class AppPages:
    def __init__(self):
        # Define base project directory
        self.base_project_dir = os.path.join('streamlit', 'annotation_app', 'projects')
        
        # Create base project directory if it doesn't exist
        os.makedirs(self.base_project_dir, exist_ok=True)
        
        # Create project type subdirectories
        for project_type in ["Detection", "Keypoint"]:
            type_dir = os.path.join(self.base_project_dir, project_type)
            os.makedirs(type_dir, exist_ok=True)
        
        # Initialize task queue file if it doesn't exist
        task_queue_file = os.path.join(self.base_project_dir, "task_queue.json")
        if not os.path.exists(task_queue_file):
            task_queue_data = {
                "available_images": [],
                "in_progress": {},
                "completed": {},
                "users": {}
            }
            FileTools.save_json(task_queue_data, task_queue_file)
        
        # Initialize managers
        self.image_manager = ImageManager(project_dir=self.base_project_dir)
        self.manager = AnnotationManager()
        self.task_manager = TaskManager(project_dir=self.base_project_dir)
        self.project_data = None
        
    def show_welcome_page(self):
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
                st.rerun()

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
                st.rerun()

    def create_project_screen(self):
        st.markdown("<h1 style='text-align: center; color: white'>Create New Project</h1>", unsafe_allow_html=True)
        
        project_type = st.radio(":orange[Project Type]", ["Detection", "Keypoints"])
        project_name = st.text_input(":orange[Project Name]")
        project_description = st.text_area(":orange[Project Description]")
        
        if project_type == "Detection":
            st.markdown("<h3 style='text-align: center; color: white'>Available Categories</h3>", unsafe_allow_html=True)
            categories = DefaultTools.init_baseball_categories()["detection"]
            selected_categories = st.multiselect(
                ":orange[Select categories to include]",
                options=[cat["name"] for cat in categories],
                default=[cat["name"] for cat in categories]
            )
        else:
            st.markdown("<h3 style='text-align: center; color: white'>Keypoint Presets</h3>", unsafe_allow_html=True)
            keypoint_presets = DefaultTools.init_baseball_categories()["keypoints"]
            selected_preset = st.selectbox(
                ":orange[Select keypoint configuration]",
                options=list(keypoint_presets.keys())
            )
        
        if st.button("Create Project"):
            if not project_name or not project_description:
                st.error("Please fill in both project name and description")
                return
                
            # Use self.base_project_dir for consistent paths
            project_dir = os.path.join(self.base_project_dir, project_type, project_name)
            st.write(f"Creating project in: {project_dir}")  # Debug print
            
            config = {
                "info": {
                    "description": project_description,
                    "type": "detection" if project_type == "Detection" else "keypoint",
                    "date_created": datetime.now().isoformat()
                },
                "categories": [cat for cat in categories if cat["name"] in selected_categories] if project_type == "Detection" else [{
                    "id": 1,
                    "name": selected_preset,
                    "keypoints": keypoint_presets[selected_preset]["keypoints"],
                    "skeleton": keypoint_presets[selected_preset]["skeleton"]
                }]
            }

            try:
                self.manager._create_project_structure(project_name, config)
                st.success(f"Project created successfully in {project_dir}")
                st.session_state.selected_project = project_name
                st.session_state.project_type = project_type
                st.session_state.page = "add_media"
                st.rerun()
            except Exception as e:
                st.error(f"Error creating project: {str(e)}")

    def show_project_selection(self):
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
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
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
                            st.rerun()
                    with col2:
                        if st.button("Details", key=f"details_{project}"):
                            st.session_state.project_details = project
                            st.rerun()

    def show_project_dashboard(self):
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
                st.rerun()

        with col2:
            st.markdown("""
                <div class='action-card' style='display: flex; flex-direction: column; align-items: center; justify-content: center; margin: auto; width: 80%;'>
                    <h3 style='color: orange; text-align: center;'>Start Annotating</h3>
                    <p style='text-align: center;'>Begin or continue annotation work</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Open Annotator", use_container_width=True):
                st.session_state.page = "annotate"
                st.rerun()

    def show_add_media_page(self):
        if not st.session_state.selected_project:
            st.error("No project selected")
            return
            
        st.title("Add Media")
        
        tab1, tab2 = st.tabs(["Upload Images", "Upload Video"])
        
        with tab1:
            uploaded_files = st.file_uploader(
                "Upload Images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("Add Images"):
                    try:
                        num_added = self.manager.add_images_to_task_queue(
                            st.session_state.selected_project,
                            uploaded_files
                        )
                        st.success(f"Added {num_added} images successfully")
                        st.session_state.page = "project_dashboard"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding images: {str(e)}")

        with tab2:
            video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
            if video_file:
                frame_interval = st.slider("Extract every nth frame", 1, 30, 5)
                if st.button("Process Video"):
                    try:
                        frames = self.manager.handle_video_upload(
                            st.session_state.selected_project,
                            video_file,
                            frame_interval
                        )
                        st.success(f"Extracted {len(frames)} frames from video")
                        st.session_state.page = "project_dashboard"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")

    def show_progress_page(self):
        st.title("Project Progress")
        
        if not st.session_state.selected_project:
            st.error("No project selected")
            return
            
        task_manager = self.manager.get_task_manager(st.session_state.selected_project)
        

        tasks_data = FileTools.load_json(task_manager.tasks_file)
        total_images = (len(tasks_data["available_images"]) + 
                       len(tasks_data["in_progress"]) + 
                       len(tasks_data["completed"]))
        
        completed = len(tasks_data["completed"])
        in_progress = len(tasks_data["in_progress"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Images", total_images)
        with col2:
            st.metric("Completed", completed)
        with col3:
            completion_rate = (completed / total_images * 100) if total_images > 0 else 0
            st.metric("Completion Rate", f"{completion_rate:.1f}%")

    def show_annotation_interface(self):
        if not st.session_state.selected_project or not st.session_state.project_type:
            st.error("No project selected")
            return
                
        # Load project data using the correct project type path
        self.project_dir = os.path.join(self.base_project_dir, 
                                        st.session_state.project_type,  # Use project_type from session state
                                        st.session_state.selected_project)
        
        try:
            with open(os.path.join(self.project_dir, "annotations.json"), "r") as f:
                self.project_data = json.load(f)
        except Exception as e:
            st.error(f"Error loading project data: {str(e)}")
            return

        # Initialize session states
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        if 'annotations' not in st.session_state:
            st.session_state.annotations = []
        if 'zoom_level' not in st.session_state:
            st.session_state.zoom_level = 1.0
        if 'rotation' not in st.session_state:
            st.session_state.rotation = 0

        # CSS for minimal spacing and optimal layout
        st.markdown("""
            <style>
            /* Reset container padding and margins */
            .block-container {
                padding: 0 !important;
                margin: 0 !important;
            }
            
            div[data-testid="stVerticalBlock"] > div {
                padding: 0 !important;
                margin: 0 !important;
            }
            
            /* Top controls bar */
            .top-controls {
                background: #1E1E1E;
                border-bottom: 1px solid #333;
                padding: 5px;
                margin: 0;
                display: flex;
                align-items: center;
            }
            
            /* Image workspace */
            .image-workspace {
                background: #2D2D2D;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 5px;
                margin: 0;
                min-height: calc(100vh - 120px);
            }
            
            /* Center align image container */
            .stImage {
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
            }
            
            /* Image info overlay */
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
            
            /* Annotation label */
            .annotation-label {
                background: #FF6B00;
                color: white;
                padding: 2px 6px;
                border-radius: 2px;
                font-size: 12px;
            }
            
            /* Button styling */
            .stButton button {
                padding: 2px 8px !important;
                height: 30px !important;
                min-height: 30px !important;
            }
            
            /* Remove selectbox label spacing */
            .stSelectbox label {
                display: none !important;
            }
            
            /* Hide streamlit extras */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
        """, unsafe_allow_html=True)

        # Top Controls
        cols = st.columns([1, 1, 1, 1, 1, 1, 2])
        
        with cols[0]:
            st.button("📂 Open", use_container_width=True)
        with cols[1]:
            if st.button("⟲ Rotate", use_container_width=True):
                st.session_state.rotation = (st.session_state.rotation - 90) % 360
        with cols[2]:
            if st.button("🔍 Zoom In", use_container_width=True):
                st.session_state.zoom_level *= 1.2
        with cols[3]:
            if st.button("🔍 Zoom Out", use_container_width=True):
                st.session_state.zoom_level /= 1.2
        with cols[4]:
            if st.button("↔ Fit", use_container_width=True):
                st.session_state.zoom_level = 1.0
        with cols[5]:
            st.button("✋ Move", use_container_width=True)
        with cols[6]:
            st.selectbox(
                "Category",
                options=[cat['name'] for cat in self.project_data.get("categories", [])],
                key="current_category",
                label_visibility="collapsed"
            )

        # Main Image Area
        if not st.session_state.current_image:
            next_task = self.manager.get_task_manager(st.session_state.selected_project).get_next_task(
                st.session_state.user_id
            )
            if next_task:
                st.session_state.current_image = next_task
            else:
                st.info("No images available for annotation")
                return

        if st.session_state.current_image:
            # Load and process image
            image = Image.open(st.session_state.current_image)
            orig_w, orig_h = image.size
            
            # Calculate optimal display size
            max_width = 1200
            max_height = 700
            scale = min(max_width/orig_w, max_height/orig_h) * st.session_state.zoom_level
            new_size = (int(orig_w * scale), int(orig_h * scale))
            
            # Apply transformations
            image = image.resize(new_size)
            if st.session_state.rotation:
                image = image.rotate(st.session_state.rotation, expand=True)

            # Prepare drawing surface
            img_draw = image.copy()
            if img_draw.mode != 'RGB':
                img_draw = img_draw.convert('RGB')
            draw = ImageDraw.Draw(img_draw)

            # Draw existing annotations
            for ann in st.session_state.annotations:
                if 'bbox' in ann:
                    x, y, width, height = ann['bbox']
                    x1, y1 = x*scale, y*scale
                    x2, y2 = (x+width)*scale, (y+height)*scale
                    
                    # Draw box
                    draw.rectangle([x1, y1, x2, y2], outline='#FF6B00', width=3)
                    
                    # Get and draw category name
                    category_name = next(
                        (cat["name"] for cat in self.project_data.get("categories", [])
                        if cat["id"] == ann["category_id"]),
                        "Unknown"
                    )
                    
                    # Draw label background and text
                    text_width = len(category_name) * 6
                    draw.rectangle([x1, y1-20, x1+text_width+4, y1], fill='#FF6B00')
                    draw.text((x1+2, y1-18), category_name, fill='white')
                    
                elif 'keypoints' in ann:
                    x, y, v = ann['keypoints']
                    draw.ellipse(
                        [(x*scale)-5, (y*scale)-5, (x*scale)+5, (y*scale)+5],
                        fill='#FF6B00'
                    )

            # Draw current box
            if st.session_state.get('bbox_start'):
                start_x, start_y = st.session_state.bbox_start
                current_x = st.session_state.get('current_point', (start_x, start_y))[0]
                current_y = st.session_state.get('current_point', (start_x, start_y))[1]
                draw.rectangle(
                    [start_x, start_y, current_x, current_y],
                    outline='#FF6B00',
                    width=2
                )

            # Display image filename and dimensions
            st.markdown(f"""
                <div class="image-info">
                    {os.path.basename(st.session_state.current_image)}<br/>
                    {orig_w} × {orig_h}px
                </div>
            """, unsafe_allow_html=True)

            # Handle image clicks
            clicked = streamlit_image_coordinates(img_draw, key="annotator")
            if clicked and clicked != st.session_state.get('last_click'):
                st.session_state.last_click = clicked
                self.handle_annotation_click(clicked, self.project_data, (scale, scale))

        # Bottom Controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.button("⬅️ Previous", use_container_width=True)
        with col2:
            st.button("➡️ Next", use_container_width=True)
        with col3:
            if st.button("🗑️ Delete Last", use_container_width=True):
                if st.session_state.annotations:
                    st.session_state.annotations.pop()
                    st.rerun()
        with col4:
            if st.button("💾 Save", use_container_width=True, type="primary"):
                if self.save_current_annotations():
                    st.session_state.current_image = None
                    st.session_state.annotations = []
                    st.rerun()
                    
    def handle_annotation_click(self, clicked, project_data, scale_factor):
        x, y = clicked['x'], clicked['y']
        scale_factor_w = scale_factor[0] if isinstance(scale_factor, tuple) else scale_factor
        scale_factor_h = scale_factor[1] if isinstance(scale_factor, tuple) else scale_factor
        
        if project_data.get("info", {}).get("type", "unknown") == "detection":
            if not st.session_state.current_category:
                st.error("Select a category first")
                return
                
            if not st.session_state.get('bbox_start'):
                st.session_state.bbox_start = (x, y)
                st.session_state.current_point = (x, y)
                st.rerun()
            else:
                x1, y1 = st.session_state.bbox_start
                x2, y2 = x, y
                
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                x1 /= scale_factor_w
                y1 /= scale_factor_h
                x2 /= scale_factor_w
                y2 /= scale_factor_h
                
                category_id = next((cat["id"] for cat in project_data.get("categories", [])
                                if cat["name"] == st.session_state.current_category), None)
                
                if category_id is not None:
                    st.session_state.annotations.append({
                        "category_id": category_id,
                        "bbox": [x1, y1, x2-x1, y2-y1]
                    })
                
                st.session_state.bbox_start = None
                st.session_state.current_point = None
                st.rerun()
        else:
            st.session_state.annotations.append({
                "keypoints": [x/scale_factor_w, y/scale_factor_h, 2]
            })
            st.rerun()

    def save_current_annotations(self):
        """Save annotations to project directory"""
        try:
            # Debug prints
            print(f"Current image: {st.session_state.current_image}")
            print(f"Project dir: {self.project_dir}")
            print(f"Current annotations: {st.session_state.annotations}")
            
            # Save to annotations.json
            annotations_file = os.path.join(self.project_dir, "annotations.json")
            print(f"Loading annotations from: {annotations_file}")
            
            with open(annotations_file, "r") as f:
                data = json.load(f)
            
            # Get image ID
            image_filename = os.path.basename(st.session_state.current_image)
            print(f"Looking for image: {image_filename}")
            
            matching_images = [img for img in data["images"] 
                             if img["file_name"] == image_filename]
            
            if not matching_images:
                st.error(f"Image {image_filename} not found in annotations.json")
                return False
            
            image_id = matching_images[0]["id"]
            print(f"Found image ID: {image_id}")
            
            # Add annotations
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
                print(f"Added annotation: {ann_data}")
            
            # Save updated annotations
            print("Saving updated annotations...")
            with open(annotations_file, "w") as f:
                json.dump(data, f, indent=4)
            
            # Update task queue
            print("Updating task queue...")
            task_manager = self.manager.get_task_manager(st.session_state.selected_project)
            success = task_manager.complete_task(
                st.session_state.current_image,
                st.session_state.user_id,
                st.session_state.annotations
            )
            
            if success:
                st.success("Annotations saved successfully!")
                return True
            else:
                st.error("Failed to update task queue")
                return False
            
        except Exception as e:
            st.error(f"Error saving annotations: {str(e)}")
            print(f"Full error details: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def app_style(self):
        st.markdown("""
            <style>
            /* Remove top padding and header */
            .block-container {
                padding-top: 0rem !important;
                padding-bottom: 0rem !important;
            }
            
            /* Hide Streamlit header and footer */
            header {display: none !important;}
            footer {display: none !important;}
            
            /* Remove top margin from all headers */
            h1, h2, h3, h4, h5, h6 {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
            
            /* Rest of your styles */
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
            }
            
            .annotation-panel {
                width: 300px;
                background-color: #252525;
                border-radius: 8px;
                padding: 1rem;
                border: 1px solid #333;
            }
            
            /* Remove default margins and padding */
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













