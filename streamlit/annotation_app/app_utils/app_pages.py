import os
import streamlit as st
from PIL import ImageDraw
import json
import html
from streamlit_image_coordinates import streamlit_image_coordinates
from .annotation_manager import AnnotationManager
from .image_manager import ImageManager
from .task_manager import TaskManager
from .default_tools import DefaultTools
from .file_tools import FileTools

class AppPages:
    def __init__(self):
        self.project_dir = os.path.join('streamlit', 'annotation_app', 'projects')
        if not os.path.exists(self.project_dir):
            os.makedirs(self.project_dir)
            
        for project_type in ["Detection", "Keypoint"]:
            type_dir = os.path.join(self.project_dir, project_type)
            if not os.path.exists(type_dir):
                os.makedirs(type_dir)
        
        task_queue_file = os.path.join(self.project_dir, "task_queue.json")
        if not os.path.exists(task_queue_file):
            task_queue_data = {
                "available_images": [],
                "in_progress": {},
                "completed": {},
                "users": {}
            }
            FileTools.save_json(task_queue_data, task_queue_file)
        
        self.image_manager = ImageManager(project_dir=self.project_dir)
        self.annotation_manager = AnnotationManager()
        self.task_manager = TaskManager(project_dir=self.project_dir)
        self.project_data = None
        
    def show_welcome_page(self):
        # Remove default streamlit margins
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
        
        project_type = st.radio(":orange[Project Type]", [":orange[Detection]", ":orange[Keypoints]"], key="project_type", label_visibility="visible")
        
        project_name = st.text_input(":orange[Project Name]", key="project_name")
        
        project_description = st.text_area(":orange[Project Description]", key="project_desc")
        
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
                
            project_dir = f"streamlit/annotation_app/projects/{project_type}/{project_name}"
            if project_type == "Detection":
                config = {
                    "info": {
                        "description": project_description,
                        "type": "detection",
                        "date_created": st.session_state.get("date_created", "")
                    },
                    "categories": [cat for cat in categories if cat["name"] in selected_categories]
                }
            else:
                config = {
                    "info": {
                        "description": project_description,
                        "type": "keypoint",
                        "date_created": st.session_state.get("date_created", "")
                    },
                    "categories": [{
                        "id": 1,
                        "name": selected_preset,
                        "keypoints": keypoint_presets[selected_preset]["keypoints"],
                        "skeleton": keypoint_presets[selected_preset]["skeleton"]
                    }]
                }

            self.manager._create_project_structure(project_name, config)
            st.success("Project created successfully!")
            st.session_state.selected_project = project_name
            st.session_state.page = "add_media"
            st.rerun()

    def show_project_selection(self):
        # Remove top padding and add bottom padding
        st.markdown("""
            <style>
            .block-container {
                padding-top: 0rem !important;
                padding-bottom: 3rem !important;
            }
            
            /* Hide Streamlit header */
            header {display: none !important;}
            
            /* Project card styling */
            .project-card {
                background-color: #252525;
                padding: 1.5rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                border: 1px solid #333;
            }
            
            /* Header styling */
            .stMarkdown h1 {
                margin-top: 0 !important;
                padding-top: 0 !important;
                color: #FF6B00;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.title(":orange[Select Project]")
        
        project_types = ["Detection", "Keypoint"]
        for project_type in project_types:
            st.header(f":gray[{project_type} Projects]", divider="gray")
            projects_path = f"streamlit2/projects/{project_type}"
            if not os.path.exists(projects_path):
                continue
                
            projects = [p for p in os.listdir(projects_path) 
                       if os.path.isdir(os.path.join(projects_path, p))]
            
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
            
        st.title(f"Project: {st.session_state.selected_project}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div class='action-card'>
                    <h3>Add Content</h3>
                    <p>Upload new images or videos to annotate</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Upload Media"):
                st.session_state.page = "add_media"
                st.rerun()

        with col2:
            st.markdown("""
                <div class='action-card'>
                    <h3>Start Annotating</h3>
                    <p>Begin or continue annotation work</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("Open Annotator"):
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

    def handle_annotation_click(self, clicked, project_data, scale_factor):
        x, y = clicked['x'], clicked['y']
        project_type = project_data.get("info", {}).get("type", "unknown")
        
        if project_type == "detection":
            if not st.session_state.get('bbox_start'):
                if not st.session_state.current_category:
                    st.error("Please select a category first")
                    return
                st.session_state.bbox_start = (x, y)
                st.rerun()
            else:
                x1, y1 = st.session_state.bbox_start
                x2, y2 = x, y
                
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                scale_factor_w = scale_factor[0] if isinstance(scale_factor, tuple) else scale_factor
                scale_factor_h = scale_factor[1] if isinstance(scale_factor, tuple) else scale_factor
                
                x1 /= scale_factor_w
                y1 /= scale_factor_h
                x2 /= scale_factor_w
                y2 /= scale_factor_h
                
                try:
                    category_id = None
                    for cat in project_data.get("categories", []):
                        if cat["name"] == st.session_state.current_category:
                            category_id = cat["id"]
                            break
                    
                    if category_id is None:
                        st.error(f"Category {st.session_state.current_category} not found")
                        return
                    
                    st.session_state.annotations.append({
                        "category_id": category_id,
                        "bbox": [x1, y1, x2-x1, y2-y1]
                    })
                    
                    st.session_state.bbox_start = None
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error creating annotation: {str(e)}")
                    st.session_state.bbox_start = None
        else:
            st.session_state.annotations.append({
                "keypoints": [x/scale_factor_w, y/scale_factor_h, 2]
            })
            st.rerun()

    def show_annotation_interface(self):
        if not st.session_state.selected_project or not st.session_state.project_type:
            st.error("No project selected")
            return
            
        self.project_dir = f"streamlit2/projects/{st.session_state.project_type}/{st.session_state.selected_project}"
        self.image_manager = ImageManager(self.project_dir)
        
        with open(f"{self.project_dir}/annotations.json", "r") as f:
            self.project_data = json.load(f)
        
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        if 'annotations' not in st.session_state:
            st.session_state.annotations = []
        
        col1, col2 = st.columns([7,3])
        
        with col1:
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
                image = self.image_manager.load_image(st.session_state.current_image)
                display_image = self.image_manager.resize_for_display()
                
                img_draw = display_image.copy()
                draw = ImageDraw.Draw(img_draw)
                
                AnnotationManager.draw_annotations(draw, st.session_state.annotations, self.project_data, 
                               (self.image_manager.resized_ratio_w, self.image_manager.resized_ratio_h))
                
                if st.session_state.get('bbox_start'):
                    AnnotationManager.draw_bbox_preview(
                        draw, 
                        st.session_state.bbox_start,
                        st.session_state.get('current_point'),
                        st.session_state.current_category
                    )
                
                clicked = streamlit_image_coordinates(img_draw, key="annotator")
                
                if clicked and clicked != st.session_state.get('last_click'):
                    st.session_state.last_click = clicked
                    orig_x, orig_y = self.image_manager.display_to_original_coords(
                        clicked['x'], 
                        clicked['y']
                    )
                    self.handle_annotation_click(
                        {"x": orig_x, "y": orig_y}, 
                        self.project_data,
                        (self.image_manager.resized_ratio_w, self.image_manager.resized_ratio_h)
                    )
        
        with col2:
            st.markdown("### Controls")
            if st.button("Delete Last"):
                if st.session_state.annotations:
                    st.session_state.annotations.pop()
                    st.rerun()
            
            if st.button("Save & Next"):
                task_manager = self.manager.get_task_manager(st.session_state.selected_project)
                if task_manager.complete_task(
                    st.session_state.current_image,
                    st.session_state.user_id,
                    st.session_state.annotations
                ):
                    st.session_state.current_image = None
                    st.session_state.annotations = []
                    st.rerun()

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



