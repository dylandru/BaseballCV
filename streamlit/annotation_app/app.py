import os
import logging
import streamlit as st
from app_utils import AppPages, FileTools, TaskManager, ImageManager, AnnotationManager, DefaultTools

# Set up logging configuration
logging.basicConfig(
    filename=os.path.join(os.path.expanduser("~"), "annotation_app.log"),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Log the current working directory
    current_working_directory = os.getcwd()
    logging.info(f"Current working directory: {current_working_directory}")

    # Log the process ID and user ID
    process_id = os.getpid()
    user_id = os.getuid()
    logging.info(f"Process ID: {process_id}, User ID: {user_id}")

    st.set_page_config(
        layout="wide",
        page_title="Baseball Annotation Tool",
        page_icon="⚾"
    )

    # Initialize project structure at startup
    DefaultTools.init_project_structure()

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "welcome"
    if 'selected_project' not in st.session_state:
        st.session_state.selected_project = None
    if 'project_type' not in st.session_state:
        st.session_state.project_type = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'current_category' not in st.session_state:
        st.session_state.current_category = None
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
    if 'bbox_start' not in st.session_state:
        st.session_state.bbox_start = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

    app_pages = AppPages()
    app_pages.app_style()

    if not st.session_state.user_id or not st.session_state.get('email'):
        with st.sidebar:
            st.markdown("<h1 style='text-align: center; font-size: 3rem;'>⚾ BASEBALLCV ⚾</h1>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("### User Login")
            user_id = st.text_input("Enter your username:")
            email = st.text_input("Enter your email:")
            if user_id and email:
                st.session_state.user_id = user_id
                if str(email).endswith(".com"):
                    st.session_state.email = email
                    st.rerun()
                else:
                    st.error("Invalid Email Address!")
        st.markdown("""
            <div style='text-align: center; padding: 7 rem; color: white;'>
                <h1 style='color: white; font-size: 5
                    rem;'>BaseballCV Annotation Tool</h1>
                <p style='font-size: 2rem; color: #FF6B00;'>
                    Please enter Username and Email in the Sidebar to Continue...
                </p>
            </div>
        """, unsafe_allow_html=True)
        return

    with st.sidebar:
        st.markdown("<h1 style='text-align: center; font-size: 3rem;'>⚾ BASEBALLCV ⚾</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: black; text-align: center'>User: {st.session_state.user_id}</h3>", unsafe_allow_html=True)
        st.markdown("---")

        if st.session_state.page != "welcome":
            if st.button("← Back to Home", key="nav_home"):
                st.session_state.page = "welcome"
                st.session_state.selected_project = None
                st.rerun()

            if st.session_state.selected_project:
                st.markdown(f"<h3 style='color: black; text-align: center'>Current Project: {st.session_state.selected_project}</h3>", unsafe_allow_html=True)
                st.markdown("---")

                if st.button("Project Dashboard", key="nav_dashboard"):
                    st.session_state.page = "project_dashboard"
                    st.rerun()
                if st.button("Upload Media", key="nav_upload"):
                    st.session_state.page = "add_media"
                    st.rerun()
                if st.button("Start Annotating", key="nav_annotate"):
                    st.session_state.page = "annotate"
                    st.rerun()
                if st.button("View Progress", key="nav_progress"):
                    st.session_state.page = "progress"
                    st.rerun()

        st.markdown("---")
        if st.button("Logout", key="logout"):
            st.session_state.user_id = None
            st.session_state.page = "welcome"
            st.session_state.selected_project = None
            st.rerun()

    if st.session_state.page == "welcome":
        app_pages.show_welcome_page()
    elif st.session_state.page == "create_project":
        app_pages.create_project_screen()
    elif st.session_state.page == "select_project":
        app_pages.show_project_selection()
    elif st.session_state.page == "project_dashboard":
        app_pages.show_project_dashboard()
    elif st.session_state.page == "add_media":
        app_pages.show_add_media_page()
    elif st.session_state.page == "progress":
        app_pages.show_progress_page()
    elif st.session_state.page == "annotate":
        app_pages.show_annotation_interface()

if __name__ == "__main__":
    main()
