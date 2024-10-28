import os
import streamlit as st

def main():
    st.set_page_config(
        layout="wide",
        page_title="Baseball Annotation Tool",
        page_icon="⚾"
    )
    
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
    
    # Handle user authentication
    if not st.session_state.user_id:
        with st.sidebar:
            st.title("⚾ Baseball CV")
            st.markdown("---")
            st.markdown("### User Login")
            user_id = st.text_input("Enter your username")
            if user_id:
                st.session_state.user_id = user_id
                st.rerun()
        st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h1>⚾ Baseball Annotation Tool</h1>
                <p style='font-size: 1.2rem; color: #FF6B00;'>
                    Please enter Username in the sidebar to continue
                </p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Add navigation in sidebar
    with st.sidebar:
        st.title("Baseball CV")
        st.markdown(f"**User:** {st.session_state.user_id}")
        st.markdown("---")
        
        if st.session_state.page != "welcome":
            if st.button("← Back to Home", key="nav_home"):
                st.session_state.page = "welcome"
                st.session_state.selected_project = None
                st.rerun()
            
            if st.session_state.selected_project:
                st.markdown(f"**Current Project:** {st.session_state.selected_project}")
                st.markdown("---")
                
                # Project navigation
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
        
        # Add logout button
        st.markdown("---")
        if st.button("Logout", key="logout"):
            st.session_state.user_id = None
            st.session_state.page = "welcome"
            st.session_state.selected_project = None
            st.rerun()

    #TODO: Create Pages Based on Needs for Project
    
    # Page routing
    # if st.session_state.page == "welcome":
    #     #show_welcome_page()
    # elif st.session_state.page == "create_project":
    #     #create_project_screen()
    # elif st.session_state.page == "select_project":
    #     #show_project_selection()
    # elif st.session_state.page == "project_dashboard":
    #     #show_project_dashboard()
    # elif st.session_state.page == "add_media":
    #     #show_add_media_page()
    # elif st.session_state.page == "progress":
    #     #show_progress_page()
    # elif st.session_state.page == "annotate":
    #     #show_annotation_interface()

if __name__ == "__main__":
    main()
