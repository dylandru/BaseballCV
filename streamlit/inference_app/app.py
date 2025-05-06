import streamlit as st
from app_utils import YOLOInference, File, DatasetCreator
import os

# TODO: Add some more models
# Check to see what file types we want to use for inference
# Need to add download and clear functionality

class InferenceApp:
    def __init__(self):
        self.acceptable_alias = {
            'Pitcher Hitter Catcher Detector': 'phc_detector'
        }

        self.app_files = []
        
        if 'click_count' not in st.session_state:
            st.session_state.click_count = 0
        
        self.file = File()
        self.dataset_creator = DatasetCreator()

    def _dropdown(self):
        alias_model_type_dict = {
            'YOLO' : ['Pitcher Hitter Catcher Detector'],
            'RFDTER': ['Hello', 'Hi']
        }
        self.model = st.sidebar.selectbox('Type of Model',
                     list(alias_model_type_dict.keys()))
        self.alias = st.sidebar.selectbox('Model Alias',
                         alias_model_type_dict.get(self.model))
        
    def _create_random_video(self):
        if st.session_state.click_count < 4:
            self.dataset_creator.generate_video(self.file.videos_dir)
            st.session_state.click_count += 1

            video_file = self.file.videos_dir

            video_file = os.path.join(video_file, os.listdir(video_file)[0])

            with open(video_file, 'rb') as video:
                st.sidebar.download_button("Download Video Here", video.read(), file_name="random_sample_video.mp4", mime="video/mp4")
            self.file.clear()

        else:
            st.error("Sorry, Max Downloads exceeded")
        
    def _css_styling(self) -> None:
        st.markdown("""
        <style>
            [data-testid='stFileUploader'] {
                width: max-content;
            }
            [data-testid='stFileUploader'] section {
                padding: 0;
                float: left;
            }
            [data-testid='stFileUploader'] section > input + div {
                display: none;
            }
            [data-testid='stFileUploader'] section + div {
                float: right;
                padding-top: 0;
            }

        </style>
        """, unsafe_allow_html=True)

    def _get_model(self, model_type, confidence):
        model_alias = self.acceptable_alias.get(self.alias)
        model = None

        if model_type == 'YOLO':
            model = YOLOInference(model_alias, confidence, self.file.models_dir_path)

        return model

    def _check_structure(self):
        all_images = all(file.name.endswith(('.jpg', '.jpeg', '.png')) for file in self.app_files)
        all_videos = all(file.name.endswith('.mp4') for file in self.app_files)

        if not (all_videos or all_images):
            st.error('All uploaded files must be an Image OR Video (in .mp4 format), not both')

        if all_videos and len(self.app_files) > 1:
            st.warning("We are only supporting 1 video")
            self.app_files = self.app_files[0]
        elif all_images and len(self.app_files) > 10:
            st.warning("We are only supporting 10 images")
            self.app_files = self.app_files[:10]

    def run(self):
        st.set_page_config(
        layout="wide",
        page_title="Baseball Inference Tool",
        page_icon=":baseball:"
        )

        st.title("Baseball Inference Tool")

        st.sidebar.header("Model Configuration")

        if st.sidebar.button("Download Random Video"):
            self._create_random_video()
            
        self._dropdown()

        files = st.sidebar.file_uploader(label='Upload File', accept_multiple_files=True, type=['jpg', 'jpeg', 'png', 'mp4'])

        confidence = float(st.sidebar.slider("Confidence", 25, 100, 50)) / 100

        self.app_files.extend(files)
       
        if files:
            self._check_structure()

        random_frames = st.sidebar.radio('Generate Random Frames from Video?', ['Yes', 'No'], index=1)
    
        if st.sidebar.button("Run Model"):
            try:

                is_video = self.app_files[0].name.endswith('.mp4') and random_frames == 'No' # True for video, false for image

                model = self._get_model(self.model, confidence)

                if is_video:
                    out, cap, length = self.file.write_video(self.app_files[0])
                    model.infer_video(out, cap, length)
                    st.video(self.file.annot_video_path)

                else:
                    if random_frames == 'Yes':
                        _, cap, length = self.file.write_video(self.app_files[0], False)
                        self.dataset_creator.generate_example_images(self.file.imgs_dir, cap, length)
                        self.app_files.clear()

                        self.app_files = [
                                            os.path.join(self.file.imgs_dir, item)
                                            for item in os.listdir(self.file.imgs_dir)
                                        ]

                    for img in self.app_files:
                        st.image(model.infer_image(img))
                
                self.file.clear()

            except Exception as e:
                st.error(f"Error. Most Likely you didn't upload a file before running. {e}")
        
        self._css_styling()


if __name__ == '__main__':
    app = InferenceApp()
    app.run()