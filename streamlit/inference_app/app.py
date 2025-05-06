import streamlit as st
from app_utils import YOLOInference
import os
from typing import Literal
from baseballcv.functions import DataTools

# TODO: Make the app run on mp4 video plus multiple images, also create tool to make custom dataset, 
# Check to see what file types we want to use for inference
# Need to add download and clear functionality
# Use this as an example: https://www.youtube.com/watch?v=tdqpPKYKl_g


class InferenceApp:
    def __init__(self):
        self.acceptable_alias = {
            'Pitcher Hitter Catcher Detector': 'phc_detector'
        }

        self.app_files = []

    def _dropdown(self):
        alias_model_type_dict = {
            'YOLO' : ['Pitcher Hitter Catcher Detector'],
            'RFDTER': ['Hello', 'Hi']
        }
        self.model = st.sidebar.selectbox('Type of Model',
                     list(alias_model_type_dict.keys()))
        self.alias = st.sidebar.selectbox('Model Alias',
                         alias_model_type_dict.get(self.model))
        
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

    def _run_inference(self, files, is_video, model_type, confidence):
        model_alias = self.acceptable_alias.get(self.alias)

        if model_type == 'YOLO':
            model = YOLOInference(model_alias, confidence)


        if is_video:
            return model.infer_video(files)
        else:
            return model.infer_image(files)
        

    def run(self):
        st.set_page_config(
        layout="wide",
        page_title="Baseball Inference Tool",
        page_icon=":baseball:"
        )

        st.title("Baseball Inference Tool")

        st.sidebar.header("Model Configuration")

        self._dropdown()

        files = st.sidebar.file_uploader(label='Upload File', accept_multiple_files=True, type=['jpg', 'jpeg', 'png', 'mp4'])

        confidence = float(st.sidebar.slider("Confidence", 25, 100, 50)) / 100

        self.app_files.extend(files)
       
        if files:
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


        if st.sidebar.button("Run Model"):
            try:
                is_video = self.app_files[0].name.endswith('.mp4') # True for video, false for image

                annot_obj = self._run_inference(self.app_files, is_video, self.model, confidence)

                if is_video:
                    st.video(annot_obj)
                    os.remove(annot_obj)

                else:
                    for img in annot_obj:
                        st.image(img)
            except Exception as e:
                st.error(f"Error. Most Likely you didn't upload a file before running. {e}")
        
        self._css_styling()


if __name__ == '__main__':
    app = InferenceApp()
    app.run()