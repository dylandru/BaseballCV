import streamlit as st
from app_utils import (File, 
                       DatasetCreator, 
                       YOLOInference, 
                       YOLOv9Inference,
                       RFDETRInference, 
                       PaligemmaInference, 
                       FlorenceInference)
import os
import zipfile
import io
from PIL import Image

class InferenceApp:
    """
    Class that handles the inference streamlit app.
    """
    def __init__(self) -> None:
        """
        Initializes the `InferenceApp` class with model options, session handled files, 
        session handled click counts, and the temporary session file structure.
        """
        self.options = {
            'YOLO': {
                'Pitcher Hitter Catcher Detector': 'phc_detector',
                'Bat Tracking' : 'bat_tracking',
                'Ball Tracking' : 'ball_tracking',
                'Ball Tracking v4' : 'ball_trackingv4',
                'Glove Tracking' : 'glove_tracking'
            },
            'YOLOv9': {
                'Homeplate Tracking' : 'homeplate_tracking'
            }
        }

        self.app_files = []
        
        if 'click_count' not in st.session_state:
            st.session_state.click_count = 0
        
        self.file = File()
        
    def _create_random_video(self, dataset_creator: DatasetCreator) -> None:
        """
        Generates a random video for the user if they don't have one to use. Also, handles
        clicks for each session to limit the number of times the button is used. (Limited to 4)

        Args:
            dataset_creator (DatasetCreator): A object that called for the `generate_video` function.
        """
        if st.session_state.click_count < 4:
            try:
                dataset_creator.generate_video(self.file.videos_dir)
                st.session_state.click_count += 1

                video_file = self.file.videos_dir

                video_file = os.path.join(video_file, os.listdir(video_file)[0])

                video_bytes = open(video_file, 'rb').read()
                st.sidebar.download_button("Download Video Here", video_bytes, file_name="random_sample_video.mp4", mime="video/mp4")
                self.file.clear()
            except Exception: 
                st.error("Failed to get video. Please try again.")

        else:
            st.error("Sorry, Max Downloads exceeded")
        
    def _css_styling(self) -> None:
        """
        Applies custom CSS styling to the Streamlit application for a
        dark theme with orange accents.
        """
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
            [data-testid='stFileUploader'] section > button {
                color: #FFFFFF;
                background-color: #FFA500;
                border: 1px solid #FFA500;
            }      
            .main {
                background-color: #000000;
                color: #FFFFFF;
            }
            body {
                background-color: #000000;
                color: #FFFFFF;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #FFFFFF;
            }
            .stButton>button {
                color: #FFFFFF;
                background-color: #FFA500;
                border: 1px solid #FFA500;
            }
            .stDownloadButton>button {
                color: #FFFFFF;
                background-color: #FFA500;
                border: 1px solid #FFA500;
            }
            .stTextInput>div>div>input {
                 background-color: #333333;
                 color: #FFFFFF;
            }
             .stTextArea>div>div>textarea {
                 background-color: #333333;
                 color: #FFFFFF;
            }
            .stExpander summary:hover svg {
                fill: #FFA500; 
            }
            .stExpander summary:hover {
                color: #FFA500;
                cursor: pointer;
            }
            .stExpander code {
                color: #FFA500;
            }
            div[data-baseweb="progress-bar"][role="progressbar"] > div > div > div {
                background-color: #FFA500 !important;
            }
        </style>
        """, unsafe_allow_html=True)

    def _display_instructions(self) -> None:
        """
        Generates a markdown of instructions to enhance the user experience. 
        """
        with st.expander("How to use this App"):
            st.markdown("""
                        Welcome to the **Baseball Inference Tool**. This is an app dedicated to 
                        allow you to play around with various computer vision tools incorporated in 
                        the `baseballcv` package. There are different ways to use this app.

                        ### Model Selection
                        On the sidebar, you have the choice of picking the different models you can use.
                        Based on the type of model, the corresponding alias will pop up to make selecting easier.

                        ### Uploading File
                        You have the ability to upload you own files for images (png, jpeg, jpg) or videos (mp4). You are limited
                        to 10 images and 1 video per inference. If you don't have a video or image, you can extract one by selecting
                        the `Download Random Video` button on the sidebar. You are limited to 4 videos to extract. **Note**: The sizes of the
                        files must be less than 200 MB.

                        ### Generate Random Frames From Video
                        This feature is used if you don't have any images to use and don't want to use up time to inference an entire
                        video. Simply select yes, and the model will pick 3 random frames of the video and output their inferences.

                        ### Run the Model
                        When you are ready and have the configuration you want, simply select this button and it will run the inference
                        on the file you uploaded. Once it's ran, you will have the option to download the desired image or video, and then
                        you can show it off to your peers.

                        **Note**: On every session instance, running the model may require installation of the model which can take an extra
                        minute so please be patient when running an inference. 
                        """)

    def _get_model(self, model_type: str, alias: str, **kwargs) -> (YOLOInference | RFDETRInference | PaligemmaInference | YOLOv9Inference | FlorenceInference | None):
        """
        Extracts the desired model based on the user input and instantiates it.

        Args:
            model_type (str): The type of model the user wants to use.
            alias (str): The alias of the respective model. Each alias is unique to each model. 
            **kwargs: Additional arguments that can be used to enhance the model experience.
            For example, confidence can be used for annotations. Hopefully, more arguments can
            be used in the future. 
        
        Returns:
            The respective instantiated model used for inferencing. 
        """
        model = None

        if model_type == 'YOLO':
            model = YOLOInference(alias, self.file.models_dir_path, self._css_styling, 
                                  confidence=kwargs.get('confidence'))

        elif model_type == 'RF-DETR':
            model = RFDETRInference(alias, self.file.models_dir_path, self._css_styling, 
                                    confidence=kwargs.get('confidence'))

        elif model_type == 'Paligemma':
            model = PaligemmaInference(alias, self.file.models_dir_path, self._css_styling, 
                                       confidence=kwargs.get('confidence'))
        
        elif model_type == 'YOLOv9':
            model = YOLOv9Inference(alias, self.file.models_dir_path, self._css_styling, 
                                    confidence=kwargs.get('confidence'))

        elif model_type == 'Florence':
            model = FlorenceInference(alias, self.file.models_dir_path, self._css_styling, 
                                      confidence=kwargs.get('confidence'))

        return model

    def _check_structure(self) -> None:
        """
        Checks the structure of the file uploads. What this is looking for:
        * All files are either images or videos
        * The length of the uploaded videos must be 1 (It will take forevor to do more than 1).
        * The length of the uploaded images must be less than 10.

        This is done to limit computational resources and time. 
        """
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

    def run(self) -> None:
        """
        Sets up the Streamlit page configuration and orchestrates the display
        and interaction logic of the application.
        """
        st.set_page_config(
        layout="wide",
        page_title="Baseball Inference Tool",
        page_icon=":baseball:"
        )

        dataset_creator = DatasetCreator(self._css_styling)
        st.title("Baseball Inference Tool")

        st.sidebar.header("Model Configuration")

        if st.sidebar.button("Download Random Video"):
            self._create_random_video(dataset_creator)
        
        self._display_instructions()

        #####################
        # Handle Model Inputs
        #####################

        model_type = st.sidebar.selectbox('Type of model', self.options.keys())
        alias = st.sidebar.selectbox('Model Alias', self.options.get(model_type).keys())

        files = st.sidebar.file_uploader(label='Upload File', accept_multiple_files=True, type=['jpg', 'jpeg', 'png', 'mp4'])

        confidence = float(st.sidebar.slider("Confidence", 0, 100, 50)) / 100

        ###########################
        # End Handling Model Inputs
        ###########################

        self.app_files.extend(files)
       
        if files:
            self._check_structure()

        random_frames = st.sidebar.radio('Generate Random Frames from Video?', ['Yes', 'No'], index=1)
    
        if st.sidebar.button("Run Model"):
            try:
                model = self._get_model(model_type, self.options.get(model_type).get(alias), confidence=confidence)

                col1, col2 = st.columns([4, 1], gap='medium')

                is_video = self.app_files[0].name.endswith('.mp4') and random_frames == 'No' # True for video, false for image

                if is_video:
                    out, cap, length = self.file.write_video(self.app_files[0])
                    model.infer_video(out, cap, length)
                    video_file = self.file.annot_video_path
                    col1.video(video_file)

                    video_bytes = open(video_file, 'rb').read()

                    col2.download_button("Download Video", data=video_bytes,
                                         file_name="processed_video.mp4", mime="video/mp4",
                                         icon=":material/download:")

                else:
                    if random_frames == 'Yes':
                        _, cap, length = self.file.write_video(self.app_files[0], False)
                        dataset_creator.generate_example_images(self.file.imgs_dir, cap, length)
                        self.app_files.clear() # Clear out the .mp4 file

                        self.app_files = [os.path.join(self.file.imgs_dir, item) 
                                          for item in os.listdir(self.file.imgs_dir)]

                    processed_img = []
                    for i, img in enumerate(self.app_files):
                        pred = model.infer_image(img)
                        col1.image(pred)
                        processed_img.append((f"processed_img{i+1}.png", Image.fromarray(pred, mode="RGB")))

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zip_file:
                        for filename, img in processed_img:
                            img_bytes = io.BytesIO()
                            img.save(img_bytes, format="PNG")
                            zip_file.writestr(filename, img_bytes.getvalue())

                    zip_buffer.seek(0)
                    col2.download_button("Download Images", data=zip_buffer, 
                                         file_name="processed_img.zip", 
                                         mime="application/zip", icon=":material/download:")
                    
                
                self.file.clear()

            except Exception as e:
                st.error(f"Error. Most Likely you didn't upload a file before running. {e}")
                self.file.clear()
        
        self._css_styling()


if __name__ == '__main__':
    app = InferenceApp()
    app.run()