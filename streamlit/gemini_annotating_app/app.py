from typing import Optional, Tuple, List, Dict, Any
import streamlit as st
import google.generativeai as genai
from PIL import Image
import json
import numpy as np
import traceback
import io
import os
import concurrent.futures
import functools
from app_utils.detection_processor import DetectionProcessor
from app_utils.data_downloader import DataDownloader
from baseballcv.utilities import BaseballCVLogger


class GeminiBaseballAnnotationApp:
    """
    A Streamlit application for annotating baseball images using Google Gemini models. It allows users to upload images, configure
    annotation parameters (model, prompt, format, temperature), process the images using a selected Gemini model, view the results
    (including visualizations), and download the annotations in various formats.

    Attributes:
        data_downloader (DataDownloader): Helper class for creating and providing
                                          download links for annotation results.
        detection_processor (Optional[DetectionProcessor]): Processor class that
                                                           handles interaction with
                                                           the Gemini API and result
                                                           parsing/visualization.
        logger (BaseballCVLogger): Logger for the application.
    """

    def __init__(self) -> None:
        """
        Initializes the Streamlit session state variables required for the app
        and instantiates helper classes.
        """
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = []
        if 'all_processed' not in st.session_state:
            st.session_state.all_processed = False

        self.logger = BaseballCVLogger.get_logger(name="GeminiAnnotationApp")
        self.data_downloader = DataDownloader()
        self.available_models = [
            'gemini-1.5-flash',
            'gemini-1.5-flash-8b',
            'gemini-1.5-pro',
            'gemini-2.0-flash',
            'gemini-2.0-flash-lite',
            'gemini-2.5-pro-exp-03-25',
            'gemini-2.5-flash-preview-04-17',
        ]
        self.detection_processor: Optional[DetectionProcessor] = None
        self.logo_path = os.path.join(os.path.dirname(__file__), "app_assets", "baseballcvlogo.png")
        self.max_workers = os.cpu_count()


    def _apply_styling(self) -> None:
        """
        Applies custom CSS styling to the Streamlit application for a
        dark theme with orange accents.
        """
        st.markdown("""
        <style>
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
        </style>
        """, unsafe_allow_html=True)

    def _initialize_gemini(self, api_key: str, model_name: str) -> Optional[genai.GenerativeModel]:
        """
        Configures the Google Generative AI client with the provided API key
        and initializes the specified Gemini model.

        Args:
            api_key: The Google Gemini API key.
            model_name: The name of the Gemini model to initialize
                        (ex. 'gemini-1.5-flash').

        Returns:
            genai.GenerativeModel: An initialized model instance if successful.
        """
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            self.logger.info(f"Successfully initialized Gemini model '{model_name}'.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model '{model_name}'. Error: {e}", exc_info=True)
            st.error(f"Failed to initialize Gemini model '{model_name}'. "
                     f"Check API key and model availability. Error: {e}")
            st.stop()
            return None

    def _display_sidebar(self) -> Tuple[Optional[str], Optional[str], Optional[str], float, Optional[str], bool]:
        """
        Displays the configuration sidebar in the Streamlit app.

        Includes the logo, widgets for API key input, model selection, output format choice,
        custom prompt input, and the 'Process Images' button.

        Returns:
            Tuple[Optional[str], Optional[str], Optional[str], float, Optional[str], bool]:
            - api_key (Optional[str]): The entered API key
            - model_name (Optional[str]): The selected Gemini model name.
            - format_choice (Optional[str]): The selected output format ('JSON', 'YOLO', 'COCO').
            - temperature (float): The selected temperature value.
            - custom_prompt (Optional[str]): The entered annotation prompt.
            - process_button_clicked (bool): True if the 'Process Images' button was clicked, False otherwise.
        """
        if os.path.exists(self.logo_path):
            st.sidebar.image(self.logo_path, width=275)
        else:
            self.logger.warning(f"Logo not found at: {self.logo_path}")
            st.sidebar.warning(f"Logo not found at: {self.logo_path}")

        st.sidebar.header("Configuration")

        api_key = st.sidebar.text_input(
            "Enter your Google Gemini API Key",
            type="password",
            key="api_key_input"
        )

        if not api_key:
            st.warning("Please enter your Google Gemini API Key in the sidebar to continue.")
            return None, None, None, 0.5, None, False

        model_name = st.sidebar.selectbox(
            "Select Gemini Model",
            options=self.available_models,
            index=0,
            key="model_select",
            help="Choose the Gemini vision model to use for annotations."
        )

        format_choice = st.sidebar.selectbox(
            "Choose output format",
            ["JSON", "YOLO", "COCO"],
            key="format_select",
            help="Select desired format for the annotation output."
        )

        temperature = st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="temperature_slider",
            help="Controls randomness. Lower values make the output more deterministic."
        )

        custom_prompt = st.sidebar.text_area(
            "Enter your annotation prompt",
            placeholder="Example: Identify all cars and pedestrians.",
            help="Describe the objects you want to detect.",
            key="prompt_input"
        )

        process_button_clicked = st.sidebar.button(
            "Process Images",
            key="process_btn",
            type="primary"
        )

        return api_key, model_name, format_choice, temperature, custom_prompt, process_button_clicked

    def _display_uploader(self) -> Optional[List[st.runtime.uploaded_file_manager.UploadedFile]]:
        """
        Displays the file uploader widget in the main area of the Streamlit app.

        Allows users to upload multiple image files (JPG, JPEG, PNG).

        Returns:
            A list of uploaded file objects, or None if no files were uploaded.
        """
        return st.file_uploader(
            "Upload Image Files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="uploader",
            help="Upload one or more JPG/PNG image files."
        )

    def _process_single_image_for_executor(self, uploaded_file, current_custom_prompt: str, current_format_choice: str) -> Dict[str, Any]:
        """
        Helper method to process a single uploaded file.
        Designed to be called by the ThreadPoolExecutor.
        Handles image reading and delegates processing to self.detection_processor.
        """
        filename = uploaded_file.name
        try:
            image_bytes = uploaded_file.getvalue()

            if not self.detection_processor:
                 self.logger.error(f"Detection processor not available when trying to process {filename}")
                 st.error(f"Detection processor not available when trying to process {filename}")
                 raise RuntimeError("Detection processor is not initialized for threaded task.")

            return self.detection_processor.process_single_image(
                image_bytes=image_bytes,
                filename=filename,
                custom_prompt=current_custom_prompt,
                format_choice=current_format_choice
            )
        except Exception as img_e:
            self.logger.error(f"Error processing image {filename} in executor task: {str(img_e)}", exc_info=True)
            return {
                "filename": filename,
                "image_bytes": None,
                "image_np": None,
                "raw_response": f"EXECUTOR_TASK_ERROR: {img_e}\n{traceback.format_exc()}",
                "annotations": None,
                "format": current_format_choice,
                "error": str(img_e)
            }

    def _process_images(self,
                        files_to_process: List[st.runtime.uploaded_file_manager.UploadedFile],
                        custom_prompt: str,
                        temperature: float,
                        format_choice: str) -> None:
        """
        Processes a list of uploaded image files using the configured Gemini model,
        utilizing parallel processing for efficiency.

        Args:
            files_to_process: A list of uploaded file objects from the uploader.
            custom_prompt: The user-defined prompt for object detection.
            temperature: The temperature setting for the Gemini model.
            format_choice: The desired output format for annotations ('JSON', 'YOLO', 'COCO').
        """
        if not self.detection_processor:
            self.logger.error("Detection processor not initialized. Cannot process images.")
            st.error("Detection processor not initialized. Cannot process images.")
            return

        self.detection_processor.temperature = temperature

        if not custom_prompt:
            self.logger.warning("Annotation prompt is empty.")
            st.warning("Enter an annotation prompt in the sidebar before processing.")
            return

        st.session_state.processing_results = []
        st.session_state.all_processed = False

        if not files_to_process:
            self.logger.error("No valid image files found in upload.")
            st.error("No valid image files found in upload.")
            return

        num_total_items = len(files_to_process)
        self.logger.info(f"Starting processing for {num_total_items} image file(s)...")
        st.info(f"Starting processing for {num_total_items} image file(s)...")
        progress_bar = st.progress(0, text="Initializing parallel processing...")

        processed_count = 0

        processing_task = functools.partial(
            self._process_single_image_for_executor,
            current_custom_prompt=custom_prompt,
            current_format_choice=format_choice
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(
                    processing_task,
                    uploaded_file
                ): uploaded_file.name
                for uploaded_file in files_to_process
            }

            for future in concurrent.futures.as_completed(future_to_file):
                filename_processed = future_to_file[future]
                try:
                    result_data = future.result()
                    st.session_state.processing_results.append(result_data)
                except Exception as exc:
                    self.logger.error(f"Future for {filename_processed} resulted in an unexpected exception: {exc}", exc_info=True)
                    st.error(f"Future for {filename_processed} resulted in an unexpected exception: {exc}")
                    current_format_choice_for_error = format_choice
                    st.session_state.processing_results.append({
                        "filename": filename_processed,
                        "image_bytes": None,
                        "image_np": None,
                        "raw_response": f"FUTURE_COMPLETION_ERROR: {exc}\n{traceback.format_exc()}",
                        "annotations": None,
                        "format": current_format_choice_for_error,
                        "error": str(exc)
                    })

                processed_count += 1
                progress_value = processed_count / num_total_items
                progress_text = f"Processed {processed_count}/{num_total_items} images... ({filename_processed} completed)"
                if progress_bar:
                    progress_bar.progress(progress_value, text=progress_text)

        st.session_state.all_processed = True
        if progress_bar:
            progress_bar.empty()
        final_processed_count = len(st.session_state.processing_results)
        self.logger.info(f"Finished parallel processing for {final_processed_count} item(s).")
        st.success(f"Finished parallel processing for {final_processed_count} item(s).")

    def _display_download_button(self) -> None:
        """
        Displays a download button for the processed annotation results. Filters valid results, checks for format consistency (especially for COCO/YOLO),
        creates a ZIP archive containing images and annotation files using `DataDownloader`.
        """
        if st.session_state.processing_results:
            st.markdown("---")
            st.subheader("Download Results")

            valid_results_for_zip = [
                r for r in st.session_state.processing_results
                if r is not None and r.get('image_bytes') is not None and r.get('error') is None
            ]

            if not valid_results_for_zip:
                self.logger.warning("No valid results with images available to create a ZIP file for download.")
                st.warning("No valid results with images available to create a ZIP file.")
                st.markdown("---")
                return

            first_format = valid_results_for_zip[0].get('format', 'unknown').lower()

            if not all(r.get('format', '').lower() == first_format for r in valid_results_for_zip):
                 self.logger.warning(f"Inconsistent formats detected in results. Attempting to create ZIP using format: {first_format.upper()}")
                 st.warning("Inconsistent formats detected in results. "
                            "Download might not work as expected, especially for COCO/YOLO. "
                            f"Attempting to create ZIP using format: {first_format.upper()}")

            try:
                zip_download_filename = f"annotations_{first_format}_batch.zip"
                zip_buffer: Optional[io.BytesIO] = self.data_downloader.create_zip_file(valid_results_for_zip)

                if zip_buffer:
                     st.download_button(
                         label=f"Download Annotations ({first_format.upper()})",
                         data=zip_buffer,
                         file_name=zip_download_filename,
                         mime="application/zip",
                         key="download_zip_btn"
                     )
                     self.logger.info(f"Download button created for {zip_download_filename}.")
                else:
                     self.logger.error("Failed to create ZIP buffer. No data to download.")
                     st.error("Failed to create ZIP buffer. No data to download.")

            except Exception as zip_e:
                self.logger.error(f"Failed to create ZIP file or download button: {zip_e}", exc_info=True)
                st.error(f"Failed to create ZIP file or download button: {zip_e}")
                st.error(traceback.format_exc())

            st.markdown("---")


    def _display_results(self) -> None:
        """
        Displays the processed image annotation results in the main area.
        """
        if not st.session_state.processing_results:
            st.info("No results to display yet. Upload images and click 'Process Images'.")
            return

        if not self.detection_processor:
            self.logger.warning("Detection processor is not available for displaying results (API key might be missing). Annotation visualization will be skipped.")
            st.warning("Detection processor is not available (API key might be missing). "
                       "Annotation visualization will be skipped.")

        num_results = len(st.session_state.processing_results)
        num_columns = min(num_results, 3)

        if num_columns <= 0:
            return

        cols = st.columns(num_columns)

        for i, result in enumerate(st.session_state.processing_results):
            if result is None:
                continue

            col_index = i % num_columns
            with cols[col_index]:
                filename = result.get('filename', f'Item {i+1}')
                st.write(f"**{filename}**")

                if result.get('error'):
                     st.error(f"Error: {result['error']}")
                     if result.get('raw_response'):
                          st.text_area("Raw Response / Error Detail:",
                                       value=str(result['raw_response']),
                                       height=100,
                                       key=f"raw_error_{i}")

                elif result.get('image_np') is not None:
                    image_np = result['image_np']
                    annotations = result.get('annotations')
                    format_choice_res = result.get('format', 'JSON')

                    try:
                        image_display = Image.fromarray(image_np)
                        st.image(image_display, caption=f"Original: {filename}", use_column_width='auto')
                    except Exception as img_disp_e:
                        self.logger.error(f"Could not display original image {filename}: {img_disp_e}", exc_info=True)
                        st.error(f"Could not display original image {filename}: {img_disp_e}")

                    if annotations is not None:
                        st.write(f"Annotations ({format_choice_res}):")
                        try:
                            json_str = json.dumps(annotations, indent=2)
                            st.code(json_str, language="json")
                        except TypeError as te:
                             self.logger.error(f"Could not serialize annotations to JSON for display for {filename}: {te}", exc_info=True)
                             st.error(f"Could not serialize annotations to JSON for display: {te}")
                             st.text(str(annotations))
                        except Exception as json_e:
                             self.logger.error(f"Error displaying annotations for {filename}: {json_e}", exc_info=True)
                             st.error(f"Error displaying annotations: {json_e}")
                             st.text(str(annotations))

                        if self.detection_processor:
                            try:
                                annotated_image = self.detection_processor.visualize_results(
                                    image_np, annotations, format_choice_res
                                )
                                if annotated_image is not None and not np.array_equal(annotated_image, image_np):
                                    st.image(annotated_image, caption=f"Annotated: {filename}", use_column_width='auto')
                                elif annotated_image is not None:
                                     st.info("Annotations processed, but no bounding boxes drawn (or boxes match original).")
                                else:
                                     self.logger.warning(f"Visualization function returned None for {filename}.")
                                     st.warning("Visualization function returned None.")
                            except Exception as vis_e:
                                self.logger.error(f"Error visualizing annotations for {filename}: {vis_e}", exc_info=True)
                                st.error(f"Error visualizing annotations for {filename}: {vis_e}")
                                st.error(traceback.format_exc())
                        else:
                            pass # Already warned about detection_processor being None

                    elif not result.get('error'):
                        self.logger.warning(f"No annotations generated for {filename}, and no error recorded.")
                        st.warning(f"No annotations generated for {filename}.")
                        if result.get('raw_response'):
                            st.text_area("Raw Response:", value=str(result['raw_response']), height=100, key=f"raw_ok_{i}")

                elif not result.get('error'):
                    self.logger.error(f"Could not load image data for {filename}, but no specific error recorded.")
                    st.error(f"Could not load image data for {filename}, but no specific error recorded.")

                st.markdown("---")

        st.markdown("---")
        if st.button("Clear Results", key="clear_btn"):
            self.logger.info("Clearing processing results.")
            st.session_state.processing_results = []
            st.session_state.all_processed = False
            st.rerun()

    def _display_instructions(self) -> None:
        """
        Displays an expandable section containing instructions on how to use the app
        and example prompts.
        """
        with st.expander("How to use this app"):
            st.markdown("""
            1.  **Enter API Key:** Input your Google Gemini API Key in the sidebar.
            2.  **Select Model:** Choose the Gemini model for Vision (ex. `gemini-1.5-flash`).
            3.  **Choose Format:** Select the desired output format (JSON, YOLO, or COCO).
            4.  **Set Temperature:** Adjust the model's creativity/randomness (0.0 = deterministic, 1.0 = max randomness).
            5.  **Write Prompt:** Clearly describe the objects you want to detect in the images.
            6.  **Upload:** Upload one or more image files (`.jpg`, `.jpeg`, `.png`).
            7.  **Process:** Click the 'Process Images' button in the sidebar.
            8.  **View & Download Results:** View results below (original image, annotations, visualized image). Download a ZIP containing images and annotation files using the download button.

            **Example Prompts:**
            *   "Identify all baseball players. There should be a maximum of 3 players in the image."
            *   "Identify a pitcher, hitter, and catcher."
            *   "Identify the catcher's glove."
            *   "Identify only the pitcher in the image."

            *Note: The quality and clarity of the input images significantly impact the annotation results.*
            """)

    def run(self) -> None:
        """
        Sets up the Streamlit page configuration and orchestrates the display
        and interaction logic of the application.
        """
        st.set_page_config(
            page_title="BaseballCV Automated Image Annotation (w/ Gemini)",
            page_icon=":baseball:",
            layout="wide"
        )

        self._apply_styling()

        st.title("BaseballCV Detection Annotation with Gemini")
        self.logger.info("Application started.")

        api_key, model_name, format_choice, temperature, custom_prompt, process_button_clicked = self._display_sidebar()

        if api_key and model_name:
            if self.detection_processor is None or self.detection_processor.model.model_name != model_name:
                self.logger.info(f"Attempting to initialize Gemini model: {model_name}")
                gemini_model = self._initialize_gemini(api_key, model_name)
                if gemini_model:
                    self.detection_processor = DetectionProcessor(gemini_model, temperature, 3)
                else:
                    self.detection_processor = None
            elif self.detection_processor:
                   self.detection_processor.temperature = temperature
        else:
             self.detection_processor = None

        uploaded_files = self._display_uploader()

        if process_button_clicked:
            self.logger.info("'Process Images' button clicked.")
            if self.detection_processor and uploaded_files and custom_prompt and format_choice:
                self._process_images(uploaded_files, custom_prompt, temperature, format_choice)
                self._display_download_button()
            elif not api_key:
                self.logger.warning("Processing attempted without API key.")
                st.error("Please enter your API key in the sidebar before processing.")
            elif not uploaded_files:
                self.logger.warning("Processing attempted without uploaded files.")
                st.warning("Please upload image files before processing.")
            elif not custom_prompt:
                self.logger.warning("Processing attempted without a custom prompt.")
                st.warning("Please enter an annotation prompt in the sidebar before processing.")
            elif not self.detection_processor:
                 self.logger.error("Processing attempted but detection_processor is not initialized.")
                 st.error("Initialization failed. Cannot process images. Check API key and model selection.")


        if st.session_state.all_processed and st.session_state.processing_results:
             if not self.detection_processor and api_key and model_name:
                 self.logger.info(f"Re-attempting to initialize Gemini model: {model_name} for displaying results.")
                 gemini_model = self._initialize_gemini(api_key, model_name)
                 if gemini_model:
                     self.detection_processor = DetectionProcessor(gemini_model, temperature)

             self._display_results()
             if not process_button_clicked:
                 self._display_download_button()

        self._display_instructions()


if __name__ == "__main__":
    app = GeminiBaseballAnnotationApp()
    app.run()