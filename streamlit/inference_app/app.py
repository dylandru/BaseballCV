import streamlit as st
from PIL import Image
import numpy as np
from app_utils.yolo import YOLOInference


def app():
    st.set_page_config(
        layout="wide",
        page_title="Baseball Inference Tool",
        page_icon="⚾"
    )

    st.title("Baseball Inference Tool")

    uploaded_file = st.file_uploader(label='Upload File', type=['jpg', 'jpeg', 'png'])
    
    yolo = YOLOInference()

    if uploaded_file is not None:
       img = np.array(Image.open(uploaded_file))

       results = yolo.inference(img)
       annot_img = results[0].plot()
       st.image(annot_img, caption="Annotated image with detections")


if __name__ == '__main__':
    app()