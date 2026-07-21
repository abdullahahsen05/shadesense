"""ShadeSense AI — Streamlit UI entrypoint.

This module contains presentation logic only. CV/matching logic lives in `src/`.
"""

import numpy as np
import streamlit as st
from PIL import Image

from src.config import APP_NAME

st.set_page_config(page_title=APP_NAME, layout="wide")

st.title(APP_NAME)
st.caption("Local AI foundation shade recommender — upload a facial photo to begin.")

uploaded_file = st.file_uploader(
    "Upload a facial image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    st.subheader("Uploaded Image")
    st.image(image_rgb, caption=f"{image.width}x{image.height}px", width=400)
else:
    st.info("Upload an image to see it displayed here.")
