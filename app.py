"""ShadeSense AI — Streamlit UI entrypoint.

This module contains presentation logic only. CV/matching logic lives in `src/`.
"""

import numpy as np
import streamlit as st
from PIL import Image

from src.config import APP_NAME
from src.face_detection import detect_face_landmarks
from src.region_masks import build_region_masks
from src.visualization import draw_all_region_masks, draw_face_landmarks, draw_region_mask

st.set_page_config(page_title=APP_NAME, layout="wide")

st.title(APP_NAME)
st.caption("Local AI foundation shade recommender — upload a facial photo to begin.")

uploaded_file = st.file_uploader(
    "Upload a facial image", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(image_rgb, caption=f"{image.width}x{image.height}px", width=400)

    face_result = detect_face_landmarks(image_rgb)

    for warning in face_result.warnings:
        st.warning(warning)

    if not face_result.success:
        st.error(face_result.error)
    else:
        with col2:
            st.subheader("Detected Face Landmarks")
            overlay = draw_face_landmarks(image_rgb, face_result.landmarks)
            st.image(overlay, caption=f"{len(face_result.landmarks)} landmarks", width=400)

        masks = build_region_masks(image_rgb.shape, face_result.landmarks)

        st.subheader("Skin Regions")
        combined_overlay = draw_all_region_masks(image_rgb, masks)
        st.image(combined_overlay, caption="Forehead / cheeks / jawline (combined)", width=450)

        region_cols = st.columns(4)
        region_labels = {
            "forehead": "Forehead",
            "left_cheek": "Left Cheek",
            "right_cheek": "Right Cheek",
            "jawline": "Jawline",
        }
        for col, (region_key, label) in zip(region_cols, region_labels.items()):
            with col:
                region_overlay = draw_region_mask(image_rgb, masks[region_key])
                pixel_count = int((masks[region_key] > 0).sum())
                st.image(region_overlay, caption=f"{label} ({pixel_count}px)", width=180)
                if pixel_count == 0:
                    st.caption("No usable pixels in this region.")
else:
    st.info("Upload an image to see it displayed here.")
