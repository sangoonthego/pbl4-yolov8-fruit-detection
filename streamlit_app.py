import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from yolov8.detect_object import ObjectDetector

st.set_page_config(page_title="Fruit Detection", layout="centered")
st.title("Fruit Detection by YOLOv8")

object_detector = ObjectDetector()

uploaded_file = st.file_uploader("Upload an image", type=["jpeg", "jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Selected Image", use_container_width=True)
    result_path = f"results/{uploaded_file.name}"
    os.makedirs("results", exist_ok=True)

    with open(result_path, "wb") as file:
        file.write(uploaded_file.read())

        st.markdown("---")

        st.subheader("Object Detection")
        detections = object_detector.object_detects(result_path)

        if detections:
            for detection in detections:
                st.write(f"Class: {detection["class"]}")
                st.write(f"Confidence: {detection["confidence"]:.2f}")
        else:
            st.warning("No Object Detected!!!")

        output_image_path = f"static/output/{uploaded_file.name}"
        if os.path.exists(output_image_path):
            st.image(output_image_path, caption=detection["class"], use_container_width=True)