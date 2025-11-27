import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Fish Freshness Detector", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("/weights/best.pt")     # ganti dengan nama model kamu

model = load_model()

st.title("🐟 Fish Freshness Detector")
st.write("Upload foto ikan untuk mendeteksi kesegaran.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar diunggah", use_column_width=True)

    # Convert to numpy
    img_array = np.array(img)

    # Run prediction
    results = model.predict(img_array)

    # Get annotated image
    annotated = results[0].plot()  # sudah return numpy array

    st.image(annotated, caption="Hasil Deteksi", use_column_width=True)

    # Show detected classes
    st.subheader("Deteksi:")
    boxes = results[0].boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"- {model.names[cls]} ({conf:.2f})")


