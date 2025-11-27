# app.py
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageOps
import os
import requests
import cv2

st.set_page_config(page_title="Fish Freshness Detector", layout="centered")

# ---------- CONFIG ----------
MODEL_FILENAME = "weights/best.pt"
MODEL_URL = st.secrets.get("MODEL_URL", None)

# ---------- HELPERS ----------
@st.cache_resource
def load_model(path=MODEL_FILENAME):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if MODEL_URL:
            st.info("Mengunduh model dari MODEL_URL...")
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            st.success("Model berhasil diunduh.")
        else:
            st.error("Model tidak ditemukan di folder 'weights/'.")
            raise FileNotFoundError("Model tidak ditemukan")

    return YOLO(path)

def load_image(file):
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)  # fix rotasi HP
    img = img.convert("RGB")
    return img

# ---------- UI ----------
st.title("🐟 Fish Freshness Detector (YOLOv8m)")
st.write("Upload foto ikan untuk mendeteksi kesegaran.")

try:
    model = load_model()
except:
    st.stop()

uploaded = st.file_uploader("Upload gambar ikan (jpg/png)", type=["jpg", "jpeg", "png"])

conf_threshold = st.slider("Confidence threshold", 0.05, 0.9, 0.25, step=0.05)
imgsz = st.slider("Image size", 256, 1280, 640, step=64)

# ---------- PROCESS ----------
if uploaded:
    pil_img = load_image(uploaded)
    st.image(pil_img, caption="Input Image", use_column_width=True)

    # Convert ke format YOLO (BGR Numpy)
    img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    with st.spinner("Memproses..."):
        results = model.predict(
            source=img_np,
            imgsz=imgsz,
            conf=conf_threshold,
            verbose=False
        )

    # Gambar hasil memakai bawaan YOLO (lebih stabil)
    result_img = results[0].plot()  # plot() = BGR
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

    # Summary
    st.subheader("Ringkasan Deteksi")
    det = results[0].boxes

    if det:
        names = results[0].names
        counts = {}
        for cls in det.cls.cpu().numpy().astype(int):
            name = names.get(cls, str(cls))
            counts[name] = counts.get(name, 0) + 1

        for k, v in counts.items():
            st.write(f"- **{k}** : {v}x")
    else:
        st.write("Tidak ada deteksi.")

else:
    st.info("Silakan upload gambar untuk mulai deteksi.")
