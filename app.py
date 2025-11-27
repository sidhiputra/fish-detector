# app.py — FINAL STREAMLIT VERSION (NO CV2)

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
import requests
import io

st.set_page_config(page_title="Fish Freshness Detector", layout="centered")

# ---------- CONFIG ----------
MODEL_FILENAME = "weights/best.pt"   # lokasi model
MODEL_URL = st.secrets.get("MODEL_URL", None)  # optional kalau mau download otomatis


# ---------- HELPERS ----------
@st.cache_resource
def load_model(path=MODEL_FILENAME):
    """Load model YOLO dari file lokal atau download via URL."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if MODEL_URL:
            st.info("Model tidak ditemukan, mengunduh dari MODEL_URL...")
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            st.success("Model berhasil diunduh.")
        else:
            st.error("Model tidak ditemukan. Upload ke folder 'weights/' atau set MODEL_URL.")
            raise FileNotFoundError("Model YOLO tidak ditemukan.")

    model = YOLO(path)
    return model


def read_imagefile(file) -> np.ndarray:
    """Convert uploaded image ke numpy array."""
    image = Image.open(file).convert("RGB")
    return np.array(image)


def draw_boxes(orig_img, results, conf_thresh=0.25):
    """Gambar bounding box menggunakan PIL (AMAN di Streamlit Cloud tanpa cv2)."""
    img = Image.fromarray(orig_img)
    draw = ImageDraw.Draw(img)

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
            if conf < conf_thresh:
                continue

            # bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # label
            label = f"{r.names.get(cls, str(cls))} {conf:.2f}"
            draw.text((x1, y1 - 15), label, fill="red")

    return np.array(img)


# ---------- UI ----------
st.title("🐟 Fish Freshness Detector (YOLOv8m)")
st.write("Upload foto ikan untuk mendeteksi kesegaran menggunakan model YOLOv8m.")

# Load model YOLO
try:
    model = load_model()
except Exception:
    st.stop()

# Upload panel
uploaded = st.file_uploader("Upload gambar ikan (jpg/png)", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider("Confidence threshold", 0.05, 0.9, 0.25, step=0.05)
imgsz = st.slider("Image size (inference)", 256, 1280, 640, step=64)

st.sidebar.header("Informasi Model")
st.sidebar.write("Model:", MODEL_FILENAME)
if MODEL_URL:
    st.sidebar.write("Model URL: tersedia")

# Processing
if uploaded:
    image_np = read_imagefile(uploaded)
    st.image(image_np, caption="Input Image", use_column_width=True)

    with st.spinner("Menjalankan inferensi..."):
        results = model.predict(
            source=image_np,
            imgsz=imgsz,
            conf=conf_threshold,
            verbose=False
        )

    drawn = draw_boxes(image_np, results, conf_thresh=conf_threshold)
    st.image(drawn, caption="Hasil Deteksi", use_column_width=True)

    # Summary
    st.subheader("📌 Ringkasan Deteksi")
    detections = {}

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        names = r.names
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for cls, conf in zip(clss, confs):
            if conf < conf_threshold:
                continue
            label = names.get(cls, str(cls))
            detections[label] = detections.get(label, 0) + 1

    if detections:
        for label, count in detections.items():
            st.write(f"- **{label}** : {count} kali")
    else:
        st.write("Tidak ada deteksi di atas threshold.")
else:
    st.info("Silakan upload gambar untuk mulai deteksi.")
