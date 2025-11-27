# app.py
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import requests
import cv2

st.set_page_config(page_title="Fish Freshness Detector", layout="centered")

# ---------- CONFIG ----------
MODEL_FILENAME = "weights/best.pt"    # ganti sesuai nama model Kamu
MODEL_URL = st.secrets.get("MODEL_URL", None)

# ---------- HELPERS ----------
@st.cache_resource
def load_model(path=MODEL_FILENAME):
    # Download model if not found
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if MODEL_URL:
            st.info("Model tidak ditemukan. Mengunduh dari MODEL_URL...")
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            st.success("Model berhasil diunduh.")
        else:
            st.error("Model tidak ditemukan. Upload ke folder 'weights/' atau set MODEL_URL di Secrets.")
            raise FileNotFoundError("Model tidak ditemukan")

    return YOLO(path)

def read_imagefile(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return np.array(img)

def draw_boxes(img, results, conf_thresh=0.25):
    out = img.copy()
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

            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)

            label = f"{r.names[cls]} {conf:.2f}" if hasattr(r, "names") else f"{cls} {conf:.2f}"
            cv2.putText(out, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    return out

# ---------- UI ----------
st.title("🐟 Fish Freshness Detector (YOLOv8m)")
st.write("Upload foto ikan untuk mendeteksi kesegaran.")

# Load model once
try:
    model = load_model()
except:
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    uploaded = st.file_uploader("Upload gambar ikan (jpg/png)", type=["jpg", "jpeg", "png"])
    conf_threshold = st.slider("Confidence threshold", 0.05, 0.9, 0.25, step=0.05)
    imgsz = st.slider("Image size", 256, 1280, 640, step=64)

with col2:
    st.sidebar.header("Pengaturan Model")
    st.sidebar.write("Model path:", MODEL_FILENAME)
    if MODEL_URL:
        st.sidebar.write("MODEL_URL: ✔️ ada")

# ---------- PROCESS ----------
if uploaded:
    img_np = read_imagefile(uploaded)
    st.image(img_np, caption="Input Image", use_column_width=True)

    with st.spinner("Memproses..."):
        results = model.predict(source=img_np, imgsz=imgsz, conf=conf_threshold, verbose=False)

    drawn = draw_boxes(img_np, results, conf_thresh=conf_threshold)
    st.image(drawn, caption="Hasil Deteksi", use_column_width=True)

    # Summary
    st.subheader("Ringkasan Deteksi")
    detections = {}
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for i, conf in enumerate(boxes.conf.cpu().numpy()):
            if conf < conf_threshold:
                continue

            cls = int(boxes.cls.cpu().numpy()[i])
            name = r.names.get(cls, str(cls)) if hasattr(r, "names") else str(cls)
            detections[name] = detections.get(name, 0) + 1

    if detections:
        for k, v in detections.items():
            st.write(f"- **{k}** : {v}x")
    else:
        st.write("Tidak ada deteksi.")

    st.markdown("---")
    st.caption("Catatan: Output 100% berdasarkan model YOLO yang Kamu gunakan.")
else:
    st.info("Silakan upload gambar untuk mulai deteksi.")
