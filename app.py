# app.py
import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import os
import requests
import tempfile
import cv2

st.set_page_config(page_title="Fish Freshness Detector", layout="centered")

# ---------- CONFIG ----------
MODEL_FILENAME = "weights/yolov8m_fish_freshness.pt"  # ganti sesuai nama model kamu
# Optionally set an external model URL in Streamlit Secrets as {"MODEL_URL": "https://.../model.pt"}
MODEL_URL = st.secrets.get("MODEL_URL", None)

# ---------- HELPERS ----------
@st.cache_resource
def load_model(path=MODEL_FILENAME):
    # If model not found locally and MODEL_URL is provided, download it
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if MODEL_URL:
            st.info("Model tidak ditemukan, mengunduh dari MODEL_URL...")
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success("Model diunduh.")
        else:
            st.error("Model tidak ditemukan dan MODEL_URL tidak di-set. Upload model ke folder 'weights/' atau set MODEL_URL di Streamlit Secrets.")
            raise FileNotFoundError("Model tidak ditemukan")
    # Load model (Ultralytics YOLO)
    model = YOLO(path)
    return model

def read_imagefile(file) -> np.ndarray:
    image = Image.open(file).convert("RGB")
    return np.array(image)

def draw_boxes(orig_img, results, conf_thresh=0.25):
    img = orig_img.copy()
    for r in results:
        # r.boxes.xyxy, r.boxes.conf, r.boxes.cls
        boxes = r.boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
        confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
        clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []
        for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, clss):
            if conf < conf_thresh:
                continue
            # draw rectangle and label
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{r.names[cls] if hasattr(r,'names') else cls} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    return img

# ---------- UI ----------
st.title("🐟 Fish Freshness Detector (YOLOv8m)")
st.write("Upload foto ikan, aplikasi akan mendeteksi label terkait kesegaran (mis. 'fresh','not_fresh') dan menampilkan bounding box serta confidence.")

# Load model (cached)
try:
    model = load_model()
except Exception as e:
    st.stop()

col1, col2 = st.columns([2,1])

with col1:
    uploaded = st.file_uploader("Upload gambar ikan (jpg, png)", type=["jpg","jpeg","png"])
    conf_threshold = st.slider("Confidence threshold", 0.05, 0.9, 0.25, step=0.05)
    imgsz = st.slider("Image size (pixels, inference)", 256, 1280, 640, step=64)

with col2:
    st.sidebar.header("Pengaturan")
    st.sidebar.write("Model:", MODEL_FILENAME)
    if MODEL_URL:
        st.sidebar.write("Model URL: set")

if uploaded:
    image_np = read_imagefile(uploaded)
    st.image(image_np, caption="Input image", use_column_width=True)
    with st.spinner("Menjalankan inferensi..."):
        # Ultralytics model.predict returns Results objects
        # We pass parameters: imgsz, conf (confidence), save=False
        results = model.predict(source=image_np, imgsz=imgsz, conf=conf_threshold, verbose=False)

    # draw boxes on image
    # results is list-like (each result per image), use results[0]
    drawn = draw_boxes(image_np, results, conf_thresh=conf_threshold)
    st.image(drawn, caption="Deteksi", use_column_width=True)

    # Summarize detections
    st.subheader("Ringkasan deteksi")
    detections = {}
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        labels = []
        if hasattr(r, "names"):
            names_map = r.names
        else:
            names_map = {}
        for i, conf in enumerate(boxes.conf.cpu().numpy()):
            if conf < conf_threshold:
                continue
            cls = int(boxes.cls.cpu().numpy()[i])
            name = names_map.get(cls, str(cls))
            detections[name] = detections.get(name, 0) + 1
    if detections:
        for k, v in detections.items():
            st.write(f"- **{k}** : {v} kali")
    else:
        st.write("Tidak ada deteksi di atas threshold.")

    # OPTIONAL: a very simple 'freshness score' heuristic if model outputs class probabilities
    st.markdown("---")
    st.caption("Catatan: jika model kamu memberi label kelas seperti 'fresh' atau 'stale', gunakan hasil deteksi di atas. Aplikasi ini tidak mengubah output model.")
else:
    st.info("Silakan upload gambar untuk memulai.")
