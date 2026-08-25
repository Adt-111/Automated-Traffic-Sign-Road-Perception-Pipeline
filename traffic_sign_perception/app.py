"""
Upload a road scene (image or video), classical detection proposes
candidate sign boxes, the CNN classifies each crop, results get drawn back
on with labels and confidence.

    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch

import config
from src.classical_detector import Candidate, draw_candidates, find_candidate_regions
from src.cnn_classifier import TrafficSignCNN, preprocess_bgr_crop

st.set_page_config(page_title="Traffic Sign & Road Perception", layout="wide")


# Model loading (cached so it only happens once per session)

@st.cache_resource(show_spinner="Loading CNN classifier...")
def load_model() -> TrafficSignCNN | None:
    """
    Loads the trained CNN from `config.MODEL_PATH`. Returns None (rather than
    raising) if no checkpoint exists yet, so the app can still demonstrate
    the classical detection stage and show a clear "train a model first"
    message instead of crashing.
    """
    if not os.path.exists(config.MODEL_PATH):
        return None
    model = TrafficSignCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# Core pipeline: detect -> crop -> classify -> annotate

def classify_candidate(model: TrafficSignCNN, candidate: Candidate, frame_bgr: np.ndarray) -> tuple[str, float]:
    """Crops a candidate region and returns (class_name, confidence) from the CNN."""
    crop = candidate.crop(frame_bgr)
    if crop.size == 0:
        return "unknown", 0.0
    try:
        input_tensor = preprocess_bgr_crop(crop).to(config.DEVICE)
        with torch.no_grad():
            probs = model.predict_proba(input_tensor)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
        class_name = config.GTSRB_CLASSES.get(pred_idx, f"class_{pred_idx}")
        return class_name, confidence
    except Exception as exc:  # pragma: no cover - defensive guard for malformed crops
        st.warning(f"Classification failed for a candidate region: {exc}")
        return "error", 0.0


def process_frame(
    frame_bgr: np.ndarray, model: TrafficSignCNN | None, confidence_threshold: float, colors: list[str]
) -> tuple[np.ndarray, list[dict]]:
    """
    Runs the full detect -> classify -> annotate pipeline on a single BGR
    frame. Returns the annotated frame (BGR) and a list of detection dicts
    for the results table.
    """
    try:
        candidates = find_candidate_regions(frame_bgr, colors=colors)
    except Exception as exc:
        st.error(f"Detection stage failed: {exc}")
        return frame_bgr, []

    annotated = frame_bgr.copy()
    detections = []

    for candidate in candidates:
        if model is not None:
            class_name, confidence = classify_candidate(model, candidate, frame_bgr)
        else:
            class_name, confidence = "N/A - load a trained model", 0.0

        if model is not None and confidence < confidence_threshold:
            continue  # skip low-confidence detections when a real model is loaded

        x, y, w, h = candidate.bbox
        box_color = (0, 255, 0) if (model is None or confidence >= confidence_threshold) else (0, 165, 255)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)

        caption = f"{class_name} ({confidence:.0%})" if model is not None else "detected region"
        text_y = max(y - 10, 15)
        cv2.putText(
            annotated, caption, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA
        )

        detections.append(
            {
                "class": class_name,
                "confidence": confidence,
                "color_cue": candidate.color,
                "bbox": candidate.bbox,
            }
        )

    return annotated, detections


# Streamlit UI

def main() -> None:
    st.title("Automated Traffic Sign & Road Perception Pipeline")
    st.caption(
        "Classical HSV/Sobel/contour detection proposes candidate sign regions; "
        "a custom CNN classifies each crop into one of 43 GTSRB sign categories."
    )

    model = load_model()
    if model is None:
        st.warning(
            f"No trained CNN checkpoint found at `{config.MODEL_PATH}`. "
            "The detector will still run and show candidate regions, but classification "
            "labels will be placeholders until you run `python -m src.cnn_classifier` "
            "(or your own training script) to produce a checkpoint."
        )

    with st.sidebar:
        st.header("Settings")
        selected_colors = st.multiselect(
            "Sign colors to detect", options=["red", "blue", "yellow"], default=["red", "blue", "yellow"]
        )
        confidence_threshold = st.slider(
            "Minimum confidence to display", min_value=0.0, max_value=1.0,
            value=config.CONFIDENCE_THRESHOLD, step=0.05,
        )
        st.markdown("---")
        st.markdown(
            "**Pipeline stages**\n"
            "1. HSV color masking (red/blue/yellow)\n"
            "2. Sobel edges + contour candidate boxes\n"
            "3. CNN classification per crop\n"
            "4. Labeled overlay with confidence"
        )

    tab_image, tab_video = st.tabs(["Image", "Video"])

    # --------------------------------------------------------------------- #
    # Image tab
    # --------------------------------------------------------------------- #
    with tab_image:
        uploaded_image = st.file_uploader(
            "Upload a road scene image", type=["jpg", "jpeg", "png", "bmp"], key="image_uploader"
        )
        if uploaded_image is not None:
            file_bytes = np.frombuffer(uploaded_image.read(), dtype=np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                st.error("Could not decode the uploaded file as an image. Please upload a valid JPG/PNG.")
            else:
                with st.spinner("Running detection + classification..."):
                    annotated, detections = process_frame(
                        frame_bgr, model, confidence_threshold, selected_colors or ["red", "blue", "yellow"]
                    )

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                with col2:
                    st.subheader("Detected & Classified")
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                st.subheader(f"Detections ({len(detections)})")
                if detections:
                    st.dataframe(
                        [
                            {
                                "Class": d["class"],
                                "Confidence": f"{d['confidence']:.1%}" if model is not None else "N/A",
                                "Color cue": d["color_cue"],
                                "BBox (x, y, w, h)": d["bbox"],
                            }
                            for d in detections
                        ],
                        use_container_width=True,
                    )
                else:
                    st.info("No candidate sign regions found. Try adjusting the color selection.")

    # --------------------------------------------------------------------- #
    # Video tab
    # --------------------------------------------------------------------- #
    with tab_video:
        uploaded_video = st.file_uploader(
            "Upload a short road scene video", type=["mp4", "avi", "mov"], key="video_uploader"
        )
        frame_stride = st.number_input(
            "Process every Nth frame (higher = faster preview)", min_value=1, max_value=30, value=5
        )
        max_frames = st.number_input(
            "Maximum frames to process", min_value=1, max_value=300, value=60,
            help="Caps processing time for long videos in this demo app.",
        )

        if uploaded_video is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as tmp:
                tmp.write(uploaded_video.read())
                video_path = tmp.name

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Could not open the uploaded video file.")
            else:
                st.info("Processing video frames — this is a demo preview, not real-time playback.")
                progress_bar = st.progress(0)
                frame_display = st.empty()
                detections_summary: dict[str, int] = {}

                frame_idx = 0
                processed_count = 0

                while cap.isOpened() and processed_count < max_frames:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break
                    if frame_idx % frame_stride == 0:
                        annotated, detections = process_frame(
                            frame_bgr, model, confidence_threshold, selected_colors or ["red", "blue", "yellow"]
                        )
                        frame_display.image(
                            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                            caption=f"Frame {frame_idx}",
                            use_container_width=True,
                        )
                        for d in detections:
                            detections_summary[d["class"]] = detections_summary.get(d["class"], 0) + 1

                        processed_count += 1
                        progress_bar.progress(min(processed_count / max_frames, 1.0))

                    frame_idx += 1

                cap.release()
                os.unlink(video_path)

                st.subheader("Detection summary across processed frames")
                if detections_summary:
                    st.bar_chart(detections_summary)
                else:
                    st.info("No detections across the processed frames.")


if __name__ == "__main__":
    main()
