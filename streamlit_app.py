#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import streamlit as st
from omegaconf import OmegaConf

from inference import InferenceEngine, ModelLoader


MODEL_EXT_TO_FORMAT = {
    ".pt": "pytorch",
    ".onnx": "onnx",
    ".mlpackage": "coreml",
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def discover_models(models_root: Path) -> List[Path]:
    if not models_root.exists():
        return []

    discovered: List[Path] = []

    for ext in ("*.pt", "*.onnx", "*.mlpackage"):
        discovered.extend(models_root.rglob(ext))

    files_only = [p for p in discovered if p.is_file() or p.suffix == ".mlpackage"]
    files_only = sorted(set(files_only), key=lambda p: str(p).lower())
    return files_only


def discover_data_videos(data_root: Path) -> List[Path]:
    if not data_root.exists():
        return []

    discovered: List[Path] = []
    for ext in VIDEO_EXTENSIONS:
        discovered.extend(data_root.rglob(f"*{ext}"))
        discovered.extend(data_root.rglob(f"*{ext.upper()}"))

    return sorted(set([p for p in discovered if p.is_file()]), key=lambda p: str(p).lower())


def build_base_config(config_path: Path):
    if config_path.exists():
        config = OmegaConf.load(config_path)
    else:
        config = OmegaConf.create(
            {
                "model": {
                    "imgsz": 640,
                    "conf_threshold": 0.5,
                    "iou_threshold": 0.45,
                },
                "inference": {
                    "display": False,
                    "save_output": False,
                    "show_conf": True,
                    "line_width": 2,
                    "video": {"fps": 30, "codec": "mp4v"},
                    "image": {"quality": 95},
                },
            }
        )

    config.inference.display = False
    config.inference.save_output = False
    return config


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str, model_format: str):
    model_path_obj = Path(model_path)
    loader = ModelLoader(
        base_path=model_path_obj.parent,
        model_paths=[{"path": model_path_obj.name, "format": model_format}],
        conf_threshold=0.25,
    )
    return loader.load()


def infer_frame(engine: InferenceEngine, frame_bgr: np.ndarray) -> np.ndarray:
    if engine.model_type == "pytorch":
        return engine._infer_pytorch(frame_bgr)
    if engine.model_type == "onnx":
        return engine._infer_onnx(frame_bgr)
    if engine.model_type == "coreml":
        return engine._infer_coreml(frame_bgr)
    return frame_bgr


def model_label(path: Path, workspace_root: Path) -> str:
    try:
        rel = path.relative_to(workspace_root)
        return str(rel)
    except ValueError:
        return str(path)


def run_image(engine: InferenceEngine, uploaded_file):
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not decode image file.")
        return

    with st.spinner("Running inference on image..."):
        result_bgr = infer_frame(engine, image_bgr)

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption="Inference result", use_container_width=True)


def run_video_from_path(engine: InferenceEngine, input_path: Path, display_name: str):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        st.error("Could not open selected video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as output_tmp:
        output_path = Path(output_tmp.name)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_slot = st.empty()
    progress = st.progress(0)
    status = st.empty()

    processed = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result_frame = infer_frame(engine, frame)
            writer.write(result_frame)

            frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            frame_slot.image(frame_rgb, channels="RGB", use_container_width=True)

            processed += 1
            if total_frames > 0:
                progress.progress(min(processed / total_frames, 1.0))
            status.write(f"Processed frames: {processed}/{total_frames if total_frames > 0 else '?'}")

    finally:
        cap.release()
        writer.release()

    st.success(f"Video inference complete ({processed} frames).")
    st.video(str(output_path))

    with open(output_path, "rb") as f:
        st.download_button(
            label="Download processed video",
            data=f.read(),
            file_name=f"processed_{Path(display_name).stem}.mp4",
            mime="video/mp4",
        )


def run_video_uploaded(engine: InferenceEngine, uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_tmp:
        input_tmp.write(uploaded_file.read())
        input_path = Path(input_tmp.name)

    run_video_from_path(engine, input_path, uploaded_file.name)


def main():
    st.set_page_config(page_title="SoccerTrack Inference", layout="wide")
    st.title("SoccerTrack - Streamed Inference")

    workspace_root = Path(__file__).resolve().parent
    models_root = workspace_root / "models"
    data_root = workspace_root / "data"
    config_path = workspace_root / "configs" / "inf_config.yaml"

    discovered_models = discover_models(models_root)
    if not discovered_models:
        st.error("No model files found under ./models (expected .pt, .onnx, .mlpackage).")
        return

    model_options: Dict[str, Path] = {model_label(path, workspace_root): path for path in discovered_models}

    selected_label = st.selectbox("Select model", list(model_options.keys()))
    selected_model_path = model_options[selected_label]
    selected_format = MODEL_EXT_TO_FORMAT.get(selected_model_path.suffix.lower())

    if selected_format is None:
        st.error("Selected model format is not supported.")
        return

    conf_threshold = st.slider("Confidence threshold", 0.5, 1.0, 0.5, 0.01)
    input_source = st.radio("Input source", ["Upload", "Select from data videos"], horizontal=True)

    uploaded = None
    selected_data_video = None
    if input_source == "Upload":
        uploaded = st.file_uploader(
            "Upload an image or a video",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "mp4", "avi", "mov", "mkv", "flv", "wmv"],
        )
        input_ready = uploaded is not None
    else:
        data_videos = discover_data_videos(data_root)
        if not data_videos:
            st.warning("No videos found under ./data.")
            input_ready = False
        else:
            video_options: Dict[str, Path] = {model_label(path, workspace_root): path for path in data_videos}
            selected_data_label = st.selectbox("Choose a video from data/", list(video_options.keys()))
            selected_data_video = video_options[selected_data_label]
            input_ready = True

    run = st.button("Run inference", type="primary", disabled=not input_ready)
    if not run:
        return

    config = build_base_config(config_path)
    config.model.conf_threshold = conf_threshold

    with st.spinner(f"Loading model: {selected_label}"):
        model, model_type = load_model_cached(str(selected_model_path), selected_format)

    engine = InferenceEngine(model, model_type, config)

    if input_source == "Upload" and uploaded is not None:
        ext = Path(uploaded.name).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            run_image(engine, uploaded)
        else:
            run_video_uploaded(engine, uploaded)
    else:
        run_video_from_path(engine, selected_data_video, selected_data_video.name)


if __name__ == "__main__":
    main()
