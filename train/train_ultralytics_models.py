#!/usr/bin/env python3

import torch
import os
import sys
import argparse
import subprocess
from omegaconf import OmegaConf

CONFIG = {}


# =========================
# CONFIG LOADING
# =========================
def load_config(config_path: str):
    global CONFIG

    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    dataset_type = cfg["dataset"]["type"]
    view_type = cfg["dataset"].get("view_type", "top_view")

    base_data_dir = cfg["paths"]["data_dir"]

    if dataset_type == "soccertrack":
        prep_dir = os.path.join(base_data_dir, f"soccertrack_prep_{view_type}")
        yaml_path = os.path.join(prep_dir, "yolo", "dataset.yaml")

        CONFIG = {
            "dataset_type": dataset_type,
            "view_type": view_type,
            "data_dir": base_data_dir,
            "dataset_zip": cfg["paths"]["soccertrack"]["dataset_zip"],
            "dataset_dir": cfg["paths"]["soccertrack"]["dataset_dir"],
            "prep_dir": prep_dir,
            "output_dir": cfg["paths"]["soccertrack"]["output_dir"],
            "yaml_path": yaml_path,
            "requires_preparation": True,
            "training": cfg["training"],
            "augmentation": cfg["augmentation"],
            "runtime": cfg["runtime"]
        }

    elif dataset_type == "football":
        CONFIG = {
            "dataset_type": dataset_type,
            "data_dir": base_data_dir,
            "dataset_dir": cfg["paths"]["football"]["dataset_dir"],
            "output_dir": cfg["paths"]["football"]["output_dir"],
            "yaml_path": os.path.join(cfg["paths"]["football"]["dataset_dir"], "data.yaml"),
            "requires_preparation": False,
            "training": cfg["training"],
            "augmentation": cfg["augmentation"],
            "runtime": cfg["runtime"]
        }

    else:
        raise ValueError("Unknown dataset type")

    print(f"\n✓ Loaded config: {config_path}")
    print(f"  Dataset: {dataset_type}")
    print(f"  Output: {CONFIG['output_dir']}")

    return CONFIG


# =========================
# ENV SETUP
# =========================
def setup_environment():
    print(f"PyTorch version: {torch.__version__}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")


def install_dependencies():
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "ultralytics", "pandas", "pillow", "tqdm", "opencv-python", "matplotlib", "omegaconf", "-q"],
        check=True
    )


# =========================
# DATASET HANDLING
# =========================
def unzip_dataset():
    if not CONFIG.get("dataset_zip"):
        return

    if os.path.isdir(CONFIG["dataset_dir"]):
        print("✓ Dataset exists")
        return

    subprocess.run(["unzip", "-q", CONFIG["dataset_zip"], "-d", CONFIG["dataset_dir"]], check=True)


def prepare_dataset():
    sys.path.insert(0, os.path.join(os.getcwd(), "utils"))

    from utils.dataset_preparation import extract_frames_from_videos, convert_soccertrack_csvs_to_yolo

    view_type = CONFIG["view_type"]

    video_root = os.path.join(CONFIG["dataset_dir"], view_type, "videos")
    annotation_root = os.path.join(CONFIG["dataset_dir"], view_type, "annotations")

    prep_dir = CONFIG["prep_dir"]
    frames = os.path.join(prep_dir, "frames")
    yolo = os.path.join(prep_dir, "yolo")

    if not os.path.isdir(frames):
        extract_frames_from_videos(video_root, frames, frame_stride=1)

    if not os.path.isdir(yolo):
        convert_soccertrack_csvs_to_yolo(annotation_root, frames, yolo, train_split=0.8)


# =========================
# TRAINING
# =========================
def start_training():
    training = CONFIG["training"]
    aug = CONFIG["augmentation"]

    model_name = training["model"]

    if model_name.startswith("yolo"):
        from ultralytics import YOLO
        model = YOLO(f"{model_name}.pt")
    else:
        from ultralytics import RTDETR
        model = RTDETR(f"{model_name}.pt")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    close_mosaic = int(training["epochs"] * aug["close_mosaic_ratio"])

    model.train(
        data=CONFIG["yaml_path"],
        epochs=training["epochs"],
        imgsz=training["img_size"],
        batch=training["batch"],
        device=device,
        project=CONFIG["output_dir"],
        name=model_name,
        patience=10,
        save=True,
        augment=True,
        deterministic=False,

        # augmentations from YAML
        degrees=aug["rotate"],
        translate=aug["translate"],
        mosaic=aug["mosaic"],
        scale=aug["scale"],
        mixup=aug["mixup"],
        close_mosaic=close_mosaic,

        amp=False,
        max_det=25
    )

    print("✅ Training complete!")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to YAML config")

    args = parser.parse_args()

    load_config(args.config)

    runtime = CONFIG["runtime"]

    if not runtime["skip_setup"]:
        setup_environment()
        install_dependencies()

    if CONFIG["requires_preparation"] and not runtime["skip_data_prep"]:
        unzip_dataset()
        prepare_dataset()

    start_training()


if __name__ == "__main__":
    main()