#!/usr/bin/env python3
"""
Training script for SoccerTrack detection model using YOLOv8s on macOS.
Data should be stored under ./data directory.
"""

import torch
import os
import sys
import argparse
import subprocess

# Configuration
DATA_DIR = "./data"
DATASET_ZIP = os.path.join(DATA_DIR, "soccertrack.zip")
DATASET_DIR = os.path.join(DATA_DIR, "soccertrack")
PREP_DIR = os.path.join(DATA_DIR, "soccertrack_prep_top_view")
OUTPUT_DIR = "./output_yolov8"


def setup_environment():
    """Check CUDA/device availability and print info."""
    print(f"PyTorch version: {torch.__version__}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device available: {device}")
    print(f"MPS available: {torch.backends.mps.is_available()}")


def install_dependencies():
    """Install YOLOv8 dependencies."""
    print("\nInstalling dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"], check=True)
    
    # YOLOv8 and data processing dependencies
    subprocess.run(
        [sys.executable, "-m", "pip", "install", 
         "ultralytics", "pandas", "pillow", "tqdm", "opencv-python", "matplotlib", "-q"],
        check=True
    )
    print("✓ Dependencies installed.")


def unzip_dataset():
    """Unzip dataset if needed."""
    if os.path.isdir(DATASET_DIR):
        print("✓ Soccertrack dataset already exists.")
        return
    
    if not os.path.isfile(DATASET_ZIP):
        print(f"ERROR: Dataset zip not found at {DATASET_ZIP}")
        print(f"Please place soccertrack.zip in {DATA_DIR}")
        sys.exit(1)
    
    print(f"Unzipping dataset to {DATASET_DIR}...")
    subprocess.run(["unzip", "-q", DATASET_ZIP, "-d", DATASET_DIR], check=True)
    print("✓ Dataset unzipped.")


def prepare_dataset():
    """Prepare dataset by extracting frames and converting to COCO format."""
    sys.path.insert(0, os.path.join(os.getcwd(), "utils"))
    
    from utils.dataset_preparation import (
        extract_frames_from_videos,
        convert_soccertrack_csvs_to_coco
    )
    
    frame_out_root = os.path.join(PREP_DIR, "frames")
    out_root = os.path.join(PREP_DIR, "coco")
    
    video_root = os.path.join(DATASET_DIR, "top_view", "videos")
    annotation_root = os.path.join(DATASET_DIR, "top_view", "annotations")
    
    train_split = 0.8
    frame_stride = 1
    
    os.makedirs(PREP_DIR, exist_ok=True)
    
    if not os.path.isdir(frame_out_root):
        print(f"Extracting frames from {video_root}")
        extract_frames_from_videos(
            video_root=video_root,
            output_root=frame_out_root,
            frame_stride=frame_stride
        )
        print("✓ Frames extracted.")
    else:
        print("✓ Frames already extracted.")
    
    if not os.path.isdir(out_root):
        print("Converting SoccerTrack CSVs to COCO format...")
        convert_soccertrack_csvs_to_coco(
            annotation_root=annotation_root,
            image_root=frame_out_root,
            output_root=out_root,
            train_split=train_split,
            frame_stride=frame_stride
        )
        print("✓ COCO format conversion complete.")
    else:
        print("✓ COCO dataset already prepared.")


def visualize_sample():
    """Visualize a sample from the dataset."""
    sys.path.insert(0, os.path.join(os.getcwd(), "utils"))
    
    try:
        from utils.visualization import visualize_coco_sample
        
        coco_root = os.path.join(PREP_DIR, "coco")
        
        print("Visualizing sample...")
        visualize_coco_sample(
            coco_json_path=os.path.join(coco_root, "annotations", "train.json"),
            image_root=os.path.join(coco_root, "images", "train")
        )
    except ImportError:
        print("⚠️  visualization module not available, skipping visualization")


def start_training(epochs: int = 50, img_size: int = 640, batch_size: int = 8, model_name: str = "yolov8s"):
    """Start the YOLOv8 training process."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Installing now...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics", "-q"], check=True)
        from ultralytics import YOLO
    
    print("\n" + "="*60)
    print(f"Starting {model_name.upper()} training...")
    print("="*60)
    
    # Load pretrained model
    model_path = f"{model_name}.pt"
    model = YOLO(model_path)
    
    # Determine device
    device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Train the model
    results = model.train(
        data=os.path.join(PREP_DIR, "coco", "dataset.yaml"),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=OUTPUT_DIR,
        name=f"soccertrack_{model_name}",
        patience=10,
        save=True,
        verbose=True,
        augment=True,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        degrees=10,
        translate=0.1,
        scale=0.5,
    )
    
    print("\n✅ Training complete!")
    print(f"Results saved to: {OUTPUT_DIR}/soccertrack_{model_name}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train SoccerTrack detection model with YOLOv8")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip dataset preparation")
    parser.add_argument("--visualize", action="store_true", help="Visualize sample from dataset")
    parser.add_argument("--downsize", action="store_true", help="Downsize all images in prep folder to 50% scale")
    parser.add_argument("--downsize-scale", type=float, default=0.5, help="Scale factor for downsizing (0.0-1.0)")
    parser.add_argument("--make-yolo-annotations", action="store_true", help="Convert COCO annotations to YOLO format in annotations_yolo folder")
    parser.add_argument("--model", type=str, default="yolov8s", choices=["yolov8s", "yolov8n"], help="YOLOv8 model size (default: yolov8s)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--img-size", type=int, default=640, help="Training image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    
    args = parser.parse_args()
    
    print(f"SoccerTrack YOLOv8 Training Pipeline ({args.model.upper()})")
    print("=" * 60)
    
    if not args.skip_setup:
        setup_environment()
        install_dependencies()
    
    if not args.skip_data_prep:
        unzip_dataset()
        prepare_dataset()
    
    if args.downsize:
        sys.path.insert(0, os.path.join(os.getcwd(), "utils"))
        from utils.dataset_preparation import downsize_images
        
        image_dir = os.path.join(PREP_DIR, "coco", "images")
        print(f"\nDownsizing images to {args.downsize_scale*100:.0f}% scale...")
        downsize_images(image_dir, scale_factor=args.downsize_scale)
    
    if args.make_yolo_annotations:
        sys.path.insert(0, os.path.join(os.getcwd(), "utils"))
        from utils.dataset_preparation import build_yolo_annotations
        
        print("\nGenerating YOLO-format annotations...")
        build_yolo_annotations(prep_root=PREP_DIR)
    
    if args.visualize:
        visualize_sample()
    
    start_training(epochs=args.epochs, img_size=args.img_size, batch_size=args.batch_size, model_name=args.model)


if __name__ == "__main__":
    main()
