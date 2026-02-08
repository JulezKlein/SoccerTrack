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

# Dataset type configurations
DATASET_CONFIGS = {
    "soccertrack": {
        "view_type": "top_view",  # Change to "wide_view" if using wide view data
        "data_dir": "./data",
        "dataset_zip": "./data/soccertrack.zip",
        "dataset_dir": "./data/soccertrack",
        "prep_dir": "./data/soccertrack_prep_top_view",
        "output_dir": "./output_yolo",
        "requires_preparation": True,
        "yaml_path": None,  # Will be set to prep_dir/yolo/dataset.yaml
    },
    "football": {
        "data_dir": "./data",
        "dataset_dir": "./data/football_player_detection",
        "output_dir": "./output_yolo_football",
        "requires_preparation": False,
        "yaml_path": "./data/football_player_detection/data.yaml",
    }
}

# Global config (will be set based on dataset choice)
CONFIG = {}


def setup_environment():
    """Check CUDA/device availability and print info."""
    print(f"PyTorch version: {torch.__version__}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device available: {device}")
    print(f"MPS available: {torch.backends.mps.is_available()}")


def configure_dataset(dataset_type: str = "football", view_type: str = "top_view"):
    """
    Configure paths and settings based on dataset type.
    
    Args:
        dataset_type (str): Either "soccertrack" or "football"
        view_type (str): For soccertrack, either "top_view" or "wide_view"
    
    Returns:
        dict: Configuration dictionary with all necessary paths
    """
    global CONFIG
    
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_type].copy()
    
    # For soccertrack, update prep_dir based on view_type
    if dataset_type == "soccertrack":
        config["view_type"] = view_type
        config["prep_dir"] = os.path.join(config["data_dir"], f"soccertrack_prep_{view_type}")
        config["yaml_path"] = os.path.join(config["prep_dir"], "yolo", "dataset.yaml")
    
    CONFIG = config
    print(f"\n✓ Configuration loaded for '{dataset_type}' dataset")
    print(f"  - Dataset directory: {config['dataset_dir']}")
    print(f"  - Output directory: {config['output_dir']}")
    print(f"  - Requires preparation: {config['requires_preparation']}")
    
    return config


def install_dependencies():
    """Install YOLO dependencies."""
    print("\nInstalling dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"], check=True)
    
    # YOLO and data processing dependencies
    subprocess.run(
        [sys.executable, "-m", "pip", "install", 
         "ultralytics", "pandas", "pillow", "tqdm", "opencv-python", "matplotlib", "-q"],
        check=True
    )
    print("✓ Dependencies installed.")


def unzip_dataset():
    """Unzip dataset if needed."""
    dataset_dir = CONFIG["dataset_dir"]
    dataset_zip = CONFIG.get("dataset_zip")
    
    if not dataset_zip:
        print("✓ Dataset is already in YOLO format, no unzipping needed.")
        return
    
    if os.path.isdir(dataset_dir):
        print("✓ Dataset already exists.")
        return
    
    if not os.path.isfile(dataset_zip):
        print(f"ERROR: Dataset zip not found at {dataset_zip}")
        print(f"Please place the dataset zip in {os.path.dirname(dataset_zip)}")
        sys.exit(1)
    
    print(f"Unzipping dataset to {dataset_dir}...")
    subprocess.run(["unzip", "-q", dataset_zip, "-d", dataset_dir], check=True)
    print("✓ Dataset unzipped.")


def prepare_dataset():
    """Prepare YOLO dataset by extracting frames and converting to YOLO format."""
    sys.path.insert(0, os.path.join(os.getcwd(), "utils"))
    
    from utils.dataset_preparation import (
        extract_frames_from_videos,
        convert_soccertrack_csvs_to_yolo
    )
    
    view_type = CONFIG["view_type"]
    dataset_dir = CONFIG["dataset_dir"]
    prep_dir = CONFIG["prep_dir"]
    
    frame_out_root = os.path.join(prep_dir, "frames")
    yolo_root = os.path.join(prep_dir, "yolo")
    
    video_root = os.path.join(dataset_dir, view_type, "videos")
    annotation_root = os.path.join(dataset_dir, view_type, "annotations")
    
    train_split = 0.8
    frame_stride = 1
    
    os.makedirs(prep_dir, exist_ok=True)
    
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
    
    if not os.path.isdir(yolo_root):
        print("Converting SoccerTrack CSVs to YOLO format...")
        convert_soccertrack_csvs_to_yolo(
            annotation_root=annotation_root,
            image_root=frame_out_root,
            output_root=yolo_root,
            train_split=train_split,
            frame_stride=frame_stride
        )
        print("✓ YOLO format conversion complete.")
    else:
        print("✓ YOLO dataset already prepared.")


def validate_dataset():
    """Validate YOLO dataset format for the configured dataset."""
    sys.path.insert(0, os.path.join(os.getcwd(), "utils"))
    
    try:
        from utils.dataset_preparation import validate_yolo_dataset
    except ImportError:
        print("Validation module not available")
        return
    
    print("\n" + "="*60)
    print("Validating YOLO dataset format...")
    print("="*60)
    
    if CONFIG["requires_preparation"]:
        # For soccertrack, validate in the prepared directory
        yolo_root = os.path.join(CONFIG["prep_dir"], "yolo")
        train_labels = os.path.join(yolo_root, "labels", "train")
        train_images = os.path.join(yolo_root, "images", "train")
        val_labels = os.path.join(yolo_root, "labels", "val")
        val_images = os.path.join(yolo_root, "images", "val")
    else:
        # For football, validate in the dataset directory
        train_labels = os.path.join(CONFIG["dataset_dir"], "train", "labels")
        train_images = os.path.join(CONFIG["dataset_dir"], "train", "images")
        val_labels = os.path.join(CONFIG["dataset_dir"], "valid", "labels")
        val_images = os.path.join(CONFIG["dataset_dir"], "valid", "images")
    
    print(f"\nValidating training set...")
    validate_yolo_dataset(train_labels, train_images)
    
    print(f"\nValidating validation set...")
    validate_yolo_dataset(val_labels, val_images)
    
    print("\n✓ Dataset validation complete!")


def visualize_yolo_sample(image_path: str = None, label_path: str = None, class_names: list = None):
    """Visualize a YOLO formatted image with bounding boxes.
    
    Args:
        image_path (str, optional): path to the image file. If None, picks a random image from training set.
        label_path (str, optional): path to the corresponding YOLO label file
        class_names (list, optional): list of class names for labeling
    """
    sys.path.insert(0, os.path.join(os.getcwd(), "utils"))
    
    try:
        from utils.visualization import visualize_yolo_sample as viz_yolo
        import random
        import glob
        
        # Determine the images directory based on dataset type
        if CONFIG["requires_preparation"]:
            # For soccertrack, use the prepared directory
            images_dir = os.path.join(CONFIG["prep_dir"], "yolo", "images", "train")
        else:
            # For football, use the direct dataset directory
            images_dir = os.path.join(CONFIG["dataset_dir"], "train", "images")
        
        # If no image path provided, pick a random image from training set
        if image_path is None:
            image_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                         glob.glob(os.path.join(images_dir, "*.png"))
            if not image_files:
                print(f"No images found in {images_dir}")
                return
            image_path = random.choice(image_files)
        
        # If no label path provided, construct it from image path
        if label_path is None and image_path:
            label_path = os.path.splitext(image_path)[0] + ".txt"
        
        print(f"Visualizing: {os.path.basename(image_path)}")
        viz_yolo(
            image_path=image_path,
            label_path=label_path if os.path.exists(label_path) else None,
            class_names=class_names
        )
    except ImportError:
        print("Visualization module not available, skipping visualization")


def make_yolo_annotations():
    """Generate YOLO-format annotations for SoccerTrack dataset."""
    if not CONFIG["requires_preparation"]:
        print("ERROR: Annotation conversion is only available for the SoccerTrack dataset.")
        print("The football dataset is already in YOLO format.")
        return
    
    sys.path.insert(0, os.path.join(os.getcwd(), "utils"))
    
    try:
        from utils.dataset_preparation import build_yolo_annotations
    except ImportError:
        print("Annotation building module not available")
        return
    
    print("\n" + "="*60)
    print("Generating YOLO-format annotations for SoccerTrack...")
    print("="*60)
    
    build_yolo_annotations(prep_root=CONFIG["prep_dir"])
    print("✓ YOLO annotations generated!")


def start_training(epochs: int = 50, img_size: int = 640, model_name: str = "yolov8s"):
    """Start the YOLO training process."""
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
    
    # Get yaml path
    yaml_path = CONFIG["yaml_path"]
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=16,
        device=device,
        project=CONFIG["output_dir"],
        name=f"{model_name}",
        patience=10,
        save=True,
        verbose=True,
        augment=True,
        mosaic=0.0,
        scale=0.0,
        mixup=0.0,
        auto_augment=None,
        cutmix=0.0,
        amp=False,
        max_det=25,
        deterministic=False,
        iou = 0.6
    )
    
    print("\n✅ Training complete!")
    print(f"Results saved to: {CONFIG['output_dir']}/soccertrack_{model_name}")
    
    return results

def resume_training(ckpt_path: str):
    """Resume YOLO training from a checkpoint."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Installing now...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics", "-q"], check=True)
        from ultralytics import YOLO
    
    print("\n" + "="*60)
    print(f"Resuming training from checkpoint: {ckpt_path}")
    print("="*60)
    
    # Load model from checkpoint
    model = YOLO(ckpt_path)
    
    # Resume training
    results = model.train(
        resume=True
    )
    
    print("\n✅ Training resumed and complete!")    
    return results

def main():
    parser = argparse.ArgumentParser(description="Train object detection model with YOLO")
    parser.add_argument("--dataset", type=str, default="football", choices=["soccertrack", "football"], 
                        help="Dataset to use for training (default: football)")
    parser.add_argument("--view-type", type=str, default="top_view", choices=["top_view", "wide_view"],
                        help="View type for SoccerTrack dataset (default: top_view)")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip dataset preparation")
    parser.add_argument("--visualize", action="store_true", help="Visualize sample from dataset")
    parser.add_argument("--validate", action="store_true", help="Validate YOLO dataset format before training")
    parser.add_argument("--make-yolo-annotations", action="store_true", help="Convert SoccerTrack annotations to YOLO format")
    parser.add_argument("--model", type=str, default="yolov8s", choices=["yolov8s", "yolov8n", "yolo26n"], 
                        help="YOLO model size (default: yolov8s)")
    parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs")
    parser.add_argument("--img-size", type=int, default=640, help="Training image size")
    parser.add_argument("--resume", action="store_true", help="Resume Training from a checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Configure dataset first
    configure_dataset(dataset_type=args.dataset, view_type=args.view_type)
    
    print(f"\nYOLO Training Pipeline ({args.model.upper()})")
    print("=" * 60)
    
    if not args.skip_setup:
        setup_environment()
        install_dependencies()
    
    # Only prepare dataset if needed (soccertrack requires preparation, football doesn't)
    if CONFIG["requires_preparation"]:
        if not args.skip_data_prep:
            unzip_dataset()
            prepare_dataset()
    else:
        print("\n✓ Dataset is already in YOLO format, skipping preparation.")
    
    # Generate YOLO annotations if requested (SoccerTrack only)
    if args.make_yolo_annotations:
        make_yolo_annotations()
    
    # Validate dataset if requested
    if args.validate:
        validate_dataset()
    
    # Visualize sample if requested
    if args.visualize:
        visualize_yolo_sample()
    
    if args.resume and args.checkpoint:
        print(f"\nResuming training from checkpoint: {args.checkpoint}")
        resume_training(ckpt_path=args.checkpoint)
        
    else:
        start_training(epochs=args.epochs, img_size=args.img_size, model_name=args.model)


if __name__ == "__main__":
    main()
