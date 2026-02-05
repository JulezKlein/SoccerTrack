#!/usr/bin/env python3
"""
Training script for SoccerTrack detection model on macOS.
Data should be stored under ./data directory.
"""

import torch
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Configuration
DATA_DIR = "./data"
DATASET_ZIP = os.path.join(DATA_DIR, "soccertrack.zip")
DATASET_DIR = os.path.join(DATA_DIR, "soccertrack")
PREP_DIR = os.path.join(DATA_DIR, "soccertrack_prep_top_view")
OUTPUT_DIR = "./output"

# RT-DETR setup
RTDETR_DIR = "./RT-DETR"
RTDETR_PYTORCH_DIR = os.path.join(RTDETR_DIR, "rtdetrv2_pytorch")


def setup_environment():
    """Check CUDA/device availability and print info."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device available: {torch.device('mps' if torch.backends.mps.is_available() else 'cpu')}")
    print(f"MPS available: {torch.backends.mps.is_available()}")


def clone_rtdetr():
    """Clone RT-DETR repository if not exists."""
    if os.path.isdir(RTDETR_DIR):
        print("✓ RT-DETR is already cloned.")
        return
    
    print("Cloning RT-DETR...")
    subprocess.run(["git", "clone", "https://github.com/lyuwenyu/RT-DETR.git"], check=True)
    print("✓ RT-DETR cloned successfully.")


def install_dependencies():
    """Install RT-DETR dependencies."""
    print("\nInstalling dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"], check=True)
    
    requirements_path = os.path.join(RTDETR_PYTORCH_DIR, "requirements.txt")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path, "-q"], check=True)
    
    # Additional dependencies for data processing
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pandas", "pillow", "tqdm", "opencv-python", "matplotlib", "-q"],
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
    
    from utils.visualization import visualize_coco_sample
    
    coco_root = os.path.join(PREP_DIR, "coco")
    
    print("Visualizing sample...")
    visualize_coco_sample(
        coco_json_path=os.path.join(coco_root, "annotations", "train.json"),
        image_root=os.path.join(coco_root, "images", "train")
    )


def create_config_files():
    """Create RT-DETR configuration files."""
    # Save current working directory
    original_cwd = os.getcwd()
    
    # Verify RT-DETR directory exists
    if not os.path.isdir(RTDETR_PYTORCH_DIR):
        print(f"ERROR: RT-DETR directory not found at {RTDETR_PYTORCH_DIR}")
        raise FileNotFoundError(f"No such directory: {RTDETR_PYTORCH_DIR}")
    
    # Compute absolute paths BEFORE changing directory
    train_images_path = os.path.abspath(os.path.join(PREP_DIR, 'coco', 'images', 'train'))
    train_ann_path = os.path.abspath(os.path.join(PREP_DIR, 'coco', 'annotations', 'train.json'))
    val_images_path = os.path.abspath(os.path.join(PREP_DIR, 'coco', 'images', 'val'))
    val_ann_path = os.path.abspath(os.path.join(PREP_DIR, 'coco', 'annotations', 'val.json'))
    output_path = os.path.abspath(OUTPUT_DIR)
    
    os.chdir(RTDETR_PYTORCH_DIR)
    
    # Dataset config
    dataset_config = f"""task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox']

num_classes: 2
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {train_images_path}
    ann_file: {train_ann_path}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 0
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {val_images_path}
    ann_file: {val_ann_path}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 0
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
"""
    
    os.makedirs("configs/dataset", exist_ok=True)
    with open("configs/dataset/soccertrack_coco.yml", "w") as f:
        f.write(dataset_config)
    print("✓ Dataset config created.")
    
    # Dataloader config
    dataloader_config = """train_dataloader:
  dataset:
    transforms:
      ops:
        - {type: RandomPhotometricDistort, p: 0.5}
        - {type: RandomZoomOut, fill: 0}
        - {type: RandomIoUCrop, p: 0.8}
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: RandomHorizontalFlip}
        - {type: Resize, size: [640, 640], }
        - {type: SanitizeBoundingBoxes, min_size: 1}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
        - {type: ConvertBoxes, fmt: 'cxcywh', normalize: True}
      policy:
        name: stop_epoch
        epoch: 30
        ops: ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']

  collate_fn:
    type: BatchImageCollateFunction
    scales: [640, 640]
    stop_epoch: 30

  shuffle: True
  total_batch_size: 2
  num_workers: 0

val_dataloader:
  dataset:
    transforms:
      ops:
        - {type: Resize, size: [640, 640]}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
  shuffle: False
  total_batch_size: 2
  num_workers: 0
"""
    
    os.makedirs("configs/rtdetrv2/include", exist_ok=True)
    with open("configs/rtdetrv2/include/dataloader_soccertrack.yml", "w") as f:
        f.write(dataloader_config)
    print("✓ Dataloader config created.")
    
    # Optimizer config
    optimizer_config = """optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*backbone)(?!.*norm).*$'
      lr: 0.0001
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
      weight_decay: 0.

  lr: 0.001
  betas: [0.9, 0.999]
  weight_decay: 0.0001

lr_scheduler:
  type: MultiStepLR
  milestones: [1000]
  gamma: 0.1

lr_warmup_scheduler:
  type: LinearWarmup
  warmup_duration: 500

clip_max_norm: 1.0
use_ema: False
use_amp: False
"""
    
    with open("configs/rtdetrv2/include/optimizer_soccertrack.yml", "w") as f:
        f.write(optimizer_config)
    print("✓ Optimizer config created.")
    
    # Training config
    training_config = f"""__include__: [
  '../dataset/soccertrack_coco.yml',
  '../runtime.yml',
  './include/dataloader_soccertrack.yml',
  './include/optimizer_soccertrack.yml',
  './include/rtdetrv2_r50vd.yml',
]

output_dir: '{output_path}'

PResNet:
  depth: 18
  freeze_at: -1
  freeze_norm: False
  pretrained: True

HybridEncoder:
  in_channels: [128, 256, 512]
  hidden_dim: 256
  expansion: 0.5

RTDETRTransformerv2:
  num_layers: 3

epoches: 50
"""
    
    with open("configs/rtdetrv2/rtdetrv2_r18vd_120e_soccertrack.yml", "w") as f:
        f.write(training_config)
    print("✓ Training config created.")
    
    # Restore original working directory
    os.chdir(original_cwd)


def start_training():
    """Start the training process."""
    # Verify RT-DETR directory exists before training
    if not os.path.isdir(RTDETR_PYTORCH_DIR):
        print(f"ERROR: RT-DETR directory not found at {RTDETR_PYTORCH_DIR}")
        print("Make sure the RT-DETR repository was cloned successfully.")
        raise FileNotFoundError(f"No such directory: {RTDETR_PYTORCH_DIR}")
    
    # Get absolute path to training script and config
    train_script = os.path.join(os.path.abspath(RTDETR_PYTORCH_DIR), "tools", "train.py")
    config_file = os.path.join(os.path.abspath(RTDETR_PYTORCH_DIR), "configs", "rtdetrv2", "rtdetrv2_r18vd_120e_soccertrack.yml")
    
    if not os.path.isfile(train_script):
        print(f"ERROR: Training script not found at {train_script}")
        raise FileNotFoundError(f"No such file: {train_script}")
    
    if not os.path.isfile(config_file):
        print(f"ERROR: Config file not found at {config_file}")
        raise FileNotFoundError(f"No such file: {config_file}")
    
    print("\n" + "="*60)
    print("Starting RT-DETR training...")
    print("="*60)
    
    # Change to RT-DETR pytorch directory for training
    original_cwd = os.getcwd()
    try:
        os.chdir(RTDETR_PYTORCH_DIR)
        cmd = [
            sys.executable,
            "tools/train.py",
            "-c", "configs/rtdetrv2/rtdetrv2_r18vd_120e_soccertrack.yml"
        ]
        subprocess.run(cmd, check=True)
    finally:
        # Always restore the original working directory
        os.chdir(original_cwd)


def main():
    parser = argparse.ArgumentParser(description="Train SoccerTrack detection model")
    parser.add_argument("--skip-setup", action="store_true", help="Skip environment setup")
    parser.add_argument("--skip-data-prep", action="store_true", help="Skip dataset preparation")
    parser.add_argument("--visualize", action="store_true", help="Visualize sample from dataset")
    parser.add_argument("--downsize", action="store_true", help="Downsize all images in prep folder to 50% scale")
    parser.add_argument("--downsize-scale", type=float, default=0.5, help="Scale factor for downsizing (0.0-1.0)")
    
    args = parser.parse_args()
    
    print("SoccerTrack Training Pipeline")
    print("=" * 60)
    
    if not args.skip_setup:
        setup_environment()
        clone_rtdetr()
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
    
    if args.visualize:
        visualize_sample()
    
    create_config_files()
    start_training()


if __name__ == "__main__":
    main()
