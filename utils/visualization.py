import json
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


def visualize_coco_sample(
    coco_json_path: str,
    image_root: str,
    image_id: int = None,
    max_boxes: int = 50,
    figsize=(12, 8)
):
    """
    Visualize one COCO image with bounding boxes.

    Args:
        coco_json_path (str): path to COCO annotation JSON
        image_root (str): root folder containing images
        image_id (int, optional): specific image_id to visualize
        max_boxes (int): limit number of boxes drawn
        figsize (tuple): matplotlib figure size
    """
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    if image_id is None:
        img_info = random.choice(images)
    else:
        img_info = next(i for i in images if i["id"] == image_id)

    img_path = os.path.join(image_root, img_info["file_name"])
    assert os.path.exists(img_path), f"Image not found: {img_path}"

    img = Image.open(img_path).convert("RGB")

    anns = [
        a for a in annotations
        if a["image_id"] == img_info["id"]
    ][:max_boxes]

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)

    for ann in anns:
        x, y, w, h = ann["bbox"]
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(
        f"{img_info['file_name']} | "
        f"Boxes: {len(anns)}"
    )
    ax.axis("off")
    plt.show()


def visualize_yolo_sample(
    image_path: str,
    label_path: str = None,
    figsize=(12, 8),
    class_names: list = None
):
    """
    Visualize an image with YOLO format bounding boxes.

    Args:
        image_path (str): path to the image file
        label_path (str, optional): path to the YOLO label file (.txt)
        figsize (tuple): matplotlib figure size
        class_names (list, optional): list of class names for labeling
            If None, uses class IDs as labels

    YOLO label format (normalized coordinates):
        <class_id> <x_center> <y_center> <width> <height>
        where coordinates are in range [0, 1]
    """
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]
    
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)
    
    boxes_count = 0
    
    # Load and draw labels if provided
    if label_path and os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # Convert from normalized to pixel coordinates
            x_center_px = x_center * img_width
            y_center_px = y_center * img_height
            width_px = width * img_width
            height_px = height * img_height
            
            # Convert from center format to top-left corner format
            x_tl = x_center_px - width_px / 2
            y_tl = y_center_px - height_px / 2
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x_tl, y_tl), width_px, height_px,
                linewidth=2,
                edgecolor="lime",
                facecolor="none"
            )
            ax.add_patch(rect)
            
            # Add class label
            if class_names and class_id < len(class_names):
                label_text = class_names[class_id]
            else:
                label_text = f"Class {class_id}"
            
            ax.text(
                x_tl, y_tl - 5,
                label_text,
                color="lime",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7)
            )
            
            boxes_count += 1
    
    # Set title with image name and box count
    img_name = os.path.basename(image_path)
    label_text = f" | Labels: {boxes_count}" if label_path else ""
    ax.set_title(f"{img_name}{label_text}")
    ax.axis("off")
    plt.tight_layout()
    plt.show()