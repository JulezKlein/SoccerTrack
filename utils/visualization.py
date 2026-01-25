import json
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


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