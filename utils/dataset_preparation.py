import os
import cv2
from tqdm import tqdm
import json
import pandas as pd
from PIL import Image
from pathlib import Path


def downsize_images(image_dir: str, scale_factor: float = 0.5):
    """
    Downsize all images in a directory by a given scale factor and overwrite them.
    
    Args:
        image_dir (str): Path to directory containing images
        scale_factor (float): Scale factor (0.5 = 50% of original size). Default: 0.5
    
    Example:
        downsize_images("./data/soccertrack_prep_top_view/coco/images", scale_factor=0.5)
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Directory not found: {image_dir}")
        return
    
    # Find all image files
    image_files = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.jpeg")) + list(image_dir.glob("**/*.png"))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Downsizing {len(image_files)} images to {scale_factor*100:.0f}% of original size...")
    
    for img_path in tqdm(image_files, desc="Downsizing images"):
        try:
            # Open image
            img = Image.open(img_path)
            
            # Calculate new dimensions
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            
            # Resize using high-quality resampling
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Overwrite original
            img_resized.save(img_path, quality=95)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print("Image downsizing complete")


def extract_frames_from_videos(
        video_root: str = "/content/soccertrack/top_view/videos",
        output_root: str = "/content/data_prep/soccertrack_frames",
        img_ext: str = ".jpg",
        frame_stride: int = 20,
):
    """
    Extract frames from all MP4 videos in video_root.

    Args:
        video_root (str): folder containing .mp4 files
        output_root (str): where extracted frames will be saved
        img_ext (str): image extension (default .jpg)
    """
    os.makedirs(output_root, exist_ok=True)

    videos = [v for v in os.listdir(video_root) if v.endswith(".mp4")]
    assert len(videos) > 0, f"No mp4 files found in {video_root}"

    for video in videos:
        name = os.path.splitext(video)[0]
        out_dir = os.path.join(output_root, name)
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(os.path.join(video_root, video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = 1

        for _ in tqdm(range(total), desc=f"Extracting {name}"):
            if frame_id % frame_stride != 0:
                frame_id += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            out_path = os.path.join(out_dir, f"{frame_id:06d}{img_ext}")
            cv2.imwrite(out_path, frame)
            frame_id += 1

        cap.release()

    print("✅ Frame extraction complete")





def convert_soccertrack_csvs_to_coco(
    annotation_root: str,
    image_root: str,
    output_root: str,
    train_split: float = 0.8,
    frame_stride: int = 20,
):
    # ---------------------------------------------------------
    # Setup
    # ---------------------------------------------------------
    os.makedirs(f"{output_root}/images/train", exist_ok=True)
    os.makedirs(f"{output_root}/images/val", exist_ok=True)
    os.makedirs(f"{output_root}/annotations", exist_ok=True)

    categories = [{"id": 1, "name": "player"}]
    images = []
    annotations = []

    img_id = 1
    ann_id = 1

    csv_files = sorted(f for f in os.listdir(annotation_root) if f.endswith(".csv"))
    assert csv_files, f"No CSV files found in {annotation_root}"

    split_idx = int(len(csv_files) * train_split)
    train_csvs = set(csv_files[:split_idx])

    # ---------------------------------------------------------
    # Process each video
    # ---------------------------------------------------------
    for csv_file in csv_files:
        video_name = os.path.splitext(csv_file)[0]
        split = "train" if csv_file in train_csvs else "val"

        csv_path = os.path.join(annotation_root, csv_file)
        img_dir = os.path.join(image_root, video_name)

        if not os.path.isdir(img_dir):
            print(f"⚠️  Image dir missing: {img_dir} — skipping")
            continue

        # Cache image list (ordered)
        img_files = sorted(
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png"))
        )

        if not img_files:
            print(f"⚠️  No images in {img_dir} — skipping")
            continue

        # ---------------------------------------------------------
        # Load CSV
        # ---------------------------------------------------------
        df = pd.read_csv(csv_path, header=None)

        if df.shape[0] < 4:
            print(f"⚠️  CSV malformed: {csv_file}")
            continue

        team_ids = df.iloc[0]
        player_ids = df.iloc[1]
        attrs = df.iloc[2]

        data = df.iloc[3:].reset_index(drop=True)
        data.rename(columns={0: "frame"}, inplace=True)

        data = data[pd.to_numeric(data["frame"], errors="coerce").notnull()]
        data["frame"] = data["frame"].astype(int)

        if data.empty:
            print(f"⚠️  No valid frames in {csv_file}")
            continue

        # ---------------------------------------------------------
        # Build player → bbox column mapping
        # ---------------------------------------------------------
        players = {}  # (team_id, player_id) -> {attr: col}

        for col in df.columns[1:]:
            team = team_ids[col]
            pid = player_ids[col]
            attr = attrs[col]

            if pd.isna(team) or pd.isna(pid) or pd.isna(attr):
                continue

            if isinstance(pid, str) and pid.upper() == "BALL":
                continue

            if not str(team).isdigit() or not str(pid).isdigit():
                continue

            if attr not in {"bb_left", "bb_top", "bb_width", "bb_height"}:
                continue

            key = (int(team), int(pid))
            players.setdefault(key, {})[attr] = col

        if not players:
            print(f"⚠️  No players found in {csv_file}")
            continue

        # ---------------------------------------------------------
        # Iterate frames
        # ---------------------------------------------------------
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"COCO {video_name}"):

            frame_num = int(row["frame"])

            if frame_stride > 1 and frame_num % frame_stride != 0:
                continue

            # SoccerTrack frame index → image index
            img_idx = frame_num - 1
            if img_idx < 0 or img_idx >= len(img_files):
                continue

            frame_img = os.path.join(img_dir, img_files[img_idx])

            coco_img_name = f"{video_name}_{frame_num:06d}.jpg"
            dst_img = os.path.join(output_root, "images", split, coco_img_name)

            if not os.path.exists(dst_img):
                Image.open(frame_img).save(dst_img)

            w_img, h_img = Image.open(frame_img).size

            images.append({
                "id": img_id,
                "file_name": coco_img_name,
                "width": w_img,
                "height": h_img,
                "video_name": video_name,
            })

            # -----------------------------------------------------
            # Annotations
            # -----------------------------------------------------
            for (team_id, pid), cols in players.items():

                if len(cols) != 4:
                    continue

                vals = {k: row[v] for k, v in cols.items()}

                if any(pd.isna(v) for v in vals.values()):
                    continue

                x = float(vals["bb_left"])
                y = float(vals["bb_top"])
                w = float(vals["bb_width"])
                h = float(vals["bb_height"])

                if w <= 1 or h <= 1:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
                ann_id += 1

            img_id += 1

    # ---------------------------------------------------------
    # Write COCO
    # ---------------------------------------------------------
    def build_split(is_train: bool):
        vids = train_csvs if is_train else set(csv_files) - train_csvs
        imgs = [i for i in images if i["video_name"] + ".csv" in vids]
        img_ids = {i["id"] for i in imgs}
        anns = [a for a in annotations if a["image_id"] in img_ids]
        return {"images": imgs, "annotations": anns, "categories": categories}

    with open(f"{output_root}/annotations/train.json", "w") as f:
        json.dump(build_split(True), f)

    with open(f"{output_root}/annotations/val.json", "w") as f:
        json.dump(build_split(False), f)

    print("✅ SoccerTrack → COCO conversion complete")
    print(f"Images: {len(images)} | Annotations: {len(annotations)}")


def coco_to_yolo_labels(coco_json_path: str, images_dir: str, labels_out_dir: str, class_mapping: dict = None, overwrite: bool = False):
    """
    Convert COCO-format annotations to YOLO (one .txt per image) labels.

    Args:
        coco_json_path: path to COCO json (train.json or val.json)
        images_dir: directory containing corresponding images
        labels_out_dir: directory where label .txt files will be written
        class_mapping: optional dict mapping COCO category_id -> yolo class index (default maps in-order starting at 0)
        overwrite: whether to overwrite existing label files (default False)
    """
    from pathlib import Path

    coco_path = Path(coco_json_path)
    images_dir = Path(images_dir)
    labels_out_dir = Path(labels_out_dir)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_path, "r") as f:
        coco = json.load(f)

    # Build category id -> name and index map
    cats = {c['id']: c['name'] for c in coco.get('categories', [])}
    if class_mapping is None:
        # default: map sorted category ids to 0..N-1
        sorted_ids = sorted(cats.keys())
        class_mapping = {cid: i for i, cid in enumerate(sorted_ids)}

    # Map image id -> (file_name, width, height)
    img_info = {img['id']: img for img in coco.get('images', [])}

    # Group annotations by image_id
    anns_by_image = {}
    for ann in coco.get('annotations', []):
        img_id = ann['image_id']
        anns_by_image.setdefault(img_id, []).append(ann)

    for img_id, info in img_info.items():
        filename = info.get('file_name')
        width = info.get('width')
        height = info.get('height')
        if not filename:
            continue

        img_path = images_dir / filename
        if not img_path.exists():
            # try plain filename without prefix
            # skip if missing
            print(f"⚠️  Image missing: {img_path} — skipping label creation")
            continue

        label_lines = []
        for ann in anns_by_image.get(img_id, []):
            bbox = ann.get('bbox')  # [x,y,w,h] absolute
            cat_id = ann.get('category_id')
            if bbox is None or cat_id is None:
                continue

            x, y, w, h = bbox
            # convert to YOLO xc yc w h (normalized)
            xc = (x + w / 2.0) / width
            yc = (y + h / 2.0) / height
            nw = w / width
            nh = h / height

            cls_ix = class_mapping.get(cat_id, 0)
            label_lines.append(f"{cls_ix} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        # Write label file (skip if exists and overwrite=False)
        out_label = labels_out_dir / (Path(filename).stem + ".txt")
        if out_label.exists() and not overwrite:
            continue
        with open(out_label, "w") as lf:
            lf.write("\n".join(label_lines))

    print(f"✓ Converted COCO -> YOLO labels: {coco_json_path} -> {labels_out_dir}")


def build_yolo_annotations(prep_root: str = "./data/soccertrack_prep_top_view", coco_subdir: str = "coco", overwrite: bool = False):
    """
    Create YOLO-style label files in the COCO folder (alongside images).

    Writes `labels/train` and `labels/val` label files derived from existing COCO annotations,
    and creates a `dataset.yaml` in the coco root.

    This does NOT modify existing COCO annotations; it only adds label `.txt` files and a `dataset.yaml`.

    Args:
        prep_root: path to the prep folder that contains the `coco` folder
        coco_subdir: name of the coco folder under `prep_root` (default 'coco')
        overwrite: whether to overwrite existing label files in the target folder
    """
    from pathlib import Path

    prep_root = Path(prep_root)
    coco_root = prep_root / coco_subdir
    if not coco_root.exists():
        raise FileNotFoundError(f"COCO root not found: {coco_root}")

    images_train = coco_root / "images" / "train"
    images_val = coco_root / "images" / "val"
    ann_train = coco_root / "annotations" / "train.json"
    ann_val = coco_root / "annotations" / "val.json"

    # Write labels into coco/labels/ (where YOLO expects them)
    labels_train = coco_root / "labels" / "train"
    labels_val = coco_root / "labels" / "val"
    labels_train.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    # Convert both splits
    if ann_train.exists():
        coco_to_yolo_labels(str(ann_train), str(images_train), str(labels_train), overwrite=overwrite)
    else:
        print(f"⚠️  COCO train json missing: {ann_train}")

    if ann_val.exists():
        coco_to_yolo_labels(str(ann_val), str(images_val), str(labels_val), overwrite=overwrite)
    else:
        print(f"⚠️  COCO val json missing: {ann_val}")

    # Create dataset.yaml in coco root
    abs_coco = os.path.abspath(str(coco_root))
    
    # Build names mapping
    if ann_train.exists():
        cats = json.load(open(ann_train))['categories']
    elif ann_val.exists():
        cats = json.load(open(ann_val))['categories']
    else:
        cats = [{"id": 1, "name": "player"}]
    
    # Map sorted category ids to 0..N-1
    sorted_cats = sorted(cats, key=lambda c: c['id'])
    num_classes = len(sorted_cats)
    
    # Build yaml content
    yaml_content = f"""path: {abs_coco}
train: images/train
val: images/val
nc: {num_classes}
names:
"""
    for i, c in enumerate(sorted_cats):
        yaml_content += f"  {i}: {c['name']}\n"
    
    yaml_path = coco_root / "dataset.yaml"
    with open(yaml_path, "w") as yf:
        yf.write(yaml_content)
    
    print(f"✓ YOLO labels written to: {coco_root / 'labels'}")
    print(f"✓ Dataset YAML created at: {yaml_path}")

if __name__ == '__main__':
    # Example: convert SoccerTrack CSVs to COCO format
    convert_soccertrack_csvs_to_coco(
        annotation_root=r"C:\Users\julia\Desktop\ML Material\SoccerTrack",
        image_root=r"C:\Users\julia\Desktop\ML Material\SoccerTrack",
        output_root=r"C:\Users\julia\Desktop\ML Material\SoccerTrack")