import os
import cv2
from tqdm import tqdm
import json
import pandas as pd
from PIL import Image

def extract_frames_from_videos(
    video_root: str = "/content/soccertrack/top_view/videos",
    output_root: str = "/content/data_prep/soccertrack_frames",
    img_ext: str = ".jpg"
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
            ret, frame = cap.read()
            if not ret:
                break

            out_path = os.path.join(out_dir, f"{frame_id:06d}{img_ext}")
            cv2.imwrite(out_path, frame)
            frame_id += 1

        cap.release()

    print("✅ Frame extraction complete")

def convert_soccertrack_csvs_to_coco(
    annotation_root: str ="/content/soccertrack/top_view/annotations",
    image_root: str ="/content/data_prep/soccertrack_frames",
    output_root: str ="/content/data_prep/soccertrack_coco",
    train_split: float = 0.8
):
    """
    Convert SoccerTrack CSV annotations (per video) to COCO detection format.

    Args:
        annotation_root (str): folder with *.csv files
        image_root (str): extracted frames root (one folder per video)
        output_root (str): output COCO folder
        train_split (float): fraction of videos for training
    """
    os.makedirs(f"{output_root}/images/train", exist_ok=True)
    os.makedirs(f"{output_root}/images/val", exist_ok=True)
    os.makedirs(f"{output_root}/annotations", exist_ok=True)

    categories = [{"id": 1, "name": "player"}]
    images = []
    annotations = []

    img_id = 1
    ann_id = 1

    csv_files = sorted([f for f in os.listdir(annotation_root) if f.endswith(".csv")])
    assert len(csv_files) > 0, "No CSV annotation files found"

    split_idx = int(len(csv_files) * train_split)
    train_csvs = set(csv_files[:split_idx])

    for csv_file in csv_files:
        video_name = os.path.splitext(csv_file)[0]
        split = "train" if csv_file in train_csvs else "val"

        df = pd.read_csv(os.path.join(annotation_root, csv_file), header=None)

        player_ids = df.iloc[1]
        attrs = df.iloc[2]
        data = df.iloc[3:].reset_index(drop=True)
        data.rename(columns={0: "frame"}, inplace=True)

        # group bbox columns by player (ignore BALL)
        players = {}
        for col in df.columns[1:]:
            pid = player_ids[col]
            if isinstance(pid, str) and "BALL" in pid:
                continue
            if attrs[col] in ["bb_left", "bb_top", "bb_width", "bb_height"]:
                players.setdefault(pid, []).append(col)

        img_dir = os.path.join(image_root, video_name)

        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"COCO {video_name}"):
            frame = int(row["frame"])
            frame_img = os.path.join(img_dir, f"{frame:06d}.jpg")
            if not os.path.exists(frame_img):
                continue

            coco_img_name = f"{video_name}_{frame:06d}.jpg"
            dst_img = os.path.join(output_root, "images", split, coco_img_name)

            if not os.path.exists(dst_img):
                Image.open(frame_img).save(dst_img)

            w_img, h_img = Image.open(frame_img).size

            images.append({
                "id": img_id,
                "file_name": coco_img_name,
                "width": w_img,
                "height": h_img
            })

            for _, cols in players.items():
                vals = {attrs[c]: row[c] for c in cols}
                if pd.isna(vals["bb_width"]) or pd.isna(vals["bb_height"]):
                    continue

                x, y, w, h = map(float, [
                    vals["bb_left"],
                    vals["bb_top"],
                    vals["bb_width"],
                    vals["bb_height"]
                ])

                if w <= 1 or h <= 1:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

            img_id += 1

    train_img_ids = {
        img["id"] for img in images
        if img["file_name"].split("_")[0] + ".csv" in train_csvs
    }

    train = {
        "images": [i for i in images if i["id"] in train_img_ids],
        "annotations": [a for a in annotations if a["image_id"] in train_img_ids],
        "categories": categories
    }

    val = {
        "images": [i for i in images if i["id"] not in train_img_ids],
        "annotations": [a for a in annotations if a["image_id"] not in train_img_ids],
        "categories": categories
    }

    with open(f"{output_root}/annotations/train.json", "w") as f:
        json.dump(train, f)

    with open(f"{output_root}/annotations/val.json", "w") as f:
        json.dump(val, f)

    print("✅ SoccerTrack CSV → COCO conversion complete")