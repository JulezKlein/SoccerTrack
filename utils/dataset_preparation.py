import os
import cv2
from tqdm import tqdm
import json
import pandas as pd
from PIL import Image


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




if __name__ == '__main__':
    convert_soccertrack_csvs_to_coco(annotation_root=r"C:\Users\julia\Desktop\ML Material\SoccerTrack",
                                     image_root=r"C:\Users\julia\Desktop\ML Material\SoccerTrack",
                                     output_root=r"C:\Users\julia\Desktop\ML Material\SoccerTrack")
