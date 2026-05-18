"""Convert handheld_dataset folders into a combined YOLO-format dataset.

Creates `data/combined_dataset/train` and `data/combined_dataset/valid` subfolders
with `images/` and `labels/`. Images are copied and label files are written in
YOLO text format: `class x_center y_center width height` (normalized).

Class mapping (final):
- 0 = player (labels starting with k, m, h, r, s)
- 1 = ref (labels starting with "ref")

Assumptions:
- Each dataset folder lives at `data/handheld_dataset/data/<name>/`
- Images are in `img1/` named zero-padded like `0001.png` and correspond to
  the `img_nr` column in `gt/gt.txt` (1-based).
- Ground-truth CSV is `gt/gt.txt` with at least columns: img_nr, label_index,
  x_pixel, y_pixel, x_dim, y_dim, ...; if column 2 is missing/invalid, column
  8 is used as a fallback label index.
- `labels.txt` lists label names (one per line) and label indices in the gt
  file are 1-based indices into this list; negative or out-of-range indices
  are skipped.

Usage: run from repository root or directly execute this file.
"""

from pathlib import Path
import shutil
import csv

ROOT = Path(__file__).resolve().parents[2]
HANDHELD_ROOT = ROOT / "data" / "handheld_dataset"
COMBINED_ROOT = ROOT / "data" / "combined_dataset"

# original image resolution (given)
IMG_W = 1920.0
IMG_H = 1080.0

PLAYER_PREFIXES = ("k", "m", "h", "r", "s")
REF_PREFIX = "ref"

CLASS_PLAYER = 0
CLASS_REF = 1


def read_split(split_file: Path):
    if not split_file.exists():
        return []
    return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]


def load_labels(labels_file: Path):
    if not labels_file.exists():
        return []
    return [label.strip() for label in labels_file.read_text().splitlines()]


def parse_gt(gt_file: Path):
    """Return dict: img_nr (int) -> list of rows (label_index(int), x,y,w,h)
    Supports both:
    - primary format where label index is in column 2
    - fallback format where class/label index is in column 8
    Caller decides which label indices to include.
    """
    mapping = {}
    if not gt_file.exists():
        return mapping
    with gt_file.open("r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                img_nr = int(float(row[0]))
            except Exception:
                continue

            label_index = None
            # Prefer legacy label index in column 2
            try:
                label_index = int(float(row[1]))
            except Exception:
                label_index = None

            # Fallback to MOT-like class index in column 8
            if label_index is None and len(row) >= 8:
                try:
                    label_index = int(float(row[7]))
                except Exception:
                    label_index = None

            if label_index is None:
                continue

            try:
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
            except Exception:
                continue

            mapping.setdefault(img_nr, []).append((label_index, x, y, w, h))
    return mapping


def label_to_class(label_name: str):
    if not label_name:
        return None
    ln = label_name.lower()
    if ln.startswith(REF_PREFIX):
        return CLASS_REF
    if ln[0] in PLAYER_PREFIXES:
        return CLASS_PLAYER
    return None


def ensure_dirs(base: Path):
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "labels").mkdir(parents=True, exist_ok=True)


def process_dataset(dataset_name: str, split_subdir: str):
    src = HANDHELD_ROOT / "data" / dataset_name
    if not src.exists():
        print(f"Warning: dataset folder not found: {src}")
        return

    labels_file = src / "labels.txt"
    labels = load_labels(labels_file)

    gt = parse_gt(src / "gt" / "gt.txt")

    img_src_dir = src / "img1"
    if not img_src_dir.exists():
        print(f"No img1 folder for {dataset_name}, skipping")
        return

    dest_base = COMBINED_ROOT / split_subdir
    ensure_dirs(dest_base)
    print(f"Processing dataset {dataset_name} with {len(gt)} annotated images and {len(labels)} labels and put it into {split_subdir} split.")
    for img_path in sorted(img_src_dir.iterdir()):
        if not img_path.is_file():
            continue
        img_name = img_path.name
        img_w, img_h = IMG_W, IMG_H

        # image number: assume filename like 0001.png -> 1
        try:
            img_nr = int(img_name.split(".")[0])
        except Exception:
            # try removing leading zeros
            try:
                img_nr = int(img_name)
            except Exception:
                continue

        dest_img_name = f"{dataset_name}_{img_name}"
        dest_img = dest_base / "images" / dest_img_name
        shutil.copy2(img_path, dest_img)

        anns = gt.get(img_nr, [])
        label_lines = []
        for label_idx, x, y, w, h in anns:
            # skip unseen labels (label index negative or zero)
            if label_idx <= 0:
                continue
            # label_idx is 1-based into labels.txt
            li = label_idx - 1
            if li < 0 or li >= len(labels):
                continue
            label_name = labels[li]
            cls = label_to_class(label_name)
            if cls is None:
                continue

            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            w_n = w / img_w
            h_n = h / img_h

            # clamp values to [0,1]
            def clamp(v):
                return max(0.0, min(1.0, float(v)))

            label_lines.append(f"{cls} {clamp(x_center):.6f} {clamp(y_center):.6f} {clamp(w_n):.6f} {clamp(h_n):.6f}")

        label_file = dest_base / "labels" / (dest_img_name.rsplit(".", 1)[0] + ".txt")
        label_file.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))


def main():
    train_split = HANDHELD_ROOT / "split_train.txt"
    val_split = HANDHELD_ROOT / "split_val.txt"

    train_sets = read_split(train_split)
    val_sets = read_split(val_split)

    (COMBINED_ROOT / "train").mkdir(parents=True, exist_ok=True)
    (COMBINED_ROOT / "valid").mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(train_sets)} train datasets and {len(val_sets)} val datasets")

    for d in train_sets:
        process_dataset(d, "train")

    for d in val_sets:
        process_dataset(d, "valid")

    print("Done. Combined dataset available at:", COMBINED_ROOT)


if __name__ == "__main__":
    main()
