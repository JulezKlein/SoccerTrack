#!/usr/bin/env python3
"""
Quick tester for exported YOLO models (ONNX or Core ML).

Iterates images in a test folder, runs inference per-image, stamps time per inference,
and visualizes one sample with bounding boxes (saved to output directory).
"""

import os
import time
import argparse
import glob
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def draw_boxes_on_image(img: Image.Image, boxes, scores=None, classes=None, class_names=None):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = None
        if scores is not None:
            label = f"{scores[i]:.2f}"
        if classes is not None:
            cls = int(classes[i])
            if class_names and cls < len(class_names):
                name = class_names[cls]
                label = f"{name} {label}" if label else name
            else:
                label = f"{cls} {label}" if label else str(cls)

        if label:
            text_pos = (x1 + 3, y1 + 3)
            draw.text(text_pos, label, fill="white", font=font)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=False,
                        default="runs/detect/output_yolo_football/football_yolov8s/weights/best.onnx",
                        help="Path to exported model (.onnx or .mlpackage)")
    parser.add_argument("--format", type=str, default="coreml", choices=["onnx", "coreml"],
                        help="Export format to test")
    parser.add_argument("--test-dir", type=str, default="data/football_player_detection/test/images",
                        help="Path to test images folder")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--save-dir", type=str, default="runs/detect/test_results")
    parser.add_argument("--visualize-index", type=int, default=0, help="Index of sample to visualize (0-based)")
    parser.add_argument("--use-yolo", action="store_true", help="Use ultralytics YOLO class for inference (if available) instead of raw ONNXRuntime or CoreML.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(args.test_dir, "*.jpg")) + glob.glob(os.path.join(args.test_dir, "*.png")))
    if not image_paths:
        print(f"No images found in {args.test_dir}")
        return

    times = []
    visualized = False

    if args.format == "onnx":
        from ultralytics import YOLO
        use_yolo = False

        if use_yolo:
            model = YOLO(args.model)
            for idx, p in enumerate(image_paths):
                t0 = time.perf_counter()
                results = model(p, imgsz=args.img_size)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0
                times.append(elapsed_ms)

                # Attempt to extract boxes, scores, classes
                try:
                    r = results[0]
                    boxes = r.boxes
                    xyxy = None
                    conf = None
                    cls = None
                    try:
                        xyxy = boxes.xyxy.cpu().numpy()
                        conf = boxes.conf.cpu().numpy()
                        cls = boxes.cls.cpu().numpy()
                    except Exception:
                        try:
                            xyxy = boxes.xyxy.numpy()
                            conf = boxes.conf.numpy()
                            cls = boxes.cls.numpy()
                        except Exception:
                            xyxy = []

                    if idx == args.visualize_index and len(xyxy) > 0:
                        img = Image.open(p).convert("RGB")
                        img = draw_boxes_on_image(img, xyxy, scores=conf, classes=cls)
                        out_viz = os.path.join(args.save_dir, f"visualized_{args.format}")
                        img.save(out_viz)
                        print(f"Saved visualization: {out_viz}")
                        visualized = True

                except Exception as e:
                    print(f"Warning: failed to parse ultralytics results for {p}: {e}")

        else:
            # Fallback to onnxruntime inference (raw outputs)
            try:
                import onnxruntime as ort
            except Exception:
                print("onnxruntime not available and ultralytics not importable. Cannot run ONNX test.")
                return

            sess = ort.InferenceSession(args.model)
            input_name = sess.get_inputs()[0].name
            for idx, p in enumerate(image_paths):
                img = Image.open(p).convert("RGB")
                orig_w, orig_h = img.size
                img_resized = img.resize((args.img_size, args.img_size))
                arr = np.array(img_resized).astype(np.float32)
                # normalize to 0-1 and transpose to NCHW
                arr = arr / 255.0
                arr = np.transpose(arr, (2, 0, 1))
                arr = np.expand_dims(arr, 0).astype(np.float32)

                t0 = time.perf_counter()
                outputs = sess.run(None, {input_name: arr})
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0
                times.append(elapsed_ms)

                # Attempt to find a detections-like output and visualize the first sample
                if idx == args.visualize_index and not visualized:
                    # Try heuristics to locate boxes in outputs
                    det = None
                    for out in outputs:
                        a = np.array(out)
                        if a.ndim == 3 and a.shape[2] >= 6:
                            # assume shape (1, N, 85) or (1, N, 6)
                            det = a[0]
                            break
                    if det is not None:
                        # assume xyxy or xywh + conf + class
                        boxes = []
                        scores = []
                        classes = []
                        for row in det:
                            # if center-x,y,w,h normalize? we assume absolute xyxy
                            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
                            conf = float(row[4])
                            cls = int(row[5]) if row.shape[0] > 5 else 0
                            # map coords from model size back to original
                            scale_x = orig_w / args.img_size
                            scale_y = orig_h / args.img_size
                            boxes.append([x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y])
                            scores.append(conf)
                            classes.append(cls)

                        img_draw = Image.open(p).convert("RGB")
                        img_draw = draw_boxes_on_image(img_draw, boxes, scores=scores, classes=classes)
                        out_viz = os.path.join(args.save_dir, f"visualized_onnx_{Path(p).name}")
                        img_draw.save(out_viz)
                        print(f"Saved visualization (onnx fallback): {out_viz}")
                        visualized = True

    elif args.format == "coreml":
        try:
            import coremltools as ct
            mlmodel = ct.models.MLModel(args.model)
        except Exception as e:
            print(f"Failed to load Core ML model: {e}")
            return

        for idx, p in enumerate(image_paths):
            # Keep original image for final visualization
            orig_img = Image.open(p).convert("RGB")
            orig_w, orig_h = orig_img.size

            # Create resized copy for model input
            inp_img = orig_img.resize((args.img_size, args.img_size))
            w, h = inp_img.size

            # -------------------------
            # Core ML inference
            # -------------------------
            t0 = time.perf_counter()
            outputs = mlmodel.predict({"image": inp_img})
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

            if idx != args.visualize_index or visualized:
                continue

            print("📤 Core ML output keys:", outputs.keys())

            if "coordinates" not in outputs or "confidence" not in outputs:
                print("❌ Unexpected Core ML outputs")
                continue

            coords = np.asarray(outputs["coordinates"])   # (N, 4) xywh normalized
            scores = np.asarray(outputs["confidence"])    # (N, C)

            if coords.shape[0] == 0:
                print("⚠️ No detections (N=0)")
                visualized = True
                continue

            # -------------------------
            # Decode detections and map to original image size
            # -------------------------
            class_ids = scores.argmax(axis=1)
            confs = scores.max(axis=1)

            boxes_xyxy = []
            vis_scores = []
            vis_classes = []

            for (xc, yc, bw, bh), conf, cls in zip(coords, confs, class_ids):

                # xywh (normalized relative to model input) → xyxy in original pixels
                x1 = (xc - bw / 2) * orig_w
                y1 = (yc - bh / 2) * orig_h
                x2 = (xc + bw / 2) * orig_w
                y2 = (yc + bh / 2) * orig_h

                boxes_xyxy.append([x1, y1, x2, y2])
                vis_scores.append(float(conf))
                vis_classes.append(int(cls))

            if not boxes_xyxy:
                print("⚠️ All detections filtered by confidence threshold")
                visualized = True
                continue

            # -------------------------
            # Visualization on original image
            # -------------------------
            img_draw = draw_boxes_on_image(
                orig_img.copy(),
                boxes_xyxy,
                scores=vis_scores,
                classes=vis_classes,
            )

            out_viz = os.path.join(
                args.save_dir,
                f"visualized_coreml_{Path(p).name}"
            )
            img_draw.save(out_viz)
            print(f"✅ Saved visualization (coreml) at original size: {out_viz}")
            visualized = True


    # Summary of timings
    if times:
        times_arr = np.array(times)
        print(f"\nRan {len(times_arr)} inferences")
        print(f"Min: {times_arr.min():.2f} ms | Mean: {times_arr.mean():.2f} ms | Max: {times_arr.max():.2f} ms")
    else:
        print("No timings recorded")

    if not visualized:
        print("No visualization was produced for the selected sample; consider using an ONNX model or ultralytics-compatible export for full visualization support.")


if __name__ == "__main__":
    main()
