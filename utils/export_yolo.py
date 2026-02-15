from ultralytics import YOLO
import coremltools as ct
import numpy as np
from PIL import Image
import os
import time

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
WEIGHTS = "models/output_yolo_football/yolov8s/weights/best.pt"  # path to your trained YOLO weights
IMG_SIZE = 640
EXPORT_FP16 = True        # set False if you want FP32
DISABLE_NMS = False     # set True to disable NMS in exported model        
EXPORT_FORMAT = "both"  # ONNX or Core ML supported in this script

CONF_THRESH = 0.4
IOU_THRESH = 0.2
# ------------------------------------------------------------------


def export_coreml():
    print("🔄 Loading YOLO model...")
    model = YOLO(WEIGHTS)

    print("📦 Exporting to Core ML...")
    model.export(
        format="coreml",
        imgsz=IMG_SIZE,
        nms=not DISABLE_NMS,
        half=EXPORT_FP16,
        conf=CONF_THRESH,  # default confidence threshold baked into model for Core ML export
        iou=IOU_THRESH,   # default IoU threshold baked into model for Core ML export
        data = "data/football_player_detection/data.yaml"
    )

    print("✅ Export finished")


def quick_test_coreml():
    print("\n🧪 Running quick Core ML test...")

    # Find exported model
    mlpackage_path = WEIGHTS.replace(".pt", ".mlpackage")

    try:
        mlmodel = ct.models.MLModel(mlpackage_path)
    except Exception as e:
        print(f"ERROR: Failed to load Core ML model at {mlpackage_path}: {e}")
        return

    dummy_input = Image.new(
        mode="RGB",
        size=(IMG_SIZE, IMG_SIZE),
        color=(128, 128, 128),
    )

    try:
        outputs = mlmodel.predict({"image": dummy_input})
        print("✅ Core ML inference successful")
        print("📤 Output keys:")
        for k, v in outputs.items():
            try:
                print(f"  - {k}: {v.shape}")
            except Exception:
                print(f"  - {k}: (non-array output)")
    except Exception as e:
        print(f"Core ML inference failed: {e}")
        return

    # Time a fresh inference
    try:
        t0 = time.perf_counter()
        _ = mlmodel.predict({"image": dummy_input})
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        print(f"⏱️ Core ML inference time: {elapsed_ms:.2f} ms")
    except Exception as e:
        print(f"Core ML timing run failed: {e}")


def export_onnx():
    print("🔄 Loading YOLO model...")
    model = YOLO(WEIGHTS)

    print("📦 Exporting to ONNX...")
    # ultralytics export handles ONNX conversion
    model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        nms=not DISABLE_NMS,
        end2end=True,  # export with postprocessing (NMS) for easier ONNX runtime usage
        conf=CONF_THRESH,  # default confidence threshold baked into model for ONNX export
        iou=IOU_THRESH,   # default IoU threshold baked into model for ONNX export
        half=EXPORT_FP16,
        data = "data/football_player_detection/data.yaml"
    )

    print("✅ ONNX export finished")


def quick_test_onnx():
    print("\n🧪 Running quick ONNX runtime test...")

    try:
        import onnxruntime as ort
    except Exception:
        print("onnxruntime not installed; skip ONNX runtime test.")
        return

    onnx_path = WEIGHTS.replace(".pt", ".onnx")
    if not os.path.isfile(onnx_path):
        print(f"ONNX model not found at {onnx_path}")
        return

    try:
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        # Build a dummy input (N, C, H, W) float32
        dummy = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        # warm-up
        sess.run(None, {input_name: dummy})
        # timed run
        t0 = time.perf_counter()
        outputs = sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        print("✅ ONNX runtime inference successful")
        print(f"⏱️ ONNX inference time: {elapsed_ms:.2f} ms")
        print("📤 Output tensors:")
        for i, out in enumerate(outputs):
            print(f"  - output[{i}]: shape={getattr(out, 'shape', 'unknown')}")
    except Exception as e:
        print(f"ONNX runtime test failed: {e}")


if __name__ == "__main__":
    fmt = EXPORT_FORMAT.strip().lower() if isinstance(EXPORT_FORMAT, str) else "coreml"
    if fmt == "coreml":
        export_coreml()
        quick_test_coreml()
    elif fmt == "onnx":
        export_onnx()
        quick_test_onnx()
    elif fmt == "both":
        export_coreml()
        quick_test_coreml()
        export_onnx()
        quick_test_onnx()
    else:
        print(f"Unsupported EXPORT_FORMAT: {EXPORT_FORMAT}. Choose 'coreml' or 'onnx'.")
