from ultralytics import YOLO
import coremltools as ct
import numpy as np
from PIL import Image
import os
import time

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# path to your trained YOLO weights
WEIGHTS = "models/output_combined_dataset/yolov8s/weights/best.pt"
IMG_SIZE = 640
EXPORT_FP16 = True  # set False if you want FP32
DISABLE_NMS = False  # set True to disable NMS in exported model
EXPORT_FORMAT = "both"  # ONNX or Core ML supported in this script
DATA_YAML = "data/combined_dataset/data.yaml"  # path to your data.yaml for class names, etc.

CONF_THRESH = 0.5
IOU_THRESH = 0.7
# ------------------------------------------------------------------


def export_coreml(weights, img_size, conf_thresh, iou_thresh, data_yaml=DATA_YAML):
    print("Loading YOLO model...")
    model = YOLO(weights)

    print("Exporting to Core ML...")
    model.export(
        format="coreml",
        imgsz=img_size,
        nms=not DISABLE_NMS,
        half=EXPORT_FP16,
        conf=conf_thresh,  # default confidence threshold baked into model for Core ML export
        iou=iou_thresh,  # default IoU threshold baked into model for Core ML export
        data=data_yaml,
    )

    print("Export finished")


def quick_test_coreml(weights, img_size):
    print("\n Running quick Core ML test...")

    # Find exported model
    mlpackage_path = weights.replace(".pt", ".mlpackage")

    try:
        mlmodel = ct.models.MLModel(mlpackage_path)
    except Exception as e:
        print(f"ERROR: Failed to load Core ML model at {mlpackage_path}: {e}")
        return

    dummy_input = Image.new(
        mode="RGB",
        size=(img_size, img_size),
        color=(128, 128, 128),
    )

    try:
        outputs = mlmodel.predict({"image": dummy_input})
        print("Core ML inference successful")
        print("Output keys:")
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
        print(f"Core ML inference time: {elapsed_ms:.2f} ms")
    except Exception as e:
        print(f"Core ML timing run failed: {e}")


def export_onnx(weights, img_size, conf_thresh, iou_thresh, data_yaml=DATA_YAML):
    print("Loading YOLO model...")
    model = YOLO(weights)

    print("Exporting to ONNX...")
    # ultralytics export handles ONNX conversion
    model.export(
        format="onnx",
        imgsz=img_size,
        nms=not DISABLE_NMS,
        # export with postprocessing (NMS) for easier ONNX runtime usage
        end2end=True,
        conf=conf_thresh,  # default confidence threshold baked into model for ONNX export
        iou=iou_thresh,  # default IoU threshold baked into model for ONNX export
        half=EXPORT_FP16,
        data=data_yaml,
    )

    print("ONNX export finished")


def quick_test_onnx(weights, img_size):
    print("\n Running quick ONNX runtime test...")

    try:
        import onnxruntime as ort
    except Exception:
        print("onnxruntime not installed; skip ONNX runtime test.")
        return

    onnx_path = weights.replace(".pt", ".onnx")
    if not os.path.isfile(onnx_path):
        print(f"ONNX model not found at {onnx_path}")
        return

    try:
        sess = ort.InferenceSession(onnx_path)
        input_name = sess.get_inputs()[0].name
        # Build a dummy input (N, C, H, W) float32
        dummy = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
        # warm-up
        sess.run(None, {input_name: dummy})
        # timed run
        t0 = time.perf_counter()
        outputs = sess.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        print("ONNX runtime inference successful")
        print(f"ONNX inference time: {elapsed_ms:.2f} ms")
        print("Output tensors:")
        for i, out in enumerate(outputs):
            print(f"  - output[{i}]: shape={getattr(out, 'shape', 'unknown')}")
    except Exception as e:
        print(f"ONNX runtime test failed: {e}")


def run_export(
    weights=WEIGHTS,
    img_size=IMG_SIZE,
    conf_thresh=CONF_THRESH,
    iou_thresh=IOU_THRESH,
    export_format=EXPORT_FORMAT,
    data_yaml=DATA_YAML,
):
    fmt = export_format.strip().lower() if isinstance(export_format, str) else "coreml"
    if fmt == "coreml":
        export_coreml(weights, img_size, conf_thresh, iou_thresh, data_yaml=data_yaml)
        quick_test_coreml(weights, img_size)
    elif fmt == "onnx":
        export_onnx(weights, img_size, conf_thresh, iou_thresh, data_yaml=data_yaml)
        quick_test_onnx(weights, img_size)
    elif fmt == "both":
        export_coreml(weights, img_size, conf_thresh, iou_thresh, data_yaml=data_yaml)
        quick_test_coreml(weights, img_size)
        export_onnx(weights, img_size, conf_thresh, iou_thresh, data_yaml=data_yaml)
        quick_test_onnx(weights, img_size)
    else:
        print(f"Unsupported EXPORT_FORMAT: {export_format}. Choose 'coreml', 'onnx', or 'both'.")


if __name__ == "__main__":
    run_export()
