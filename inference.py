#!/usr/bin/env python3
"""
Soccer Track Inference Script
Loads YOLO models with fallback logic (CoreML -> ONNX -> PyTorch)
and runs inference on images or videos.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Union, Optional, Tuple
import cv2
import numpy as np
from omegaconf import OmegaConf, DictConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handle model loading with fallback logic."""
    
    def __init__(self, base_path: Union[str, Path], model_paths: list, conf_threshold: float = 0.25):
        """
        Initialize the model loader.
        
        Args:
            base_path: Base directory containing model files
            model_paths: List of model paths with format specification
            conf_threshold: Confidence threshold for detections
        """
        self.base_path = Path(base_path)
        self.model_paths = model_paths
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_type = None
        
    def load(self) -> Tuple[object, str]:
        """
        Load model with fallback logic.
        
        Returns:
            Tuple of (model, model_type)
            
        Raises:
            RuntimeError: If no model can be loaded
        """
        for model_config in self.model_paths:
            model_path = self.base_path / model_config['path']
            model_format = model_config['format']
            
            try:
                if model_format == 'coreml':
                    self.model = self._load_coreml(model_path)
                elif model_format == 'onnx':
                    self.model = self._load_onnx(model_path)
                elif model_format == 'pytorch':
                    self.model = self._load_pytorch(model_path)
                else:
                    logger.warning(f"Unknown model format: {model_format}")
                    continue
                
                self.model_type = model_format
                logger.info(f"Successfully loaded {model_format} model from {model_path}")
                return self.model, model_format
                
            except Exception as e:
                logger.warning(f"Failed to load {model_format} model from {model_path}: {e}")
                continue
        
        raise RuntimeError("Could not load any model with the specified fallback paths")
    
    def _load_coreml(self, model_path: Path) -> object:
        """Load CoreML model."""
        try:
            import coremltools
            logger.debug(f"Loading CoreML model from {model_path}")
            
            # For .mlpackage format
            if model_path.is_dir():
                model_path = model_path / 'Data' / 'com.apple.CoreML' / 'model.mlmodel'
            
            model = coremltools.models.MLModel(str(model_path))
            logger.info("CoreML model loaded successfully")
            return model
        except ImportError:
            raise ImportError("coremltools not installed. Install with: pip install coremltools")
    
    def _load_onnx(self, model_path: Path) -> object:
        """Load ONNX model."""
        try:
            import onnxruntime as rt
            logger.debug(f"Loading ONNX model from {model_path}")
            
            sess_options = rt.SessionOptions()
            sess_options.log_severity_level = 3  # Suppress warnings
            
            # Try to use GPU if available
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            model = rt.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)
            logger.info(f"ONNX model loaded successfully (Provider: {model.get_providers()[0]})")
            return model
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
    
    def _load_pytorch(self, model_path: Path) -> object:
        """Load PyTorch model using Ultralytics."""
        try:
            from ultralytics import YOLO
            logger.debug(f"Loading PyTorch model from {model_path}")
            
            model = YOLO(str(model_path))
            logger.info("PyTorch model loaded successfully")
            return model
        except ImportError:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")


class InferenceEngine:
    """Handle inference for images and videos."""
    
    def __init__(self, model: object, model_type: str, config: DictConfig):
        """
        Initialize inference engine.
        
        Args:
            model: Loaded model object
            model_type: Type of model ('pytorch', 'onnx', 'coreml')
            config: OmegaConf configuration
        """
        self.model = model
        self.model_type = model_type
        self.config = config
        self.inference_config = config.inference
        self.model_config = config.model
    
    def run_on_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image with bounding boxes drawn
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Running inference on image: {image_path}")
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run inference based on model type
        if self.model_type == 'pytorch':
            result_image = self._infer_pytorch(image)
        elif self.model_type == 'onnx':
            result_image = self._infer_onnx(image)
        elif self.model_type == 'coreml':
            result_image = self._infer_coreml(image)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Save output if configured
        if self.inference_config.save_output:
            self._save_image(result_image, image_path)
        
        # Display if configured
        if self.inference_config.display:
            self._display_image(result_image)
        
        return result_image
    
    def run_on_video(self, video_path: Union[str, Path]) -> None:
        """
        Run inference on a video file.
        
        Args:
            video_path: Path to video file
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        logger.info(f"Running inference on video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or self.inference_config.video.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties - FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")
        
        # Setup video writer if saving
        out = None
        if self.inference_config.save_output:
            output_path = Path(self.inference_config.output_dir) / f"output_{video_path.stem}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*self.inference_config.video.codec)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            logger.info(f"Video output will be saved to: {output_path}")
        
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Run inference
                if self.model_type == 'pytorch':
                    result_frame = self._infer_pytorch(frame)
                elif self.model_type == 'onnx':
                    result_frame = self._infer_onnx(frame)
                elif self.model_type == 'coreml':
                    result_frame = self._infer_coreml(frame)
                else:
                    result_frame = frame
                
                # Write frame if saving
                if out is not None:
                    out.write(result_frame)
                
                # Display if configured
                if self.inference_config.display:
                    cv2.imshow('Soccer Track Inference', result_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Video playback interrupted by user")
                        break
                
                frame_idx += 1
                if frame_idx % 30 == 0:
                    logger.info(f"Processed {frame_idx}/{total_frames} frames")
        
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            logger.info(f"Video inference complete. Processed {frame_idx} frames")
    
    def _infer_pytorch(self, image: np.ndarray) -> np.ndarray:
        """Run inference using PyTorch model."""
        results = self.model.predict(
            image,
            conf=self.model_config.conf_threshold,
            iou=self.model_config.iou_threshold,
            imgsz=self.model_config.imgsz,
            verbose=False
        )
        
        # Draw bounding boxes
        result_image = results[0].plot(
            line_width=self.inference_config.line_width,
            conf=self.inference_config.show_conf
        )
        
        return result_image
    
    def _infer_onnx(self, image: np.ndarray) -> np.ndarray:
        """Run inference using ONNX model."""
        # Prepare input
        imgsz = self.model_config.imgsz
        h, w = image.shape[:2]
        
        # Resize and normalize
        scale = imgsz / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(image, (new_w, new_h))
        img_padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        pad_h = (imgsz - new_h) // 2
        pad_w = (imgsz - new_w) // 2
        img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized
        
        # Normalize and prepare for model
        img_normalized = img_padded.astype(np.float32) / 255.0
        img_input = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_input, 0)
        
        # Run inference
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: img_batch})
        predictions = outputs[0]  

        logger.info(f"ONNX inference completed. Output shape: {predictions.shape}")
        
        # Post-processing (simplified)
        result_image = self._draw_predictions_onnx(image, predictions, scale, pad_h, pad_w)
        
        return result_image
    
    def _infer_coreml(self, image: np.ndarray) -> np.ndarray:
        """Run inference using CoreML model."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow not installed. Install with: pip install Pillow")
        
        # CoreML requires PIL Image input
        imgsz = self.model_config.imgsz
        h, w = image.shape[:2]
        
        # Resize
        scale = imgsz / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = cv2.resize(image, (new_w, new_h))
        img_padded = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        pad_h = (imgsz - new_h) // 2
        pad_w = (imgsz - new_w) // 2
        img_padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = img_resized
        
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(img_rgb)
        
        # Run CoreML inference
        try:
            predictions = self.model.predict({'image': pil_image})
            logger.info(f"CoreML inference completed. Output keys: {predictions.keys()}")
            result_image = self._draw_predictions_coreml(image, predictions, scale, pad_h, pad_w)
        except Exception as e:
            logger.warning(f"CoreML inference failed: {e}, falling back to drawing input image")
            result_image = image.copy()
        
        return result_image
    
    def _draw_predictions_onnx(self, image: np.ndarray, predictions: np.ndarray, 
                               scale: float, pad_h: int, pad_w: int) -> np.ndarray:
        """Draw ONNX predictions on image.
        
        ONNX output format: [x1, y1, x2, y2, confidence, class_id]
        Coordinates are in the padded 640x640 image space - need to convert back to original.
        
        Classes:
        - 0: Player (Green)
        - 1: Referee (Red)
        """
        result_image = image.copy()
        h, w = image.shape[:2]
        
        # Color mapping: class_id -> (B, G, R)
        class_colors = {
            0: (0, 255, 0),      # Green for class 0 (Player)
            1: (0, 0, 255),      # Red for class 1 (Other)
        }
        
        class_names = {
            0: "Player",
            1: "Referee",
        }
        
        # Handle batch dimension if present
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Take first (and only) batch
        
        detections = []
        for pred in predictions:
            # Skip empty predictions
            if np.all(pred == 0):
                continue
            
            conf = float(pred[4])
            if conf < self.model_config.conf_threshold:
                continue
            
            # Extract corner coordinates from padded image space
            x1_padded, y1_padded = float(pred[0]), float(pred[1])
            x2_padded, y2_padded = float(pred[2]), float(pred[3])
            class_id = int(pred[5]) if pred.shape[0] > 5 else 0
            
            # Convert from padded 640x640 space to original image space
            # Remove padding offset, then scale
            x1 = int((x1_padded - pad_w) / scale)
            y1 = int((y1_padded - pad_h) / scale)
            x2 = int((x2_padded - pad_w) / scale)
            y2 = int((y2_padded - pad_h) / scale)
            
            # Clip to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            # Ensure x1 < x2 and y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Only add if box has positive area
            if x2 > x1 and y2 > y1:
                detections.append((x1, y1, x2, y2, conf, class_id))
        
        # Draw boxes on image
        for x1, y1, x2, y2, conf, class_id in detections:
            # Get color and label based on class_id
            color = class_colors.get(class_id, (0, 255, 0))
            class_name = class_names.get(class_id, f"Class {class_id}")
            
            # Draw rectangle
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, self.inference_config.line_width)
            
            # Draw label
            if self.inference_config.show_conf:
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background for better visibility
                cv2.rectangle(result_image, (x1, y1 - label_size[1] - 4), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(result_image, label, (x1, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        logger.debug(f"ONNX: Found {len(detections)} detections above confidence threshold")
        return result_image
    
    def _draw_predictions_coreml(self, image: np.ndarray, predictions: dict,
                                 scale: float, pad_h: int, pad_w: int) -> np.ndarray:
        """Draw CoreML predictions on image.
        
        CoreML output format:
        - 'coordinates': bounding box coordinates [[x1, y1, x2, y2], ...]
        - 'confidence': confidence scores [conf1, conf2, ...]
        - 'class_ids' (optional): class IDs (0=player green, 1=other red)
        """
        result_image = image.copy()
        h, w = image.shape[:2]
        
        # Color mapping: class_id -> (B, G, R)
        class_colors = {
            0: (0, 255, 0),      # Green for class 0
            1: (0, 0, 255),      # Red for class 1
        }
        
        class_names = {
            0: "Player",
            1: "Referee",
        }
        
        try:
            # Extract predictions
            coordinates = predictions.get('coordinates', [])
            confidences = predictions.get('confidence', [])
            class_ids = predictions.get('class_ids', None)
            
            if len(coordinates) == 0:
                logger.info("CoreML: No detections found")
                return result_image
            
            # Convert to numpy arrays if needed
            if not isinstance(coordinates, np.ndarray):
                coordinates = np.array(coordinates)
            if not isinstance(confidences, np.ndarray):
                confidences = np.array(confidences)
            if class_ids is not None and not isinstance(class_ids, np.ndarray):
                class_ids = np.array(class_ids)
            
            # Ensure coordinates is 2D
            if len(coordinates.shape) == 1:
                coordinates = coordinates.reshape((len(confidences), -1))
            
            detections = []
            
            # Process each detection
            for idx, (coord, conf) in enumerate(zip(coordinates, confidences)):
                class_id = np.argmax(conf) if isinstance(conf, np.ndarray) else 0
                conf = np.max(conf) if isinstance(conf, np.ndarray) else float(conf)
                
                if conf < self.model_config.conf_threshold:
                    continue
                
                # Extract normalized coordinates (center_x, center_y, width, height)
                if len(coord) >= 4:
                    x_center_norm = float(coord[0])
                    y_center_norm = float(coord[1])
                    width_norm = float(coord[2])
                    height_norm = float(coord[3])
                else:
                    logger.warning(f"Unexpected coordinate format: {coord}")
                    continue
                
                # Convert from normalized [0, 1] to pixel coordinates on 640x640 padded image
                imgsz = self.model_config.imgsz
                x_center_padded = x_center_norm * imgsz
                y_center_padded = y_center_norm * imgsz
                width_padded = width_norm * imgsz
                height_padded = height_norm * imgsz
                
                # Convert from center format to corner format
                x1_padded = x_center_padded - width_padded / 2
                y1_padded = y_center_padded - height_padded / 2
                x2_padded = x_center_padded + width_padded / 2
                y2_padded = y_center_padded + height_padded / 2
                
                # Convert from padded 640x640 space to original image space
                # Remove padding offset, then scale
                x1 = int((x1_padded - pad_w) / scale)
                y1 = int((y1_padded - pad_h) / scale)
                x2 = int((x2_padded - pad_w) / scale)
                y2 = int((y2_padded - pad_h) / scale)
                
                # Clip to image bounds
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))
                
                # Ensure x1 < x2 and y1 < y2
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                # Only add if box has positive area
                if x2 > x1 and y2 > y1:
                    detections.append((x1, y1, x2, y2, conf, class_id))
            
            # Draw boxes on image
            for x1, y1, x2, y2, conf, class_id in detections:
                # Get color and label based on class_id
                color = class_colors.get(class_id, (0, 255, 0))
                class_name = class_names.get(class_id, f"Class {class_id}")
                
                # Draw rectangle
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, self.inference_config.line_width)
                
                # Draw label
                if self.inference_config.show_conf:
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # Draw label background for better visibility
                    cv2.rectangle(result_image, (x1, y1 - label_size[1] - 4), 
                                 (x1 + label_size[0], y1), color, -1)
                    cv2.putText(result_image, label, (x1, y1 - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            logger.info(f"CoreML: Found {len(detections)} detections above confidence threshold")
            
        except Exception as e:
            logger.error(f"Error processing CoreML predictions: {e}")
        
        return result_image
    
    def _save_image(self, image: np.ndarray, original_path: Path) -> None:
        """Save result image."""
        output_dir = Path(self.inference_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"output_{original_path.stem}.jpg"
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, self.inference_config.image.quality])
        logger.info(f"Result saved to: {output_path}")
    
    def _display_image(self, image: np.ndarray) -> None:
        """Display image in a window."""
        cv2.imshow('Soccer Track Inference', image)
        logger.info("Displaying image. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Soccer Track Inference - YOLO model inference with fallback logic"
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='Path to input image or video file',
        default='data/football_player_detection/test/images/57508_003904_Sideline_frame173_jpg.rf.f88ab45efa2c9d2bf20ebe0cb4d47092.jpg'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to OmegaConf config file (default: config.yaml)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable display of results'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Disable saving of results'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        help='Override confidence threshold from config'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        if not Path(args.config).exists():
            logger.error(f"Config file not found: {args.config}")
            sys.exit(1)
        
        config = OmegaConf.load(args.config)
        logger.debug(f"Configuration loaded: {OmegaConf.to_yaml(config)}")
        
        # Override config with command line arguments
        if args.no_display:
            config.inference.display = False
        if args.no_save:
            config.inference.save_output = False
        if args.conf_threshold is not None:
            config.model.conf_threshold = args.conf_threshold
        
        # Load model with fallback logic
        loader = ModelLoader(
            base_path=config.model.base_path,
            model_paths=config.model.model_paths,
            conf_threshold=config.model.conf_threshold
        )
        model, model_type = loader.load()
        
        # Create inference engine
        engine = InferenceEngine(model, model_type, config)
        
        # Determine input type and run inference
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"Input path not found: {input_path}")
            sys.exit(1)
        
        # Define supported extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        
        # Handle folder input
        if input_path.is_dir():
            logger.info(f"Detected input type: FOLDER with {input_path}")
            
            # Find all images and videos in the folder
            image_files = []
            video_files = []
            
            for ext in image_extensions:
                image_files.extend(input_path.rglob(f'*{ext}'))
                image_files.extend(input_path.rglob(f'*{ext.upper()}'))
            
            for ext in video_extensions:
                video_files.extend(input_path.rglob(f'*{ext}'))
                video_files.extend(input_path.rglob(f'*{ext.upper()}'))
            
            # Remove duplicates
            image_files = list(set(image_files))
            video_files = list(set(video_files))
            
            logger.info(f"Found {len(image_files)} images and {len(video_files)} videos")
            
            if len(image_files) == 0 and len(video_files) == 0:
                logger.error("No images or videos found in the specified folder")
                sys.exit(1)
            
            # Process images
            for idx, image_path in enumerate(sorted(image_files), 1):
                try:
                    logger.info(f"Processing image {idx}/{len(image_files)}: {image_path.name}")
                    engine.run_on_image(image_path)
                except Exception as e:
                    logger.error(f"Failed to process image {image_path}: {e}")
            
            # Process videos
            for idx, video_path in enumerate(sorted(video_files), 1):
                try:
                    logger.info(f"Processing video {idx}/{len(video_files)}: {video_path.name}")
                    engine.run_on_video(video_path)
                except Exception as e:
                    logger.error(f"Failed to process video {video_path}: {e}")
        
        # Handle single file input
        else:
            file_ext = input_path.suffix.lower()
            
            if file_ext in image_extensions:
                logger.info("Detected input type: IMAGE")
                engine.run_on_image(input_path)
            elif file_ext in video_extensions:
                logger.info("Detected input type: VIDEO")
                engine.run_on_video(input_path)
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                logger.info(f"Supported image formats: {image_extensions}")
                logger.info(f"Supported video formats: {video_extensions}")
                sys.exit(1)
        
        logger.info("Inference completed successfully!")
    
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    main()
