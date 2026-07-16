from ultralytics import YOLO
import os
import logging
import torch

logger = logging.getLogger(__name__)


def get_optimal_batch_size():
    """
    Determines optimal batch size based on available VRAM.

    Returns:
        int: Optimal batch size (16, 8, or 4)
    """
    if torch.cuda.is_available():
        # Get GPU memory in GB
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_mem >= 8:
            return 16
        elif gpu_mem >= 4:
            return 8
        else:
            return 4
    else:
        # CPU training - use smaller batch
        return 4


def train_model(data_yaml_path, model_type=None):
    """
    Trains the YOLO model using the annotated dataset.

    Args:
        data_yaml_path: Path to the data.yaml file containing dataset configuration
        model_type: Optional weights path/name. Defaults to the YOLO_MODEL
                    environment variable or accuracy-first yolov8m.pt.

    Returns:
        Path to the trained model
    """
    try:
        # The medium backbone materially improves localization accuracy over
        # nano. Resource-constrained deployments can set YOLO_MODEL=yolov8n.pt.
        model_weights = model_type or os.getenv('YOLO_MODEL', 'yolov8m.pt')
        if model_weights == 'yolov8':  # Backward compatibility with old caller.
            model_weights = os.getenv('YOLO_MODEL', 'yolov8m.pt')
        model = YOLO(model_weights)

        # Determine optimal batch size based on available VRAM
        batch_size = get_optimal_batch_size()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Train with specific parameters
        results = model.train(
            data=data_yaml_path,
            epochs=75,
            imgsz=640,            # Image size
            batch=batch_size,     # Auto batch size based on VRAM
            patience=10,          # Early stopping patience
            save=True,           # Save model
            device=device        # Use GPU if available, else CPU
        )

        # Get the best model path
        metrics = getattr(results, "results_dict", {}) or {}
        map50 = metrics.get("metrics/mAP50(B)")
        map50_95 = metrics.get("metrics/mAP50-95(B)")
        if map50 is not None:
            logger.info("Validation mAP50: %.1f%%", float(map50) * 100)
        if map50_95 is not None:
            logger.info("Validation mAP50-95: %.1f%%", float(map50_95) * 100)

        save_dir = str(getattr(results, "save_dir", ""))
        if save_dir and os.path.isdir(save_dir):
            model_path = os.path.join(save_dir, "weights", "best.pt")

            if os.path.exists(model_path):
                logger.info(f"Model trained and saved at {model_path}")
                return model_path
            else:
                logger.error(f"Model file not found at {model_path}")
                # Try to find if last.pt exists as fallback
                last_path = os.path.join(
                    save_dir, "weights", "last.pt")
                if os.path.exists(last_path):
                    logger.info(
                        f"best.pt not found, using last.pt at {last_path}")
                    return last_path
                return None
        else:
            logger.error(
                "Training output directory was not created. Training may have failed.")
            return None

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None
