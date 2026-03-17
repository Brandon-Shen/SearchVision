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


def train_model(data_yaml_path, model_type='yolov8'):
    """
    Trains the YOLO model using the annotated dataset.

    Args:
        data_yaml_path: Path to the data.yaml file containing dataset configuration
        model_type: Type of YOLO model to train ('yolov8' recommended)

    Returns:
        Path to the trained model
    """
    try:
        # Initialize YOLO model
        model = YOLO('yolov8n.pt')  # Start with pre-trained model

        # Determine optimal batch size based on available VRAM
        batch_size = get_optimal_batch_size()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Train with specific parameters
        results = model.train(
            data=data_yaml_path,
            epochs=25,            # Default epochs for training
            imgsz=640,            # Image size
            batch=batch_size,     # Auto batch size based on VRAM
            patience=10,          # Early stopping patience
            save=True,           # Save model
            device=device        # Use GPU if available, else CPU
        )

        # Get the best model path
        model_dir = "runs/detect"
        if os.path.exists(model_dir):
            train_dirs = [
                os.path.join(
                    model_dir,
                    d) for d in os.listdir(model_dir) if os.path.isdir(
                    os.path.join(
                        model_dir,
                        d)) and d.startswith('train')]

            if not train_dirs:
                logger.error("No training directories found in runs/detect")
                return None

            latest_train_dir = max(train_dirs, key=os.path.getmtime)
            model_path = os.path.join(latest_train_dir, "weights", "best.pt")

            if os.path.exists(model_path):
                logger.info(f"Model trained and saved at {model_path}")
                return model_path
            else:
                logger.error(f"Model file not found at {model_path}")
                # Try to find if last.pt exists as fallback
                last_path = os.path.join(
                    latest_train_dir, "weights", "last.pt")
                if os.path.exists(last_path):
                    logger.info(
                        f"best.pt not found, using last.pt at {last_path}")
                    return last_path
                return None
        else:
            logger.error(
                "Training directory not found at runs/detect. Training may have failed.")
            return None

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None
