from ultralytics import YOLO
import os
import logging

logger = logging.getLogger(__name__)


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

        # Train with specific parameters
        results = model.train(
            data=data_yaml_path,
            epochs=25,            # Reduced epochs for faster training
            imgsz=640,            # Image size
            batch=8,              # Batch size (reduce if memory issues)
            patience=10,          # Early stopping patience
            save=True,           # Save model
            device='cpu'         # Change to 'cuda' if GPU available
        )

        # Get the best model path
        model_dir = "runs/detect/train"
        if os.path.exists(model_dir):
            latest_train_dir = max([os.path.join(model_dir, d) for d in os.listdir(model_dir)
                                   if os.path.isdir(os.path.join(model_dir, d))],
                                   key=os.path.getmtime)
            model_path = os.path.join(latest_train_dir, "weights", "best.pt")
            logger.info(f"Model trained and saved at {model_path}")
            return model_path
        else:
            logger.error(
                "Training directory not found. Training may have failed.")
            return None

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None
