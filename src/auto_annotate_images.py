from ultralytics import YOLO
import os
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def auto_annotate_images(image_folder, labels_folder):
    """
    Auto-annotate images using YOLOv8 and save annotations in YOLO format.

    Args:
        image_folder: Directory containing images
        labels_folder: Directory where annotation files will be saved
    """
    # Load the pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')

    os.makedirs(labels_folder, exist_ok=True)

    # Track statistics
    processed_count = 0
    annotated_count = 0
    error_count = 0

    # Loop through images in the specified folder
    for image_file in os.listdir(image_folder):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, image_file)

            # Check if the image can be opened and processed
            try:
                # Get image dimensions
                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                # Run detection
                results = model(image_path)
                processed_count += 1

                # Create annotation filename
                label_filename = os.path.splitext(image_file)[0] + ".txt"
                label_path = os.path.join(labels_folder, label_filename)

                # Process the results
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        # Extract bounding boxes
                        boxes = result.boxes.xyxy.cpu().numpy()

                        with open(label_path, 'w') as f:
                            for box in boxes:
                                if len(box) >= 4:
                                    x_min, y_min, x_max, y_max = box[:4]

                                    # Calculate YOLO format (normalized)
                                    x_center = (
                                        (x_min + x_max) / 2) / img_width
                                    y_center = (
                                        (y_min + y_max) / 2) / img_height
                                    width = (x_max - x_min) / img_width
                                    height = (y_max - y_min) / img_height

                                    # Class 0 for all objects (single class)
                                    f.write(
                                        f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                                    annotated_count += 1

                        logger.info(f"Saved YOLO annotations for {image_file}")
                    else:
                        logger.warning(f"No objects detected in {image_file}")

            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                error_count += 1
                continue

    logger.info(
        f"Auto-annotation complete: processed {processed_count} images, "
        f"created {annotated_count} annotations, {error_count} errors")

    return processed_count, annotated_count, error_count
