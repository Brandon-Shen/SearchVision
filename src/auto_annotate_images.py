from ultralytics import YOLO
import os
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def _target_names(target_class):
    """Map common user phrases to COCO class names without broad substring matching."""
    normalized = " ".join((target_class or "").lower().replace("-", " ").split())
    aliases = {
        "coffee cup": "cup", "mug": "cup", "automobile": "car",
        "vehicle": "car", "bike": "bicycle", "motorbike": "motorcycle",
        "sofa": "couch", "aeroplane": "airplane", "television": "tv",
    }
    names = {normalized, aliases.get(normalized, normalized)}
    if normalized.endswith("s"):
        names.add(normalized[:-1])
    return names


def auto_annotate_images(image_folder, labels_folder, target_class=None, confidence=0.35):
    """
    Auto-annotate images using YOLOv8 and save annotations in YOLO format.

    Args:
        image_folder: Directory containing images
        labels_folder: Directory where annotation files will be saved
    """
    # Load the pre-trained YOLOv8 model
    annotation_model = os.getenv(
        'YOLO_ANNOTATION_MODEL', os.getenv('YOLO_MODEL', 'yolov8m.pt'))
    model = YOLO(annotation_model)

    os.makedirs(labels_folder, exist_ok=True)

    # Track statistics
    processed_count = 0
    annotated_count = 0
    error_count = 0

    allowed_names = _target_names(target_class) if target_class else None
    model_names = {str(name).lower() for name in model.names.values()}
    if allowed_names and not (allowed_names & model_names):
        logger.warning(
            "'%s' is not a class known by the pretrained annotator; "
            "scraped images will not be given potentially false labels.",
            target_class)
        return 0, 0, 0

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
                label_filename = os.path.splitext(image_file)[0] + ".txt"
                label_path = os.path.join(labels_folder, label_filename)
                if os.path.exists(label_path):
                    # Human annotations are ground truth and must never be replaced.
                    continue

                results = model(image_path, conf=confidence, verbose=False)
                processed_count += 1

                # Create annotation filename
                # Process the results
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        # Extract bounding boxes
                        boxes = result.boxes.xyxy.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy().astype(int)

                        lines = []
                        for box, class_id in zip(boxes, classes):
                            detected_name = str(result.names[class_id]).lower()
                            if allowed_names and detected_name not in allowed_names:
                                continue
                            if len(box) >= 4:
                                x_min, y_min, x_max, y_max = box[:4]
                                x_center = ((x_min + x_max) / 2) / img_width
                                y_center = ((y_min + y_max) / 2) / img_height
                                width = (x_max - x_min) / img_width
                                height = (y_max - y_min) / img_height
                                lines.append(
                                    f"0 {x_center:.6f} {y_center:.6f} "
                                    f"{width:.6f} {height:.6f}\n")

                        # Do not turn uncertain target misses into false-negative
                        # training examples. Unlabeled scraped images are excluded
                        # when the dataset split is assembled.
                        if lines:
                            with open(label_path, 'w') as f:
                                f.writelines(lines)
                            annotated_count += len(lines)
                            logger.info(
                                f"Saved YOLO annotations for {image_file}")
                        else:
                            logger.warning(
                                f"No target objects detected in {image_file}")
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
