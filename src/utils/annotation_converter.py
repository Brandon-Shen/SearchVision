import os
import json
from PIL import Image

def convert_to_yolo_format(json_annotation, img_width, img_height):
    """
    Convert JSON bbox annotation to YOLO format (normalized coordinates).
    
    Args:
        json_annotation: JSON string with x, y, width, height
        img_width: Width of the image
        img_height: Height of the image
        
    Returns:
        YOLO format string: "<class> <x_center> <y_center> <width> <height>"
    """
    # Parse JSON string to dict
    bbox = json.loads(json_annotation)
    
    # YOLO uses center coordinates and normalized dimensions
    x_center = (bbox['x'] + bbox['width'] / 2) / img_width
    y_center = (bbox['y'] + bbox['height'] / 2) / img_height
    width = abs(bbox['width'] / img_width)  # Use abs to handle negative width from right-to-left drawing
    height = abs(bbox['height'] / img_height)  # Use abs to handle negative height from bottom-to-top drawing
    
    # Clamp values between 0 and 1
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    # Return YOLO format string (class 0)
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)