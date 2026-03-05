import os
import json
from PIL import Image

def convert_to_yolo_format(json_annotation, img_width, img_height):
    """
    Convert JSON bbox annotation to YOLO format (normalized coordinates).
    
    Args:
        json_annotation: JSON string with x, y, width, height (simple format)
                       or dict with rects array (complex format from canvas)
        img_width: Width of the image
        img_height: Height of the image
        
    Returns:
        YOLO format string: "<class> <x_center> <y_center> <width> <height>"
    """
    # Parse JSON string to dict
    data = json.loads(json_annotation)
    
    # Check if it's the new format with rects array
    if 'rects' in data and isinstance(data['rects'], list) and len(data['rects']) > 0:
        rects = data['rects']
        canvas_width = data.get('canvasWidth', img_width)
        canvas_height = data.get('canvasHeight', img_height)
        
        # Get image element info if available
        img_info = data.get('imageElement')
        if img_info:
            display_width = img_info.get('displayWidth', img_width)
            display_height = img_info.get('displayHeight', img_height)
            offset_x = img_info.get('offsetX', 0)
            offset_y = img_info.get('offsetY', 0)
        else:
            display_width = img_width
            display_height = img_height
            offset_x = 0
            offset_y = 0
        
        yolo_lines = []
        for rect in rects:
            x = rect['x']
            y = rect['y']
            width = rect['width']
            height = rect['height']
            
            # Scale from canvas coordinates to image coordinates
            scale_x = display_width / canvas_width
            scale_y = display_height / canvas_height
            
            x_scaled = (x - offset_x) / scale_x
            y_scaled = (y - offset_y) / scale_y
            width_scaled = width / scale_x
            height_scaled = height / scale_y
            
            # Convert to YOLO format (normalized)
            x_center = (x_scaled + width_scaled / 2) / img_width
            y_center = (y_scaled + height_scaled / 2) / img_height
            width_norm = abs(width_scaled / img_width)
            height_norm = abs(height_scaled / img_height)
            
            # Clamp values between 0 and 1
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))
            
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}")
        
        return "\n".join(yolo_lines)
    
    # Legacy format: simple x, y, width, height
    bbox = data
    
    # YOLO uses center coordinates and normalized dimensions
    x_center = (bbox['x'] + bbox['width'] / 2) / img_width
    y_center = (bbox['y'] + bbox['height'] / 2) / img_height
    width = abs(bbox['width'] / img_width)
    height = abs(bbox['height'] / img_height)
    
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