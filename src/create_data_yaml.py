import os
import yaml

def create_data_yaml(annotations_path, object_name="object"):
    """
    Create a YAML configuration file for YOLOv8 training.
    
    Args:
        annotations_path: Path to annotations (used for determining dataset path)
        object_name: Name of the object class
    
    Returns:
        Path to the created YAML file
    """
    # Use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(annotations_path)))
    train_images_path = os.path.join(base_dir, "train", "images")
    train_labels_path = os.path.join(base_dir, "train", "labels")
    
    # Dataset structure expected by YOLOv8 with absolute paths
    data = {
        'path': base_dir,  # Absolute base path
        'train': train_images_path,  # Absolute train images path
        'val': train_images_path,  # Using same images for validation
        'names': {
            0: object_name  # Single class detection
        }
    }
    
    # Create the YAML file
    yaml_path = os.path.join(base_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path