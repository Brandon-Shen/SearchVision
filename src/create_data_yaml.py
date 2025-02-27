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
    # Base dataset directory
    dataset_dir = os.path.dirname(os.path.dirname(annotations_path))
    
    # Dataset structure expected by YOLOv8
    data = {
        'path': dataset_dir,
        'train': 'train/images',
        'val': 'train/images',  # Using same images for validation
        'names': {
            0: object_name  # Single class detection
        }
    }
    
    # Create the YAML file
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path