import os
import yaml
import shutil


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
    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(annotations_path)))
    train_images_path = os.path.join(base_dir, "train", "images")
    train_labels_path = os.path.join(base_dir, "train", "labels")
    val_images_path = os.path.join(base_dir, "val", "images")
    val_labels_path = os.path.join(base_dir, "val", "labels")
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)

    # Hold out a deterministic 20% of labeled images. Using the training images
    # as validation leaks data and produces an inflated, unusable metric.
    image_extensions = {'.jpg', '.jpeg', '.png'}
    candidates = sorted(
        filename for filename in os.listdir(train_images_path)
        if os.path.splitext(filename)[1].lower() in image_extensions
        and os.path.exists(os.path.join(
            train_labels_path, os.path.splitext(filename)[0] + '.txt'))
    )
    if len(candidates) < 5:
        raise ValueError("At least 5 labeled images are required for a train/validation split")
    val_count = max(1, round(len(candidates) * 0.2))
    # Evenly spaced selection avoids making the split depend on filename source.
    val_indices = {
        round(i * (len(candidates) - 1) / max(1, val_count - 1))
        for i in range(val_count)
    }
    for index in sorted(val_indices):
        filename = candidates[index]
        stem = os.path.splitext(filename)[0]
        shutil.move(os.path.join(train_images_path, filename),
                    os.path.join(val_images_path, filename))
        shutil.move(os.path.join(train_labels_path, stem + '.txt'),
                    os.path.join(val_labels_path, stem + '.txt'))

    # Dataset structure expected by YOLOv8 with absolute paths
    data = {
        'path': base_dir,  # Absolute base path
        'train': train_images_path,  # Absolute train images path
        'val': val_images_path,
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
