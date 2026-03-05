from src.train_model import train_model
import os
import pytest


def test_train_model():
    """Test the train_model function with sample data."""
    # Create test data.yaml
    test_yaml = "test_data.yaml"
    with open(test_yaml, "w") as f:
        f.write("""
path: dataset
train: train/images
val: train/images
names:
  0: object
        """)

    try:
        train_model(test_yaml)
    except Exception as e:
        assert False, f"train_model raised an exception: {e}"
    finally:
        if os.path.exists(test_yaml):
            os.remove(test_yaml)
