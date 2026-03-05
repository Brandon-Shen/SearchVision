import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from PIL import Image
from torchvision import models, transforms
import torch
from src.download_images import download_images

# Load a pre-trained model (e.g., ResNet) for feature extraction
# Updated to use the correct 'weights' parameter
model = models.resnet50(weights='IMAGENET1K_V1')
model = model.eval()  # Set the model to evaluation mode

# Transformation for input images (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_features(image_path):
    """
    Extracts features from an image using a pre-trained model.

    :param image_path: Path to the image.
    :return: Feature vector of the image.
    """
    try:
        image = Image.open(image_path).convert(
            'RGB')  # Ensure the image is in RGB format
        image_tensor = transform(image).unsqueeze(
            0)  # Transform and add batch dimension
        with torch.no_grad():
            # Use the model to extract features
            # Convert tensor to numpy array and flatten
            features = model(image_tensor).flatten().numpy()
        return features
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None


def select_most_dissimilar_images(image_urls, num_images):
    """
    Selects the most dissimilar images from a list.

    :param image_urls: List of image URLs.
    :param num_images: Number of most dissimilar images to select.
    :return: List of the most dissimilar image URLs.
    """
    # Download images to a local directory
    # Temporary directory to save downloaded images
    download_path = "dataset/train/images"
    image_paths = download_images(image_urls, download_path=download_path)

    # Validate that images were downloaded
    if not image_paths:
        print("No images were downloaded.")
        return []

    # Extract features from each downloaded image
    features = []
    valid_image_paths = []
    for path in image_paths:
        feature = extract_features(path)
        if feature is not None:
            features.append(feature)
            valid_image_paths.append(path)

    # Ensure there are enough features to proceed
    if len(features) < num_images:
        print("Not enough images to select the most dissimilar ones.")
        return image_urls[:len(features)]  # Return as many images as available

    # Convert features list to a numpy array
    features = np.array(features)

    # Compute the cosine distance matrix between image features
    distance_matrix = cosine_distances(features)

    # Sum the distances for each image and sort them in descending order
    dissimilarity_scores = np.sum(distance_matrix, axis=1)
    most_dissimilar_indices = np.argsort(dissimilarity_scores)[-num_images:]

    # Select the most dissimilar image URLs
    most_dissimilar_images = [image_urls[idx]
                              for idx in most_dissimilar_indices]

    return most_dissimilar_images
