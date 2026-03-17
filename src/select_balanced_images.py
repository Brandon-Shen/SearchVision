"""
Balanced image selection combining search relevance with visual dissimilarity.

Strategy:
1. Images are initially ranked by search engine (relevance score based on position)
2. Extract visual features from all images using ResNet50
3. Select images that balance:
   - High relevance (early in search results)
   - Visual dissimilarity (diverse appearance)

This ensures training data is both relevant to the search query and diverse in appearance.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from PIL import Image
from torchvision import models, transforms
import torch
import logging

logger = logging.getLogger(__name__)

# Load a pre-trained ResNet50 model for feature extraction
model = models.resnet50(weights='IMAGENET1K_V1')
model = model.eval()  # Set the model to evaluation mode

# Remove the final classification layer to extract 2048-dim features from
# avgpool
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor = feature_extractor.eval()

# Transformation for input images (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_features(image_path):
    """
    Extracts features from an image using a pre-trained ResNet50 model.

    Args:
        image_path: Path to the image.

    Returns:
        Feature vector of the image (2048-dim from avgpool layer), or None if failed.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = feature_extractor(image_tensor)
            features = features.flatten().numpy()
        return features
    except Exception as e:
        logger.warning(f"Error extracting features from {image_path}: {e}")
        return None


def select_balanced_images(
        image_urls,
        image_paths,
        num_images=9,
        relevance_weight=0.7):
    """
    Selects images that balance search relevance with visual dissimilarity.

    Args:
        image_urls: List of image URLs (in order of relevance from search engine)
        image_paths: List of local file paths corresponding to image_urls
        num_images: Number of images to select (default 9)
        relevance_weight: Weight for relevance score (0-1). Dissimilarity weight = 1 - relevance_weight
                         Default 0.7 means 70% relevance, 30% dissimilarity

    Returns:
        List of selected image URLs, balanced between relevance and dissimilarity
    """

    if len(image_urls) < num_images:
        logger.warning(
            f"Requested {num_images} images but only {len(image_urls)} available")
        return image_urls

    # Extract features from all images
    features_list = []
    valid_indices = []

    for idx, path in enumerate(image_paths):
        feature = extract_features(path)
        if feature is not None:
            features_list.append(feature)
            valid_indices.append(idx)
        else:
            logger.debug(f"Skipping image {idx} - could not extract features")

    if len(features_list) < num_images:
        logger.warning(
            f"Only {len(features_list)} images have valid features, returning top {min(len(image_urls), num_images)}")
        return image_urls[:min(len(image_urls), num_images)]

    features = np.array(features_list)

    # Calculate dissimilarity scores based on visual features
    # Compute cosine distance matrix between image features
    distance_matrix = cosine_distances(features)

    # Calculate dissimilarity score for each image (sum of distances to all
    # others)
    dissimilarity_scores = np.sum(distance_matrix, axis=1)

    # Normalize both scores to 0-1 range
    dissimilarity_weight = 1 - relevance_weight

    # Relevance score: images earlier in search results have higher relevance
    # Map position (0 to len-1) to relevance (1.0 to 0.0)
    relevance_scores = 1.0 - \
        np.arange(len(features_list)) / max(1, len(features_list) - 1)

    # Normalize dissimilarity scores to 0-1 range
    if dissimilarity_scores.max() > dissimilarity_scores.min():
        dissimilarity_scores_norm = (
            dissimilarity_scores - dissimilarity_scores.min()) / (
            dissimilarity_scores.max() - dissimilarity_scores.min())
    else:
        dissimilarity_scores_norm = dissimilarity_scores

    # Combined score: weighted combination of relevance and dissimilarity
    combined_scores = (relevance_weight * relevance_scores +
                       dissimilarity_weight * dissimilarity_scores_norm)

    # Select top num_images indices by combined score
    selected_feature_indices = np.argsort(combined_scores)[-num_images:][::-1]

    # Map back to original image indices
    selected_indices = [valid_indices[idx] for idx in selected_feature_indices]

    # Return selected image URLs
    selected_images = [image_urls[idx] for idx in selected_indices]

    logger.info(
        f"Selected {len(selected_images)} images using balanced strategy "
        f"(relevance_weight={relevance_weight}, dissimilarity_weight={dissimilarity_weight})")

    return selected_images
