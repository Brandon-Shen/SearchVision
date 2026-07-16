"""
Balanced image selection combining search relevance, caption relevance,
and visual dissimilarity.

Strategy:
1. Images are initially ranked by search engine (position-based relevance)
2. Compute caption relevance scores based on keyword matching
3. Extract visual features from all images using ResNet50
4. Select images that balance:
   - High relevance (early in search results + relevant captions)
   - Caption relevance (semantic match with query)
   - Visual dissimilarity (diverse appearance)

This ensures training data is relevant to the search query both semantically
(via captions) and visually (via ResNet50 features), while maintaining diversity.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from PIL import Image
from torchvision import models, transforms
import torch
import logging
from src.utils.caption_relevance import compute_batch_caption_relevance

logger = logging.getLogger(__name__)

feature_extractor = None


def _get_feature_extractor():
    """Load the optional ranking model on first use, not during API startup."""
    global feature_extractor
    if feature_extractor is None:
        model = models.resnet50(weights='IMAGENET1K_V1').eval()
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).eval()
    return feature_extractor

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
            features = _get_feature_extractor()(image_tensor)
            features = features.flatten().numpy()
        return features
    except Exception as e:
        logger.warning(f"Error extracting features from {image_path}: {e}")
        return None


def select_balanced_images(
        image_results,
        image_paths,
        query="",
        num_images=9,
        popularity_weight=0.6,
        caption_weight=0.25,
        dissimilarity_weight=0.15):
    """
    Selects images balancing search popularity, caption relevance,
    and visual dissimilarity.

    Strategy: Prioritize quality (popularity/search ranking) over diversity. 
    The search engine has already ranked results by relevance, so we heavily weight
    position-based popularity. Caption relevance adds semantic understanding,
    and visual dissimilarity is used as a tiebreaker for minor diversity.

    Args:
        image_results: List of result dicts with 'url', 'title', 'snippet' keys
                      OR list of URLs (for backward compatibility)
        image_paths: List of tuples (original_index, file_path) from download_images()
                    Each tuple preserves which position in image_results this came from
        query: Original search query (required for caption relevance scoring)
        num_images: Number of images to select (default 9)
        popularity_weight: Weight for search result position (default 0.6)
        caption_weight: Weight for caption relevance (default 0.25)
        dissimilarity_weight: Weight for visual dissimilarity (default 0.15)

    Returns:
        List of selected image URLs, prioritizing popularity and quality
    """

    if len(image_results) < num_images:
        logger.warning(
            f"Requested {num_images} images but only {len(image_results)} available")
        return _extract_urls(image_results)

    # Normalize weights to sum to 1.0
    total_weight = popularity_weight + caption_weight + dissimilarity_weight
    popularity_weight = popularity_weight / total_weight
    caption_weight = caption_weight / total_weight
    dissimilarity_weight = dissimilarity_weight / total_weight

    # Extract features from downloaded images, preserving original indices
    features_list = []
    original_indices = []  # Track which image_results index each feature came from

    for original_idx, file_path in image_paths:
        feature = extract_features(file_path)
        if feature is not None:
            features_list.append(feature)
            original_indices.append(original_idx)
        else:
            logger.debug(f"Skipping image from index {original_idx} - could not extract features")

    if len(features_list) < num_images:
        logger.warning(
            f"Only {len(features_list)} images have valid features, returning top {min(len(image_results), num_images)}")
        urls = _extract_urls(image_results)
        return urls[:min(len(urls), num_images)]

    features = np.array(features_list)

    # Calculate dissimilarity scores based on visual features
    # Compute cosine distance matrix between image features
    distance_matrix = cosine_distances(features)

    # Calculate dissimilarity score for each image (sum of distances to all)
    dissimilarity_scores = np.sum(distance_matrix, axis=1)

    # Popularity score: based on ORIGINAL position in search results
    # Map original_index (0 to n-1) to popularity (1.0 to 0.0)
    # Higher original_index = lower popularity, lower score
    max_original_idx = max(original_indices) if original_indices else 0
    popularity_scores = 1.0 - np.array(original_indices) / max(1, max_original_idx)

    # Caption relevance scores (if query provided)
    caption_scores = np.zeros(len(features_list))
    if query and _is_metadata_format(image_results):
        caption_scores_list = compute_batch_caption_relevance(
            [image_results[i] for i in original_indices], query)
        caption_scores = np.array(caption_scores_list)

    # Normalize all scores to 0-1 range
    if dissimilarity_scores.max() > dissimilarity_scores.min():
        dissimilarity_scores_norm = (
            dissimilarity_scores - dissimilarity_scores.min()) / (
            dissimilarity_scores.max() - dissimilarity_scores.min())
    else:
        dissimilarity_scores_norm = dissimilarity_scores

    if caption_scores.max() > caption_scores.min():
        caption_scores_norm = (
            caption_scores - caption_scores.min()) / (
            caption_scores.max() - caption_scores.min())
    else:
        caption_scores_norm = caption_scores

    # Combined score: heavily weighted toward popularity (search engine ranking),
    # with caption relevance for semantic understanding, and dissimilarity as
    # a tiebreaker for minor diversity
    combined_scores = (
        popularity_weight * popularity_scores +
        caption_weight * caption_scores_norm +
        dissimilarity_weight * dissimilarity_scores_norm
    )

    # Select top num_images indices by combined score
    selected_feature_indices = np.argsort(combined_scores)[-num_images:][::-1]

    # Map back to original image_results indices
    selected_original_indices = [original_indices[idx] for idx in selected_feature_indices]

    # Return selected image URLs
    urls = _extract_urls(image_results)
    selected_images = [urls[idx] for idx in selected_original_indices]

    logger.info(
        f"Selected {len(selected_images)} images using quality-first strategy "
        f"(popularity={popularity_weight:.2f}, caption={caption_weight:.2f}, "
        f"dissimilarity={dissimilarity_weight:.2f})")

    return selected_images


def _extract_urls(image_results):
    """
    Extract URLs from image results.

    Handles both metadata format (list of dicts) and legacy format (list of strings).
    """
    if not image_results:
        return []

    if isinstance(image_results[0], dict):
        return [r['url'] for r in image_results]
    else:
        return image_results


def _is_metadata_format(image_results):
    """Check if image_results are in metadata format (dicts) or legacy format (strings)."""
    if not image_results:
        return False
    return isinstance(image_results[0], dict)
