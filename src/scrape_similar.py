from src.search_images import search_images
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def scrape_similar_images(
        selected_image_urls,
        original_query,
        api_key,
        search_engine_id,
        num_results_per_image=10,
        total_images_to_download=50):
    """
    Scrape similar images for training augmentation.
    Uses multiple query variations to find diverse training images.
    Falls back gracefully if search fails.

    Returns:
        List of image URLs (strips metadata for compatibility)
    """
    similar_images = []

    # Query variations to try - gradually simpler if advanced searches fail
    selected_domains = []
    for url in selected_image_urls:
        domain = urlparse(url).netloc.lower().removeprefix("www.")
        if domain and domain not in selected_domains:
            selected_domains.append(domain)

    # Searching domains chosen by the user makes expansion reflect their
    # selections as well as the original text query.
    query_variations = [
        *(f"{original_query} site:{domain}" for domain in selected_domains[:3]),
        f"{original_query} filetype:jpg OR filetype:png",  # Specific file types
        f"{original_query} clear photo",
        # Descriptive quality
        f"{original_query} high resolution",
        f"{original_query} isolated",
        f"{original_query} product photo",
        f"{original_query} professional photo",
        f"{original_query} detailed",
        f"{original_query} close-up",
        original_query,                                     # Fallback: plain query
    ]

    # Try each query variation
    for query_idx, query in enumerate(query_variations):
        if len(similar_images) >= total_images_to_download:
            logger.info(f"Reached target of {total_images_to_download} images")
            break

        try:
            logger.debug(f"Attempting search with query: {query}")

            results = search_images(
                query,
                api_key,
                search_engine_id,
                num_results=num_results_per_image
            )

            if results:
                # Extract URLs from metadata dicts
                urls = [r['url'] for r in results]
                logger.info(f"Got {len(urls)} images from query: {query}")
                similar_images.extend(urls)
            else:
                logger.debug(f"No images from query: {query}")

        except Exception as e:
            logger.warning(f"Failed to search for '{query}': {str(e)[:100]}")
            # Continue to next query variation
            continue

    # Remove duplicates while preserving order
    selected_set = set(selected_image_urls)
    similar_images = [
        url for url in dict.fromkeys(similar_images) if url not in selected_set]

    final_count = min(len(similar_images), total_images_to_download)
    logger.info(
        f"Scrape similar images: collected {final_count}/{total_images_to_download} images after removing duplicates")

    return similar_images[:total_images_to_download]
