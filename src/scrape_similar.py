from src.search_images import search_images


def scrape_similar_images(
        selected_image_urls,
        original_query,
        api_key,
        search_engine_id,
        num_results_per_image=10,
        total_images_to_download=50):
    similar_images = []

    # Enhanced fallback queries for better results
    fallback_queries = [
        f"{original_query} clear photo",
        f"{original_query} high resolution",
        f"{original_query} isolated",
        f"{original_query} product photo",
        f"{original_query} professional photo"
    ]

    # Add specific filters to the query
    base_query = f"{original_query} filetype:jpg OR filetype:png"
    images = search_images(
        base_query,
        api_key,
        search_engine_id,
        num_results=num_results_per_image)
    similar_images.extend(images)

    # Use fallback queries only if needed
    for fallback_query in fallback_queries:
        if len(similar_images) >= total_images_to_download:
            break

        fallback_images = search_images(
            f"{fallback_query} filetype:jpg OR filetype:png",
            api_key,
            search_engine_id,
            num_results=num_results_per_image)
        similar_images.extend(fallback_images)

    # Remove duplicates while preserving order
    similar_images = list(dict.fromkeys(similar_images))

    return similar_images[:total_images_to_download]
