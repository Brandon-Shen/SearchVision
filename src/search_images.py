import requests
import json
import logging
import re
import time

logger = logging.getLogger(__name__)


def search_images(query, api_key, search_engine_id, num_results=10):
    """
    Search for images using Google Custom Search API.
    Falls back to Bing Images if Google fails (no API key needed).
    """
    images = []
    google_error = None
    
    # Try Google Custom Search first
    try:
        images = _search_google_custom_search(query, api_key, search_engine_id, num_results)
        if images:
            logger.info(f"Successfully retrieved {len(images)} images from Google Custom Search")
            return images
    except Exception as e:
        google_error = str(e)
        logger.warning(f"Google Custom Search failed: {google_error}")
    
    # Fallback to Bing Images (free, no API key needed)
    try:
        logger.info("Falling back to Bing Images for search")
        images = _search_bing_images(query, num_results)
        if images:
            logger.info(f"Successfully retrieved {len(images)} images from Bing Images")
            return images
    except Exception as e:
        logger.error(f"Bing Images fallback also failed: {str(e)}")
    
    # If both fail, raise an error with helpful message
    if google_error:
        raise Exception(
            f"Unable to search for images. Google API error: {google_error}\n\n"
            f"The app attempted to use a fallback image source (Bing Images) but it also failed. "
            f"Please check your internet connection and try again."
        )
    else:
        raise Exception("No image search service is available")


def _search_google_custom_search(query, api_key, search_engine_id, num_results=10):
    """Search using Google Custom Search API"""
    images = []
    results_per_page = 10
    start_index = 1

    while len(images) < num_results:
        search_url = (
            f"https://www.googleapis.com/customsearch/v1?"
            f"q={query}&searchType=image&key={api_key}&cx={search_engine_id}"
            f"&start={start_index}&num={min(results_per_page, num_results - len(images))}")

        try:
            response = requests.get(search_url, timeout=10)
            
            logger.debug(f"Google API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                error_message = _parse_google_api_error(response)
                raise Exception(error_message)

            data = response.json()
            if 'items' not in data:
                break

            for item in data['items']:
                images.append(item['link'])

            start_index += results_per_page

            if len(data['items']) < results_per_page:
                break
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error while searching: {str(e)}")

    return images


def _search_bing_images(query, num_results=10):
    """
    Search using Bing Images (free, no API key required)
    Scrapes image URLs from Bing image search with retry logic.
    Strips problematic filter syntax before searching.
    """
    images = []
    max_retries = 3
    retry_count = 0
    
    # Clean up query: remove Google-style filters that Bing doesn't understand
    clean_query = query
    clean_query = clean_query.replace(" filetype:jpg", "").replace(" filetype:png", "")
    clean_query = clean_query.replace(" OR ", " ")  # Replace OR with space
    clean_query = clean_query.strip()
    
    logger.debug(f"Bing Images search query cleaned: '{query}' -> '{clean_query}'")
    
    while retry_count < max_retries:
        try:
            # Bing Images search URL
            search_url = "https://www.bing.com/images/search"
            
            params = {
                "q": clean_query,
                "count": min(num_results, 35),
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(search_url, params=params, headers=headers, timeout=10)
            logger.debug(f"Bing Images Response Status: {response.status_code}")
            
            if response.status_code != 200:
                raise Exception(f"Bing Images returned status {response.status_code}")
            
            # Extract image URLs from the HTML response using regex
            # Bing stores lazy-loaded images in data-src attributes
            # These are Bing image proxy URLs (tse1.mm.bing.net, etc.)
            image_pattern = r'<img[^>]+data-src="([^"]+)"'
            matches = re.findall(image_pattern, response.text)
            
            if not matches:
                logger.debug(f"No images found on Bing Images (attempt {retry_count + 1}/{max_retries})")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)  # Wait before retrying
                    continue
                raise Exception("No images found on Bing Images after retries")
            
            # Process URLs and decode HTML entities
            for url in matches:
                if url.startswith('http') and len(images) < num_results:
                    # Decode HTML entities (e.g., &amp; to &)
                    url = url.replace('&amp;', '&')
                    url = url.replace('\\/', '/')
                    images.append(url)
            
            if not images:
                logger.debug(f"No valid image URLs found (attempt {retry_count + 1}/{max_retries})")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)
                    continue
                raise Exception("No valid image URLs found after retries")
            
            logger.info(f"Bing Images search returned {len(images)} images for query: {clean_query}")
            return images[:num_results]
            
        except Exception as e:
            if retry_count < max_retries - 1:
                logger.debug(f"Bing Images error (attempt {retry_count + 1}/{max_retries}): {str(e)}")
                retry_count += 1
                time.sleep(1)
            else:
                logger.error(f"Bing Images error after {max_retries} attempts: {str(e)}")
                raise


def _parse_google_api_error(response):
    """Parse Google API error response and return a user-friendly message"""
    try:
        data = response.json()
        if 'error' in data:
            error_obj = data['error']
            
            if isinstance(error_obj, dict):
                message = error_obj.get('message', 'Unknown error')
                code = error_obj.get('code', response.status_code)
                status = error_obj.get('status', 'UNKNOWN')
                
                if status == 'PERMISSION_DENIED' or code == 403:
                    return (
                        f"Google Custom Search API Access Denied (403): {message}\n\n"
                        f"This usually means:\n"
                        f"• The Custom Search JSON API is not enabled in your Google Cloud project\n"
                        f"• Your API key doesn't have the right permissions\n"
                        f"• The search engine ID (CX) is incorrect or disabled\n\n"
                        f"The app will use Bing Images as a fallback image source."
                    )
                elif status == 'INVALID_ARGUMENT' or code == 400:
                    return f"Invalid Request: {message}"
                elif status == 'UNAUTHENTICATED' or code == 401:
                    return f"Authentication Failed: {message}"
                elif status == 'RESOURCE_EXHAUSTED' or code == 429:
                    return f"Rate Limited: {message} - Daily quota exceeded"
                else:
                    return f"API Error ({code}): {message}"
            else:
                return f"API Error: {str(error_obj)}"
    except:
        pass
    
    return f"Google API failed with status {response.status_code}. Using Bing Images as fallback."
