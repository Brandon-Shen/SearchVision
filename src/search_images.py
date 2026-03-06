import requests
import json


def search_images(query, api_key, search_engine_id, num_results=10):
    images = []
    # Google Custom Search allows a maximum of 10 results per page
    results_per_page = 10
    start_index = 1

    while len(images) < num_results:
        # Adjust the start index for pagination
        search_url = (
            f"https://www.googleapis.com/customsearch/v1?"
            f"q={query}&searchType=image&key={api_key}&cx={search_engine_id}"
            f"&start={start_index}&num={min(results_per_page, num_results - len(images))}")

        response = requests.get(search_url)
        if response.status_code != 200:
            error_message = _parse_api_error(response)
            raise Exception(error_message)

        data = response.json()
        if 'items' not in data:
            break  # No more results

        for item in data['items']:
            images.append(item['link'])  # Get the image URL

        # Increment the start index for the next batch of results
        start_index += results_per_page

        if len(data['items']) < results_per_page:
            break  # No more results available

    return images


def _parse_api_error(response):
    """Parse Google API error response and return a user-friendly message"""
    try:
        data = response.json()
        if 'error' in data:
            error_obj = data['error']
            
            # Handle different error formats
            if isinstance(error_obj, dict):
                message = error_obj.get('message', 'Unknown error')
                code = error_obj.get('code', response.status_code)
                status = error_obj.get('status', 'UNKNOWN')
                
                # Map common errors to user-friendly messages
                if status == 'PERMISSION_DENIED' or code == 403:
                    return f"Access Denied: {message} - Please check your Google API credentials and ensure the Custom Search JSON API is enabled in your Google Cloud project."
                elif status == 'INVALID_ARGUMENT' or code == 400:
                    return f"Invalid Request: {message} - Please verify your search query and API configuration."
                elif status == 'UNAUTHENTICATED' or code == 401:
                    return f"Authentication Failed: {message} - Your API key may be invalid or expired."
                elif status == 'RESOURCE_EXHAUSTED' or code == 429:
                    return f"Rate Limited: {message} - You've exceeded your daily search quota. Please try again later."
                else:
                    return f"API Error ({code}): {message}"
            else:
                return f"API Error: {str(error_obj)}"
    except:
        pass
    
    # Fallback error message
    return f"Failed to fetch images: Status code {response.status_code}. The image search service returned an error. Please verify your API keys and search query."
