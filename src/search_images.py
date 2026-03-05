import requests


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
            raise Exception(
                f"Failed to fetch images: Status code {response.status_code}, Response: {response.text}")

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
