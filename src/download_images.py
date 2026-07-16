# src/download_images.py

import requests
import os
from io import BytesIO
from PIL import Image


def download_images(image_urls, download_path="dataset/train/images", filename_prefix="image"):
    """
    Downloads images from a list of URLs and saves them to the specified directory.

    Maintains index alignment by returning tuples of (original_index, file_path)
    so that ranking algorithms can correctly map back to original URLs.

    :param image_urls: List of image URLs to download.
    :param download_path: Directory to save downloaded images.
    :return: List of tuples (original_index, file_path) for successfully downloaded images,
             preserving which position in the input list each downloaded image came from.
    """
    print("Starting image download...")  # Debugging statement

    # Ensure the download directory exists
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # List to hold (original_index, file_path) tuples
    # This preserves alignment between downloaded images and input URLs
    downloaded_paths = []

    # Iterate over the image URLs and download each image
    for idx, url in enumerate(image_urls):
        print(f"Attempting to download ({idx}/{len(image_urls)}): {url}")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                response.raise_for_status()
                # Reject HTML/error pages returned with a 200 response and
                # normalize all inputs to a format YOLO can reliably open.
                with Image.open(BytesIO(response.content)) as image:
                    image = image.convert("RGB")
                    image.load()
                file_path = os.path.join(download_path, f"{filename_prefix}_{idx}.jpg")
                with open(file_path, "wb") as f:
                    image.save(f, format="JPEG", quality=95)
                print(f"Downloaded: {file_path}")
                # Store both the original index and the file path
                downloaded_paths.append((idx, file_path))
            else:
                print(f"Failed to download {url}: status {response.status_code}")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

    # Check if any images were downloaded
    if not downloaded_paths:
        print("No images were downloaded.")

    return downloaded_paths

