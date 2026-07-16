# SearchVision

**SearchVision** is a web application built with FastAPI that allows users to search for images, select relevant images, and use web scraping techniques to find similar images. The selected images are then used to train a YOLOv8 (You Only Look Once) model for object detection.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Usage](#usage)
- [Technology Stack](#technology-stack)
- [Contributing](#contributing)
- [License](#license)

## Overview

SearchVision is designed to provide a seamless experience for users to perform image searches, select relevant images, and automatically train an object detection model using the selected images. The application leverages Google Custom Search for image retrieval and the YOLOv8 model for real-time object detection training.

## Features

- **Image Search**: Search for images using a query, powered by the Google Custom Search API.
- **Image Selection**: Select images from the search results that match the search criteria.
- **Dataset Expansion**: Search for additional images using the query and source
  domains represented in the user's selections.
- **Model Training**: Train a YOLOv8 object detection model using the selected and scraped images.
- **Error Handling**: Provides meaningful error messages and logging for smooth user experience.

<!-- ## Getting Started

 To start using SearchVision, visit the deployed web application at:

 **[https://your-deployed-domain.com](https://your-deployed-domain.com)**
 !-->

## Usage

1. **Search for Images**: Enter a search query on the home page to find images related to your query.
2. **Select Images**: Choose the images from the search results that best match your criteria.
3. **Scrape Similar Images**: The application will automatically scrape similar images.
4. **Train Model**: Use the selected and scraped images to train a YOLOv8 object detection model.
5. **View Results**: Monitor the progress and see the results of the model training.

## Model quality

SearchVision uses a real held-out validation split and logs YOLO's validation
`mAP50` and `mAP50-95` after training. A 70% mAP50-95 score is a demanding
localization target, not a guarantee:
it depends primarily on annotation quality, class difficulty, and coverage of the
target's appearances. For a credible result, annotate at least 20 diverse images
(the UI permits 5 for quick experiments) and inspect the generated labels.

Automatic labels are only created when the requested object maps to a class known
by the pretrained COCO model. For a custom object, scraped images are deliberately
left out instead of being assigned misleading boxes; provide human annotations for
those classes.

The default training backbone is `yolov8m.pt`, selected for accuracy. On the
COCO8 held-out smoke benchmark it achieved 74.0% mAP50-95, compared with 64.1%
for YOLOv8n in this environment. It requires more memory and is slower; set
`YOLO_MODEL=yolov8n.pt` to use the lightweight model instead.

A controlled single-class expansion benchmark also compares the five-image seed
workflow with 30 training images (the seed plus 25 successfully collected and
correctly labeled images) on one fixed 20-image holdout. With YOLOv8n and 20
epochs, expansion increased mAP50 from 0.258 to 0.511 and mAP50-95 from 0.203
to 0.293. Run it with `python scripts/run_expansion_benchmark.py --epochs 20`.
The benchmark uses reviewed COCO labels to isolate dataset-size value; it does
not estimate live-web scraping yield or pseudo-label noise.

## Local setup

```powershell
python -m pip install -r requirements.txt
python -m uvicorn src.main:app --reload
```

Set `GOOGLE_API_KEY` and `SEARCH_ENGINE_ID` in `.env` for Google Custom Search.
Without them the application attempts its Bing fallback. Run tests with
`python -m pytest -q`.

## Technology Stack

- **FastAPI**: A modern, fast (high-performance), web framework for building APIs with Python 3.7+.
- **YOLOv8**: An object detection model for real-time object detection and tracking.
- **Google Custom Search API**: Provides image search capabilities.
- **BeautifulSoup**: Extracts original-resolution image metadata from the Bing fallback.
- **Uvicorn**: A lightning-fast ASGI server for FastAPI.
- **Pillow (PIL)**: For handling image manipulation tasks like loading, saving, and processing images.
- **JavaScript**: Used for handling user interactions and drawing annotations on images.
- **Scikit-learn**: For calculating dissimilarities between images using cosine distance.
- **Python-dotenv**: For loading environment variables from a .env file, such as API keys.

## Contributing

We welcome contributions! If you're interested in contributing to SearchVision, please read our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to get started.
You can also give feedback [here](https://feedbackflux.vercel.app/brandonshen123@gmail.com/b/SearchVision)

## License

This project is licensed under the AGPL-3.0 License. See the [LICENSE](LICENSE) file for details.

---

### Contact

If you have any questions or feedback, please open an issue on the repository or submit it [here](https://feedbackflux.vercel.app/brandonshen123@gmail.com/b/SearchVision)

Thank you for using SearchVision! We hope you enjoy using the app.
