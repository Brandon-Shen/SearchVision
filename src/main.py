import os
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import logging
from dotenv import load_dotenv

from src.auto_annotate_images import auto_annotate_images
from src.download_images import download_images
from src.search_images import search_images
from src.search_most_dissimilar_images import select_most_dissimilar_images
from src.train_model import train_model
from src.scrape_similar import scrape_similar_images
from PIL import Image
from src.create_data_yaml import create_data_yaml
import shutil
from src.utils.annotation_converter import convert_to_yolo_format, ensure_directory

load_dotenv()

app = FastAPI()

templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

log_file_path = os.path.join(os.getcwd(), 'app_logs.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

images_path = "dataset/train/images"
labels_path = "dataset/train/labels"
download_path = "dataset/train/images"

os.makedirs(download_path, exist_ok=True)
app.mount("/images", StaticFiles(directory=download_path), name="images")

training_status = {
    "step": 0,
    "status": "Idle",
    "detail": "",
    "completed": False,
    "success": False,
    "model_path": "",
    "error": "",
    "query": ""
}


def clear_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def reset_training_status(query):
    global training_status
    training_status = {
        "step": 0,
        "status": "Starting",
        "detail": "Initializing...",
        "completed": False,
        "success": False,
        "model_path": "",
        "error": "",
        "query": query
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    clear_directory(images_path)
    clear_directory(labels_path)
    return templates.TemplateResponse("search.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("SEARCH_ENGINE_ID")

        if not api_key or not search_engine_id:
            return templates.TemplateResponse("search.html", {
                "request": request,
                "error": "API keys not configured. Please set GOOGLE_API_KEY and SEARCH_ENGINE_ID in .env"
            })

        images = search_images(query, api_key, search_engine_id)

        if not images:
            return templates.TemplateResponse("search.html", {
                "request": request,
                "error": "No images found. Try a different search term."
            })

        selected_images = select_most_dissimilar_images(images, 9)

        return templates.TemplateResponse("select.html", {
            "request": request,
            "query": query,
            "images": selected_images
        })
    except Exception as e:
        logger.error(f"Error during image search: {e}")
        return templates.TemplateResponse("search.html", {
            "request": request,
            "error": f"Search failed: {str(e)}"
        })


@app.post("/select", response_class=HTMLResponse)
async def select(
        request: Request,
        selected_images: list[str] = Form(...),
        original_query: str = Form(...)):
    try:
        if not selected_images or len(selected_images) < 3:
            raise HTTPException(status_code=400,
                                detail="Please select at least 3 images.")

        clear_directory(images_path)
        clear_directory(labels_path)

        local_image_paths = download_images(selected_images, download_path)

        images_data = [
            (path, os.path.basename(path))
            for path in local_image_paths
        ]

        return templates.TemplateResponse("annotate.html", {
            "request": request,
            "query": original_query,
            "images": images_data
        })
    except Exception as e:
        logger.error(f"Error during image selection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/start-training", response_class=HTMLResponse)
async def start_training(
        request: Request,
        image_urls: list[str] = Form(...),
        annotations: list[str] = Form(...),
        original_query: str = Form(...)):
    try:
        reset_training_status(original_query)

        asyncio.create_task(
            run_training(
                image_urls,
                annotations,
                original_query))

        return templates.TemplateResponse("training.html", {
            "request": request,
            "query": original_query
        })
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_training(image_urls, annotations, original_query):
    global training_status
    try:
        training_status["status"] = "Downloading"
        training_status["detail"] = "Saving your annotated images"
        training_status["step"] = 0

        ensure_directory(
            labels_dir := os.path.join(
                "dataset", "train", "labels"))

        for image_url, annotation_json in zip(image_urls, annotations):
            try:
                image_name = os.path.basename(image_url)
                image_path = os.path.join(images_path, image_name)

                with Image.open(image_path) as img:
                    img_width, img_height = img.size

                yolo_annotation = convert_to_yolo_format(
                    annotation_json, img_width, img_height)

                label_filename = os.path.splitext(image_name)[0] + ".txt"
                label_path = os.path.join(labels_dir, label_filename)

                with open(label_path, 'w') as f:
                    f.write(yolo_annotation)
            except Exception as e:
                logger.error(f"Error processing annotation: {e}")

        training_status["step"] = 1
        training_status["status"] = "Scraping"
        training_status["detail"] = "Finding similar images..."

        api_key = os.getenv("GOOGLE_API_KEY")
        search_engine_id = os.getenv("SEARCH_ENGINE_ID")

        similar_images = scrape_similar_images(
            image_urls,
            original_query,
            api_key,
            search_engine_id,
            num_results_per_image=10,
            total_images_to_download=30
        )

        training_status["step"] = 2
        training_status["status"] = "Downloading"
        training_status["detail"] = f"Downloading {len(similar_images)} similar images"

        download_images(similar_images, images_path)

        training_status["step"] = 3
        training_status["status"] = "Annotating"
        training_status["detail"] = "Auto-annotating scraped images"

        auto_annotate_images(images_path, labels_dir)

        training_status["step"] = 4
        training_status["status"] = "Training"
        training_status["detail"] = "Training YOLOv8 model (this may take a few minutes)"

        data_yaml_path = create_data_yaml(labels_dir, original_query)

        model_path = train_model(data_yaml_path, 'yolov8')

        if model_path and os.path.exists(model_path):
            training_status["completed"] = True
            training_status["success"] = True
            training_status["model_path"] = model_path
            training_status["status"] = "Complete"
            training_status["detail"] = "Model trained successfully!"
        else:
            raise Exception("Model training failed - no output model found")

    except Exception as e:
        logger.error(f"Training error: {e}")
        training_status["completed"] = True
        training_status["success"] = False
        training_status["error"] = str(e)
        training_status["status"] = "Error"
        training_status["detail"] = str(e)


@app.get("/training-status")
async def get_training_status():
    return JSONResponse(training_status)


@app.get("/results", response_class=HTMLResponse)
async def results(request: Request, model: str = Query(...)):
    query = training_status.get("query", "Unknown")

    images_count = len([f for f in os.listdir(images_path)
                       if f.endswith(('.jpg', '.png'))])
    labels_count = len([f for f in os.listdir(
        labels_path) if f.endswith('.txt')])

    stats = {
        "images": images_count,
        "annotations": labels_count,
        "epochs": 25
    }

    return templates.TemplateResponse("results.html", {
        "request": request,
        "query": query,
        "model_path": model,
        "stats": stats
    })


@app.get("/error", response_class=HTMLResponse)
async def error_page(request: Request, message: str = Query(...)):
    return templates.TemplateResponse("error.html", {
        "request": request,
        "error": message
    })


@app.post("/save_annotations", response_class=HTMLResponse)
async def save_annotations(
        request: Request,
        image_urls: list[str] = Form(...),
        annotations: list[str] = Form(...),
        original_query: str = Form(...)):
    return await start_training(request, image_urls, annotations, original_query)
