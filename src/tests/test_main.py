# src/tests/test_main.py

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_index():
    """Test the index page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "What would you like to detect?" in response.text


def test_search_endpoint(monkeypatch):
    """Test the search endpoint with a sample query."""
    results = [
        {"url": f"https://example.com/cat-{i}.jpg", "title": "cat", "snippet": "cat"}
        for i in range(9)
    ]
    monkeypatch.setattr("src.main.search_images", lambda *args, **kwargs: results)
    # Exercise the endpoint fallback without downloading external images.
    monkeypatch.setattr("src.main.download_images", lambda *args, **kwargs: [])
    response = client.post("/search", data={"query": "cat"})
    assert response.status_code == 200
    assert 'Select images for: "cat"' in response.text


def test_select_endpoint(monkeypatch):
    """Test the select endpoint with a sample selection."""
    paths = [(i, f"dataset/train/images/selected_{i}.jpg") for i in range(5)]
    monkeypatch.setattr("src.main.download_images", lambda *args, **kwargs: paths)
    response = client.post(
        "/select",
        data={
            "selected_images": [f"http://example.com/image{i}.jpg" for i in range(5)],
            "original_query": "cat",
        })
    assert response.status_code == 200
    assert 'Annotate: "cat"' in response.text
