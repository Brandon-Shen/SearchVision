# src/tests/test_scrape_similar.py

from src.scrape_similar import scrape_similar_images


def test_scrape_similar_images(monkeypatch):
    monkeypatch.setattr(
        "src.scrape_similar.search_images",
        lambda *args, **kwargs: [{"url": "https://example.com/image.jpg"}])
    results = scrape_similar_images(
        ["http://example.com/image1.jpg"], "cat", None, None)
    assert isinstance(results, list)
    assert results == ["https://example.com/image.jpg"]
    assert all(isinstance(url, str) for url in results)
