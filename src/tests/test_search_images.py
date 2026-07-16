# src/tests/test_search_images.py

from src.search_images import search_images


def test_search_images_uses_fallback_without_credentials(monkeypatch):
    expected = [{"url": "https://example.com/cat.jpg", "title": "cat", "snippet": "cat"}]
    monkeypatch.setattr("src.search_images._search_bing_images", lambda query, count: expected)
    results = search_images("cat", None, None)
    assert isinstance(results, list)
    assert results == expected
