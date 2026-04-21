"""
Caption relevance scoring for image search results.

This module provides functions to compute semantic and keyword-based relevance
scores for image captions against the original search query.
"""

import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)


def compute_caption_relevance(caption, query):
    """
    Compute caption relevance score to the search query (0-1).

    This uses a multi-factor approach:
    1. Keyword overlap: Percentage of query words present in caption
    2. Keyword position: Earlier occurrences weighted more heavily
    3. Length normalization: Balanced against caption length

    Args:
        caption: Image caption or title string
        query: Original search query string

    Returns:
        float: Relevance score between 0.0 and 1.0
    """
    if not caption or not query:
        return 0.0

    # Normalize text: lowercase, remove punctuation
    caption_clean = _normalize_text(caption)
    query_clean = _normalize_text(query)

    # Split into words
    caption_words = set(caption_clean.split())
    query_words = set(query_clean.split())

    # Remove common stop words to focus on meaningful terms
    query_words = query_words - _get_stop_words()
    caption_words = caption_words - _get_stop_words()

    if not query_words:
        return 0.0

    # Calculate keyword overlap
    matching_words = query_words & caption_words
    overlap_ratio = len(matching_words) / len(query_words)

    # Calculate position-weighted score (earlier matches count more)
    position_score = _compute_position_score(
        caption_clean, query_clean, matching_words)

    # Length normalization: penalize very long captions with few matches
    length_penalty = min(1.0, 100 / len(caption_words)) if caption_words else 0.0

    # Combined score: average with length penalty
    relevance_score = (overlap_ratio * 0.5 +
                       position_score * 0.3 +
                       length_penalty * 0.2)

    return min(1.0, max(0.0, relevance_score))


def _normalize_text(text):
    """Normalize text for comparison (lowercase, remove punctuation)."""
    text = text.lower()
    # Remove punctuation and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _get_stop_words():
    """Return a set of common English stop words."""
    return {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'as',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how'
    }


def _compute_position_score(caption, query, matching_words):
    """
    Compute position-weighted score.

    Matching words that appear early in the caption score higher.
    """
    if not matching_words:
        return 0.0

    caption_words = caption.split()
    position_scores = []

    for word in matching_words:
        # Find first occurrence of the word
        for i, caption_word in enumerate(caption_words):
            if caption_word == word:
                # Earlier words (lower indices) get higher scores
                # Position 0 = 1.0, position increases = score decreases
                position_score = 1.0 / (1.0 + i / 10.0)
                position_scores.append(position_score)
                break

    return sum(position_scores) / len(
        matching_words) if position_scores else 0.0


def compute_batch_caption_relevance(results, query):
    """
    Compute caption relevance scores for a batch of search results.

    Args:
        results: List of result dicts with 'title' and 'snippet' keys
        query: Search query string

    Returns:
        List of relevance scores corresponding to input results
    """
    scores = []

    for result in results:
        # Combine title and snippet for more context
        caption = f"{result.get('title', '')} {result.get('snippet', '')}"
        score = compute_caption_relevance(caption, query)
        scores.append(score)

    return scores
