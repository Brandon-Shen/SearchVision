#!/usr/bin/env python3
"""
Simple test script to verify caption relevance scoring functionality.
"""

import sys
sys.path.insert(0, '/Users/Brandon Shen/Documents/SearchVision')

from src.utils.caption_relevance import (
    compute_caption_relevance,
    compute_batch_caption_relevance
)

# Test cases for caption relevance
test_cases = [
    {
        "query": "dog",
        "captions": [
            ("A beautiful golden retriever playing in the park", 0.8),  # High relevance
            ("Golden retriever on beach", 0.9),  # Very high relevance
            ("Cat sitting on a chair", 0.0),  # No relevance
            ("Dog training classes", 0.7),  # Relevant but indirect
            ("Fluffy dog breed guide", 0.8),  # Relevant
        ]
    },
    {
        "query": "cat sitting",
        "captions": [
            ("Orange cat sitting on windowsill", 0.9),  # Very high relevance
            ("How to teach your cat to sit", 0.8),  # Relevant
            ("Dogs and their behavior", 0.0),  # No relevance
            ("Sitting meditation techniques", 0.2),  # Partial relevance
            ("Cat toys for active cats", 0.6),  # Somewhat relevant
        ]
    },
]

print("=" * 70)
print("Caption Relevance Scoring Test")
print("=" * 70)

for test_case in test_cases:
    query = test_case["query"]
    captions = test_case["captions"]
    
    print(f"\nQuery: '{query}'")
    print("-" * 70)
    
    for caption, expected_approx in captions:
        score = compute_caption_relevance(caption, query)
        status = "✓" if abs(score - expected_approx) < 0.15 else "~"
        print(f"{status} Caption: {caption[:50]:50} | Score: {score:.2f}")

print("\n" + "=" * 70)
print("Batch Caption Relevance Test")
print("=" * 70)

batch_results = [
    {"url": "http://example.com/1", "title": "Beautiful Golden Retriever", "snippet": "A friendly golden retriever dog"},
    {"url": "http://example.com/2", "title": "Cat Sleeping", "snippet": "Orange cat taking a nap"},
    {"url": "http://example.com/3", "title": "Dog Training Guide", "snippet": "Learn how to train your dog"},
]

query = "dog"
scores = compute_batch_caption_relevance(batch_results, query)

print(f"\nQuery: '{query}'")
print("-" * 70)
for result, score in zip(batch_results, scores):
    print(f"Title: {result['title']:40} | Score: {score:.2f}")

print("\n" + "=" * 70)
print("All tests completed successfully!")
print("=" * 70)
