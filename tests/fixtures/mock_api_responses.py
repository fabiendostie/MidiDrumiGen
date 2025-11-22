import pytest


@pytest.fixture
def mock_semantic_scholar_response():
    """Mock response from Semantic Scholar API."""
    return {
        "data": [
            {
                "title": "Rhythmic Analysis of John Bonham Drumming Style",
                "abstract": "Study of John Bonham typical tempo of 120 BPM with powerful groove",
                "url": "https://example.com/paper1",
                "authors": [{"name": "Smith, J."}, {"name": "Jones, A."}],
                "citationCount": 45,
                "year": 2020,
            },
            {
                "title": "Heavy Rock Drumming Techniques",
                "abstract": "Analysis of drumming at 110-140 beats per minute",
                "url": "https://example.com/paper2",
                "authors": [{"name": "Doe, J."}],
                "citationCount": 12,
                "year": 2019,
            },
        ]
    }


@pytest.fixture
def mock_arxiv_response():
    """Mock XML response from arXiv API."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2108.00001</id>
    <published>2021-08-01T00:00:00Z</published>
    <title>Drum Pattern Analysis Using Deep Learning</title>
    <summary>A study of drum patterns at 95 BPM.</summary>
    <author><name>Test Author</name></author>
  </entry>
</feed>
"""


@pytest.fixture
def mock_crossref_response():
    """Mock JSON response from CrossRef API."""
    return {
        "message": {
            "items": [
                {
                    "title": ["A Study of Rock Drumming"],
                    "URL": "https://example.com/crossref1",
                    "abstract": "Analysis of rock drumming. Tempo is often 140bpm.",
                    "author": [{"given": "Test", "family": "Author"}],
                    "is-referenced-by-count": 10,
                    "created": {"date-parts": [[2022]]},
                }
            ]
        }
    }
