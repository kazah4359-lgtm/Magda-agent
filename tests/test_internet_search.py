import pytest
from unittest.mock import patch, MagicMock

from magda_agent.skills.internet_search import search_internet

def test_search_internet_success():
    """Test successful internet search with mocked DDGS."""
    mock_results = [
        {'title': 'Python (programming language) - Wikipedia', 'href': 'https://en.wikipedia.org/wiki/Python_(programming_language)', 'body': 'Python is a high-level, general-purpose programming language.'},
        {'title': 'Welcome to Python.org', 'href': 'https://www.python.org/', 'body': 'The official home of the Python Programming Language'}
    ]

    with patch('magda_agent.skills.internet_search.DDGS') as MockDDGS:
        mock_instance = MockDDGS.return_value
        mock_instance.text.return_value = mock_results

        results = search_internet("python", max_results=2)

        assert len(results) == 2
        assert results[0]['title'] == 'Python (programming language) - Wikipedia'
        mock_instance.text.assert_called_once_with("python", max_results=2)

def test_search_internet_exception():
    """Test internet search handling an exception."""
    with patch('magda_agent.skills.internet_search.DDGS') as MockDDGS:
        mock_instance = MockDDGS.return_value
        mock_instance.text.side_effect = Exception("Network error")

        results = search_internet("python", max_results=2)

        assert results == []
