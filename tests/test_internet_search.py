import pytest
from unittest.mock import patch, MagicMock
from magda_agent.skills.internet_search import search_internet

def test_search_internet_success():
    mock_results = [
        {
            'title': 'Test Title 1',
            'href': 'http://test1.com',
            'body': 'Test Body 1'
        },
        {
            'title': 'Test Title 2',
            'href': 'http://test2.com',
            'body': 'Test Body 2'
        }
    ]

    with patch('magda_agent.skills.internet_search.DDGS') as mock_ddgs_class:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = mock_results

        result = search_internet("test query", max_results=2)

        expected_result = "1. Test Title 1\n   Ссылка: http://test1.com\n   Описание: Test Body 1\n\n2. Test Title 2\n   Ссылка: http://test2.com\n   Описание: Test Body 2"
        assert result == expected_result
        mock_ddgs_instance.text.assert_called_once_with("test query", max_results=2)

def test_search_internet_empty():
    with patch('magda_agent.skills.internet_search.DDGS') as mock_ddgs_class:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.return_value = []

        result = search_internet("empty query")

        assert result == "По запросу 'empty query' ничего не найдено."

def test_search_internet_error():
    with patch('magda_agent.skills.internet_search.DDGS') as mock_ddgs_class:
        mock_ddgs_instance = MagicMock()
        mock_ddgs_class.return_value = mock_ddgs_instance
        mock_ddgs_instance.text.side_effect = Exception("Network Error")

        result = search_internet("error query")

        assert result == "Произошла ошибка при выполнении поиска: Network Error"

def test_search_internet_ddgs_missing():
    with patch('magda_agent.skills.internet_search.DDGS', new=None):
        result = search_internet("test query")
        assert result == "Ошибка: библиотека duckduckgo-search не установлена. Поиск невозможен."
