import pytest
from unittest.mock import patch, MagicMock
from magda_agent.skills.web_navigation import load_url, click_element, type_text, web_navigate

@patch('urllib.request.urlopen')
def test_load_url_success(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = b"<html><body><h1 id='header'>Hello World</h1><p id='para'>Test content.</p></body></html>"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    result = load_url("https://example.com")
    assert "URL: https://example.com" in result
    assert "Text Content: Hello World Test content." in result
    assert "<h1 id='header'>" in result
    assert "<p id='para'>" in result

@patch('urllib.request.urlopen')
def test_load_url_failure(mock_urlopen):
    mock_urlopen.side_effect = Exception("Network error")
    result = load_url("https://example.com")
    assert "Error loading URL https://example.com: Network error" in result

def test_click_element():
    result = click_element("submit-button")
    assert "Simulated click on element 'submit-button'" in result

def test_type_text():
    result = type_text("search-box", "hello world")
    assert "Simulated typing 'hello world' into element 'search-box'" in result

@patch('magda_agent.skills.web_navigation.load_url')
def test_web_navigate_load(mock_load):
    mock_load.return_value = "Mocked Load"
    result = web_navigate("load", url="https://test.com")
    assert result == "Mocked Load"
    mock_load.assert_called_with("https://test.com")

def test_web_navigate_load_missing_url():
    result = web_navigate("load")
    assert "Error: 'url' is required" in result

def test_web_navigate_click():
    result = web_navigate("click", element_id="btn-1")
    assert "Simulated click" in result

def test_web_navigate_click_missing_element():
    result = web_navigate("click")
    assert "Error: 'element_id' is required" in result

def test_web_navigate_type():
    result = web_navigate("type", element_id="input-1", text="some text")
    assert "Simulated typing" in result

def test_web_navigate_type_missing_args():
    result = web_navigate("type", element_id="input-1")
    assert "Error: 'element_id' and 'text' are required" in result

def test_web_navigate_unknown_action():
    result = web_navigate("unknown")
    assert "Error: Unknown action" in result
