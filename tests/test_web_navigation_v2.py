import socket
import pytest
import json
from unittest.mock import patch, MagicMock
from magda_agent.skills.web_navigation_v2 import load_url, click_element, type_text, scroll, submit_form, web_navigate_v2

@patch('urllib.request.urlopen')

@patch('socket.getaddrinfo')
def test_load_url_success(mock_getaddrinfo, mock_urlopen):
    mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))]
    mock_response = MagicMock()
    mock_response.read.return_value = b"<html><body><form id='login-form'></form><h1 id='header'>Hello World</h1></body></html>"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    result_json = load_url("https://example.com")
    result = json.loads(result_json)
    assert result["status"] == "success"
    assert result["url"] == "https://example.com"
    assert "Hello World" in result["text"]
    assert any(e["id"] == "header" for e in result["elements"])
    assert "login-form" in result["forms"]

@patch('urllib.request.urlopen')

@patch('socket.getaddrinfo')
def test_load_url_failure(mock_getaddrinfo, mock_urlopen):
    mock_getaddrinfo.return_value = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))]
    mock_urlopen.side_effect = Exception("Network error")
    result_json = load_url("https://example.com")
    result = json.loads(result_json)
    assert result["status"] == "error"
    assert "Network error" in result["message"]

def test_click_element():
    result = json.loads(click_element("btn-1"))
    assert result["status"] == "success"
    assert result["action"] == "click"

def test_type_text():
    result = json.loads(type_text("input-1", "test"))
    assert result["status"] == "success"
    assert result["action"] == "type"

def test_scroll():
    result = json.loads(scroll("down"))
    assert result["status"] == "success"
    assert result["action"] == "scroll"

def test_submit_form():
    result = json.loads(submit_form("form-1"))
    assert result["status"] == "success"
    assert result["action"] == "submit_form"

@patch('magda_agent.skills.web_navigation_v2.load_url')
def test_web_navigate_v2_dispatch(mock_load):
    mock_load.return_value = '{"status": "success"}'
    res = web_navigate_v2("load", url="https://test.com")
    assert json.loads(res)["status"] == "success"

def test_web_navigate_v2_errors():
    assert "required" in web_navigate_v2("load")
    assert "required" in web_navigate_v2("click")
    assert "required" in web_navigate_v2("type", element_id="1")
    assert "required" in web_navigate_v2("submit")
    assert "Unknown" in web_navigate_v2("invalid_action")
