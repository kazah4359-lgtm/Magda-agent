"""
Web navigation skill.
Provides capabilities to load a URL, click on elements, and type text.
Simulates a browser interaction for the agent.
"""
import ipaddress
import logging
import socket
from typing import Any
import urllib.request
import urllib.error
import urllib.parse
from html.parser import HTMLParser



def _is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_public_http_url(url: str) -> str:
    """Validate that a URL is HTTP(S) and resolves only to public IP addresses."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve hostname '{parsed.hostname}'") from exc

    resolved_ips = {entry[4][0] for entry in addresses}
    if not resolved_ips:
        raise ValueError(f"Could not resolve hostname '{parsed.hostname}'")
    blocked = sorted(ip for ip in resolved_ips if not _is_public_ip(ip))
    if blocked:
        raise ValueError(f"URL resolves to blocked private or local address(es): {', '.join(blocked)}")
    return url

class SimpleDOMParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_content = []
        self.elements = []

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        element_id = attr_dict.get('id')
        if element_id:
            self.elements.append({"tag": tag, "id": element_id})

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.text_content.append(text)

def load_url(url: str) -> str:
    """
    Loads a URL and outputs a semantic DOM representation.

    Args:
        url (str): The URL to load.

    Returns:
        str: Semantic DOM text.
    """
    logging.info(f"Loading URL: {url}")
    try:
        validated_url = validate_public_http_url(url)
        req = urllib.request.Request(validated_url, headers={'User-Agent': 'Magda-Agent-Web-Navigator/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
            parser = SimpleDOMParser()
            parser.feed(html)
            text_repr = " ".join(parser.text_content)
            elements_repr = ", ".join([f"<{e['tag']} id='{e['id']}'>" for e in parser.elements])
            return f"URL: {url}\nText Content: {text_repr[:500]}...\nInteractive Elements: {elements_repr}"
    except Exception as e:
        return f"Error loading URL {url}: {e}"

def click_element(element_id: str) -> str:
    """
    Simulates a click action on a specific element identified by element_id.

    Args:
        element_id (str): The identifier of the element to click.

    Returns:
        str: Result of the click action.
    """
    logging.info(f"Clicking element: {element_id}")
    # Simulate a click by logging it. Real interactive DOM state requires a full headless browser.
    return f"Simulated click on element '{element_id}'. In a real browser, this would trigger associated events."

def type_text(element_id: str, text: str) -> str:
    """
    Simulates typing text into a specific element identified by element_id.

    Args:
        element_id (str): The identifier of the element.
        text (str): The text to type.

    Returns:
        str: Result of the typing action.
    """
    logging.info(f"Typing '{text}' into element: {element_id}")
    # Simulate typing.
    return f"Simulated typing '{text}' into element '{element_id}'."

def web_navigate(action: str, **kwargs: Any) -> str:
    """
    Entrypoint for web navigation. Dispatches to the corresponding action method.

    Args:
        action (str): The action to perform ('load', 'click', or 'type').
        **kwargs: Arguments for the specific action (url, element_id, text).

    Returns:
        str: The result of the action.
    """
    if action == 'load':
        url = kwargs.get('url')
        if not url:
            return "Error: 'url' is required for load action."
        return load_url(str(url))
    elif action == 'click':
        element_id = kwargs.get('element_id')
        if not element_id:
            return "Error: 'element_id' is required for click action."
        return click_element(str(element_id))
    elif action == 'type':
        element_id = kwargs.get('element_id')
        text = kwargs.get('text')
        if not element_id or text is None:
            return "Error: 'element_id' and 'text' are required for type action."
        return type_text(str(element_id), str(text))
    else:
        return f"Error: Unknown action '{action}'."
