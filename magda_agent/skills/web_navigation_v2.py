"""
Web navigation skill v2 (WebArena inspired).
Provides capabilities to load a URL, click on elements, type text, hover, clear, scroll, and submit forms.
"""
import ipaddress
import logging
import socket
from typing import Any
import urllib.request
import urllib.error
import urllib.parse
import json
from html.parser import HTMLParser

def _is_public_ip(address: str) -> bool:
    """Check if an IP address is public."""
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

class AdvancedDOMParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_content = []
        self.elements = []
        self.forms = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle the start tag of an HTML element."""
        attr_dict = dict(attrs)
        element_id = attr_dict.get('id')
        if element_id:
            self.elements.append({"tag": tag, "id": element_id})
        if tag == 'form':
            self.forms.append(attr_dict.get('id', 'unnamed_form'))

    def handle_data(self, data: str) -> None:
        """Handle data between HTML tags."""
        text = data.strip()
        if text:
            self.text_content.append(text)

def load_url(url: str) -> str:
    """
    Loads a URL and outputs a semantic DOM representation.
    """
    logging.info(f"Loading URL: {url}")
    try:
        validated_url = validate_public_http_url(url)
        req = urllib.request.Request(validated_url, headers={'User-Agent': 'Magda-Agent-Web-Navigator-v2/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')
            parser = AdvancedDOMParser()
            parser.feed(html)
            text_repr = " ".join(parser.text_content)
            return json.dumps({
                "status": "success",
                "url": url,
                "text": text_repr[:500] + "...",
                "elements": parser.elements,
                "forms": parser.forms,
                "viewport": "top"
            })
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Error loading URL {url}: {e}"})

def click_element(element_id: str) -> str:
    """
    Simulates a click action on a specific element identified by element_id.
    """
    logging.info(f"Clicking element: {element_id}")
    return json.dumps({"status": "success", "action": "click", "element_id": element_id})

def type_text(element_id: str, text: str) -> str:
    """
    Simulates typing text into a specific element identified by element_id.
    """
    logging.info(f"Typing '{text}' into element: {element_id}")
    return json.dumps({"status": "success", "action": "type", "element_id": element_id, "text": text})


def hover_element(element_id: str) -> str:
    """
    Simulates a hover action on a specific element identified by element_id.
    """
    logging.info(f"Hovering element: {element_id}")
    return json.dumps({"status": "success", "action": "hover", "element_id": element_id})

def clear_text(element_id: str) -> str:
    """
    Simulates clearing text from a specific element identified by element_id.
    """
    logging.info(f"Clearing text from element: {element_id}")
    return json.dumps({"status": "success", "action": "clear", "element_id": element_id})

def scroll(direction: str) -> str:
    """
    Simulates scrolling the viewport.
    """
    logging.info(f"Scrolling {direction}")
    return json.dumps({"status": "success", "action": "scroll", "direction": direction})

def submit_form(form_id: str) -> str:
    """
    Simulates submitting a form by form_id.
    """
    logging.info(f"Submitting form: {form_id}")
    return json.dumps({"status": "success", "action": "submit_form", "form_id": form_id})

def web_navigate_v2(action: str, **kwargs: Any) -> str:
    """
    Advanced entrypoint for web navigation.
    """
    if action == 'load':
        url = kwargs.get('url')
        if not url:
            return json.dumps({"status": "error", "message": "'url' is required for load action."})
        return load_url(str(url))
    elif action == 'click':
        element_id = kwargs.get('element_id')
        if not element_id:
            return json.dumps({"status": "error", "message": "'element_id' is required for click action."})
        return click_element(str(element_id))
    elif action == 'type':
        element_id = kwargs.get('element_id')
        text = kwargs.get('text')
        if not element_id or text is None:
            return json.dumps({"status": "error", "message": "'element_id' and 'text' are required for type action."})
        return type_text(str(element_id), str(text))

    elif action == 'hover':
        element_id = kwargs.get('element_id')
        if not element_id:
            return json.dumps({"status": "error", "message": "'element_id' is required for hover action."})
        return hover_element(str(element_id))
    elif action == 'clear':
        element_id = kwargs.get('element_id')
        if not element_id:
            return json.dumps({"status": "error", "message": "'element_id' is required for clear action."})
        return clear_text(str(element_id))
    elif action == 'scroll':
        direction = kwargs.get('direction', 'down')
        return scroll(str(direction))
    elif action == 'submit':
        form_id = kwargs.get('form_id')
        if not form_id:
            return json.dumps({"status": "error", "message": "'form_id' is required for submit action."})
        return submit_form(str(form_id))
    else:
        return json.dumps({"status": "error", "message": f"Unknown action '{action}'."})
