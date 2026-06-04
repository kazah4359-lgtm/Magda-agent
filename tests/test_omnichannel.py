import pytest
from magda_agent.skills.omnichannel import send_message

def test_send_message_telegram():
    result = send_message("telegram", "12345678", "Hello from test!")
    assert "Success" in result
    assert "Telegram" in result
    assert "12345678" in result

def test_send_message_whatsapp():
    result = send_message("whatsapp", "+1234567890", "Hello from test!")
    assert "Success" in result
    assert "WhatsApp" in result
    assert "+1234567890" in result

def test_send_message_email():
    result = send_message("email", "test@example.com", "Hello from test!")
    assert "Success" in result
    assert "Email" in result
    assert "test@example.com" in result

def test_send_message_unsupported():
    result = send_message("carrier_pigeon", "roof", "Hello?")
    assert "Error" in result
    assert "not supported" in result
