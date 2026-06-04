import logging

def send_message(platform: str, recipient: str, message: str) -> str:
    """
    Sends a message to a recipient on a specified platform.
    Supported platforms (mocked): telegram, whatsapp, email.

    Args:
        platform (str): The communication platform (e.g., 'telegram', 'whatsapp', 'email').
        recipient (str): The recipient's ID, phone number, or email address.
        message (str): The message content to send.

    Returns:
        str: A status message indicating the result of the operation.
    """
    platform = platform.lower()
    logging.info(f"Omnichannel: Sending {platform} message to {recipient}: {message}")

    if platform == "telegram":
        # In a real scenario, this would use the bot's token and telegram API
        return f"Success: Telegram message sent to {recipient}."

    elif platform == "whatsapp":
        # Mocking WhatsApp sending logic
        return f"Success: WhatsApp message sent to {recipient} via mock gateway."

    elif platform == "email":
        # Mocking email sending logic
        return f"Success: Email sent to {recipient} via mock SMTP."

    else:
        return f"Error: Platform '{platform}' is not supported."
