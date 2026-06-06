from typing import Optional, Any
import logging
from magda_agent.gateway.router import GatewayRouter
from magda_agent.user_model.model import UserModel

class NotificationManager:
    """
    Manages proactive notifications sent from the agent to the user.
    Respects user preferences like quiet hours and notification types.
    """
    def __init__(self, gateway: GatewayRouter, user_model: Optional[UserModel] = None):
        self.gateway = gateway
        self.user_model = user_model

    async def send_notification(self, user_id: str, channel_id: str, text: str, notification_type: str = "general", urgency: str = "normal") -> bool:
        """
        Send a notification to a specific user on a specific channel.
        Checks user preferences before sending.

        Returns:
            bool: True if the notification was sent, False if it was skipped due to preferences or error.
        """
        # Default behavior: send if no user model
        if self.user_model:
            try:
                # Expecting user_id to be numeric for UserModel, but it's string in channels
                uid = int(user_id) if user_id.isdigit() else 0
                model_data = self.user_model.get_model(uid)
                prefs = model_data.get("preferences", {})

                # Check quiet hours constraint
                # A simple check: if quiet_hours is True and urgency is not high/critical, skip
                if prefs.get("quiet_hours", False) and urgency not in ["high", "critical"]:
                    logging.info(f"Skipping notification for user {user_id} due to quiet hours.")
                    return False

                # Check ignored notification types
                ignored_types = prefs.get("ignored_notifications", [])
                if notification_type in ignored_types and urgency not in ["high", "critical"]:
                    logging.info(f"Skipping notification for user {user_id} due to ignored type {notification_type}.")
                    return False

            except Exception as e:
                logging.error(f"Error checking user preferences for notification: {e}")

        # Route the message to the appropriate channel
        channel = self.gateway.get_channel(channel_id)
        if not channel:
            logging.error(f"Cannot send notification: Channel {channel_id} not found.")
            return False

        try:
            await channel.send(recipient_id=user_id, text=text, metadata={"notification_type": notification_type, "urgency": urgency})
            return True
        except Exception as e:
            logging.error(f"Failed to send notification via {channel_id}: {e}")
            return False
