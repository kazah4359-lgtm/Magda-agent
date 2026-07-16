"""
Cross-Platform Unified Notification Broadcast Skill.
Allows broadcasting critical notifications to a user across all their registered platforms via the ChannelHub.
"""

import asyncio
import logging
import threading
from typing import Any, Dict, Optional, List
from magda_agent.channels.hub import ChannelHub


_BACKGROUND_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BACKGROUND_THREAD: Optional[threading.Thread] = None
_BACKGROUND_LOCK = threading.Lock()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """
    Retrieve or start a background asyncio event loop for running sync wrappers.

    Returns:
        asyncio.AbstractEventLoop: The background event loop instance.
    """
    global _BACKGROUND_LOOP, _BACKGROUND_THREAD
    with _BACKGROUND_LOCK:
        if _BACKGROUND_LOOP and _BACKGROUND_LOOP.is_running():
            return _BACKGROUND_LOOP

        loop = asyncio.new_event_loop()

        def run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        thread = threading.Thread(target=run_loop, name="magda-broadcast-loop", daemon=True)
        thread.start()
        _BACKGROUND_LOOP = loop
        _BACKGROUND_THREAD = thread
        return loop


async def broadcast_notification_async(
    message: str,
    recipient_id: str,
    hub: Optional[ChannelHub] = None,
    recipients: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Asynchronously broadcasts a notification message to a user across all registered platforms in ChannelHub.

    Args:
        message (str): The critical notification message text.
        recipient_id (str): The default recipient ID.
        hub (Optional[ChannelHub]): The ChannelHub instance managing registered channel adapters.
        recipients (Optional[Dict[str, str]]): Specific overrides for recipient IDs per channel (e.g. {"telegram": "uid1"}).
        metadata (Optional[Dict[str, Any]]): Additional metadata to attach to the broadcast.

    Returns:
        Dict[str, Any]: A dictionary containing overall status and detailed results per channel.
    """
    if not message:
        return {"status": "error", "message": "Notification message cannot be empty."}

    if not hub:
        logging.warning("No ChannelHub provided to broadcast_notification_async.")
        return {"status": "error", "message": "ChannelHub is required for broadcasting."}

    channel_ids: List[str] = list(hub._adapters.keys())
    if not channel_ids:
        logging.warning("ChannelHub has no registered adapters.")
        return {"status": "error", "message": "No registered channels available in the ChannelHub."}

    results: Dict[str, Any] = {}
    tasks = []

    async def _send_to_single_channel(channel_id: str, target_rec_id: str) -> None:
        try:
            # We route the message via the hub's send_to_channel method
            res = await hub.send_to_channel(channel_id, target_rec_id, message, metadata)
            results[channel_id] = {"status": "success", "result": res}
        except Exception as e:
            logging.error(f"Error broadcasting to channel {channel_id}: {e}")
            results[channel_id] = {"status": "error", "message": str(e)}

    for cid in channel_ids:
        target_recipient = recipient_id
        if recipients and cid in recipients:
            target_recipient = recipients[cid]

        tasks.append(_send_to_single_channel(cid, target_recipient))

    if tasks:
        await asyncio.gather(*tasks)

    # Determine overall status
    any_success = any(res["status"] == "success" for res in results.values())
    overall_status = "success" if any_success else "failed"

    return {
        "status": overall_status,
        "results": results
    }


def broadcast_notification(
    message: str,
    recipient_id: str,
    hub: Optional[ChannelHub] = None,
    recipients: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Synchronous compatibility wrapper for broadcast_notification_async.

    Args:
        message (str): The critical notification message text.
        recipient_id (str): The default recipient ID.
        hub (Optional[ChannelHub]): The ChannelHub instance managing registered channel adapters.
        recipients (Optional[Dict[str, str]]): Specific overrides for recipient IDs per channel.
        metadata (Optional[Dict[str, Any]]): Additional metadata to attach to the broadcast.

    Returns:
        Dict[str, Any]: A dictionary containing overall status and detailed results per channel.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = _get_background_loop()
        future = asyncio.run_coroutine_threadsafe(
            broadcast_notification_async(message, recipient_id, hub=hub, recipients=recipients, metadata=metadata),
            loop,
        )
        return future.result(timeout=30)

    # If already running in an event loop, we must run the coroutine directly via a task or raise an error.
    return {
        "status": "error",
        "message": "broadcast_notification called from an active event loop; use broadcast_notification_async instead."
    }
