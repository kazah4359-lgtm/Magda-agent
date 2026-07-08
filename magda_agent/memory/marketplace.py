import logging
from typing import List, Dict, Any

def export_episodic_to_marketplace_format(events: List[Dict[str, Any]], author_id: str = "magda") -> Dict[str, Any]:
    """
    Export a list of episodic memory events to the Hermes marketplace standard JSON format.

    Args:
        events: A list of episodic memory event dictionaries. Each should have 'id', 'text', and optional 'metadata'.
        author_id: The identifier for the agent creating the export. Defaults to "magda".

    Returns:
        A dictionary representing the marketplace JSON structure containing the memory events.
    """
    try:
        marketplace_events = []
        for event in events:
            marketplace_events.append({
                "event_id": event.get("id", ""),
                "text": event.get("text", ""),
                "metadata": event.get("metadata", {}),
                "timestamp": event.get("metadata", {}).get("timestamp", None)
            })

        return {
            "version": "1.0",
            "author_id": author_id,
            "events": marketplace_events
        }
    except Exception as e:
        logging.error(f"Error exporting episodic memory to marketplace format: {e}")
        return {"version": "1.0", "author_id": author_id, "events": []}

def import_episodic_from_marketplace_format(marketplace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Import episodic memory events from the Hermes marketplace standard JSON format.

    Args:
        marketplace_data: A dictionary representing the marketplace JSON structure.

    Returns:
        A list of episodic memory event dictionaries compatible with the local system.
    """
    try:
        if not isinstance(marketplace_data, dict) or "events" not in marketplace_data:
            logging.error("Invalid marketplace data format. Must be a dict with an 'events' key.")
            return []

        events = []
        for marketplace_event in marketplace_data.get("events", []):
            event = {
                "id": marketplace_event.get("event_id", ""),
                "text": marketplace_event.get("text", ""),
                "metadata": marketplace_event.get("metadata", {})
            }
            # Maintain timestamp if it was present explicitly in the marketplace event
            if "timestamp" in marketplace_event and marketplace_event["timestamp"] is not None:
                if "timestamp" not in event["metadata"]:
                    event["metadata"]["timestamp"] = marketplace_event["timestamp"]

            events.append(event)

        return events
    except Exception as e:
        logging.error(f"Error importing episodic memory from marketplace format: {e}")
        return []
