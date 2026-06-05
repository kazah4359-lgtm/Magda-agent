import json
from typing import Dict, Any, List
from magda_agent.skills.registry import SkillRegistry

def generate_agent_card(registry: SkillRegistry, endpoint: str) -> Dict[str, Any]:
    """
    Generates an A2A Agent Card describing Magda's capabilities, modalities, and endpoint.

    Args:
        registry (SkillRegistry): The initialized skill registry containing Magda's skills.
        endpoint (str): The URL or identifier of the agent's communication endpoint.

    Returns:
        Dict[str, Any]: A dictionary representing the A2A Agent Card JSON schema.
    """
    capabilities: List[Dict[str, str]] = []

    for name, desc in registry.descriptions.items():
        capabilities.append({
            "name": name,
            "description": desc
        })

    card = {
        "agent_name": "Magda",
        "description": "A sophisticated multi-modal agent with long-term memory, emotional intelligence, and dynamic skill execution capabilities.",
        "endpoint": endpoint,
        "supported_modalities": ["text", "voice", "image"],
        "capabilities": capabilities,
        "version": "1.0.0",
        "protocol": "A2A"
    }

    return card

def generate_agent_card_json(registry: SkillRegistry, endpoint: str) -> str:
    """
    Generates an A2A Agent Card as a JSON formatted string.

    Args:
        registry (SkillRegistry): The initialized skill registry containing Magda's skills.
        endpoint (str): The URL or identifier of the agent's communication endpoint.

    Returns:
        str: A JSON string representing the A2A Agent Card.
    """
    card = generate_agent_card(registry, endpoint)
    return json.dumps(card, indent=2, ensure_ascii=False)
