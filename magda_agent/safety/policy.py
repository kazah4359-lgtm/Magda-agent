import logging
import re
from typing import Any, Dict, List, Tuple

from magda_agent.skills.names import SkillNames
from magda_agent.tracing.audit import AuditLogger


_SENSITIVE_CODE_PATTERNS = (
    re.compile(r"(^|[/\\])\.env($|[/\\]|[\'\")])"),
    re.compile(r"(^|[/\\])secrets?($|[/\\]|[\'\")])", re.IGNORECASE),
    re.compile(r"api[_-]?key|token|password|private[_-]?key", re.IGNORECASE),
)


class PolicyLayer:
    """
    Policy Layer for all tool/action execution.
    Evaluates every action with an external effect before execution.
    Logs an audit trail of all allowed and denied actions.
    """

    def __init__(self) -> None:
        self.audit_logger = AuditLogger()

    def evaluate(self, tool_name: str, **kwargs: Any) -> Tuple[bool, str]:
        allow = True
        explanation = f"Action '{tool_name}' is allowed."

        canonical_tool = self._canonical_tool_name(tool_name)
        if canonical_tool == SkillNames.PROGRAMMER:
            code = str(kwargs.get("code", ""))
            if self._mentions_sensitive_material(code):
                allow = False
                explanation = (
                    "Action denied: 'programmer' cannot access sensitive paths or secret-like material "
                    "such as .env, secrets, tokens, passwords, or private keys."
                )
        elif canonical_tool == SkillNames.OMNICHANNEL_SEND:
            recipient = kwargs.get("recipient", "")
            if recipient == "blocked_user":
                allow = False
                explanation = "Action denied: Cannot send message to a blocked recipient."

        self.audit_logger.log_call(
            tool_name=tool_name,
            kwargs=kwargs,
            why=kwargs.get("why", "No reason provided"),
            result={"allowed": allow, "explanation": explanation},
            duration=0.0,
        )

        if allow:
            logging.info(f"PolicyLayer: ALLOW - {tool_name} with args {kwargs}")
        else:
            logging.warning(f"PolicyLayer: DENY - {tool_name} with args {kwargs}. Reason: {explanation}")

        return allow, explanation

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        return self.audit_logger.get_all()

    @staticmethod
    def _canonical_tool_name(tool_name: str) -> str:
        aliases = {
            SkillNames.SYSTEM_EXECUTE_CODE: SkillNames.PROGRAMMER,
            SkillNames.SEND_MESSAGE: SkillNames.OMNICHANNEL_SEND,
        }
        return aliases.get(tool_name, tool_name)

    @staticmethod
    def _mentions_sensitive_material(value: str) -> bool:
        lowered = value.lower()
        if ".env" in lowered or "secret" in lowered:
            return True
        return any(pattern.search(value) for pattern in _SENSITIVE_CODE_PATTERNS)
