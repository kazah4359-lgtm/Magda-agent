import pytest
from datetime import datetime, timezone, timedelta
from magda_agent.codex_bridge import is_claimed

def test_is_claimed_missing_field():
    assert is_claimed({}) is False

def test_is_claimed_invalid_date_format():
    assert is_claimed({"claimed_at": "not-a-date"}) is False

def test_is_claimed_active_claim():
    now = datetime.now(timezone.utc)
    # Claimed 1 hour ago
    claimed_at = (now - timedelta(hours=1)).isoformat().replace('+00:00', 'Z')
    assert is_claimed({"claimed_at": claimed_at}) is True

def test_is_claimed_expired_claim():
    now = datetime.now(timezone.utc)
    # Claimed 3 hours ago
    claimed_at = (now - timedelta(hours=3)).isoformat().replace('+00:00', 'Z')
    assert is_claimed({"claimed_at": claimed_at}) is False
