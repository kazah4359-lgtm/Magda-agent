import pytest
from unittest.mock import patch, MagicMock
import datetime

from magda_agent.rhythms.pineal_gland import PinealGland

@pytest.fixture
def pineal_gland():
    return PinealGland()

@pytest.mark.parametrize("hour, expected_context, expected_modifier", [
    (8, "morning", 1.2),    # Morning (6-11)
    (11, "morning", 1.2),
    (14, "afternoon", 1.0), # Afternoon (12-17)
    (17, "afternoon", 1.0),
    (19, "evening", 0.9),   # Evening (18-21)
    (21, "evening", 0.9),
    (23, "night", 0.7),     # Night (22-5)
    (2, "night", 0.7),
    (5, "night", 0.7)
])
def test_time_context_and_energy(pineal_gland, hour, expected_context, expected_modifier):
    with patch.object(pineal_gland, '_get_current_time') as mock_time:
        # Create a mock datetime object with the specified hour
        mock_dt = MagicMock()
        mock_dt.hour = hour
        mock_time.return_value = mock_dt

        assert pineal_gland.get_time_context() == expected_context
        assert pineal_gland.get_energy_modifier() == expected_modifier
