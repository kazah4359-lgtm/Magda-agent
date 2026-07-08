import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from magda_agent.agents.heartbeat import SubagentHeartbeatMonitor

def test_monitor_registration() -> None:
    """Tests that a subagent can be registered and its heartbeat is tracked."""
    monitor = SubagentHeartbeatMonitor()
    agent_id = "agent_1"

    with patch("magda_agent.agents.heartbeat.time.time", return_value=100.0):
        monitor.register_agent(agent_id)

    assert agent_id in monitor._heartbeats
    assert monitor._heartbeats[agent_id] == 100.0

def test_monitor_record_heartbeat() -> None:
    """Tests that a heartbeat updates the timestamp."""
    monitor = SubagentHeartbeatMonitor()
    agent_id = "agent_1"

    with patch("magda_agent.agents.heartbeat.time.time", return_value=100.0):
        monitor.register_agent(agent_id)

    with patch("magda_agent.agents.heartbeat.time.time", return_value=110.0):
        monitor.record_heartbeat(agent_id)

    assert monitor._heartbeats[agent_id] == 110.0

def test_monitor_record_heartbeat_unregistered() -> None:
    """Tests recording heartbeat for an unregistered agent."""
    monitor = SubagentHeartbeatMonitor()
    agent_id = "agent_1"

    with patch("magda_agent.agents.heartbeat.logging.warning") as mock_warning:
        monitor.record_heartbeat(agent_id)
        mock_warning.assert_called_once_with(f"Received heartbeat for unregistered agent {agent_id}.")
    assert agent_id not in monitor._heartbeats

def test_monitor_check_deadlocks() -> None:
    """Tests deadlock detection logic."""
    monitor = SubagentHeartbeatMonitor()
    monitor._heartbeats = {
        "agent_active": 100.0,
        "agent_dead": 50.0
    }

    # Current time is 120.0
    # Timeout is 30.0
    # agent_active last heartbeat was 20.0 seconds ago (alive)
    # agent_dead last heartbeat was 70.0 seconds ago (deadlocked)
    with patch("magda_agent.agents.heartbeat.time.time", return_value=120.0):
        deadlocked = monitor.check_deadlocks(timeout_seconds=30.0)

    assert "agent_dead" in deadlocked
    assert "agent_active" not in deadlocked

@pytest.mark.asyncio
async def test_monitor_background_loop() -> None:
    """Tests the background monitoring loop with mocking."""
    monitor = SubagentHeartbeatMonitor()

    # We will patch check_deadlocks to return a deadlocked agent
    with patch.object(monitor, "check_deadlocks", return_value=["agent_dead"]) as mock_check:
        with patch.object(monitor, "_handle_deadlocked_agents") as mock_handle:
            # We want the loop to run once and then exit.
            def side_effect(*args, **kwargs):
                monitor._is_running = False
                return ["agent_dead"]
            mock_check.side_effect = side_effect

            # Since check_interval is passed to sleep, let's patch sleep to avoid waiting
            with patch("asyncio.sleep", new_callable=AsyncMock):
                # Run the loop directly for testing
                monitor._is_running = True
                await monitor._monitor_loop(timeout_seconds=30.0, check_interval=0.1)

            mock_check.assert_called_with(30.0)
            mock_handle.assert_called_with(["agent_dead"])

@pytest.mark.asyncio
async def test_monitor_start_stop() -> None:
    """Tests starting and stopping the monitor task without patching asyncio.create_task globally."""
    monitor = SubagentHeartbeatMonitor()

    # Create a quick loop that finishes itself fast to not hang
    # Note: _monitor_loop is called as self._monitor_loop(timeout, interval), so it takes timeout and interval.
    # When patching an object method with a function, we must not include 'self' as the first param if we replace the bound method,
    # but patch.object on an instance handles this trickily. Let's patch it properly.

    async def fast_dummy_loop(*args, **kwargs):
        monitor._is_running = True
        await asyncio.sleep(0.01)
        monitor._is_running = False

    with patch.object(monitor, '_monitor_loop', side_effect=fast_dummy_loop):
        monitor.start_monitor()
        assert monitor._monitor_task is not None
        assert not monitor._monitor_task.done()

        await monitor.stop_monitor()
        assert not monitor._is_running
        # Let asyncio scheduler clean up the cancelled task
        await asyncio.sleep(0.01)
        assert monitor._monitor_task.cancelled() or monitor._monitor_task.done()
