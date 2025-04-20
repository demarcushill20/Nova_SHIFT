"""
Tests for SwarmDispatcher Fault Tolerance mechanisms (Heartbeat, Reallocation).
Corresponds to TASK.md P4.T5.4.
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, call, patch # Added patch

# Adjust imports based on project structure
from core.dispatcher import SwarmDispatcher, HEARTBEAT_CHECK_INTERVAL, HEARTBEAT_TIMEOUT
from core.shared_memory_interface import SharedMemoryInterface

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio

# --- Fixtures ---

@pytest.fixture
def mock_shared_memory() -> MagicMock:
    """Provides a mock SharedMemoryInterface with controllable reads."""
    mock = MagicMock(spec=SharedMemoryInterface)
    mock_data = {} # Simulate Redis store

    async def mock_read(key):
        return mock_data.get(key)

    async def mock_write(key, value, expiry_seconds=None):
        mock_data[key] = value
        # Basic simulation of expiry if needed, but not critical for these tests
        # print(f"Mock Write: {key} = {value} (Expiry: {expiry_seconds})") # Debug

    async def mock_delete(key):
        if key in mock_data:
            del mock_data[key]
            return True
        return False

    mock.read = AsyncMock(side_effect=mock_read)
    mock.write = AsyncMock(side_effect=mock_write)
    mock.delete = AsyncMock(side_effect=mock_delete)
    # Add mock for read_score needed by SpecialistAgent init via Dispatcher tests
    mock.read_score = AsyncMock(return_value=0.5)
    return mock, mock_data # Return mock and its data store

@pytest.fixture
def dispatcher(mock_shared_memory) -> SwarmDispatcher:
    """Provides a SwarmDispatcher instance with mocked shared memory."""
    # mock_shared_memory fixture returns a tuple (mock, data), we only need the mock
    dispatcher = SwarmDispatcher(shared_memory=mock_shared_memory[0])
    return dispatcher

@pytest.fixture
def mock_agent():
    """Provides a mock agent instance."""
    agent = MagicMock()
    agent.agent_id = "mock_agent_001"
    agent.execute_task = AsyncMock()
    return agent

# --- Test Cases ---

async def test_heartbeat_check_detects_timeout(dispatcher: SwarmDispatcher, mock_shared_memory, mock_agent):
    """Verify dispatcher detects a timed-out agent via heartbeat check."""
    mock_sm, mock_sm_data = mock_shared_memory
    agent_id = mock_agent.agent_id
    dispatcher.register_agent(agent_id, mock_agent)

    # Simulate an old heartbeat timestamp
    heartbeat_key = f"agent:{agent_id}:heartbeat"
    old_timestamp = time.time() - HEARTBEAT_TIMEOUT - 5 # 5 seconds past timeout
    mock_sm_data[heartbeat_key] = str(old_timestamp)

    # Start monitoring and wait for a check cycle
    await dispatcher.start_monitoring()
    await asyncio.sleep(HEARTBEAT_CHECK_INTERVAL + 1) # Ensure at least one check runs

    # Assert agent is marked as failed
    assert agent_id in dispatcher._failed_agents
    assert dispatcher._agents[agent_id]["status"] == "failed"

    await dispatcher.stop_monitoring()

async def test_heartbeat_check_ignores_recent_heartbeat(dispatcher: SwarmDispatcher, mock_shared_memory, mock_agent):
    """Verify dispatcher does not fail an agent with a recent heartbeat."""
    mock_sm, mock_sm_data = mock_shared_memory
    agent_id = mock_agent.agent_id
    dispatcher.register_agent(agent_id, mock_agent)

    # Simulate a recent heartbeat timestamp
    heartbeat_key = f"agent:{agent_id}:heartbeat"
    recent_timestamp = time.time() - 5 # 5 seconds ago
    mock_sm_data[heartbeat_key] = str(recent_timestamp)

    # Start monitoring and wait
    await dispatcher.start_monitoring()
    await asyncio.sleep(HEARTBEAT_CHECK_INTERVAL + 1)

    # Assert agent is NOT marked as failed
    assert agent_id not in dispatcher._failed_agents
    assert dispatcher._agents[agent_id]["status"] == "available" # Should remain available

    await dispatcher.stop_monitoring()

async def test_heartbeat_check_ignores_available_agent_no_heartbeat(dispatcher: SwarmDispatcher, mock_shared_memory, mock_agent):
    """Verify dispatcher ignores an available agent with no heartbeat key yet."""
    mock_sm, mock_sm_data = mock_shared_memory
    agent_id = mock_agent.agent_id
    dispatcher.register_agent(agent_id, mock_agent)
    # No heartbeat key is set in mock_sm_data

    # Start monitoring and wait
    await dispatcher.start_monitoring()
    await asyncio.sleep(HEARTBEAT_CHECK_INTERVAL + 1)

    # Assert agent is NOT marked as failed
    assert agent_id not in dispatcher._failed_agents
    assert dispatcher._agents[agent_id]["status"] == "available"

    await dispatcher.stop_monitoring()


async def test_task_reallocation_on_failure(dispatcher: SwarmDispatcher, mock_shared_memory, mock_agent):
    """Verify tasks assigned to a failed agent are identified for requeueing."""
    mock_sm, mock_sm_data = mock_shared_memory
    agent_id = mock_agent.agent_id
    dispatcher.register_agent(agent_id, mock_agent)

    # Assign a task
    task1 = {"subtask_id": "task_fail_1", "description": "Task for failed agent"}
    await dispatcher.dispatch_subtasks([task1])
    assert dispatcher._task_assignments.get("task_fail_1") == agent_id
    assert dispatcher._agents[agent_id]["status"] == "busy"

    # Simulate failure detection (directly call handler for test isolation)
    with patch.object(dispatcher, '_pending_tasks', new=asyncio.Queue()) as mock_queue: # Mock queue for check
         await dispatcher._handle_agent_failure(agent_id, reason="simulated failure")

    # Assert
    assert agent_id in dispatcher._failed_agents
    assert dispatcher._agents[agent_id]["status"] == "failed"
    # Check if the task assignment was removed (basic reallocation)
    assert "task_fail_1" not in dispatcher._task_assignments
    # TODO: Enhance test when full subtask definition retrieval is implemented for requeueing

async def test_get_next_available_agent_skips_failed(dispatcher: SwarmDispatcher, mock_shared_memory):
    """Verify _get_next_available_agent skips agents marked as failed."""
    mock_sm, _ = mock_shared_memory
    agent1_id = "agent_ok"
    agent2_id = "agent_failed"
    agent3_id = "agent_busy"

    dispatcher.register_agent(agent1_id, MagicMock(agent_id=agent1_id))
    dispatcher.register_agent(agent2_id, MagicMock(agent_id=agent2_id))
    dispatcher.register_agent(agent3_id, MagicMock(agent_id=agent3_id))

    # Mark agent 2 as failed, agent 3 as busy
    dispatcher._failed_agents.add(agent2_id)
    dispatcher._agents[agent2_id]["status"] = "failed"
    dispatcher._agents[agent3_id]["status"] = "busy"

    # Act
    next_agent = dispatcher._get_next_available_agent()

    # Assert
    assert next_agent == agent1_id # Only agent1 should be available and not failed

    # Make agent1 busy and check again (should return None)
    dispatcher._agents[agent1_id]["status"] = "busy"
    next_agent_none = dispatcher._get_next_available_agent()
    assert next_agent_none is None

async def test_unregister_failed_agent(dispatcher: SwarmDispatcher, mock_shared_memory, mock_agent):
    """Verify unregistering an agent correctly handles failure state."""
    mock_sm, _ = mock_shared_memory
    agent_id = mock_agent.agent_id
    dispatcher.register_agent(agent_id, mock_agent)

    # Assign a task
    task1 = {"subtask_id": "task_unregister_1", "description": "Task for agent to be unregistered"}
    await dispatcher.dispatch_subtasks([task1])
    assert dispatcher._task_assignments.get("task_unregister_1") == agent_id

    # Unregister the agent (which now calls _handle_agent_failure)
    await dispatcher.unregister_agent(agent_id)

    # Assert
    assert agent_id not in dispatcher._agents # Agent should be fully removed
    assert agent_id not in dispatcher._agent_ids_round_robin
    assert agent_id in dispatcher._failed_agents # Should still be marked as failed internally for tracking
    # Check if task was removed from assignments (basic reallocation)
    assert "task_unregister_1" not in dispatcher._task_assignments