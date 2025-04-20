"""
Pytest configuration file for adding the project root to sys.path.
"""

import sys
import os
import pathlib
import pytest # Added import
from unittest.mock import MagicMock, AsyncMock # Added imports
from core.dispatcher import SwarmDispatcher # Added import
from core.shared_memory_interface import SharedMemoryInterface # Added import
from core.ltm_interface import LTMInterface # Added import

# Add the project root directory (parent of the 'tests' directory) to sys.path
# This ensures that pytest can find the 'nova_shift' package during test collection and execution.
project_root = pathlib.Path(__file__).parent.parent.resolve()
print(f"\n[conftest.py] Adding project root to sys.path: {project_root}")
sys.path.insert(0, str(project_root))
print(f"[conftest.py] Current sys.path[0]: {sys.path[0]}")

# Load environment variables from .env file at the project root
try:
    from dotenv import load_dotenv
    dotenv_path = project_root / '.env'
    if dotenv_path.is_file():
        load_dotenv(dotenv_path=dotenv_path, override=True)
        print(f"[conftest.py] Loaded environment variables from: {dotenv_path}")
    else:
        print(f"[conftest.py] .env file not found at {dotenv_path}, skipping loading.")
except ImportError:
    print("[conftest.py] python-dotenv not installed, skipping loading of .env file.")
print() # Add a newline for cleaner output separation

# You can also define project-wide fixtures here if needed later.

@pytest.fixture
def mock_dispatcher() -> MagicMock:
    """Provides a mock SwarmDispatcher, shared across tests."""
    mock = MagicMock(spec=SwarmDispatcher)
    # Add common mocked methods if needed, e.g.:
    mock.dispatch_subtasks = MagicMock(return_value={})
    mock.update_task_status = MagicMock()
    mock.get_agent_status = MagicMock(return_value={})
    mock.reallocate_task = MagicMock(return_value=None)
    return mock

@pytest.fixture
def mock_ltm_interface() -> MagicMock:
    """Provides a mock LTMInterface, shared across tests."""
    mock = MagicMock(spec=LTMInterface)
    # Simulate finding some relevant context by default
    mock.retrieve = AsyncMock(return_value=[
        {"id": "doc1", "score": 0.9, "metadata": {"original_text": "LTM context about topic A.", "source": "ltm_doc"}}
    ])
    mock.store = AsyncMock()
    # Mock formatting to return a string similar to what the agent might produce
    expected_formatted_context = "Retrieved Context:\n1. (Source: ltm_doc): LTM context about topic A."
    mock._format_ltm_context = MagicMock(return_value=expected_formatted_context)
    return mock

@pytest.fixture
def mock_shared_memory() -> MagicMock:
    """Provides a mock SharedMemoryInterface using an internal dict."""
    mock = MagicMock(spec=SharedMemoryInterface)
    mock_data = {} # Internal dictionary to simulate Redis

    async def mock_write_side_effect(key, value, expiry_seconds=None):
        mock_data[key] = str(value) # Store as string like Redis
    mock.write = AsyncMock(side_effect=mock_write_side_effect)

    async def mock_set_side_effect(key, value, expiry_seconds=None):
         mock_data[key] = str(value) # Store as string like Redis
    mock.set = AsyncMock(side_effect=mock_set_side_effect)

    async def mock_read_side_effect(key):
        # Simulate dependency checks if needed, otherwise return from dict
        if key.endswith(":status") and key not in mock_data:
             # return "completed" # Default behavior removed, rely on test setup
             pass
        if key.endswith(":result") and key not in mock_data:
             # return "mock dependency result" # Default behavior removed
             pass
        return mock_data.get(key)
    mock.read = AsyncMock(side_effect=mock_read_side_effect)

    async def mock_delete_side_effect(key):
        mock_data.pop(key, None)
    mock.delete = AsyncMock(side_effect=mock_delete_side_effect)

    mock.publish = AsyncMock()
    mock.increment_counter = AsyncMock(return_value=1)
    mock.increment_float = AsyncMock(return_value=1.0)
    mock.ping = AsyncMock(return_value=True)

    # Mock read_score needed by SpecialistAgent - use side effect to ensure float return
    async def mock_read_score_side_effect(tool_name, default_score=0.5):
        # Look up the score in the mock data dictionary
        score_key = f"tool:{tool_name}:score"
        stored_value = mock_data.get(score_key)
        if stored_value is not None:
            try:
                return float(stored_value)
            except (ValueError, TypeError):
                # Handle cases where stored value isn't a valid float
                pass # Fall through to return default
        # Return default if key not found or stored value is invalid
        return float(default_score)
    mock.read_score = AsyncMock(side_effect=mock_read_score_side_effect)

    # Add mock scan_iter and mget for learning engine
    # Add mock scan_iter and mget for learning engine
    async def mock_scan_iter(match="tool:*:usage_count"):
        # Simulate Redis SCAN with MATCH more accurately
        parts = match.split('*', 1) # Split only on the first '*'
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        for key in list(mock_data.keys()):
            # Check if key starts with prefix and ends with suffix
            if key.startswith(prefix) and key.endswith(suffix):
                # Ensure the part between prefix and suffix exists if '*' was present
                if '*' in match:
                    if len(key) > len(prefix) + len(suffix):
                         yield key
                elif key == prefix: # Handle cases without '*'
                     yield key
    mock.scan_iter = mock_scan_iter # Assign the async generator function directly

    async def mock_mget(keys):
        return [mock_data.get(k) for k in keys]
    mock.mget = AsyncMock(side_effect=mock_mget)

    # Add a way for tests to access the internal data if absolutely needed
    # (though interacting via the mock methods is preferred)
    mock._get_internal_data = lambda: mock_data

    return mock

@pytest.fixture
def mock_llm() -> MagicMock:
    """Provides a mock LLM client, shared across tests."""
    mock = MagicMock()
    # Mock the async invoke method used by agents
    mock.ainvoke = AsyncMock()
    # Set a default return value if needed, or configure in tests
    # mock.ainvoke.return_value = {"output": "Default mock LLM response"}
    return mock