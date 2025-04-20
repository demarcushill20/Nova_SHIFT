"""
Integration tests for the Phase 2 workflow:
Planner -> Dispatcher -> Concurrent Agents -> Shared Memory Communication.
"""

import pytest
import asyncio
import os
import asyncio # Added for sleep
from unittest.mock import MagicMock, AsyncMock
from langchain_core.language_models.chat_models import BaseChatModel # Added import
# Assuming project root is added to PYTHONPATH for imports
# Adjust relative paths if necessary based on test execution context
from core.dispatcher import SwarmDispatcher
from core.shared_memory_interface import SharedMemoryInterface
from core.tool_registry import ToolRegistry
from agents.planner_agent import PlannerAgent
from agents.specialist_agent import SpecialistAgent, initialize_specialist_agent_instance # Removed load_toolkits_from_directory
# Import mock data for direct loading using absolute paths from project root
from tests.agents.test_specialist_agent import MOCK_CALCULATOR_TOOLKIT_DATA, MOCK_SEARCH_TOOLKIT_DATA
from tests.agents.test_specialist_agent_dynamic_ltm import MOCK_FILE_TOOLKIT_DATA

# --- Test Setup ---

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio

# Fixture for Redis connection (requires running Redis instance)
# Consider using 'fakeredis' for isolated testing without a live Redis server
@pytest.fixture(scope="function") # Change scope to function
async def redis_interface():
    """Provides a SharedMemoryInterface instance connected to test Redis."""
    # Use a different DB for testing if possible
    test_redis_db = 1
    interface = SharedMemoryInterface(db=test_redis_db)
    # Clear the test database before starting tests
    try:
        client = await interface._get_client() # Access internal client for setup
        await client.flushdb()
        print(f"\nFlushed Redis DB {test_redis_db}")
    except Exception as e:
        pytest.fail(f"Failed to connect to or flush Redis DB {test_redis_db}: {e}")

    yield interface
    # Teardown: clear DB again and close connection
    try:
        client = await interface._get_client()
        await client.flushdb()
        print(f"\nFlushed Redis DB {test_redis_db} after tests.")
    except Exception as e:
         print(f"\nError flushing Redis DB {test_redis_db} during teardown: {e}")
    finally:
        await interface.close()

# Fixture for Tool Registry
@pytest.fixture(scope="function") # Change scope to function
def tool_registry():
    """Provides a ToolRegistry instance loaded with default tools."""
    registry = ToolRegistry()
    # Load toolkits directly from mock data dictionaries
    registry.load_toolkit_from_dict(MOCK_CALCULATOR_TOOLKIT_DATA)
    registry.load_toolkit_from_dict(MOCK_SEARCH_TOOLKIT_DATA)
    registry.load_toolkit_from_dict(MOCK_FILE_TOOLKIT_DATA)
    return registry

# Fixture for Mock LLM Client
@pytest.fixture
def mock_llm_client():
    """Provides a mock LLM client."""
    client = AsyncMock()

    # Define mock responses based on expected prompts (needs refinement)
    async def mock_generate(prompt: str):
        await asyncio.sleep(0.05) # Simulate async delay
        if "decompose" in prompt.lower() and "User Goal:" in prompt:
            # Mock Planner response
            return """
            ```json
            [
                {"subtask_id": "t1_search", "description": "Search web for 'asyncio benefits'.", "depends_on": []},
                {"subtask_id": "t2_summarize", "description": "Summarize the key benefits found.", "depends_on": ["t1_search"]},
                {"subtask_id": "t3_calculate", "description": "Calculate 25 * 4.", "depends_on": []}
            ]
            ```
            """
        elif "AgentExecutor" in prompt: # Heuristic for Specialist Agent prompt
             # Mock Specialist response (e.g., for search task)
             # This needs to be more sophisticated based on actual tool calls
             if "Search web" in prompt:
                 return {"output": "Asyncio provides concurrency using an event loop..."}
             elif "Summarize" in prompt:
                 return {"output": "Key benefits include efficient I/O handling and cooperative multitasking."}
             elif "Calculate" in prompt:
                 return {"output": "100"} # Mock calculator result
             else:
                 return {"output": "Mocked specialist response."}
        else:
            return {"output": "Unknown mock prompt."} # Default mock

    client.generate.side_effect = mock_generate
    # For LangChain's ChatOpenAI which uses .ainvoke
    client.ainvoke = AsyncMock()
    async def mock_ainvoke(input_data): # Input can be dict or str
         await asyncio.sleep(0.05)
         # Handle both dict (agent execution) and str (tool analysis) inputs
         if isinstance(input_data, dict):
             desc = input_data.get("input", "") # Agent execution uses 'input' key
             if "Search web" in desc:
                 return {"output": "Asyncio provides concurrency using an event loop..."}
             elif "Summarize" in desc:
                 return {"output": "Key benefits include efficient I/O handling and cooperative multitasking."}
             elif "Calculate" in desc:
                 return {"output": "100"}
             else:
                 return {"output": f"Mocked specialist response for: {desc[:30]}..."}
         elif isinstance(input_data, str):
             # Tool analysis prompt is just a string
             if "tool requirement analyzer" in input_data:
                 # Simulate returning required tools based on description
                 if "Search web" in input_data:
                     return '["search_internet"]'
                 elif "Calculate" in input_data:
                     return '["calculate"]'
                 else:
                     return '[]' # Default: no tools required
             else:
                 return "Mock LLM Response for string input" # Default for other string inputs
         else:
             return "Unexpected input type to mock_ainvoke"
    client.ainvoke.side_effect = mock_ainvoke

    return client


# --- Integration Test ---

async def test_planner_dispatcher_agent_shared_memory_flow(
    redis_interface: SharedMemoryInterface,
    tool_registry: ToolRegistry,
    mock_llm_client: AsyncMock,
    mock_ltm_interface: MagicMock # Ensure mock LTM fixture is requested
):
    """
    Tests the full flow from planning to concurrent execution and shared memory updates.
    """
    # Ensure necessary API keys are set for tool usage (if tools make real calls)
    # For this test, ensure TAVILY_API_KEY is set if WebSearch is used.
    assert os.environ.get("TAVILY_API_KEY"), "TAVILY_API_KEY must be set for web search tool"
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY must be set for LLM"


    # 1. Initialize Components
    dispatcher = SwarmDispatcher(shared_memory=redis_interface) # Pass shared memory interface
    # Use the actual LLM client for Specialist Agents, but mock for Planner
    # (or mock both if preferred for faster/isolated tests)
    planner_llm_mock = mock_llm_client # Use AsyncMock for planner decomposition
    # Use MagicMock for specialist LLM to potentially avoid TypeError during agent creation in test
    specialist_llm = MagicMock(spec=BaseChatModel)
    # Set up async behavior specifically for ainvoke if needed by the test logic later
    async def mock_ainvoke(*args, **kwargs):
        # Define mock responses based on expected calls if agent execution were to proceed
        # This part might need adjustment based on how the test mocks agent execution later
        input_data = args[0] # Assuming prompt/input is the first arg
        if isinstance(input_data, dict):
             desc = input_data.get("input", "") # Agent execution uses 'input' key
             if "Search web" in desc:
                 return {"output": "Asyncio provides concurrency using an event loop..."}
             elif "Summarize" in desc:
                 return {"output": "Key benefits include efficient I/O handling and cooperative multitasking."}
             elif "Calculate" in desc:
                 return {"output": "100"}
             else:
                 return {"output": f"Mocked specialist response for: {desc[:30]}..."}
        elif isinstance(input_data, str):
             # Tool analysis prompt is just a string
             if "tool requirement analyzer" in input_data:
                 # Simulate returning required tools based on description
                 if "Search web" in input_data:
                     return '["search_internet"]'
                 elif "Calculate" in input_data:
                     return '["calculate"]'
                 else:
                     return '[]' # Default: no tools required
             else:
                 return "Mock LLM Response for string input" # Default for other string inputs
        else:
             return "Unexpected input type to mock_ainvoke"
    specialist_llm.ainvoke = AsyncMock(side_effect=mock_ainvoke)

    # Need LTM interface for PlannerAgent init
    # Use the global mock fixture from conftest.py
    planner = PlannerAgent(
        dispatcher=dispatcher,
        llm_client=planner_llm_mock,
        ltm_interface=mock_ltm_interface # Pass the mock LTM interface
    )

    # Initialize and register Specialist Agents
    agent1 = initialize_specialist_agent_instance(
        agent_id="test_agent_001",
        registry=tool_registry,
        shared_memory=redis_interface,
        ltm_interface=mock_ltm_interface, # Pass mock LTM
        dispatcher=dispatcher,
        llm=specialist_llm # Use mock LLM here
    )
    agent2 = initialize_specialist_agent_instance(
        agent_id="test_agent_002",
        registry=tool_registry,
        shared_memory=redis_interface,
        ltm_interface=mock_ltm_interface, # Pass mock LTM
        dispatcher=dispatcher,
        llm=specialist_llm # Use mock LLM here
    )
    assert agent1 is not None, "Failed to initialize agent 1"
    assert agent2 is not None, "Failed to initialize agent 2"
    dispatcher.register_agent(agent1.agent_id, agent1)
    dispatcher.register_agent(agent2.agent_id, agent2)
    # Explicitly create executors after initialization for the test
    await agent1._create_agent_executor()
    await agent2._create_agent_executor()

    # 2. Define Goal and Trigger Planner
    user_goal = "Research asyncio benefits, summarize them, and calculate 25 * 4."
    print(f"\n--- Starting Test Flow with Goal: '{user_goal}' ---")
    dispatch_initiated = await planner.decompose_and_dispatch(user_goal)
    assert dispatch_initiated, "Planner failed to initiate dispatch"

    # 3. Wait for tasks to complete (allow asyncio tasks to run)
    # This duration needs to be sufficient for mocked delays + processing
    print("\n--- Waiting for agent tasks to complete ---")
    await asyncio.sleep(2) # Adjust sleep time as needed

    # 4. Verify Results in Shared Memory
    print("\n--- Verifying results in Shared Memory ---")
    # Check status and results based on the mocked planner output
    expected_tasks = ["t1_search", "t2_summarize", "t3_calculate"]
    results = {}
    all_completed = True

    for task_id in expected_tasks:
        status = await redis_interface.read(f"task:{task_id}:status")
        result = await redis_interface.read(f"task:{task_id}:result")
        error = await redis_interface.read(f"task:{task_id}:error")

        print(f"  Task {task_id}: Status='{status}'")
        if status != "completed":
            all_completed = False
            print(f"    ERROR: Task {task_id} did not complete. Error: {error}")
        else:
            results[task_id] = result
            print(f"    Result: {str(result)[:100]}...") # Print snippet

    assert all_completed, "Not all tasks completed successfully."

    # Basic checks on expected results (based on mocks)
    assert "Asyncio provides concurrency" in results.get("t1_search", "")
    assert "efficient I/O handling" in results.get("t2_summarize", "")
    assert results.get("t3_calculate") == "100" # Calculator mock returns string

    print("\n--- Test Flow Completed Successfully ---")
