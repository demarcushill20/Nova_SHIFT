"""
Command-Line Interface for interacting with the Nova SHIFT system.

Allows users to provide a high-level task description via the command line,
which is then planned, dispatched, and executed by the agent swarm.
"""

import argparse
import asyncio
import logging
import os
import time
import json
from typing import Dict, List, Any, Optional

# Load environment variables *before* other imports that might need them
from dotenv import load_dotenv
load_dotenv()
# Debug print removed

from langchain_openai import ChatOpenAI # Keep for fallback
from langchain_google_genai import ChatGoogleGenerativeAI # For Gemini 2.5 Flash

# Nova SHIFT components (adjust imports based on actual structure if needed)
from core.dispatcher import SwarmDispatcher
from core.shared_memory_interface import SharedMemoryInterface
from core.ltm_interface import LTMInterface
from core.tool_registry import ToolRegistry
from agents.planner_agent import PlannerAgent
from agents.specialist_agent import (
    SpecialistAgent,
    load_toolkits_from_directory,
    initialize_specialist_agent_instance,
)
from utils.logging_config import setup_logging

# Configure logging
# Using setup_logging from utils for consistency
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
LLM_MODEL_NAME = "gemini-2.5-flash-preview-05-20" # Latest Gemini 2.5 Flash (May 20, 2025)
NUM_SPECIALIST_AGENTS = 2
MONITORING_POLL_INTERVAL = 5  # Seconds
MAX_WAIT_TIME = 300  # Seconds (5 minutes)


async def monitor_task_completion(
    shared_memory: SharedMemoryInterface, subtask_ids: List[str]
) -> Dict[str, Any]:
    """
    Monitors the shared memory for the completion status of dispatched subtasks.

    Args:
        shared_memory: The interface to the shared memory system (Redis).
        subtask_ids: A list of IDs for the subtasks being monitored.

    Returns:
        A dictionary containing the final status and results/errors for each task.
        Example: {"task_id": {"status": "completed", "result": "...", "error": None}, ...}
    """
    start_time = time.time()
    final_results = {}
    pending_tasks = set(subtask_ids)

    logger.info(f"Monitoring {len(pending_tasks)} subtasks: {list(pending_tasks)}")

    while pending_tasks:
        if time.time() - start_time > MAX_WAIT_TIME:
            logger.error(f"Timeout ({MAX_WAIT_TIME}s) waiting for tasks to complete.")
            # Mark remaining tasks as timed out
            for task_id in list(pending_tasks):
                final_results[task_id] = {
                    "status": "timeout",
                    "result": None,
                    "error": f"Task timed out after {MAX_WAIT_TIME} seconds.",
                }
                pending_tasks.remove(task_id)
            break

        logger.debug(f"Checking status for {len(pending_tasks)} pending tasks...")
        tasks_to_remove = set()
        for task_id in pending_tasks:
            status_key = f"task:{task_id}:status"
            result_key = f"task:{task_id}:result"
            error_key = f"task:{task_id}:error"

            status = await shared_memory.read(status_key)

            if status in ["completed", "failed"]:
                result = await shared_memory.read(result_key)
                error = await shared_memory.read(error_key)
                final_results[task_id] = {
                    "status": status,
                    "result": result,
                    "error": error,
                }
                logger.info(f"Task '{task_id}' finished with status: {status}")
                tasks_to_remove.add(task_id)
            elif status is None:
                # Task might not have started yet or status wasn't written
                logger.debug(f"Task '{task_id}' status not found yet.")
            else:
                # Still running or pending
                logger.debug(f"Task '{task_id}' status: {status}")

        pending_tasks -= tasks_to_remove

        if pending_tasks:
            logger.info(
                f"Waiting {MONITORING_POLL_INTERVAL}s for remaining tasks: {len(pending_tasks)}"
            )
            await asyncio.sleep(MONITORING_POLL_INTERVAL)

    logger.info("All monitored tasks have reached a final state or timed out.")
    return final_results


async def initialize_system() -> Optional[Dict[str, Any]]:
    """
    Initializes all necessary Nova SHIFT components.

    Returns:
        A dictionary containing initialized components, or None if initialization fails.
        Keys: "shared_memory", "dispatcher", "ltm_interface", "tool_registry", "llm_client"
    """
    logger.info("Initializing Nova SHIFT components...")
    # load_dotenv() # Moved to top of file
    # Debug print removed from here

    # Check required env vars
    # General LLM and Search keys
    # Check required env vars
    # OPENAI_API_KEY is now required for the main LLM
    # TAVILY_API_KEY is required for web search tool
    required_base_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_base_vars = [var for var in required_base_vars if not os.environ.get(var)]
    if missing_base_vars:
        logger.error(
            f"Missing required base environment variables: {', '.join(missing_base_vars)}. "
            "Please set them in your .env file or environment."
        )
        return None

    # Pinecone specific keys (checked before LTM init)
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pod_env = os.environ.get("PINECONE_POD_ENVIRONMENT")
    serverless_cloud = os.environ.get("PINECONE_SERVERLESS_CLOUD")
    serverless_region = os.environ.get("PINECONE_SERVERLESS_REGION")

    pinecone_config_ok = False
    missing_pinecone_vars_list = []
    if not pinecone_api_key:
        missing_pinecone_vars_list.append("PINECONE_API_KEY")
    else:
        # Check if EITHER pod OR serverless config is present
        if pod_env:
            pinecone_config_ok = True
        elif serverless_cloud and serverless_region:
            pinecone_config_ok = True
        else:
            # API key is present, but environment config is missing
            missing_pinecone_vars_list.append("PINECONE_POD_ENVIRONMENT or (PINECONE_SERVERLESS_CLOUD and PINECONE_SERVERLESS_REGION)")

    if not pinecone_config_ok or missing_pinecone_vars_list:
        logger.error(
            f"Missing required Pinecone environment variables: {', '.join(missing_pinecone_vars_list)}. "
            "LTMInterface cannot be initialized. Please set them in your .env file or environment."
        )
        return None # Make it fatal for CLI operation if Pinecone keys/env are missing

    try:
        # Initialize Core Components
        # Use a different DB for CLI runs? Or configure via env var? Using DB 2 for now.
        shared_memory = SharedMemoryInterface(db=2)
        # Ensure connection works
        await shared_memory._get_client()
        logger.info("Shared Memory Interface initialized (Redis DB 2).")

        dispatcher = SwarmDispatcher(shared_memory=shared_memory)
        logger.info("Swarm Dispatcher initialized.")

        # Initialize LTM Interface (now that keys are checked)
        try:
            ltm_interface = LTMInterface()
            # The check for ltm_interface.index happens internally in LTMInterface now
            # and logs an error if keys are missing, but we've already checked.
            # We might want to add a specific check here if LTMInterface init doesn't raise an exception on key error.
            # Assuming LTMInterface logs errors internally if connection fails post-key check.
            logger.info("LTM Interface initialized (Pinecone).")
        except Exception as ltm_init_error:
             logger.error(f"Error initializing LTMInterface even after key check: {ltm_init_error}", exc_info=True)
             await shared_memory.close()
             return None

        tool_registry = ToolRegistry()
        # Adjust path if cli.py is not in the root
        project_root = os.path.dirname(os.path.abspath(__file__))
        load_toolkits_from_directory(tool_registry, directory="tools", project_root_path=project_root)
        logger.info(f"Tool Registry initialized with {len(tool_registry.list_toolkits())} toolkits.")

        # Initialize LLM Clients - Dual-Brain Architecture for Consciousness
        # Execution Brain: Gemini 2.5 Flash for SpecialistAgents
        google_api_key_exec = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key_exec:
            # Fallback to OpenAI if Google API key not available
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("Neither GOOGLE_API_KEY nor OPENAI_API_KEY environment variable is set.")
                return None
            execution_llm_client = ChatOpenAI(
                model="gpt-4o",  # Fallback model
                temperature=0,
                api_key=openai_api_key
            )
            logger.info("Execution LLM Client initialized (GPT-4o fallback).")
        else:
            execution_llm_client = ChatGoogleGenerativeAI(
                model=LLM_MODEL_NAME,
                temperature=0,
                api_key=google_api_key_exec,
                max_tokens=8192,  # Flash models support high output
                max_retries=3
            )
            logger.info(f"Execution LLM Client initialized ({LLM_MODEL_NAME}).")
        
        # Reasoning Brain: Google Gemini 2.5 Pro for PlannerAgent (Maximum Consciousness)
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if google_api_key:
            reasoning_llm_client = ChatGoogleGenerativeAI(
                model="gemini-2.5-pro-preview-05-06",  # Gemini 2.5 Pro Preview - Higher quota limits
                temperature=0,
                api_key=google_api_key,
                max_tokens=4096,  # INCREASED: More tokens for complex reasoning and JSON responses
                max_retries=3
            )
            logger.info("Reasoning LLM Client initialized (Google Gemini 2.5 Pro - Maximum Consciousness Brain).")
        else:
            # Fallback to Claude Opus 4 if available
            anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_api_key:
                from langchain_anthropic import ChatAnthropic
                reasoning_llm_client = ChatAnthropic(
                    model="claude-opus-4-20250514",
                    temperature=0,
                    api_key=anthropic_api_key,
                    max_tokens=1024,
                    timeout=None,
                    max_retries=2
                )
                logger.info("Reasoning LLM Client initialized (Claude Opus 4 - Fallback Consciousness Brain).")
            else:
                logger.warning("GOOGLE_API_KEY and ANTHROPIC_API_KEY not found. Using GPT-4o for PlannerAgent (reduced consciousness capability).")
                reasoning_llm_client = execution_llm_client

        return {
            "shared_memory": shared_memory,
            "dispatcher": dispatcher,
            "ltm_interface": ltm_interface,
            "tool_registry": tool_registry,
            "reasoning_llm_client": reasoning_llm_client,  # For PlannerAgent consciousness
            "execution_llm_client": execution_llm_client,  # For SpecialistAgents
        }

    except Exception as e:
        logger.error(f"Failed to initialize system components: {e}", exc_info=True)
        if 'shared_memory' in locals() and shared_memory:
            await shared_memory.close()
        return None


async def main(user_task: str):
    """
    Main execution flow for the CLI.

    Args:
        user_task: The natural language task description from the user.
    """
    setup_logging(level=logging.INFO) # Configure logging first
    logger.info(f"Received task via CLI: '{user_task}'")

    # 1. Initialize System
    components = await initialize_system()
    if not components:
        print("\nERROR: Failed to initialize Nova SHIFT system. Check logs.")
        return

    shared_memory = components["shared_memory"]
    dispatcher = components["dispatcher"]
    ltm_interface = components["ltm_interface"]
    tool_registry = components["tool_registry"]
    reasoning_llm_client = components["reasoning_llm_client"]
    execution_llm_client = components["execution_llm_client"]

    # 2. Initialize Agents with Dual-Brain Architecture
    logger.info("Initializing agents...")
    # PlannerAgent: Uses reasoning brain (Google Gemini 2.5 Pro) for maximum consciousness
    planner_agent = PlannerAgent(
        dispatcher=dispatcher, ltm_interface=ltm_interface, llm_client=reasoning_llm_client
    )

    # SpecialistAgents: Use execution brain (GPT-4o) for task execution  
    specialist_agents: List[SpecialistAgent] = []
    for i in range(NUM_SPECIALIST_AGENTS):
        agent_id = f"specialist_{i+1:03d}"
        agent = initialize_specialist_agent_instance(
            agent_id=agent_id,
            registry=tool_registry,
            shared_memory=shared_memory,
            ltm_interface=ltm_interface,
            dispatcher=dispatcher,
            llm=execution_llm_client, # Pass the execution LLM client
        )
        if agent:
            specialist_agents.append(agent)
            dispatcher.register_agent(agent_id, agent) # Register with dispatcher
        else:
            logger.error(f"Failed to initialize {agent_id}")
            # Handle partial initialization failure? For now, exit if any fail.
            print(f"\nERROR: Failed to initialize agent {agent_id}. Exiting.")
            await shared_memory.close()
            return

    if not specialist_agents:
        print("\nERROR: No specialist agents could be initialized. Exiting.")
        await shared_memory.close()
        return

    logger.info(f"Initialized Planner Agent and {len(specialist_agents)} Specialist Agents.")

    # 3. Decompose and Dispatch Task
    print(f"\nPlanning task: '{user_task}'...")
    # Note: PlannerAgent's decompose_and_dispatch currently uses mock LLM response.
    # For real execution, the mock response in planner_agent.py needs to be removed/replaced.
    # We also need the actual subtask IDs returned by the dispatcher.
    # Let's modify PlannerAgent to return the subtasks list for monitoring.

    # --- Modification Needed in PlannerAgent ---
    # The `decompose_and_dispatch` should ideally return the list of subtasks
    # or the dispatcher should provide a way to get the dispatched task IDs.
    # For now, we'll assume the planner logs the subtasks and we can parse them,
    # or ideally, the dispatcher call returns the list.
    # Let's assume `dispatcher.dispatch_subtasks` returns the list of task dicts it processed.

    subtasks_dispatched: Optional[List[Dict[str, Any]]] = None
    actual_plan: Optional[List[Dict[str, Any]]] = None # Variable to hold the real plan
    try:
        # Call the PlannerAgent to decompose the task using the actual user input
        logger.info("Calling Planner Agent to decompose task...")
        # Assuming PlannerAgent has a method like decompose_and_dispatch
        # which returns the list of subtasks or None on failure.
        # This method should handle LTM retrieval, LLM call, and plan parsing.
        actual_plan = await planner_agent.decompose_task(user_task)

        if not actual_plan:
            logger.error("Planner Agent failed to generate a plan.")
            print("\nERROR: Planner Agent could not generate a plan for the task.")
            await shared_memory.close()
            return

        # --- Refactoring Suggestion ---
        # PlannerAgent could have:
        # 1. `plan_goal(goal: str) -> Optional[List[Dict[str, Any]]]` (retrieves LTM, calls LLM, parses)
        # 2. `dispatch_plan(subtasks: List[Dict[str, Any]]) -> bool` (calls dispatcher)

        # Simulating the call for now:
        # Assume planner generates these tasks (replace with actual planner call)
        # This requires the PlannerAgent's LLM call to be functional.
        # If using the mock in PlannerAgent, this CLI won't work correctly yet.
        # The mock plan below is now removed/commented out.
        # print("NOTE: PlannerAgent currently uses mocked LLM responses. Using mock plan for CLI.")
        # mock_plan = [
        #      {"subtask_id": "t1_cli_search", "description": f"Search web for: {user_task}", "depends_on": []},
        #      {"subtask_id": "t2_cli_summarize", "description": "Summarize findings from the web search.", "depends_on": ["t1_cli_search"]}
        # ]
        # logger.info(f"Planner generated mock plan with {len(mock_plan)} subtasks.")
        logger.info(f"Planner generated plan with {len(actual_plan)} subtasks.")

        # Check if this is a consciousness synthesis (direct answer from memory)
        if (len(actual_plan) == 1 and 
            actual_plan[0].get("consciousness_mode") and 
            "direct_answer_content" in actual_plan[0]):
            
            consciousness_answer = actual_plan[0]["direct_answer_content"]
            print("\n*** CONSCIOUSNESS SYNTHESIS COMPLETE ***")
            print(f"Generated directly from comprehensive memory without external tools:\n")
            print(consciousness_answer)
            print("\n>>> Status: CONSCIOUSNESS MODE - Memory Sufficiency Recognized")
            print("No external searches required - answer synthesized from existing knowledge.")
            print("--------------------------------------------")
            
            # Close resources and exit successfully
            await shared_memory.close()
            logger.info("Consciousness synthesis completed successfully.")
            return

        print("Dispatching subtasks to specialist agents...")
        assignments = await dispatcher.dispatch_subtasks(actual_plan) # Use the actual plan
        if assignments:
            subtasks_dispatched = actual_plan # Store the dispatched plan for monitoring
            logger.info(f"Successfully dispatched tasks. Assignments: {assignments}")
            print("Subtasks dispatched successfully.")
        else:
            logger.error("Dispatcher failed to assign tasks.")
            print("\nERROR: Failed to dispatch tasks (e.g., no available agents).")
            await shared_memory.close()
            return

    except Exception as e:
        logger.error(f"Error during planning or dispatch: {e}", exc_info=True)
        print(f"\nERROR: An error occurred during planning/dispatch: {e}")
        await shared_memory.close()
        return

    # 4. Monitor Task Completion
    if subtasks_dispatched:
        subtask_ids = [task["subtask_id"] for task in subtasks_dispatched]

        # --- ADDED: Clear previous task states before monitoring ---
        logger.info(f"Clearing previous states for tasks: {subtask_ids}")
        for task_id in subtask_ids:
            try:
                await shared_memory.delete(f"task:{task_id}:status")
                await shared_memory.delete(f"task:{task_id}:result")
                await shared_memory.delete(f"task:{task_id}:error")
            except Exception as del_err:
                 # Log error but continue, monitoring might still work if keys didn't exist
                 logger.warning(f"Error clearing state for task {task_id}: {del_err}")
        # ---------------------------------------------------------

        print(f"\nMonitoring completion of {len(subtask_ids)} subtasks...")
        final_statuses = await monitor_task_completion(shared_memory, subtask_ids)

        # 5. Synthesize and Display Final Result
        print("\n--- Task Execution Summary ---")
        all_successful = True
        task_results_for_synthesis = {}

        for task_id, outcome in final_statuses.items():
            print(f"  Task ID: {task_id}")
            print(f"    Status: {outcome['status']}")
            if outcome["result"]:
                print(f"    Result: {str(outcome['result'])[:200]}...") # Truncate long results
                task_results_for_synthesis[task_id] = outcome['result'] # Store result for synthesis
            if outcome["error"]:
                print(f"    Error: {outcome['error']}")
                all_successful = False
                task_results_for_synthesis[task_id] = f"Error: {outcome['error']}" # Store error for synthesis context
            if outcome['status'] != 'completed':
                 all_successful = False
                 if task_id not in task_results_for_synthesis: # Store status if no result/error yet
                     task_results_for_synthesis[task_id] = f"Status: {outcome['status']}"

        print("-----------------------------")

        if all_successful:
            print("\nSynthesizing final output (including tool context)...")
            try:
                # Call PlannerAgent to synthesize the final output
                final_output = await planner_agent.synthesize_final_output(
                    goal=user_task,
                    subtasks=actual_plan, # Pass the original plan with suggested_tool
                    results=task_results_for_synthesis
                )
                print("\n--- Final Synthesized Output ---")
                print(final_output)
                print("--------------------------------")
                print("\nOverall Task Status: COMPLETED")

            except Exception as synth_err:
                logger.error(f"Error during final synthesis: {synth_err}", exc_info=True)
                print(f"\nERROR: Failed to synthesize final output: {synth_err}")
                print("\nOverall Task Status: COMPLETED (but synthesis failed)")

        else:
            print("\nOverall Task Status: FAILED or TIMEOUT (see details above)")
            # Optionally, still attempt synthesis with partial/error results?
            # print("\nAttempting synthesis based on partial results...")
            # final_output = await planner_agent.synthesize_final_output(...)
            # print(final_output)

    # 6. Cleanup
    logger.info("Closing shared memory connection.")
    await shared_memory.close()
    logger.info("CLI execution finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Nova SHIFT system with a natural language task."
    )
    parser.add_argument(
        "task_description", type=str, help="The high-level task for the AI swarm."
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(args.task_description))
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        logger.error(f"Unhandled exception in main execution: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")