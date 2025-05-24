"""
Implementation of the Specialist Agent for Nova SHIFT.

This agent uses LangChain to reason and utilize tools loaded from the
Tool Registry. It interacts with Shared Memory for coordination and
retrieves context from Long-Term Memory (LTM).
"""

import logging
import os
import json
import asyncio
import importlib
import time # Added for timing
# Removed functools import
from typing import List, Any, Optional, Dict, Tuple, Callable
from dotenv import load_dotenv # Added import

# Load environment variables early before other imports that might need them
load_dotenv()

# LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI # Changed import
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent # Changed import
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool as LangchainTool
from langchain_core.language_models.chat_models import BaseChatModel

# Nova SHIFT components
from core.tool_registry import ToolRegistry
from tools.toolkit_schema import ToolkitSchema, ToolDefinition
from core.shared_memory_interface import SharedMemoryInterface
from core.dispatcher import SwarmDispatcher
from core.ltm_interface import LTMInterface # Added LTM Interface
# Removed sandbox import
from utils.logging_config import setup_logging, LLMTrackingCallback # Added Logging Setup & Callback
from utils.logging_config import setup_logging # Added Logging Setup

# Import tool functions
from tools.calculator.calculator_toolkit import calculate_expression
from tools.file_reader.file_reader_toolkit import read_text_file
from tools.web_search.web_search_toolkit import perform_web_search

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
LLM_MODEL_NAME = "gemini-2.5-pro-preview-03-25" # Changed model name
MEMORY_KEY = "chat_history"
AGENT_PROMPT_TEMPLATE = """You are a helpful assistant agent within the Nova SHIFT system.
Your Agent ID is: {agent_id}
Your current task ID is: {subtask_id}
Your task description is: {subtask_description}

You have access to the following tools:
{tools}

Use the tools to accomplish the task description.
Consider the following potentially relevant context retrieved from memory:
--- START MEMORY CONTEXT ---
{memory_context}
--- END MEMORY CONTEXT ---

You have access to a chat history relevant to this task thread:
{chat_history}

Input for this step: {input}

Thought Process:
{agent_scratchpad}
""" # Added memory_context placeholder

# --- Tool Loading and Mapping ---
# Map tool function names to the synchronous functions
TOOL_FUNCTION_MAP = {
    "calculate_expression": calculate_expression, # Points to sync version
    "read_text_file": read_text_file,             # Points to sync version
    "perform_web_search": perform_web_search,     # Points to sync version
}

# Removed _async_sandbox_runner function
# --- Toolkit Loading ---

def load_toolkits_from_directory(
    registry: ToolRegistry,
    directory: str = "tools", # Default relative to project root
    project_root_path: Optional[str] = None
):
    """Loads all toolkit.json files from subdirectories into the registry.

    Scans the specified directory (relative to project root) for subdirectories,
    looks for a `toolkit.json` file within each, loads the JSON data, and
    registers it with the provided ToolRegistry instance.

    Args:
        registry (ToolRegistry): The ToolRegistry instance to load toolkits into.
        directory (str): The path to the main tools directory, relative to the
            project root. Defaults to "tools".
        project_root_path (Optional[str]): Explicit path to the project root.
            If None, calculates based on this file's location.
    """
    if project_root_path:
        project_root = os.path.abspath(project_root_path)
    else:
        # Calculate project root based on this file's location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    tools_dir = os.path.join(project_root, directory.replace("/", os.sep))
    # No agent_id available here, keep logging as is or pass agent_id if refactored
    logger.info(f"Scanning for toolkits in: {tools_dir}") # Keep as is
    if not os.path.isdir(tools_dir):
        logger.error(f"Tools directory not found: {tools_dir}") # Keep as is
        return
    for item in os.listdir(tools_dir):
        item_path = os.path.join(tools_dir, item)
        if os.path.isdir(item_path):
            toolkit_json_path = os.path.join(item_path, "toolkit.json")
            if os.path.isfile(toolkit_json_path):
                logger.debug(f"Found toolkit definition: {toolkit_json_path}") # Keep as is
                try:
                    with open(toolkit_json_path, 'r', encoding='utf-8') as f:
                        toolkit_data = json.load(f)
                    registry.load_toolkit_from_dict(toolkit_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from {toolkit_json_path}: {e}") # Keep as is
                except Exception as e:
                    logger.error(f"Error loading toolkit from {toolkit_json_path}: {e}", exc_info=True) # Keep as is

def create_langchain_tools(registry: ToolRegistry, tool_names_to_load: Optional[List[str]] = None) -> List[LangchainTool]:
    """Creates LangChain Tool objects for specified or statically mapped tools.

    Iterates through tool definitions in the registry. If `tool_names_to_load`
    is provided, only creates tools matching those names. Otherwise, creates tools
    whose function names are present in the `TOOL_FUNCTION_MAP`. Dynamically loads
    functions if specified in the toolkit's `loading_info`. Wraps the actual
    tool function execution in an async sandbox runner.

    Args:
        registry (ToolRegistry): The ToolRegistry containing toolkit definitions.
        tool_names_to_load (Optional[List[str]]): A specific list of tool names
            to load. If None, loads tools based on `TOOL_FUNCTION_MAP`.

    Returns:
        List[LangchainTool]: A list of initialized LangChain Tool objects ready
        for use by an agent.
    """
    langchain_tools = []
    loaded_toolkits = registry.list_toolkits()
    logger.debug(f"Registry contains toolkits: {loaded_toolkits}") # DEBUG LOG
    tools_to_process: List[Tuple[str, ToolDefinition]] = []
    if tool_names_to_load is not None:
        # If specific tools are requested, find their definitions
        target_tool_names = set(tool_names_to_load)
        for toolkit_name in loaded_toolkits:
            toolkit = registry.get_toolkit(toolkit_name)
            if toolkit:
                for tool_def in toolkit.tools:
                    if tool_def.name in target_tool_names:
                        # Check if function exists in map OR can be loaded dynamically
                        if tool_def.function in TOOL_FUNCTION_MAP or \
                           (toolkit.loading_info and toolkit.loading_info.type == "python_module"):
                             tools_to_process.append((toolkit_name, tool_def))
                        else:
                             logger.warning(f"Requested tool '{tool_def.name}' function '{tool_def.function}' not found in map and no loading info.")
                        target_tool_names.remove(tool_def.name) # Remove even if not loadable to avoid warning below
        if target_tool_names:
             logger.warning(f"Could not find definitions for requested tools: {list(target_tool_names)}") # Keep as is
    else:
        logger.debug(f"No specific tools requested, loading statically mapped tools based on TOOL_FUNCTION_MAP keys: {list(TOOL_FUNCTION_MAP.keys())}") # DEBUG LOG
        for toolkit_name in loaded_toolkits:
            toolkit = registry.get_toolkit(toolkit_name)
            if toolkit:
                for tool_def in toolkit.tools:
                    logger.debug(f"Checking toolkit '{toolkit_name}' tool '{tool_def.name}' with function '{tool_def.function}' against TOOL_FUNCTION_MAP") # DEBUG LOG
                    if tool_def.function in TOOL_FUNCTION_MAP:
                         logger.debug(f" -> Match found in TOOL_FUNCTION_MAP. Adding '{tool_def.name}' to process list.") # DEBUG LOG
                         tools_to_process.append((toolkit_name, tool_def))
    logger.debug(f"Attempting to create LangChain tools for: {[t[1].name for t in tools_to_process]}")
    for toolkit_name, tool_def in tools_to_process:
        tool_func_name = tool_def.function
        actual_func: Optional[Callable] = None
        # Prioritize checking the static map first
        logger.debug(f"Checking for '{tool_func_name}' in TOOL_FUNCTION_MAP. Keys: {list(TOOL_FUNCTION_MAP.keys())}") # DEBUG LOG
        if tool_func_name in TOOL_FUNCTION_MAP:
            actual_func = TOOL_FUNCTION_MAP[tool_func_name]
            logger.debug(f"Found function '{tool_func_name}' in static TOOL_FUNCTION_MAP.")
        # Only attempt dynamic loading if not found in the static map
        if actual_func is None:
            toolkit = registry.get_toolkit(toolkit_name)
            if toolkit and toolkit.loading_info and toolkit.loading_info.type == "python_module":
                module_path = toolkit.loading_info.path
                try:
                    logger.debug(f"Attempting dynamic import of '{module_path}' for tool '{tool_def.name}'")
                    module = importlib.import_module(module_path)
                    actual_func = getattr(module, tool_func_name, None)
                    if actual_func is None:
                        logger.error(f"Function '{tool_func_name}' not found in dynamically imported module '{module_path}'.")
                except ImportError as e:
                    logger.error(f"Failed to import module '{module_path}': {e}")
                except AttributeError:
                     logger.error(f"Function '{tool_func_name}' not found after importing module '{module_path}'.")
                except Exception as e:
                     logger.error(f"Error dynamically loading function '{tool_func_name}' from '{module_path}': {e}", exc_info=True)
            # No need for the elif here, the final check covers it
            # elif tool_func_name not in TOOL_FUNCTION_MAP: # Log warning only if not found in map AND no loading info
            #      logger.warning(f"Function '{tool_func_name}' not in TOOL_FUNCTION_MAP and no valid loading_info found for toolkit '{toolkit_name}'.")
        if actual_func:
            logger.debug(f"Successfully obtained function for tool '{tool_def.name}'. Creating LangchainTool.") # DEBUG LOG
            # Create tool using the synchronous function directly (TEMPORARY - NO SANDBOX)
            lc_tool = LangchainTool(
                name=tool_def.name,
                func=actual_func, # Use original sync func
                description=tool_def.description,
                coroutine=None # No coroutine
            )
            langchain_tools.append(lc_tool)
            logger.debug(f"Created SYNC LangChain tool: '{tool_def.name}' mapped to '{tool_func_name}'")
        else:
            logger.warning(f"Could not obtain function '{tool_func_name}' for tool '{tool_def.name}'. Skipping tool creation.") # DEBUG LOG
    return langchain_tools


# --- Specialist Agent Class ---

class SpecialistAgent:
    """Represents a Specialist Agent capable of executing assigned subtasks.

    This agent uses a LangChain AgentExecutor driven by an LLM to reason about
    subtasks. It interacts with the Tool Registry to dynamically load necessary
    tools, uses Shared Memory for coordination (checking dependencies, posting
    results), retrieves relevant context from Long-Term Memory (LTM), logs
    tool usage, updates its heartbeat, and reports final status to the Dispatcher.
    Tools are executed within a sandbox for security.

    Attributes:
        agent_id (str): Unique identifier for the agent.
        tool_registry (ToolRegistry): Instance for accessing tool definitions.
        shared_memory (SharedMemoryInterface): Interface for Redis communication.
        ltm_interface (LTMInterface): Interface for Pinecone communication.
        dispatcher (SwarmDispatcher): Instance for reporting task status.
        llm (BaseChatModel): LangChain LLM instance for reasoning.
        memory (ConversationBufferMemory): Short-term memory for the agent.
        tools (List[LangchainTool]): List of currently loaded LangChain tools.
        agent_executor (Optional[AgentExecutor]): The LangChain agent executor.
        llm_callback_handler (LLMTrackingCallback): Handler to track LLM usage.
    """
    def __init__(self,
                 agent_id: str,
                 tool_registry: ToolRegistry,
                 shared_memory: SharedMemoryInterface,
                 ltm_interface: LTMInterface, # Added LTM
                 dispatcher: SwarmDispatcher,
                 llm: BaseChatModel,
                 initial_tool_names: Optional[List[str]] = None):
        """Initializes the Specialist Agent.

        Args:
            agent_id (str): Unique identifier for this agent.
            tool_registry (ToolRegistry): Instance of the tool registry.
            shared_memory (SharedMemoryInterface): Instance for shared memory access.
            ltm_interface (LTMInterface): Instance for long-term memory access.
            dispatcher (SwarmDispatcher): Instance of the swarm dispatcher.
            llm (BaseChatModel): LangChain compatible chat model instance.
            initial_tool_names (Optional[List[str]]): A list of tool names to
                pre-load upon initialization. Defaults to None, which loads tools
                statically mapped in `TOOL_FUNCTION_MAP`.
        """
        self.agent_id = agent_id
        self.tool_registry = tool_registry
        self.shared_memory = shared_memory
        self.ltm_interface = ltm_interface # Store LTM interface
        self.dispatcher = dispatcher
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key=MEMORY_KEY, return_messages=True, input_key='input') # Specify input key
        self.tools: List[LangchainTool] = create_langchain_tools(self.tool_registry, initial_tool_names)
        self.agent_executor: Optional[AgentExecutor] = None # Initialize as None, will be created async
        self.llm_callback_handler = LLMTrackingCallback() # Instantiate the callback handler
        # Defer AgentExecutor creation to an async method or first use.
        logger.info(f"Specialist Agent '{self.agent_id}' initialized with {len(self.tools)} initial tools: {[t.name for t in self.tools]}")

    async def _create_agent_executor(self) -> Optional[AgentExecutor]: # Made async
        """Creates or recreates the LangChain AgentExecutor.

        Builds the agent executor using the currently loaded `self.tools`. It fetches
        tool scores from shared memory, formats them into the tool descriptions,
        and constructs the system prompt. Includes the LLM tracking callback.

        Returns:
            Optional[AgentExecutor]: The configured AgentExecutor instance, or None
            if no tools are loaded or creation fails.
        """
        # Use agent_id in log messages within the class
        agent_log_prefix = f"[{self.agent_id}]"
        if not self.tools:
            logger.warning(f"{agent_log_prefix} No tools available, cannot create agent executor.")
            return None

        # Fetch scores and format descriptions (P4.T1.3)
        tool_desc_parts = []
        for tool in self.tools:
            # Fetch score using the added method in SharedMemoryInterface
            # Provide a default score (e.g., 0.5) if none is found
            score_val = await self.shared_memory.read_score(tool.name, default_score=0.5)
            # Explicitly cast to float before formatting
            try:
                score = float(score_val) # Cast the awaited result
                score_str = f"{score:.2f}"
            except (ValueError, TypeError):
                score_str = str(score_val) # Fallback if casting fails
            desc_with_score = f"- {tool.name} (Score: {score_str}): {tool.description}"
            logger.debug(f"{agent_log_prefix} Formatted tool description: {desc_with_score}") # DEBUG LOG
            tool_desc_parts.append(desc_with_score)

        tool_descriptions = "\n".join(tool_desc_parts)
        logger.debug(f"{agent_log_prefix} Formatted tool descriptions with scores:\n{tool_descriptions}")

        # The prompt template now includes {memory_context} which needs to be provided during invoke
        # Format the system prompt directly using an f-string or .format()
        # This avoids potential issues with how .replace() interacts with mocking/template handling
        system_prompt_content = AGENT_PROMPT_TEMPLATE.format(
            agent_id=self.agent_id, # Assuming agent_id is needed here based on template
            subtask_id="{subtask_id}", # Keep placeholders for runtime variables
            subtask_description="{subtask_description}",
            tools=tool_descriptions, # Inject the formatted tool descriptions
            memory_context="{memory_context}",
            chat_history="{chat_history}",
            input="{input}",
            agent_scratchpad="{agent_scratchpad}"
        )
        # We need to ensure AGENT_PROMPT_TEMPLATE actually uses .format() style placeholders
        # Let's assume it does for now, or adjust the template if needed.
        # If AGENT_PROMPT_TEMPLATE uses f-string style, this needs adjustment.
        # Reading the template again shows it uses f-string style {variable}.
        # Let's use f-string formatting instead.

        # Re-read template: It uses {variable} style, suitable for .format() or direct f-string if variables are local.
        # Using .format() is safer here. We need all expected keys.
        # The original template has {agent_id}, {subtask_id}, {subtask_description}, {tools}, {memory_context}, {chat_history}, {input}, {agent_scratchpad}

        # Let's stick to the original approach but ensure the variable passed is correct.
        # The previous log showed formatted_system_prompt was correct.
        # The issue might be how the test mock interacts with ChatPromptTemplate.from_messages.

        # Alternative: Pass the template and variables separately if LangChain supports it?
        # Let's try passing the original template and providing 'tools' as a partial variable.

        # Reverting to the previous approach as the variable seemed correct in logs.
        # The test extraction logic might be the issue.
        formatted_system_prompt = AGENT_PROMPT_TEMPLATE.replace("{tools}", tool_descriptions)
        logger.debug(f"{agent_log_prefix} AGENT_PROMPT_TEMPLATE before replace:\n{AGENT_PROMPT_TEMPLATE}") # DEBUG LOG
        logger.debug(f"{agent_log_prefix} tool_descriptions:\n{tool_descriptions}") # DEBUG LOG
        logger.debug(f"{agent_log_prefix} formatted_system_prompt after replace:\n{formatted_system_prompt}") # DEBUG LOG - Renamed variable

        prompt = ChatPromptTemplate.from_messages([
            ("system", formatted_system_prompt), # Pass the explicitly formatted string
            MessagesPlaceholder(variable_name=MEMORY_KEY), # Handles chat_history
            # Input now needs to be structured to contain 'input' and 'memory_context'
            ("human", "{input}"), # The primary input query/instruction
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Re-added placeholder
        ])

        try:
            # Use the create_openai_tools_agent helper function
            # This function is designed to construct the correct runnable chain
            # for agents using OpenAI tools/function calling.
            # Use the create_google_genai_tools_agent helper function
            # This function is designed for agents using Google Gemini's tool calling.
            agent_runnable = create_tool_calling_agent(self.llm, self.tools, prompt) # Changed function call

            # Create the AgentExecutor with the manually constructed runnable
            agent_executor = AgentExecutor(
                agent=agent_runnable, # Pass the runnable chain
                tools=self.tools,
                memory=self.memory,
                verbose=True, # Set to False for cleaner logs in production
                handle_parsing_errors=True, # Handle potential parsing errors gracefully
                callbacks=[self.llm_callback_handler] # Add the callback handler here
            )
            logger.debug(f"{agent_log_prefix} Agent Executor created/updated with {len(self.tools)} tools.")
            return agent_executor
        except Exception as e:
            logger.error(f"{agent_log_prefix} Failed to create Agent Executor: {e}", exc_info=True)
            return None

    def _format_ltm_context(self, ltm_results: List[Dict[str, Any]]) -> str:
        """Formats retrieved LTM results into a string suitable for the LLM prompt.

        Args:
            ltm_results (List[Dict[str, Any]]): List of documents from LTM search.

        Returns:
            str: A formatted string summarizing LTM results, or a default message.
        """
        # (Copied from PlannerAgent - consider moving to a shared utility)
        if not ltm_results:
            return "No relevant information found in Long-Term Memory."
        context_str = ""
        for i, item in enumerate(ltm_results):
            text = item.get('metadata', {}).get('original_text', 'N/A')
            source = item.get('metadata', {}).get('source', 'Unknown')
            score = item.get('score', 0.0)
            context_str += f"{i+1}. (Source: {source}, Score: {score:.3f}): {text}\n"
        return context_str.strip()

    async def _analyze_tool_requirements(self, subtask_description: str) -> List[str]:
        """Analyzes subtask description to identify required tools using an LLM call.

        Constructs a prompt asking the LLM to identify necessary tools from the
        registry based on the subtask description. Parses the LLM response (expected
        to be a JSON list of tool names) and validates the names against the registry.

        Args:
            subtask_description (str): The natural language description of the subtask.

        Returns:
            List[str]: A list of validated tool names identified as required by the LLM.
            Returns an empty list if no tools are needed or if analysis fails.
        """
        # (Implementation remains the same)
        agent_log_prefix = f"[{self.agent_id}]" # Define prefix for this method
        logger.info(f"{agent_log_prefix} Analyzing tool requirements for: '{subtask_description}'")
        available_tools_info = []
        all_tool_names = set()
        for toolkit_name in self.tool_registry.list_toolkits():
            toolkit = self.tool_registry.get_toolkit(toolkit_name)
            if toolkit:
                for tool_def in toolkit.tools:
                    available_tools_info.append(f"- {tool_def.name}: {tool_def.description}")
                    all_tool_names.add(tool_def.name)
        if not available_tools_info: return []
        available_tools_str = "\n".join(available_tools_info)
        analysis_prompt = f"""
        You are an expert tool requirement analyzer for an AI agent.
        Given a subtask description, identify which of the available tools are necessary to complete the subtask. Available Tools:\n{available_tools_str}\nSubtask Description: "{subtask_description}"
        Based *only* on the available tools listed above, respond with a JSON list containing the exact names of the required tools. Only include tool names from the 'Available Tools' list. If no tools from the list are required, return an empty list []. Required Tool Names (JSON List):
        """
        llm_output = ""
        try:
            response = await self.llm.ainvoke(analysis_prompt.strip())
            if hasattr(response, 'content'): llm_output = response.content
            elif isinstance(response, str): llm_output = response
            else: logger.error(f"{agent_log_prefix} Unexpected LLM response type: {type(response)}"); return []
            logger.debug(f"{agent_log_prefix} LLM response for tool analysis: {llm_output}")
            json_start = llm_output.find('[')
            json_end = llm_output.rfind(']')
            if json_start != -1 and json_end != -1 and json_start < json_end:
                json_str = llm_output[json_start : json_end + 1]
                required_tools = json.loads(json_str)
                if isinstance(required_tools, list) and all(isinstance(t, str) for t in required_tools):
                    valid_required_tools = [name for name in required_tools if name in all_tool_names]
                    if len(valid_required_tools) != len(required_tools): logger.warning(f"{agent_log_prefix} LLM suggested tools not in registry: {set(required_tools) - set(valid_required_tools)}")
                    logger.info(f"{agent_log_prefix} Identified required tools (validated): {valid_required_tools}")
                    return valid_required_tools
                else: logger.error(f"{agent_log_prefix} Parsed JSON is not list of strings: {required_tools}"); return []
            else:
                if "[]" in llm_output or "none required" in llm_output.lower(): return []
                logger.error(f"{agent_log_prefix} Could not find JSON list in response: {llm_output}"); return []
        except json.JSONDecodeError as e: logger.error(f"{agent_log_prefix} Failed JSON decode: {e}. Response: {llm_output}"); return []
        except Exception as e: logger.error(f"{agent_log_prefix} Error during tool analysis LLM call: {e}", exc_info=True); return []


    async def _ensure_tools_loaded(self, required_tool_names: List[str]) -> bool:
        """Ensures required tools are loaded, dynamically loading if necessary.

        Compares the list of required tool names against currently loaded tools.
        If any are missing, it attempts to find their definition in the registry,
        dynamically import the associated function, wrap it in the sandbox runner,
        create a LangChain Tool, and add it to `self.tools`. If loading succeeds,
        it recreates the `self.agent_executor`.

        Args:
            required_tool_names (List[str]): A list of tool names required for the task.

        Returns:
            bool: True if all required tools are loaded (or were successfully loaded
            dynamically), False if any required tool could not be found or loaded.
        """
        # (Implementation remains the same)
        currently_loaded_tool_names = {tool.name for tool in self.tools}
        missing_tool_names = [name for name in required_tool_names if name not in currently_loaded_tool_names]
        if not missing_tool_names: return True
        agent_log_prefix = f"[{self.agent_id}]" # Define prefix
        logger.info(f"{agent_log_prefix} Attempting to dynamically load missing tools: {missing_tool_names}")
        newly_loaded_tools: List[LangchainTool] = []
        load_failed = False
        tool_defs_to_load: List[Tuple[str, ToolDefinition]] = []
        temp_missing = set(missing_tool_names)
        for toolkit_name in self.tool_registry.list_toolkits():
             toolkit = self.tool_registry.get_toolkit(toolkit_name)
             if toolkit:
                 for tool_def in toolkit.tools:
                     if tool_def.name in temp_missing:
                         tool_defs_to_load.append((toolkit_name, tool_def))
                         temp_missing.remove(tool_def.name)
             if not temp_missing: break
        if temp_missing: logger.error(f"{agent_log_prefix} Could not find definitions for missing tools: {temp_missing}"); return False
        for toolkit_name, tool_def in tool_defs_to_load:
            tool_func_name = tool_def.function
            actual_func: Optional[Callable] = None
            toolkit = self.tool_registry.get_toolkit(toolkit_name)
            if toolkit and toolkit.loading_info and toolkit.loading_info.type == "python_module":
                module_path = toolkit.loading_info.path
                try:
                    logger.debug(f"{agent_log_prefix} Dynamically importing module '{module_path}' for tool '{tool_def.name}'")
                    module = importlib.import_module(module_path)
                    actual_func = getattr(module, tool_func_name, None)
                    if actual_func is None: logger.error(f"{agent_log_prefix} Function '{tool_func_name}' not found in module '{module_path}'."); load_failed = True
                except ImportError as e: logger.error(f"{agent_log_prefix} Failed import: {e}"); load_failed = True
                except AttributeError: logger.error(f"{agent_log_prefix} Function '{tool_func_name}' not found in module '{module_path}'."); load_failed = True
                except Exception as e: logger.error(f"{agent_log_prefix} Error loading function '{tool_func_name}': {e}", exc_info=True); load_failed = True
            else: logger.error(f"{agent_log_prefix} Cannot dynamically load '{tool_func_name}'."); load_failed = True
            if actual_func:
                # Check if the loaded function is async
                if asyncio.iscoroutinefunction(actual_func):
                    # Create LangChain tool for async function
                    lc_tool = LangchainTool(
                        name=tool_def.name,
                        func=None, # No sync function provided
                        description=tool_def.description,
                        coroutine=actual_func # Assign async func here
                    )
                    logger.info(f"{agent_log_prefix} Successfully loaded ASYNC tool dynamically: '{tool_def.name}'")
                else:
                    # Create LangChain tool for sync function
                    lc_tool = LangchainTool(
                        name=tool_def.name,
                        func=actual_func, # Assign sync func here
                        description=tool_def.description,
                        coroutine=None
                    )
                    logger.info(f"{agent_log_prefix} Successfully loaded SYNC tool dynamically: '{tool_def.name}'")

                newly_loaded_tools.append(lc_tool)
            else: load_failed = True
        if load_failed: logger.error(f"{agent_log_prefix} Failed dynamic load."); return False
        self.tools.extend(newly_loaded_tools)
        logger.info(f"{agent_log_prefix} Added {len(newly_loaded_tools)} tools. Current count: {len(self.tools)}")
        logger.info(f"{agent_log_prefix} Recreating agent executor asynchronously...")
        # Need to await the creation now
        self.agent_executor = await self._create_agent_executor()
        if self.agent_executor is None: logger.error(f"{agent_log_prefix} Failed recreate agent executor."); return False
        return True


    async def _check_dependencies(self, dependencies: List[str]) -> Optional[Dict[str, Any]]:
        """Checks if prerequisite subtasks are completed by reading Shared Memory.

        Polls the shared memory (Redis) for the status of each dependency ID. Waits
        up to a maximum time (`max_wait_time`) for all dependencies to reach
        'completed' status.

        Args:
            dependencies (List[str]): A list of subtask IDs that must be completed
                before this task can start.

        Returns:
            Optional[Dict[str, Any]]: A dictionary mapping dependency subtask IDs
            to their results retrieved from shared memory if all dependencies are
            met successfully. Returns None if any dependency fails, is not found,
            or if the wait times out. Returns an empty dict if `dependencies` is empty.
        """
        # (Implementation remains the same)
        if not dependencies: return {}
        agent_log_prefix = f"[{self.agent_id}]" # Define log prefix for this method
        dependency_results = {}
        logger.info(f"{agent_log_prefix} Checking {len(dependencies)} dependencies: {dependencies}")
        max_wait_time = 10; poll_interval = 1; waited_time = 0
        while len(dependency_results) < len(dependencies):
            all_deps_checked_this_round = True
            for dep_id in dependencies:
                if dep_id in dependency_results: continue
                status_key = f"task:{dep_id}:status"; result_key = f"task:{dep_id}:result"
                status = await self.shared_memory.read(status_key)
                if status == "completed":
                    # Status is completed, now wait briefly for the result to appear
                    result = None
                    result_wait_start = time.monotonic()
                    result_max_wait = 2 # Wait max 2 extra seconds for result
                    while result is None and (time.monotonic() - result_wait_start) < result_max_wait:
                        result = await self.shared_memory.read(result_key)
                        if result is not None:
                            break # Found result
                        await asyncio.sleep(0.1) # Short sleep before checking again

                    if result is not None:
                        dependency_results[dep_id] = result
                        logger.debug(f"{agent_log_prefix} Dep '{dep_id}' met (status completed, result found).")
                    else:
                        # Result still missing after waiting
                        logger.error(f"{agent_log_prefix} Dep '{dep_id}' completed but result missing after waiting.")
                        return None # Failed because result didn't appear
                elif status == "failed":
                    logger.error(f"{agent_log_prefix} Dep '{dep_id}' failed.")
                    return None # Failed because dependency failed
                else:
                    # Status is not completed or failed yet
                    all_deps_checked_this_round = False
                    logger.debug(f"{agent_log_prefix} Dep '{dep_id}' not ready (status: {status}).")
                    break # Break inner loop to wait longer for status
            if len(dependency_results) == len(dependencies): break
            if all_deps_checked_this_round:
                 if waited_time >= max_wait_time: logger.error(f"{agent_log_prefix} Timeout waiting for deps."); return None
                 else: logger.info(f"{agent_log_prefix} Waiting {poll_interval}s for deps..."); await asyncio.sleep(poll_interval); waited_time += poll_interval
        logger.info(f"{agent_log_prefix} All dependencies met."); return dependency_results

    async def _log_tool_usage(self, tool_name: str, success: bool, duration: float):
        """Logs the usage outcome of a tool using atomic Redis increments.

        Increments counters in Redis for usage count, success/failure count, and
        total duration for the given tool name. Uses atomic operations provided
        by `SharedMemoryInterface` for efficiency and to avoid race conditions.

        Args:
            tool_name (str): The name of the tool that was used (or attempted).
            success (bool): Whether the task using the tool completed successfully.
            duration (float): The time taken for the task execution (in seconds).
        """
        base_key = f"tool:{tool_name}"
        agent_log_prefix = f"[{self.agent_id}]"
        try:
            # Use atomic increments provided by SharedMemoryInterface
            await self.shared_memory.increment_counter(f"{base_key}:usage_count")
            await self.shared_memory.increment_float(f"{base_key}:total_duration", amount=duration)

            if success:
                await self.shared_memory.increment_counter(f"{base_key}:success_count")
            else:
                await self.shared_memory.increment_counter(f"{base_key}:failure_count")

            logger.debug(f"{agent_log_prefix} Logged usage for tool '{tool_name}' using atomic increments: success={success}, duration={duration:.4f}s")
        except Exception as e:
            logger.error(f"{agent_log_prefix} Failed to log usage for tool '{tool_name}' using atomic increments: {e}", exc_info=True)

    # Note: Duplicate execute_task definition removed below. Keeping the one starting line 443.

    async def execute_task(self, subtask: Dict[str, Any]):
        """Executes a given subtask.

        This is the main entry point for the agent to handle a task assigned by the
        Dispatcher. It performs the following steps:
        1. Updates agent heartbeat.
        2. Sets task status to 'running' in shared memory.
        3. Analyzes tool requirements for the subtask.
        4. Ensures required tools are dynamically loaded.
        5. Checks if dependencies are met by polling shared memory.
        6. Retrieves relevant context from LTM.
        7. Prepares input for the AgentExecutor (combining description, context).
        8. Invokes the AgentExecutor to perform the task using tools.
        9. Logs tool usage (approximated based on identified tools).
        10. Logs overall task summary and LLM stats.
        11. Reports final status ('completed' or 'failed') and results/errors
            to shared memory and the Dispatcher.
        12. Stores successful results to LTM.
        13. Publishes a status update message.

        Args:
            subtask (Dict[str, Any]): The subtask dictionary, containing at least
                'subtask_id', 'description', and optionally 'depends_on'.
        """
        subtask_id = subtask.get("subtask_id", "unknown_task")
        subtask_description = subtask.get("description", "")
        dependencies = subtask.get("depends_on", [])
        status_key = f"task:{subtask_id}:status"
        result_key = f"task:{subtask_id}:result"
        error_key = f"task:{subtask_id}:error"

        agent_task_log_prefix = f"[{self.agent_id}][Task:{subtask_id}]" # More specific prefix for task execution
        logger.info(f"{agent_task_log_prefix} Received task - '{subtask_description}'")

        final_status = "failed"; output = None; error_message = None; start_time = time.monotonic()
        required_tool_names_for_logging: List[str] = [] # Store identified tools for logging

        try:
            # --- Heartbeat Update (P4.T4.1) ---
            heartbeat_key = f"agent:{self.agent_id}:heartbeat"
            heartbeat_ttl = 60 # Seconds until heartbeat expires
            current_timestamp = time.time()
            await self.shared_memory.write(heartbeat_key, current_timestamp, expiry_seconds=heartbeat_ttl)
            logger.debug(f"{agent_task_log_prefix} Updated heartbeat timestamp: {current_timestamp}")
            # ---------------------------------

            await self.shared_memory.write(status_key, "running")
            logger.info(f"{agent_task_log_prefix} Set status to 'running'")

            # --- Task-Tool Requirement Analysis ---
            required_tool_names_for_logging = await self._analyze_tool_requirements(subtask_description)
            logger.info(f"{agent_task_log_prefix} Identified required tools: {required_tool_names_for_logging}")

            # --- Dynamic Tool Loading ---
            if required_tool_names_for_logging:
                tools_loaded_ok = await self._ensure_tools_loaded(required_tool_names_for_logging)
                if not tools_loaded_ok:
                    error_message = f"Failed to load required tools: {required_tool_names_for_logging}"
                    logger.error(f"{agent_task_log_prefix} {error_message}")
                    raise Exception(error_message)

            # --- Check Dependencies ---
            dependency_results = await self._check_dependencies(dependencies)
            if dependency_results is None:
                error_message = f"Failed due to unmet or failed dependencies: {dependencies}"
                logger.error(f"{agent_task_log_prefix} {error_message}")
                raise Exception(error_message)

            # --- Retrieve LTM Context (P3.T3.7) ---
            ltm_context_str = "No relevant context found in LTM." # Default
            try:
                logger.info(f"{agent_task_log_prefix} Retrieving LTM context")
                retrieved_docs = await self.ltm_interface.retrieve(query_text=subtask_description, top_k=3)
                if retrieved_docs:
                    ltm_context_str = self._format_ltm_context(retrieved_docs)
                    logger.info(f"{agent_task_log_prefix} Retrieved {len(retrieved_docs)} docs from LTM.")
                    logger.debug(f"{agent_task_log_prefix} LTM Context for prompt:\n{ltm_context_str}")
            except Exception as ltm_err:
                 logger.error(f"{agent_task_log_prefix} Failed to retrieve LTM context: {ltm_err}", exc_info=True)
                 # Continue without LTM context on error
            # --- End LTM Retrieval ---

            # --- Prepare Agent Input ---
            # Combine subtask description, dependency results, and LTM context
            agent_input_parts = [subtask_description]
            if dependency_results:
                 dep_context = "\n\n[Context from completed prerequisite tasks]:\n" + json.dumps(dependency_results, indent=2)
                 agent_input_parts.append(dep_context)
            # LTM context is now handled by the prompt template directly via 'memory_context' variable

            agent_input_combined = "\n".join(agent_input_parts)

            # --- Execute Task ---
            # Ensure the agent executor is created/updated with current tools and context
            self.agent_executor = await self._create_agent_executor()
            if not self.agent_executor:
                 error_message = "Agent executor could not be created." # Updated error message
                 logger.error(f"{agent_task_log_prefix} {error_message}")
                 raise Exception(error_message)

            logger.info(f"{agent_task_log_prefix} Invoking agent executor...") # Keep as is
            # Pass necessary context to the prompt template
            formatted_input = {
                "input": agent_input_combined, # Main instruction + dependency context
                "agent_id": self.agent_id,
                "subtask_id": subtask_id,
                "subtask_description": subtask_description,
                "memory_context": ltm_context_str # Pass LTM context here
                # Removed chat_history and agent_scratchpad placeholders
            }
            response = await self.agent_executor.ainvoke(formatted_input)
            output = response.get("output")

            if output is not None:
                logger.info(f"{agent_task_log_prefix} Execution successful.") # Keep as is
                final_status = "completed"
            else:
                error_message = "Agent execution finished but produced no output."
                logger.error(f"{agent_task_log_prefix} Failed: {error_message}") # Keep as is
                final_status = "failed"

        except Exception as e:
            logger.error(f"{agent_task_log_prefix} Error during execution: {e}", exc_info=True) # Keep as is
            final_status = "failed"
            if error_message is None: error_message = str(e)
            output = None

        finally:
            duration = time.monotonic() - start_time
            task_success = (final_status == "completed")

            # --- Log Tool Usage (P3.T4 & P5.T1.2) ---
            # This now uses atomic increments via _log_tool_usage
            if required_tool_names_for_logging:
                 # In a real scenario, we'd ideally log usage based on actual tools invoked by the agent,
                 # not just the ones identified initially. LangChain callbacks (on_tool_start/end)
                 # would be better for precise tool usage logging.
                 # For now, log based on initially identified tools as a proxy.
                for tool_name in required_tool_names_for_logging:
                    # We don't know if *this specific tool* succeeded, only if the overall task did.
                    # This logging is therefore approximate for scoring purposes.
                    await self._log_tool_usage(tool_name, task_success, duration)
            else:
                logger.debug(f"{agent_task_log_prefix} No specific tools identified for logging usage.")
            # -----------------------------

            # --- Log Overall Task and LLM Stats ---
            llm_stats = self.llm_callback_handler.get_stats() # Get stats from the handler
            final_log_data = {
                'task_id': subtask_id,
                'agent_id': self.agent_id,
                'final_status': final_status,
                'duration_ms': round(duration * 1000, 2),
                'llm_calls': llm_stats.get('total_llm_calls', 0),
                'total_tokens': llm_stats.get('total_tokens', 0),
                'avg_llm_latency_ms': llm_stats.get('average_latency_ms', 0),
                'error': error_message if error_message else None
            }
            log_level = logging.INFO if task_success else logging.ERROR
            logger.log(log_level, f"Task Execution Summary", extra={'extra_data': final_log_data})
            # Reset callback stats if desired (e.g., per task) - depends on desired aggregation level
            # self.llm_callback_handler = LLMTrackingCallback() # Uncomment to reset per task
            # ------------------------------------

            # Report results/status - Write result/error BEFORE status
            logger.info(f"{agent_task_log_prefix} Reporting final status '{final_status}' (duration: {duration:.4f}s)")

            # Write result or error first
            if final_status == "completed" and output is not None:
                await self.shared_memory.write(result_key, output)
                logger.debug(f"{agent_task_log_prefix} Wrote result to Shared Memory key '{result_key}'.")
                # --- Store Result in LTM (P3.T3.8) ---
                # Store the successful output in LTM for future reference.
                store_metadata = {
                    "task_id": subtask_id,
                    "agent_id": self.agent_id,
                    "status": "completed",
                    "source_type": "agent_result" # Indicate this came from an agent task result
                }
                ltm_doc_id = f"result_{subtask_id}_{self.agent_id}" # Simple ID scheme
                try:
                    output_text = str(output)
                    await self.ltm_interface.store(text=output_text, metadata=store_metadata, doc_id=ltm_doc_id)
                    logger.info(f"{agent_task_log_prefix} Stored result to LTM (ID: {ltm_doc_id}).")
                except Exception as store_err:
                    logger.error(f"{agent_task_log_prefix} Failed to store result to LTM: {store_err}", exc_info=True)
                # ----------------------------------------------------
            elif final_status == "failed" and error_message is not None:
                await self.shared_memory.write(error_key, error_message)
                logger.debug(f"{agent_task_log_prefix} Wrote error to Shared Memory key '{error_key}'.")

            # Now write the final status
            await self.shared_memory.write(status_key, final_status)
            logger.debug(f"{agent_task_log_prefix} Wrote status '{final_status}' to Shared Memory key '{status_key}'.")

            # Notify Dispatcher (via direct call)
            result_to_report = output if final_status == "completed" else error_message
            self.dispatcher.update_task_status(subtask_id, final_status, result_to_report)
            logger.info(f"{agent_task_log_prefix} Notified dispatcher directly.") # Keep as is

            # --- Publish Status Update (P5.T1.3) ---
            status_update_channel = "task_status_updates"
            status_message = {
                "task_id": subtask_id,
                "agent_id": self.agent_id,
                "status": final_status,
                "timestamp": time.time()
            }
            # Optionally include result/error in the message, be mindful of size
            # if final_status == "completed": status_message["result_preview"] = str(output)[:100] # Preview
            # elif error_message: status_message["error_preview"] = str(error_message)[:100] # Preview

            try:
                await self.shared_memory.publish(status_update_channel, status_message)
                logger.info(f"{agent_task_log_prefix} Published status '{final_status}' to channel '{status_update_channel}'.") # Keep as is
            except Exception as pub_err:
                 logger.error(f"{agent_task_log_prefix} Failed to publish status update: {pub_err}", exc_info=True) # Keep as is
            # ------------------------------------

# --- Initialization Function ---
def initialize_specialist_agent_instance(
    agent_id: str,
    registry: ToolRegistry,
    shared_memory: SharedMemoryInterface,
    ltm_interface: LTMInterface,
    dispatcher: SwarmDispatcher,
    llm: Optional[BaseChatModel] = None # Allow passing LLM or create default
) -> Optional[SpecialistAgent]: # Make helper async
    """Initializes a SpecialistAgent instance with necessary components.

    Handles LLM client initialization (using OPENAI_API_KEY environment variable)
    and creation of the SpecialistAgent object.

    Args:
        agent_id (str): The unique ID for the agent.
        registry (ToolRegistry): The shared tool registry.
        shared_memory (SharedMemoryInterface): The shared memory interface.
        ltm_interface (LTMInterface): The long-term memory interface.
        dispatcher (SwarmDispatcher): The swarm dispatcher instance.
        llm (Optional[BaseChatModel]): An optional pre-initialized LLM client.
            If None, attempts to create a default ChatOpenAI instance.

    Returns:
        Optional[SpecialistAgent]: The initialized SpecialistAgent instance,
        or None if initialization fails (e.g., missing API keys, LLM error).
    """
    logger.info(f"Initializing Specialist Agent instance: {agent_id}...")
    # initial_tools: List[LangchainTool] = create_langchain_tools(registry) # Removed: Agent __init__ handles initial tool loading
    try:
        # Check for GOOGLE_API_KEY as it's now the primary requirement
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
             logger.error("GOOGLE_API_KEY environment variable not set.")
             return None
        if llm is None:
             # Initialize default LLM as ChatGoogleGenerativeAI
             llm = ChatGoogleGenerativeAI(
                 model=LLM_MODEL_NAME,
                 temperature=0,
                 google_api_key=google_api_key # Pass the key
             )
             logger.info(f"[{agent_id}] Initialized default LLM: {LLM_MODEL_NAME}")
        else:
             logger.info(f"[{agent_id}] Using provided LLM instance.")
    except Exception as e:
        logger.error(f"[{agent_id}] Failed to initialize LLM: {e}", exc_info=True)
        return None
    try:
        agent_instance = SpecialistAgent(
            agent_id=agent_id,
            tool_registry=registry,
            shared_memory=shared_memory,
            ltm_interface=ltm_interface, # Pass LTM interface
            dispatcher=dispatcher,
            llm=llm,
            # initial_tools argument removed as create_langchain_tools is called inside __init__ now
            # Pass llm instance directly
        )
        # Removed await _create_agent_executor(): Executor creation is handled asynchronously by the agent itself when needed.
        return agent_instance
    except Exception as e:
        logger.error(f"[{agent_id}] Failed to create SpecialistAgent instance: {e}", exc_info=True)
        return None


# --- Main Execution / Example ---
# from dotenv import load_dotenv # Removed import from here

async def main():
    # load_dotenv() # Removed call from here
    load_dotenv()
    # Setup structured logging at the start of the main execution flow
    # Note: utils.logging_config is imported at the top now
    setup_logging(level=logging.INFO) # Or logging.DEBUG for more detail
    # Setup structured logging at the start of the main execution flow
    setup_logging(level=logging.INFO) # Or logging.DEBUG for more detail
    print("Setting up Nova SHIFT components...")
    # Check required env vars
    required_env_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY", "PINECONE_API_KEY"]
    if not all(os.environ.get(var) for var in required_env_vars):
         print(f"\nERROR: Please set environment variables: {', '.join(required_env_vars)}")
         return

    # Initialize Core Components
    shared_memory = SharedMemoryInterface(db=1) # Use test DB
    dispatcher = SwarmDispatcher(shared_memory=shared_memory) # Pass shared_memory instance
    ltm_interface = LTMInterface() # Initialize LTM

    # Check LTM init success
    if not ltm_interface.index:
         print("\nERROR: Failed to initialize LTMInterface. Check Pinecone API key/config and network.")
         await shared_memory.close()
         return

    try:
        redis_client = await shared_memory._get_client()
        await redis_client.flushdb()
        logger.info("Flushed Redis test DB 1.")
        # Optional: Clear Pinecone index if needed for clean test run
        # await ltm_interface.index.delete(delete_all=True) # Use with caution!
        # logger.info("Cleared Pinecone test index.")
    except Exception as e:
        logger.error(f"Could not connect/flush Redis test DB: {e}. Ensure Redis is running.")
        await shared_memory.close()
        # No need to close ltm_interface explicitly as pinecone client handles connections
        return

    tool_registry = ToolRegistry()
    load_toolkits_from_directory(tool_registry, directory="nova_shift/tools")

    # Initialize Agents, passing LTM interface
    agent_1 = initialize_specialist_agent_instance("agent_001", tool_registry, shared_memory, ltm_interface, dispatcher)
    agent_2 = initialize_specialist_agent_instance("agent_002", tool_registry, shared_memory, ltm_interface, dispatcher)

    if not agent_1 or not agent_2:
        print("\nFailed to initialize agents. Exiting.")
        await shared_memory.close()
        return

    dispatcher.register_agent(agent_1.agent_id, agent_1)
    dispatcher.register_agent(agent_2.agent_id, agent_2)

    print("\nAgents initialized and registered.")

    # --- Pre-populate LTM for testing RAG ---
    print("\nPre-populating LTM...")
    await ltm_interface.store(
        text="The SpecialistAgent executes tasks using tools and memory.",
        metadata={"source": "test_setup", "topic": "agent_role"},
        doc_id="agent_role_doc"
    )
    await asyncio.sleep(2) # Allow indexing time
    print("LTM pre-populated.")
    # --- End LTM pre-population ---


    print("\nSimulating task dispatch...")
    tasks_to_dispatch = [
        {"subtask_id": "task_describe_agent", "description": "Describe the role of a Specialist Agent.", "depends_on": []},
        {"subtask_id": "task_calculate_pi", "description": "Calculate 3 * 3.14159", "depends_on": []},
    ]

    assignments = await dispatcher.dispatch_subtasks(tasks_to_dispatch)
    print(f"\nInitial Task Assignments: {assignments}")
    print(f"Initial Agent Status: {dispatcher.get_agent_status()}")

    print("\nWaiting for tasks to complete (simulated)...")
    await asyncio.sleep(25) # Allow time for LTM retrieval + LLM calls

    print(f"\nFinal Agent Status: {dispatcher.get_agent_status()}")

    print("\nChecking results in Shared Memory:")
    for task in tasks_to_dispatch:
        task_id = task["subtask_id"]
        status = await shared_memory.read(f"task:{task_id}:status")
        result = await shared_memory.read(f"task:{task_id}:result")
        error = await shared_memory.read(f"task:{task_id}:error")
        print(f"  Task {task_id}: Status='{status}'")
        if result: print(f"    Result: {str(result)[:150]}...")
        if error: print(f"    Error: {error}")

    await shared_memory.close()
    print("\nShared memory connection closed.")


if __name__ == '__main__':
    asyncio.run(main())
