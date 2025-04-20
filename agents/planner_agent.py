"""
Planner Agent: Responsible for decomposing high-level user goals into actionable subtasks.
Retrieves relevant context from Long-Term Memory (LTM) to aid planning.
"""

import logging
from typing import Any, Dict, List, Optional
import json
import asyncio # Added for example usage

# Assuming LLM client setup is handled elsewhere
# Assuming Dispatcher and LTM interfaces are available
# from ..core.dispatcher import SwarmDispatcher
# from ..core.ltm_interface import LTMInterface
# from ..core.llm_client import get_llm_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Placeholder type hints
DispatcherInterface = Any
LTMInterface = Any
LLMClient = Any

class PlannerAgent:
    """Analyzes user goals, retrieves context, and decomposes goals into subtasks.

    This agent takes a high-level user goal, queries the Long-Term Memory (LTM)
    for relevant context, uses a Language Model (LLM) to break the goal down
    into a structured plan (list of subtasks with dependencies), and then sends
    this plan to the SwarmDispatcher for execution by Specialist Agents.

    Attributes:
        _dispatcher: An instance of the SwarmDispatcher interface.
        _ltm_interface: An instance of the LTMInterface.
        _llm_client: An LLM client instance for generating the plan.
    """

    def __init__(self,
                 dispatcher: DispatcherInterface,
                 ltm_interface: LTMInterface, # Added LTM interface
                 llm_client: LLMClient):
        """Initializes the PlannerAgent.

        Args:
            dispatcher (DispatcherInterface): An object implementing the dispatcher
                interface, used to send the generated subtask list.
            ltm_interface (LTMInterface): An object implementing the LTM interface,
                used to retrieve context relevant to the user goal.
            llm_client (LLMClient): An object capable of invoking a language model
                (e.g., via `ainvoke`) to perform the decomposition task.
        """
        self._dispatcher = dispatcher
        self._ltm_interface = ltm_interface # Store LTM interface
        self._llm_client = llm_client
        logger.info("PlannerAgent initialized.")

    def _format_ltm_context(self, ltm_results: List[Dict[str, Any]]) -> str:
        """Formats retrieved LTM results into a string suitable for an LLM prompt.

        Args:
            ltm_results (List[Dict[str, Any]]): A list of documents retrieved
                from LTM, typically containing 'metadata' and 'score'.

        Returns:
            str: A formatted string summarizing the LTM results, or a message
            indicating no relevant information was found.
        """
        if not ltm_results:
            return "No relevant information found in Long-Term Memory."

        context_str = "Relevant Information from Long-Term Memory:\n"
        for i, item in enumerate(ltm_results):
            text = item.get('metadata', {}).get('original_text', 'N/A')
            source = item.get('metadata', {}).get('source', 'Unknown')
            score = item.get('score', 0.0)
            context_str += f"{i+1}. (Source: {source}, Score: {score:.3f}): {text}\n"
        return context_str.strip()

    def _create_decomposition_prompt(self, goal: str, ltm_context: Optional[str] = None) -> str: # Added ltm_context
        """Creates the LLM prompt for goal decomposition.

        Constructs a detailed prompt instructing the LLM to break down the user's
        goal into a JSON list of subtasks with dependencies. Includes the
        formatted LTM context if provided.

        Args:
            goal (str): The high-level user goal.
            ltm_context (Optional[str]): Formatted string of relevant information
                retrieved from LTM, or None.

        Returns:
            str: The complete, formatted prompt string ready for the LLM.
        """
        context_section = ""
        if ltm_context:
            context_section = f"""
        Use the following relevant information retrieved from memory to inform your plan:
        --- START MEMORY CONTEXT ---
        {ltm_context}
        --- END MEMORY CONTEXT ---
        """

        prompt = f"""
        You are a meticulous planning agent. Your task is to decompose the following high-level user goal into a sequence of smaller, actionable subtasks that directly lead to achieving the goal.
        Focus on creating subtasks that involve DIRECTLY USING a specific tool (like 'browser_use_tool', 'search_internet', 'calculate', 'read_file', etc.) to perform an action, rather than instructing how to use a tool or application. For example, instead of "Open Brave Search and enter query", create a subtask like "Use search_internet with query '...'".
        {context_section}
        Output the final plan as a JSON list of objects. Each object must have the following keys:
        - "subtask_id": A unique identifier string for the subtask (e.g., "t1", "t2").
        - "description": A clear, concise description of the subtask.
        - "suggested_tool": The name of the primary tool expected to be used for this subtask (e.g., "browser_use_tool", "search_internet", "calculate"). If no specific tool is needed (e.g., for a final summarization step using only the LLM), use "None".
        - "depends_on": A list of "subtask_id" strings that this subtask depends on. Use an empty list [] if there are no dependencies.

        Ensure the subtasks are logically ordered based on dependencies. Break down the goal into the minimum necessary tool-using steps. Avoid conversational or instructional steps.

        User Goal: "{goal}"

        JSON Plan:
        """
        return prompt.strip()

    def _parse_llm_output(self, llm_response: str) -> Optional[List[Dict[str, Any]]]:
        """Parses the LLM's JSON output containing the subtask list.

        Attempts to extract a valid JSON list from the LLM's potentially noisy
        response string. Validates the structure of the parsed list and its items.

        Args:
            llm_response (str): The raw string response from the LLM.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of subtask dictionaries if parsing
            and validation are successful, otherwise None. Each dictionary should
            contain 'subtask_id', 'description', and 'depends_on'.
        """
        # (Implementation remains the same as previous version)
        try:
            json_start = llm_response.find('[')
            json_end = llm_response.rfind(']')
            if json_start != -1 and json_end != -1 and json_start < json_end:
                json_str = llm_response[json_start : json_end + 1]
                subtasks = json.loads(json_str)
                # Validate structure including the new 'suggested_tool' field
                if isinstance(subtasks, list) and all(
                    isinstance(task, dict) and
                    "subtask_id" in task and
                    "description" in task and
                    "suggested_tool" in task and # Added check
                    "depends_on" in task and
                    isinstance(task["depends_on"], list)
                    for task in subtasks
                ):
                    logger.info(f"Successfully parsed {len(subtasks)} subtasks from LLM response.")
                    return subtasks
                else: logger.error(f"Parsed JSON does not match expected format: {subtasks}"); return None
            else: logger.error(f"Could not find valid JSON list in LLM response: {llm_response}"); return None
        except json.JSONDecodeError as e: logger.error(f"Failed JSON decode: {e}. Response: {llm_response}"); return None
        except Exception as e: logger.error(f"Unexpected error parsing LLM response: {e}. Response: {llm_response}"); return None


    async def decompose_and_dispatch(self, goal: str) -> Optional[List[Dict[str, Any]]]:
        """Decomposes a user goal into subtasks using LTM and LLM.

        This method orchestrates the planning process:
        1. Retrieves relevant context from LTM based on the goal.
        2. Creates a decomposition prompt including the goal and LTM context.
        3. Calls the LLM to generate the subtask plan (JSON list).
        4. Parses the LLM response.
        5. Returns the parsed subtask list or None if any step fails.
           (Dispatching is handled separately, e.g., in the calling CLI script).

        Args:
            goal (str): The high-level user goal provided by the user or system.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of subtask dictionaries if planning
            and parsing were successful, otherwise None.
        """
        logger.info(f"Received goal for decomposition: '{goal}'")

        # --- Retrieve Context from LTM (P3.T3.7) ---
        ltm_context_str: Optional[str] = None
        try:
            logger.info("Retrieving relevant context from LTM...")
            # Assuming ltm_interface has an async retrieve method
            retrieved_docs = await self._ltm_interface.retrieve(query_text=goal, top_k=3) # Retrieve top 3 relevant docs
            if retrieved_docs:
                ltm_context_str = self._format_ltm_context(retrieved_docs)
                logger.info("Formatted LTM context for prompt.")
                logger.debug(f"LTM Context:\n{ltm_context_str}")
            else:
                logger.info("No relevant context found in LTM.")
        except Exception as e:
            logger.error(f"Failed to retrieve or format LTM context: {e}", exc_info=True)
            # Continue without LTM context if retrieval fails
        # --- End LTM Retrieval ---

        # 1. Create Prompt (Now includes LTM context)
        prompt = self._create_decomposition_prompt(goal, ltm_context=ltm_context_str)
        logger.debug(f"Generated decomposition prompt:\n{prompt}")

        # 2. Call LLM
        try:
            # Replace with actual async LLM call using the stored client
            # Call the actual LLM client
            logger.info("Invoking LLM for plan generation...")
            llm_response_content = await self._llm_client.ainvoke(prompt)
            llm_response = llm_response_content.content if hasattr(llm_response_content, 'content') else str(llm_response_content)
            logger.info("Received LLM response.")
            logger.debug(f"LLM Response:\n{llm_response}")

            # Mock response removed
            # mock_llm_response = """..."""
            # llm_response = mock_llm_response # Use mock response
            # logger.info("Received LLM response (mocked).")
            # logger.debug(f"LLM Response:\n{llm_response}")

        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return False

        # 3. Parse LLM Output
        subtasks = self._parse_llm_output(llm_response)
        if not subtasks:
            logger.error("Failed to parse subtasks from LLM response.")
            return None # Return None on parsing failure

        # 4. Return Parsed Subtasks (Dispatching moved to caller)
        logger.info(f"Successfully generated and parsed plan with {len(subtasks)} subtasks.")
        return subtasks

        # Dispatching logic removed from here
        # logger.info(f"Dispatching {len(subtasks)} subtasks...")
        # try:
        #     assignments = await self._dispatcher.dispatch_subtasks(subtasks)
        #     if assignments:
        #         logger.info(f"Successfully dispatched tasks. Assignments: {assignments}")
        #         return True # Or maybe return subtasks here too?
        #     else:
        #         logger.error("Dispatcher failed to assign tasks (e.g., no available agents).")
        #         return False
        # except Exception as e:
        #     logger.error(f"Error during dispatch: {e}", exc_info=True)
        #     return False

    async def synthesize_final_output(self, goal: str, subtasks: List[Dict[str, Any]], results: Dict[str, Any]) -> str:
        """Synthesizes the final output based on the goal, plan, and subtask results.

        Constructs a prompt for the LLM that includes the original goal, the
        generated plan (including the suggested tool for each step), and the
        results (or errors) obtained for each subtask. Asks the LLM to generate
        a final, coherent response that addresses the original goal and explicitly
        mentions the key tools used in the process.

        Args:
            goal (str): The original user goal.
            subtasks (List[Dict[str, Any]]): The list of subtask dictionaries generated
                by the planner (should include 'subtask_id', 'description', 'suggested_tool').
            results (Dict[str, Any]): A dictionary mapping subtask IDs to their
                results or error messages.

        Returns:
            str: The synthesized final response generated by the LLM.
        """
        logger.info(f"Synthesizing final output for goal: '{goal}'")

        # Format the plan and results for the prompt
        plan_summary = "\nExecution Plan and Results:\n"
        for task in subtasks:
            task_id = task.get("subtask_id", "N/A")
            desc = task.get("description", "N/A")
            tool = task.get("suggested_tool", "None")
            result = results.get(task_id, "Result not available.")
            plan_summary += f"- Task ID: {task_id}\n"
            plan_summary += f"  - Description: {desc}\n"
            plan_summary += f"  - Suggested Tool: {tool}\n"
            plan_summary += f"  - Result/Status: {str(result)[:300]}...\n" # Truncate long results

        synthesis_prompt = f"""
        You are an expert synthesis agent. Your task is to generate a final, coherent response to the user's original goal based on the execution plan and the results obtained for each subtask.

        Original User Goal: "{goal}"

        {plan_summary}

        Instructions:
        1. Review the original goal and the results of the executed subtasks.
        2. Generate a comprehensive, well-structured response that directly addresses the original goal.
        3. IMPORTANT: In your response, explicitly mention the key tools that were used to gather or process the information (e.g., "Using web search...", "Based on browser agent findings...", "After calculating..."). Refer to the 'Suggested Tool' field in the plan summary. Do not just list the tools; integrate the mention naturally into your explanation of how the information was obtained.
        4. If some subtasks failed or produced errors, acknowledge this appropriately in the response, but focus on synthesizing the successful results if possible.
        5. Ensure the final response is clear, concise, and directly answers the user's request.

        Final Synthesized Response:
        """

        logger.debug(f"Synthesis prompt:\n{synthesis_prompt}")

        try:
            logger.info("Invoking LLM for final synthesis...")
            llm_response_content = await self._llm_client.ainvoke(synthesis_prompt.strip())
            final_output = llm_response_content.content if hasattr(llm_response_content, 'content') else str(llm_response_content)
            logger.info("Received synthesis from LLM.")
            return final_output
        except Exception as e:
            logger.error(f"LLM call failed during synthesis: {e}", exc_info=True)
            return f"Error: Failed to synthesize the final output due to an LLM error: {e}"

# Example Usage (Updated)
async def main():
    # Mock dispatcher, LTM, and LLM client
    class MockDispatcher:
        async def dispatch_subtasks(self, subtasks):
            print("\n--- Dispatcher Received Subtasks ---")
            for task in subtasks: print(task)
            print("------------------------------------")
            return {task["subtask_id"]: f"agent_{(i%2)+1:03d}" for i, task in enumerate(subtasks)}

    class MockLTMInterface:
        async def retrieve(self, query_text: str, top_k: int = 5, filter_metadata: Optional[Dict] = None):
            print(f"\n--- LTM Retrieve Called (Query: '{query_text[:50]}...') ---")
            await asyncio.sleep(0.05)
            if "async" in query_text.lower():
                return [
                    {"id": "doc1", "score": 0.9, "metadata": {"original_text": "Asyncio uses an event loop for concurrency.", "source": "ltm_doc"}},
                    {"id": "doc2", "score": 0.8, "metadata": {"original_text": "Await is used to pause execution in async functions.", "source": "ltm_doc"}}
                ]
            return []

    class MockLLMClient:
        async def ainvoke(self, prompt: str):
            print(f"\n--- LLM Called (Prompt Snippet) ---\n{prompt[:200]}...\n---------------------------------")
            await asyncio.sleep(0.1)
            if "JSON Plan:" in prompt: # Planner decomposition
                 return """
                 ```json
                 [
                     {"subtask_id": "t1_review_ltm", "description": "Review LTM context about async libraries.", "suggested_tool": "None", "depends_on": []},
                     {"subtask_id": "t2_search", "description": "Search web for 'alternatives to asyncio'.", "suggested_tool": "search_internet", "depends_on": ["t1_review_ltm"]},
                     {"subtask_id": "t3_compare", "description": "Compare asyncio with 1-2 alternatives based on LTM and search.", "suggested_tool": "None", "depends_on": ["t1_review_ltm", "t2_search"]}
                 ]
                 ```
                 """
            elif "Synthesizing final output" in prompt: # Synthesis call
                return "Based on web search results using search_internet, alternatives to asyncio include Trio and Curio. LTM context confirmed asyncio uses an event loop."
            else:
                return "Mock LLM Response"


    mock_dispatcher = MockDispatcher()
    mock_ltm = MockLTMInterface()
    mock_llm = MockLLMClient()

    planner = PlannerAgent(dispatcher=mock_dispatcher, ltm_interface=mock_ltm, llm_client=mock_llm)

    user_goal = "Find alternatives to Python's asyncio library and compare them."
    # Test decomposition
    plan = await planner.decompose_and_dispatch(user_goal)

    if plan:
        print("\nPlanner successfully decomposed the goal.")
        # Simulate results for synthesis test
        mock_results = {
            "t1_review_ltm": "Reviewed LTM: Asyncio uses event loop, await pauses.",
            "t2_search": "Web search found Trio and Curio as alternatives.",
            "t3_compare": "Comparison complete: Trio focuses on structured concurrency, Curio on low-level control."
        }
        # Test synthesis
        final_summary = await planner.synthesize_final_output(user_goal, plan, mock_results)
        print("\n--- Final Synthesized Output (Test) ---")
        print(final_summary)
        print("---------------------------------------")
    else:
        print("\nPlanner failed to process the goal.")

if __name__ == "__main__":
    # Need to run this from the project root directory for relative imports to work
    # Example: python -m nova_shift.agents.planner_agent
    asyncio.run(main())
