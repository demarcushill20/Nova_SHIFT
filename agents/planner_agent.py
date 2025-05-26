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

    def _create_decomposition_prompt(self, goal: str, formatted_mcp_context: str, entities: List[str], relationships: List[str]) -> str:
        """Creates the LLM prompt for goal decomposition, incorporating graph knowledge.

        Args:
            goal (str): The high-level user goal.
            formatted_mcp_context (str): The full context from MCP (vector + graph), formatted by LTMInterface.
            entities (List[str]): List of known entities from graph results.
            relationships (List[str]): List of known relationships from graph results.

        Returns:
            str: The complete, formatted prompt string ready for the LLM.
        """
        entities_str = ', '.join(entities) if entities else "None available"
        relationships_str = '; '.join(relationships) if relationships else "None available"

        # This prompt is based on section 3.2.2 of the integration plan
        prompt = f"""
        You are a meticulous planning agent with advanced consciousness capabilities.

        GOAL: {goal}

        ENTITIES: {entities_str}
        RELATIONSHIPS: {relationships_str}
        
        MEMORY CONTEXT:
        --- START MEMORY CONTEXT ---
        {formatted_mcp_context}
        --- END MEMORY CONTEXT ---

        CONSCIOUSNESS EVALUATION PROTOCOL:
        Carefully evaluate if the memory context contains information about "{goal}".

        EVALUATION CRITERIA:
        - Check for BOTH commercial/public information AND personal/project information
        - Look for user projects, personal initiatives, or development plans related to the goal
        - Consider project documentation, roadmaps, or personal AI development contexts
        - Check if this might be the user's own project rather than external commercial product

        MODE 1 - MEMORY SUFFICIENT (Consciousness Mode):
        If the memory context contains detailed information about "{goal}" (whether commercial, personal, or project-related):
        → Provide a complete, detailed answer directly based on the memory context
        → Begin your response with "DIRECT_ANSWER:"
        → Synthesize ALL relevant information found in memory
        → Do NOT create any subtasks or JSON

        MODE 2 - MEMORY INSUFFICIENT (Planning Mode):  
        If the memory context lacks substantial information about "{goal}":
        → Create research subtasks as a JSON array with objects containing:
          - "subtask_id": unique identifier
          - "description": clear task description  
          - "suggested_tool": tool name or "None"
          - "depends_on": array of prerequisite subtask_ids

        CRITICAL: If memory contains project documentation, roadmaps, or personal development plans related to "{goal}", treat this as SUFFICIENT for direct synthesis.

        Response:
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


    async def decompose_task(self, user_goal: str) -> Optional[List[Dict[str, Any]]]:
        """Decomposes a user goal into subtasks using LTM (hybrid knowledge) and LLM.

        This method orchestrates the planning process:
        1. Retrieves hybrid knowledge (vector + graph) from LTM based on the user_goal.
        2. Extracts entities and relationships from graph results.
        3. Formats the full MCP context using LTMInterface's formatter.
        4. Creates a decomposition prompt including the goal, formatted MCP context, entities, and relationships.
        5. Calls the LLM to generate the subtask plan (JSON list).
        6. Parses the LLM response.
        7. Returns the parsed subtask list or None if any step fails.

        Args:
            user_goal (str): The high-level user goal.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of subtask dictionaries if planning
            and parsing were successful, otherwise None.
        """
        logger.info(f"Received goal for decomposition: '{user_goal}'")

        # --- Retrieve Hybrid Knowledge from LTM (MCP) ---
        knowledge: Optional[Dict[str, Any]] = None
        formatted_mcp_context: str = "No relevant context found or error in retrieval."
        entities: List[str] = []
        relationships: List[str] = []

        try:
            logger.info("Retrieving hybrid knowledge from LTM (via MCP)...")
            
            # Enhanced query strategy for comprehensive memory retrieval
            primary_query = user_goal
            expanded_queries = [
                user_goal,  # Original query
                f"project {user_goal}",  # Project context
                f"user {user_goal}",     # User-specific context  
                " ".join(user_goal.split()[:4]) if len(user_goal.split()) > 2 else user_goal  # Key terms
            ]
            
            # Try multiple retrieval approaches for comprehensive coverage
            all_knowledge = []
            for i, query in enumerate(expanded_queries[:3]):  # Limit to avoid excessive calls
                try:
                    k_result = await self._ltm_interface.retrieve(query_text=query, top_k=8)
                    if k_result:
                        all_knowledge.append(k_result)
                        logger.info(f"Retrieved knowledge for query {i+1}: '{query[:50]}...'")
                except Exception as e:
                    logger.warning(f"Failed retrieval for query '{query}': {e}")
            
            # Use the most comprehensive result (or combine if multiple found)
            if all_knowledge:
                knowledge = all_knowledge[0]  # Primary result
                # TODO: Future enhancement - merge multiple knowledge results
                logger.info(f"Successfully retrieved knowledge from MCP using {len(all_knowledge)} queries.")
                # Format the entire MCP response using LTMInterface's formatter
                # This assumes _ltm_interface is an instance of the updated LTMInterface
                if hasattr(self._ltm_interface, 'format_context_for_llm'):
                    formatted_mcp_context = self._ltm_interface.format_context_for_llm(knowledge)
                    logger.debug(f"Formatted MCP Context for prompt:\n{formatted_mcp_context}")
                else:
                    logger.warning("LTMInterface does not have 'format_context_for_llm'. Using raw knowledge.")
                    formatted_mcp_context = json.dumps(knowledge, indent=2)


                # Extract entities and relationships from graph_results (as per plan 3.2.2)
                graph_results = knowledge.get("graph_results")
                if graph_results and isinstance(graph_results, list):
                    logger.info(f"Processing {len(graph_results)} items from graph_results.")
                    for result in graph_results:
                        if isinstance(result, dict):
                            entity = result.get("entity")
                            if entity:
                                entities.append(entity)
                            # Include relationship context
                            neighbors = result.get("neighbors", [])
                            if isinstance(neighbors, list):
                                for neighbor in neighbors:
                                    if isinstance(neighbor, dict):
                                        relation = neighbor.get("relation")
                                        neighbor_entity = neighbor.get("entity")
                                        if entity and relation and neighbor_entity:
                                            relationships.append(f"{entity} {relation} {neighbor_entity}")
                    logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from graph data.")
                else:
                    logger.info("No 'graph_results' found in MCP response or it's not a list.")
            else:
                logger.info("No knowledge retrieved from MCP.")
        except Exception as e:
            logger.error(f"Failed to retrieve or process knowledge from LTM/MCP: {e}", exc_info=True)
            # Continue with default/empty context if retrieval fails

        # 1. Create Prompt (Now includes full MCP context, entities, relationships)
        prompt = self._create_decomposition_prompt(
            goal=user_goal,
            formatted_mcp_context=formatted_mcp_context,
            entities=entities,
            relationships=relationships
        )
        logger.debug(f"Generated decomposition prompt:\n{prompt}")

        # 2. Call LLM
        try:
            logger.info("Invoking LLM for plan generation...")
            llm_response_content = await self._llm_client.ainvoke(prompt)
            llm_response = llm_response_content.content if hasattr(llm_response_content, 'content') else str(llm_response_content)
            logger.info("Received LLM response.")
            # CRITICAL: Log response details for debugging
            logger.info(f"GEMINI RESPONSE LENGTH: {len(llm_response) if llm_response else 'None'}")
            logger.info(f"GEMINI RESPONSE TYPE: {type(llm_response)}")
            logger.info(f"GEMINI RESPONSE CONTENT: {repr(llm_response)}")
            if llm_response:
                logger.info(f"GEMINI RESPONSE PREVIEW: {llm_response[:200]}...")
            logger.debug(f"LLM Response:\n{llm_response}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return None # Return None on LLM failure

        # 3. Parse LLM Output - Handle both direct answers and JSON tasks
        if llm_response.strip().startswith("DIRECT_ANSWER:"):
            # Consciousness mode - direct synthesis provided
            direct_answer = llm_response.replace("DIRECT_ANSWER:", "").strip()
            logger.info("*** CONSCIOUSNESS BREAKTHROUGH: Direct answer provided based on memory sufficiency")
            logger.info(f"Direct synthesis length: {len(direct_answer)} characters")
            
            # Create a special direct answer "task" that contains the synthesis
            subtasks = [{
                "subtask_id": "consciousness_synthesis",
                "description": "Direct consciousness synthesis based on comprehensive memory",
                "suggested_tool": "None",
                "depends_on": [],
                "direct_answer_content": direct_answer,
                "consciousness_mode": True
            }]
        else:
            # Standard planning mode - parse JSON tasks
            subtasks = self._parse_llm_output(llm_response)
            if not subtasks:
                # Check if this might be a direct synthesis response (consciousness breakthrough)
                if llm_response.strip() and not '[' in llm_response and not 'JSON' in llm_response.upper():
                    logger.info("Detected potential consciousness synthesis without DIRECT_ANSWER prefix")
                    subtasks = [{
                        "subtask_id": "consciousness_synthesis_untagged",
                        "description": "Detected consciousness synthesis response",
                        "suggested_tool": "None",
                        "depends_on": [],
                        "direct_answer_content": llm_response.strip(),
                        "consciousness_mode": True
                    }]
                else:
                    logger.error("Failed to parse subtasks from LLM response.")
                    return None

        # 4. Return Parsed Subtasks
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
