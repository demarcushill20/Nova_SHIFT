"""
Architect Agent: Responsible for designing high-level strategies, workflows,
or suggesting new tool combinations for complex or novel goals.
(Corresponds to TASK.md P4.T2)
"""

import logging
from typing import Any, Dict, Optional

# Assuming LLM client setup is handled elsewhere
# from ..core.llm_client import get_llm_client
# from ..core.ltm_interface import LTMInterface # May need LTM later
# from ..core.tool_registry import ToolRegistry # May need Tool Registry later

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Placeholder type hints
LLMClient = Any
LTMInterface = Any
ToolRegistry = Any

# Basic prompt template for the Architect Agent
ARCHITECT_PROMPT_TEMPLATE = """
You are an expert AI System Architect within the Nova SHIFT framework.
Your role is to devise high-level strategies and abstract plans for complex, novel, or ambiguous goals.
Leverage collective agent knowledge and known solution patterns.

COMPLEX GOAL: {complex_goal}

PREVIOUS AGENT EXPERIENCES:
{agent_experiences_context}

KNOWN SOLUTION PATTERNS:
{solution_patterns_context}

Available Toolkits (Summary - for general awareness, but prioritize learned experiences and patterns):
{tool_registry_summary}

Based on the goal, past experiences, and known patterns, design a comprehensive solution.
The output should be a conceptual blueprint or a sequence of major phases/strategies.
Focus on *how* to approach the problem, leveraging collective knowledge.

Strategic Plan:
"""
# Note: tool_registry_summary and ltm_context (now agent_experiences_context) are still useful.
# The new prompt structure is from plan 3.3.2

class ArchitectAgent:
    """
    Designs high-level solution strategies and workflows for complex goals.
    (Prototype for Phase 4)
    """

    def __init__(self,
                 llm_client: LLMClient,
                 tool_registry: Optional[ToolRegistry] = None, # Optional for now
                 ltm_interface: Optional[LTMInterface] = None): # Optional for now
        """
        Initializes the ArchitectAgent.

        Args:
            llm_client: An instance of the LLM client for reasoning.
            tool_registry: Optional instance of the ToolRegistry.
            ltm_interface: Optional instance of the LTMInterface.
        """
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._ltm_interface = ltm_interface
        logger.info("ArchitectAgent initialized (Prototype).")

    def _get_tool_registry_summary(self) -> str:
        """Generates a brief summary of available toolkits."""
        if not self._tool_registry:
            return "Tool registry information is unavailable."
        try:
            toolkit_names = self._tool_registry.list_toolkits()
            if not toolkit_names:
                return "No toolkits are currently registered."
            summary = "Available Toolkits:\n"
            for name in toolkit_names:
                toolkit = self._tool_registry.get_toolkit(name)
                if toolkit:
                    summary += f"- {name}: {toolkit.description}\n"
            return summary.strip()
        except Exception as e:
            logger.error(f"Failed to generate tool registry summary: {e}")
            return "Error retrieving tool registry summary."

    async def _get_ltm_context(self, goal: str) -> str:
        """Retrieves relevant context from LTM based on the goal."""
        if not self._ltm_interface:
            return "LTM interface is unavailable."
        try:
            # Use the proper MCP formatting method
            mcp_response = await self._ltm_interface.retrieve(query_text=goal, top_k=2) # Fetch top 2
            if not mcp_response:
                return "No relevant context found in LTM."

            # Use the proper MCP formatting method
            return self._ltm_interface.format_context_for_llm(mcp_response)
        except Exception as e:
            logger.error(f"Failed to retrieve LTM context for Architect: {e}")
            return "Error retrieving LTM context."

    async def design_solution(self, complex_goal: str) -> Optional[str]:
        """
        Generates a high-level solution design for the given complex goal,
        leveraging cross-agent knowledge from LTM. (Plan 3.3.2)

        Args:
            complex_goal: The complex user goal requiring architectural planning.

        Returns:
            A string containing the strategic plan/solution design, or None if generation fails.
        """
        logger.info(f"ArchitectAgent received complex_goal for solution design: '{complex_goal}'")

        agent_experiences_context = "No previous agent experiences found or LTM unavailable."
        solution_patterns_context = "No known solution patterns found."
        solution_patterns_list: list[str] = [] # Explicitly type as list of strings

        if self._ltm_interface:
            try:
                logger.info(f"Querying LTM for agent experiences related to: 'agent solutions {complex_goal}'")
                # Query for relevant agent experiences
                agent_experiences_mcp_results = await self._ltm_interface.retrieve(
                    f"agent solutions {complex_goal}"
                )

                if agent_experiences_mcp_results:
                    # Format the full MCP response for the prompt
                    if hasattr(self._ltm_interface, 'format_context_for_llm'):
                        agent_experiences_context = self._ltm_interface.format_context_for_llm(agent_experiences_mcp_results)
                        logger.info("Formatted agent experiences from LTM.")
                    else:
                        logger.warning("LTMInterface does not have 'format_context_for_llm'. Using raw JSON for agent_experiences_context.")
                        agent_experiences_context = json.dumps(agent_experiences_mcp_results, indent=2)
                    
                    logger.debug(f"Agent Experiences Context:\n{agent_experiences_context}")

                    # Leverage graph relationships for solution patterns
                    graph_results = agent_experiences_mcp_results.get("graph_results")
                    if graph_results and isinstance(graph_results, list):
                        for result in graph_results:
                            if isinstance(result, dict):
                                properties = result.get("properties", {})
                                if isinstance(properties, dict): # Ensure properties is a dict
                                    pattern = properties.get("solution_pattern")
                                    if pattern and isinstance(pattern, str): # Ensure pattern is a string
                                        solution_patterns_list.append(pattern)
                        if solution_patterns_list:
                            solution_patterns_context = "\n- " + "\n- ".join(solution_patterns_list)
                            logger.info(f"Extracted solution patterns: {solution_patterns_list}")
                        else:
                            logger.info("No 'solution_pattern' found in graph_results properties.")
                    else:
                        logger.info("No 'graph_results' in LTM response for solution patterns or not a list.")
                else:
                    logger.info("No agent experiences found in LTM for this goal.")

            except Exception as e:
                logger.error(f"Error retrieving or processing agent experiences from LTM: {e}", exc_info=True)
        else:
            logger.warning("LTM interface not available to ArchitectAgent for retrieving agent experiences.")

        tool_summary = self._get_tool_registry_summary()

        prompt = ARCHITECT_PROMPT_TEMPLATE.format(
            complex_goal=complex_goal,
            agent_experiences_context=agent_experiences_context,
            solution_patterns_context=solution_patterns_context,
            tool_registry_summary=tool_summary
        ).strip()

        logger.debug(f"Architect prompt (design_solution):\n{prompt}")

        # Call LLM
        try:
            response = await self._llm_client.ainvoke(prompt)
            solution_design = response.content if hasattr(response, 'content') else str(response)

            if not solution_design:
                 logger.warning("Architect LLM returned an empty solution design.")
                 return None

            logger.info(f"Architect generated solution design for goal '{complex_goal}'.")
            logger.debug(f"Generated Solution Design:\n{solution_design}")
            return solution_design

        except Exception as e:
            logger.error(f"Architect LLM call failed for goal '{complex_goal}': {e}", exc_info=True)
            return None

# Example Usage (Conceptual - requires setting up mocks/real components)
async def main():
    # Mock components needed for the example
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value="1. Analyze goal components.\n2. Identify required high-level capabilities (e.g., data analysis, report generation).\n3. Suggest combining ToolA and ToolB.\n4. Note potential need for a new visualization tool.")

    mock_registry = MagicMock(spec=ToolRegistry)
    mock_registry.list_toolkits = MagicMock(return_value=["CalculatorToolkit", "WebSearchToolkit"])
    mock_registry.get_toolkit = MagicMock(side_effect=lambda name: MagicMock(description=f"{name} description"))

    mock_ltm = MagicMock(spec=LTMInterface)
    mock_ltm.retrieve = AsyncMock(return_value=[{"metadata": {"original_text": "Previous similar project used ToolA effectively."}}])

    # Initialize Architect
    architect = ArchitectAgent(llm_client=mock_llm, tool_registry=mock_registry, ltm_interface=mock_ltm)

    # Design strategy
    complex_goal = "Develop a system to predict stock market trends based on news sentiment and historical data, providing visual reports."
    strategy_output = await architect.design_strategy(complex_goal)

    if strategy_output:
        print("\n--- Architect Strategy Output ---")
        print(strategy_output)
        print("-------------------------------")
    else:
        print("\nArchitect failed to generate a strategy.")

if __name__ == "__main__":
    # This example requires mocks or actual components to run fully.
    # Need to install `unittest.mock` if not present (`pip install mock` for older python, usually built-in)
    from unittest.mock import MagicMock, AsyncMock
    import asyncio
    asyncio.run(main())