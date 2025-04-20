"""
Developer Agent: Responsible for generating Python code for new tools
and their corresponding toolkit.json definitions based on specifications.
(Corresponds to TASK.md P4.T3)
"""

import logging
import json
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

# Assuming LLM client setup is handled elsewhere
# from ..core.llm_client import get_llm_client
# from ..core.tool_registry import ToolRegistry # May need Tool Registry later

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Placeholder type hints
LLMClient = Any
ToolRegistry = Any

# --- Prompt Templates ---

CODE_GENERATION_PROMPT_TEMPLATE = """
You are an expert Python Developer Agent within the Nova SHIFT framework.
Your task is to write a single, self-contained Python function based on the provided specification.

Tool Specification:
---
{specification}
---

Constraints:
- The function should only use standard Python libraries unless specified otherwise.
- It must be purely functional, taking inputs and returning outputs without side effects (like modifying global state or external files unless that's the explicit purpose).
- Include a Google-style docstring explaining what the function does, its arguments, and what it returns.
- Ensure the function handles potential errors gracefully (e.g., using try-except blocks) and returns informative error messages as strings if it fails.
- The function name must be: `{function_name}`
- The function signature should align with the inputs/outputs described in the spec.

Python Function Code:
```python
# Start your Python code here
```
"""

TOOLKIT_JSON_GENERATION_PROMPT_TEMPLATE = """
You are an expert AI Agent Configuration Specialist within the Nova SHIFT framework.
Your task is to generate a valid `toolkit.json` definition for a new tool, based on its specification and the generated Python function name.

Tool Specification:
---
{specification}
---

Generated Python Function Name: `{function_name}`
Target Python Module Path: `{module_path}`

Constraints:
- The `toolkit.json` must strictly adhere to the ToolkitSchema.
- The `name` field should be a concise, descriptive CamelCase name for the toolkit (e.g., StringReverserToolkit).
- The `version` should be "1.0.0".
- The `description` should briefly explain the toolkit's purpose based on the specification.
- The `tools` list should contain exactly one tool definition.
- The tool's `name` should be a concise snake_case name relevant to the function (e.g., `reverse_string`).
- The tool's `function` must be the exact generated Python function name provided above.
- The tool's `description` should explain what the tool does based on the specification.
- The `inputs` and `outputs` lists should be derived from the specification (use format 'name:type'). Assume 'string' type if not specified.
- Set `requirements` to `null` unless specific Python packages or API keys are mentioned in the spec.
- The `loading_info.type` must be "python_module".
- The `loading_info.path` must be the exact target Python module path provided above.

Output only the raw JSON object, starting with `{` and ending with `}`. Do not include any explanatory text before or after the JSON.

toolkit.json:
```json
{{
    # Start your JSON here
}}
```
"""

class DeveloperAgent:
    """
    Generates Python code for new tools and their toolkit definitions.
    (Prototype for Phase 4)
    """

    def __init__(self,
                 llm_client: LLMClient,
                 tool_registry: Optional[ToolRegistry] = None): # Optional for now
        """
        Initializes the DeveloperAgent.

        Args:
            llm_client: An instance of the LLM client for code/JSON generation.
            tool_registry: Optional instance of the ToolRegistry (needed later for registration).
        """
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        logger.info("DeveloperAgent initialized (Prototype).")

    def _extract_python_code(self, llm_response: str) -> Optional[str]:
        """Extracts Python code block from LLM response."""
        try:
            code_block_start = llm_response.find("```python")
            if code_block_start == -1:
                logger.warning("Could not find start of Python code block (```python).")
                # Fallback: Assume the whole response might be code if it looks like it
                if "def " in llm_response and ":" in llm_response:
                     logger.warning("Assuming entire response might be Python code.")
                     return llm_response.strip()
                return None

            code_start_index = code_block_start + len("```python\n")
            code_end_index = llm_response.find("```", code_start_index)
            if code_end_index == -1:
                logger.warning("Could not find end of Python code block (```).")
                # Fallback: Take everything after the start marker
                return llm_response[code_start_index:].strip()

            return llm_response[code_start_index:code_end_index].strip()
        except Exception as e:
            logger.error(f"Error extracting Python code: {e}", exc_info=True)
            return None

    def _extract_json_object(self, llm_response: str) -> Optional[Dict]:
        """Extracts JSON object block from LLM response."""
        try:
            json_block_start = llm_response.find("```json")
            if json_block_start == -1: json_block_start = llm_response.find("{") # Fallback: find first '{'
            if json_block_start == -1: logger.warning("Could not find start of JSON block."); return None

            # Adjust start index if ```json marker was found
            code_start_index = json_block_start + len("```json\n") if "```json" in llm_response else json_block_start

            json_end_index = llm_response.rfind("}") # Find the last '}'
            if json_end_index == -1 or json_end_index < code_start_index:
                 logger.warning("Could not find end of JSON block ('}}')."); return None

            json_str = llm_response[code_start_index : json_end_index + 1].strip()

            # Basic cleanup: remove potential leading/trailing ```
            if json_str.startswith("```"): json_str = json_str[3:]
            if json_str.endswith("```"): json_str = json_str[:-3]
            json_str = json_str.strip()

            # Parse the JSON string
            json_object = json.loads(json_str)
            return json_object
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode extracted JSON: {e}. String was: '{json_str}'")
            return None
        except Exception as e:
            logger.error(f"Error extracting JSON object: {e}", exc_info=True)
            return None

    async def generate_tool(self, specification: str, function_name: str, module_path: str) -> Optional[Tuple[str, Dict]]:
        """
        Generates Python code and toolkit.json for a new tool based on specification.

        Args:
            specification: Natural language description of the tool's purpose, inputs, and outputs.
            function_name: The exact Python function name to generate.
            module_path: The target Python module path for loading_info (e.g., 'nova_shift.tools.new_tool.new_tool_toolkit').

        Returns:
            A tuple containing (generated_python_code, generated_toolkit_json_dict)
            or None if generation fails at any step.
        """
        logger.info(f"DeveloperAgent received request to generate tool '{function_name}' based on spec: '{specification[:100]}...'")

        # 1. Generate Python Code
        code_prompt = CODE_GENERATION_PROMPT_TEMPLATE.format(
            specification=specification,
            function_name=function_name
        ).strip()
        logger.debug(f"Code Generation Prompt:\n{code_prompt}")
        try:
            code_response = await self._llm_client.ainvoke(code_prompt)
            code_response_content = code_response.content if hasattr(code_response, 'content') else str(code_response)
            logger.debug(f"Code Generation LLM Response:\n{code_response_content}")
            python_code = self._extract_python_code(code_response_content)
            if not python_code:
                logger.error("Failed to extract Python code from LLM response.")
                return None
            logger.info(f"Successfully generated Python code for function '{function_name}'.")
            # TODO (P4.T3.5): Add optional sandbox testing here
            # test_passed = self._test_generated_code(python_code, function_name, test_cases)
            # if not test_passed: return None

        except Exception as e:
            logger.error(f"Code generation LLM call failed: {e}", exc_info=True)
            return None

        # 2. Generate toolkit.json
        json_prompt = TOOLKIT_JSON_GENERATION_PROMPT_TEMPLATE.format(
            specification=specification,
            function_name=function_name,
            module_path=module_path
        ).strip()
        logger.debug(f"Toolkit JSON Generation Prompt:\n{json_prompt}")
        try:
            json_response = await self._llm_client.ainvoke(json_prompt)
            json_response_content = json_response.content if hasattr(json_response, 'content') else str(json_response)
            logger.debug(f"Toolkit JSON Generation LLM Response:\n{json_response_content}")
            toolkit_json_dict = self._extract_json_object(json_response_content)
            if not toolkit_json_dict:
                logger.error("Failed to extract toolkit JSON from LLM response.")
                return None
            # TODO: Add Pydantic validation using ToolkitSchema here
            logger.info(f"Successfully generated toolkit JSON for function '{function_name}'.")

        except Exception as e:
            logger.error(f"Toolkit JSON generation LLM call failed: {e}", exc_info=True)
            return None

        # 3. Return results
        return python_code, toolkit_json_dict

    def _test_generated_code(self, code: str, function_name: str, test_cases: List[Dict]) -> bool:
        """
        (Optional - P4.T3.5) Executes generated code in a restricted sandbox
        against provided test cases.

        Args:
            code: The generated Python function code string.
            function_name: The name of the function to test.
            test_cases: A list of dicts, each with 'input' and 'expected_output'.

        Returns:
            True if all test cases pass, False otherwise.
        """
        logger.info(f"Attempting to sandbox test function '{function_name}'...")
        # WARNING: Executing LLM-generated code is inherently risky.
        # Use a robust sandboxing library or method in production.
        # This is a very basic example using exec, which is NOT secure.
        temp_file_path = None
        try:
            # Write code to a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
                tmp.write(code)
                temp_file_path = tmp.name
            logger.debug(f"Generated code written to temporary file: {temp_file_path}")

            # Execute tests using subprocess for isolation (slightly safer than direct exec)
            all_passed = True
            for i, case in enumerate(test_cases):
                input_arg = case['input']
                expected_output = case['expected_output']

                # Prepare a script to run the test case
                test_script_code = f"""
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("generated_module", "{temp_file_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

func = getattr(module, "{function_name}")
input_arg = {repr(input_arg)} # Use repr to handle string quoting etc.
expected_output = {repr(expected_output)}

try:
    result = func(input_arg)
    # Use JSON for potentially complex type comparison
    result_json = json.dumps(result, sort_keys=True)
    expected_json = json.dumps(expected_output, sort_keys=True)
    if result_json == expected_json:
        print("PASS")
    else:
        print(f"FAIL: Input={{input_arg}}, Expected={{expected_output}}, Got={{result}}")
        sys.exit(1) # Indicate failure
except Exception as e:
    print(f"FAIL: Input={{input_arg}}, Expected={{expected_output}}, Error={{e}}")
    sys.exit(1) # Indicate failure
"""
                # Run the test script in a subprocess
                process = subprocess.run(
                    [sys.executable, '-c', test_script_code],
                    capture_output=True, text=True, timeout=5 # Add timeout
                )

                logger.debug(f"Test Case {i+1} Output:\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}")

                if process.returncode != 0 or "FAIL" in process.stdout:
                    logger.error(f"Sandbox test case {i+1} failed for function '{function_name}'.")
                    all_passed = False
                    break # Stop on first failure
                else:
                     logger.info(f"Sandbox test case {i+1} passed for function '{function_name}'.")

            return all_passed

        except FileNotFoundError:
             logger.error(f"Python executable not found at '{sys.executable}'. Cannot run sandbox test.")
             return False
        except subprocess.TimeoutExpired:
             logger.error(f"Sandbox test timed out for function '{function_name}'.")
             return False
        except Exception as e:
            logger.error(f"Error during sandbox testing of '{function_name}': {e}", exc_info=True)
            return False
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Removed temporary file: {temp_file_path}")
                except OSError as e:
                    logger.error(f"Error removing temporary file {temp_file_path}: {e}")


# Example Usage (Conceptual)
async def main():
    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock()

    # Define side effects for the two LLM calls
    def llm_side_effect(prompt):
        if "Python Function Code:" in prompt:
            # Return mock Python code
            return """```python
import logging

logger = logging.getLogger(__name__)

def reverse_string_tool(input_string: str) -> str:
    \"\"\"
    Reverses the input string.

    Args:
        input_string (str): The string to reverse.

    Returns:
        str: The reversed string or an error message.
    \"\"\"
    logger.debug(f"Reversing string: {input_string}")
    if not isinstance(input_string, str):
        return "Error: Input must be a string."
    try:
        return input_string[::-1]
    except Exception as e:
        logger.error(f"Error reversing string: {e}")
        return f"Error: Could not reverse string - {e}"

```"""
        elif "toolkit.json:" in prompt:
            # Return mock JSON
            return """```json
{
    "name": "StringReverserToolkit",
    "version": "1.0.0",
    "description": "A toolkit to reverse strings.",
    "tools": [
        {
            "name": "reverse_string",
            "function": "reverse_string_tool",
            "description": "Reverses the input string.",
            "inputs": ["input_string:string"],
            "outputs": ["reversed_string:string | error:string"]
        }
    ],
    "requirements": null,
    "loading_info": {
        "type": "python_module",
        "path": "nova_shift.tools.string_reverser.string_reverser_toolkit"
    }
}
```"""
        else:
            return "Unknown prompt"
    mock_llm.ainvoke.side_effect = llm_side_effect

    # Initialize Developer Agent
    developer = DeveloperAgent(llm_client=mock_llm)

    # Define tool specification
    spec = "Create a tool that takes a string as input and returns the reversed string."
    func_name = "reverse_string_tool"
    mod_path = "nova_shift.tools.string_reverser.string_reverser_toolkit"

    # Generate the tool
    result = await developer.generate_tool(spec, func_name, mod_path)

    if result:
        python_code, toolkit_json = result
        print("\n--- Generated Python Code ---")
        print(python_code)
        print("-----------------------------")
        print("\n--- Generated toolkit.json ---")
        print(json.dumps(toolkit_json, indent=2))
        print("----------------------------")

        # Example of basic sandbox test (P4.T3.5)
        # test_cases = [
        #     {'input': 'hello', 'expected_output': 'olleh'},
        #     {'input': '', 'expected_output': ''},
        #     {'input': '123', 'expected_output': '321'},
        # ]
        # test_passed = developer._test_generated_code(python_code, func_name, test_cases)
        # print(f"\nSandbox Test Passed: {test_passed}")

    else:
        print("\nDeveloper Agent failed to generate the tool.")


if __name__ == "__main__":
    # This example requires mocks or actual components to run fully.
    from unittest.mock import MagicMock, AsyncMock
    import asyncio
    import sys # Needed for sandbox test example
    asyncio.run(main())