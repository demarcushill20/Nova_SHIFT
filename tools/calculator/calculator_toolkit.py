"""
Implementation of the Calculator Toolkit for Nova SHIFT.

Provides a safe way to evaluate mathematical expressions.
"""

import logging
from typing import Union, Dict, Any
from simpleeval import simple_eval, NameNotDefined, FunctionNotDefined, InvalidExpression
# Removed sandbox imports

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define allowed functions/names if any (start with none for safety)
# We could add math functions here later if needed, e.g., {'sqrt': math.sqrt}
ALLOWED_NAMES: Dict[str, Any] = {}
ALLOWED_FUNCTIONS: Dict[str, Any] = {}

# --- Synchronous Tool Function ---
def calculate_expression(expression: str) -> Union[float, int, str]:
    """
    Safely evaluates a mathematical expression string.

    Uses the 'simpleeval' library to prevent unsafe operations. Supports
    basic arithmetic operators (+, -, *, /) and parentheses.

    Args:
        expression: The mathematical expression string to evaluate.
                    Example: "2 + 2 * 5 / (3-1)"

    Returns:
        The numerical result (float or int) of the evaluation if successful.
        An error message string if the evaluation fails due to invalid
        syntax, disallowed names/functions, or other errors.
    """
    if not isinstance(expression, str):
        return "Error: Input expression must be a string."

    logger.info(f"Attempting to evaluate expression: {expression}")
    try:
        # Evaluate the expression safely
        result = simple_eval(
            expression,
            names=ALLOWED_NAMES,
            functions=ALLOWED_FUNCTIONS
        )
        # Ensure the result is a number (int or float)
        if isinstance(result, (int, float)):
            logger.info(f"Evaluation successful: {expression} = {result}")
            return result
        else:
            # Should not happen with basic arithmetic, but as a safeguard
            logger.warning(f"Evaluation resulted in non-numeric type for expression '{expression}': {type(result)}")
            return f"Error: Evaluation resulted in non-numeric type ({type(result)})."

    except (NameNotDefined, FunctionNotDefined) as e:
        logger.error(f"Evaluation failed for expression '{expression}': Disallowed name or function used. Details: {e}")
        return f"Error: Expression contains disallowed names or functions: {e}"
    except InvalidExpression as e:
        logger.error(f"Evaluation failed for expression '{expression}': Invalid syntax. Details: {e}")
        return f"Error: Invalid expression syntax: {e}"
    except SyntaxError as e:
        logger.error(f"Evaluation failed for expression '{expression}': Syntax error. Details: {e}")
        return f"Error: Syntax error in expression: {e}"
    except ZeroDivisionError: # Catch division by zero explicitly
        logger.error(f"Evaluation failed for expression '{expression}': Division by zero.")
        return "Error: Division by zero."
    except Exception as e:
        # Catch any other unexpected errors during evaluation
        logger.error(f"An unexpected error occurred during evaluation of '{expression}': {e}", exc_info=True)
        return f"Error: An unexpected error occurred during evaluation: {e}"

# Example Usage (can be removed or moved to tests later)
if __name__ == '__main__':
    test_expressions = [
        "2 + 2",
        "10 / 2 * 5",
        "(3 + 4) * 2",
        "100 / (5 - 3)",
        "2.5 * 4",
        "1 / 0", # Division by zero
        "sqrt(4)", # Disallowed function
        "a + 1", # Disallowed name
        "2 +", # Syntax error
        "import os", # Unsafe attempt
        "__import__('os').system('echo unsafe')", # Unsafe attempt
        5 # Invalid input type
    ]

    for expr in test_expressions:
        print(f"Evaluating: '{expr}'")
        output = calculate_expression(str(expr) if not isinstance(expr, str) else expr) # Ensure string for testing non-str input
        print(f"Result: {output} (Type: {type(output)})")
        print("-" * 20)