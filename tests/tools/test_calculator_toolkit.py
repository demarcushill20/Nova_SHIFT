"""
Unit tests for the Calculator Toolkit.
"""

import pytest
import math # Import math for potential future use if functions are added

# Adjust the import path based on the project structure
# Assumes tests are run from the project root (Desktop/NOVA_SHIFT)
from tools.calculator.calculator_toolkit import calculate_expression, ALLOWED_NAMES, ALLOWED_FUNCTIONS

# --- Test Cases ---

@pytest.mark.parametrize(
    "expression, expected_result",
    [
        # Expected Use Cases (Basic Arithmetic)
        ("2 + 2", 4),
        ("10 - 3", 7),
        ("5 * 6", 30),
        ("100 / 4", 25.0),
        ("2.5 * 2", 5.0),
        ("10 / 4", 2.5),
        # Order of Operations
        ("2 + 3 * 4", 14),
        (" (2 + 3) * 4", 20),
        ("100 / (5 - 3)", 50.0),
        ("10 + 2 * 6 / 3 - 1", 13.0),
        # Negative Numbers
        ("-5 + 10", 5),
        ("10 * -2", -20),
        ("-10 / -2", 5.0),
    ],
)
def test_calculate_expression_expected(expression, expected_result):
    """Tests expected successful calculations."""
    result = calculate_expression(expression) # Removed await
    assert isinstance(result, (int, float))
    assert result == expected_result

@pytest.mark.parametrize(
    "expression, expected_error_substring",
    [
        # Failure Cases (Invalid Syntax)
        ("2 +", "Syntax error in expression"), # Corrected expected substring
        ("10 /", "Syntax error in expression"), # Corrected expected substring
        ("(3 + 4", "Syntax error in expression"), # Corrected expected substring - likely '(' was never closed
        ("3 + * 4", "Syntax error in expression"), # Corrected expected substring
        # Failure Cases (Disallowed Names/Functions)
        ("a + 1", "Expression contains disallowed names or functions"), # Corrected expected substring
        ("pi", "Expression contains disallowed names or functions"), # Corrected expected substring
        ("sqrt(4)", "Expression contains disallowed names or functions"), # Corrected expected substring
        ("print('hello')", "Expression contains disallowed names or functions"), # Corrected expected substring
        # Failure Cases (Unsafe Operations - simpleeval should block these)
        ("__import__('os')", "Expression contains disallowed names or functions"), # Corrected expected substring
        ("open('file.txt')", "Expression contains disallowed names or functions"), # Corrected expected substring
        # Failure Cases (Type Errors)
        ("1 + '2'", "unsupported operand type"), # simpleeval might catch this differently
    ],
)
def test_calculate_expression_failures(expression, expected_error_substring):
    """Tests calculations that should fail with specific errors."""
    result = calculate_expression(expression) # Removed await
    assert isinstance(result, str)
    assert "Error:" in result
    assert expected_error_substring in result

@pytest.mark.parametrize(
    "expression, expected_error_substring",
    [
        # Edge Cases (Division by Zero)
        ("1 / 0", "division by zero"),
        ("100 / (2 - 2)", "division by zero"),
        ("5.0 / 0.0", "division by zero"),
    ],
)
def test_calculate_expression_division_by_zero(expression, expected_error_substring):
    """Tests division by zero edge cases."""
    # simpleeval raises ZeroDivisionError, which our wrapper catches
    result = calculate_expression(expression) # Removed await
    assert isinstance(result, str)
    assert "Error:" in result
    # The exact error message might vary slightly, check for substring
    assert expected_error_substring in result.lower() # Check lower case for flexibility

def test_calculate_expression_invalid_input_type():
    """Tests providing non-string input."""
    result = calculate_expression(12345) # type: ignore # Removed await
    assert isinstance(result, str)
    assert result == "Error: Input expression must be a string."

def test_calculate_expression_empty_string():
    """Tests providing an empty string."""
    result = calculate_expression("") # Removed await
    assert isinstance(result, str)
    assert "Error: Invalid expression syntax" in result # simpleeval treats empty as invalid

# Example of how to test if specific functions/names were allowed (currently none are)
# def test_allowed_names_and_functions():
#     """Verify that only expected names/functions are allowed (currently none)."""
#     assert not ALLOWED_NAMES # Should be empty
#     assert not ALLOWED_FUNCTIONS # Should be empty
#
#     # Example if 'pi' and 'sqrt' were added later:
#     # ALLOWED_NAMES['pi'] = math.pi
#     # ALLOWED_FUNCTIONS['sqrt'] = math.sqrt
#     # assert calculate_expression("pi * 2") == math.pi * 2
#     # assert calculate_expression("sqrt(16)") == 4.0
#     # # Remember to reset them if modifying globals in tests, or use fixtures.