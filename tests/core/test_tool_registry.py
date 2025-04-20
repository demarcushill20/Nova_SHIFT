"""
Unit tests for the Tool Registry.
"""

import pytest
import logging

# Adjust the import path based on the project structure
# Assumes tests are run from the project root (Desktop/NOVA_SHIFT)
from core.tool_registry import ToolRegistry
from tools.toolkit_schema import ToolkitSchema # Import the actual schema

# --- Test Data ---

VALID_TOOLKIT_DATA_1 = {
    "name": "CalculatorToolkit",
    "version": "1.0.0",
    "description": "Provides basic arithmetic calculation capabilities.",
    "tools": [
        {
            "name": "calculate",
            "function": "calculate_expression",
            "description": "Evaluates a mathematical expression.",
            "inputs": ["expression:string"],
            "outputs": ["result:float | result:string"]
        }
    ],
    "requirements": None,
    "loading_info": {
        "type": "python_module",
        "path": "tools.calculator.calculator_toolkit"
    }
}

VALID_TOOLKIT_DATA_2 = {
    "name": "FileReaderToolkit",
    "version": "1.1.0",
    "description": "Reads local files.",
    "tools": [
        {
            "name": "read_file",
            "function": "read_text_file",
            "description": "Reads a text file.",
            "inputs": ["file_path:string"],
            "outputs": ["content:string | error:string"]
        }
    ],
    "requirements": None,
    "loading_info": {
        "type": "python_module",
        "path": "tools.file_reader.file_reader_toolkit"
    }
}

INVALID_TOOLKIT_DATA_MISSING_FIELD = {
    "name": "IncompleteToolkit",
    "version": "1.0",
    # Missing 'description'
    "tools": [],
    "loading_info": {
        "type": "python_module",
        "path": "some.path"
    }
}

INVALID_TOOLKIT_DATA_WRONG_TYPE = {
    "name": "WrongTypeToolkit",
    "version": "1.0",
    "description": "Description",
    "tools": "not a list", # Should be a list of ToolDefinition
    "loading_info": {
        "type": "python_module",
        "path": "some.path"
    }
}

INVALID_TOOLKIT_DATA_EXTRA_FIELD = {
    "name": "ExtraFieldToolkit",
    "version": "1.0",
    "description": "Description",
    "tools": [],
    "loading_info": {
        "type": "python_module",
        "path": "some.path"
    },
    "extra_unwanted_field": "some value" # Schema forbids extra fields
}


# --- Test Cases ---

@pytest.fixture
def registry() -> ToolRegistry:
    """Provides a fresh ToolRegistry instance for each test."""
    return ToolRegistry()

def test_registry_initialization(registry: ToolRegistry):
    """Tests that the registry initializes empty."""
    assert registry.list_toolkits() == []
    assert registry.get_toolkit("AnyName") is None

def test_load_valid_toolkit(registry: ToolRegistry, caplog):
    """Tests loading a single valid toolkit definition."""
    caplog.set_level(logging.INFO)
    success = registry.load_toolkit_from_dict(VALID_TOOLKIT_DATA_1)
    assert success is True
    assert "Successfully loaded and validated toolkit: 'CalculatorToolkit'" in caplog.text
    assert registry.list_toolkits() == ["CalculatorToolkit"]

def test_load_multiple_valid_toolkits(registry: ToolRegistry):
    """Tests loading multiple valid toolkits."""
    success1 = registry.load_toolkit_from_dict(VALID_TOOLKIT_DATA_1)
    success2 = registry.load_toolkit_from_dict(VALID_TOOLKIT_DATA_2)
    assert success1 is True
    assert success2 is True
    assert set(registry.list_toolkits()) == {"CalculatorToolkit", "FileReaderToolkit"}

def test_get_existing_toolkit(registry: ToolRegistry):
    """Tests retrieving a toolkit that has been loaded."""
    registry.load_toolkit_from_dict(VALID_TOOLKIT_DATA_1)
    toolkit = registry.get_toolkit("CalculatorToolkit")
    assert toolkit is not None
    assert isinstance(toolkit, ToolkitSchema)
    assert toolkit.name == "CalculatorToolkit"
    assert toolkit.version == "1.0.0"
    assert len(toolkit.tools) == 1
    assert toolkit.tools[0].name == "calculate"

def test_get_non_existent_toolkit(registry: ToolRegistry, caplog):
    """Tests retrieving a toolkit that has not been loaded."""
    caplog.set_level(logging.WARNING)
    registry.load_toolkit_from_dict(VALID_TOOLKIT_DATA_1) # Load something else
    toolkit = registry.get_toolkit("NonExistentToolkit")
    assert toolkit is None
    assert "Toolkit 'NonExistentToolkit' not found in registry." in caplog.text

def test_load_duplicate_toolkit(registry: ToolRegistry, caplog):
    """Tests loading a toolkit with the same name again (should overwrite)."""
    caplog.set_level(logging.WARNING)
    registry.load_toolkit_from_dict(VALID_TOOLKIT_DATA_1) # First load
    # Modify slightly for the second load
    modified_data = VALID_TOOLKIT_DATA_1.copy()
    modified_data["version"] = "1.0.1"
    success = registry.load_toolkit_from_dict(modified_data)
    assert success is True
    assert "Toolkit 'CalculatorToolkit' already exists. Overwriting" in caplog.text
    assert registry.list_toolkits() == ["CalculatorToolkit"]
    # Verify the new version is stored
    retrieved = registry.get_toolkit("CalculatorToolkit")
    assert retrieved is not None
    assert retrieved.version == "1.0.1"

@pytest.mark.parametrize(
    "invalid_data, expected_log_substring",
    [
        (INVALID_TOOLKIT_DATA_MISSING_FIELD, "Validation failed for toolkit 'IncompleteToolkit'"),
        (INVALID_TOOLKIT_DATA_WRONG_TYPE, "Validation failed for toolkit 'WrongTypeToolkit'"),
        (INVALID_TOOLKIT_DATA_EXTRA_FIELD, "Validation failed for toolkit 'ExtraFieldToolkit'"),
        ("not a dictionary", "Input data is not a dictionary"),
        ({}, "Validation failed for toolkit 'Unknown'"), # Empty dict fails validation
        (None, "Input data is not a dictionary"),
    ]
)
def test_load_invalid_toolkit_data(registry: ToolRegistry, invalid_data, expected_log_substring, caplog):
    """Tests loading various forms of invalid toolkit data."""
    caplog.set_level(logging.ERROR)
    success = registry.load_toolkit_from_dict(invalid_data) # type: ignore # Intentionally passing invalid types
    assert success is False
    assert expected_log_substring in caplog.text
    # Ensure registry remains unchanged or only contains previously valid items
    assert "IncompleteToolkit" not in registry.list_toolkits()
    assert "WrongTypeToolkit" not in registry.list_toolkits()
    assert "ExtraFieldToolkit" not in registry.list_toolkits()

def test_list_toolkits(registry: ToolRegistry):
    """Tests the list_toolkits method."""
    assert registry.list_toolkits() == []
    registry.load_toolkit_from_dict(VALID_TOOLKIT_DATA_1)
    assert registry.list_toolkits() == ["CalculatorToolkit"]
    registry.load_toolkit_from_dict(VALID_TOOLKIT_DATA_2)
    # Order might not be guaranteed, so use set for comparison
    assert set(registry.list_toolkits()) == {"CalculatorToolkit", "FileReaderToolkit"}