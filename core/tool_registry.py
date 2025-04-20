"""
Implements the in-memory Tool Registry for Nova SHIFT.

This registry loads, validates, and provides access to toolkit definitions.
"""

import logging
from typing import Dict, Optional, List # Added List
from typing import Dict, Optional

# Assuming toolkit_schema.py is accessible via PYTHONPATH or relative import
# For initial setup, let's assume it's importable. Adjust if needed.
# Need to create __init__.py files later for proper packaging.
try:
    # Attempt relative import if core and tools are siblings under a common root (e.g., src)
    # This might fail depending on how the project is run initially.
    from ..tools.toolkit_schema import ToolkitSchema
except ImportError:
    # Fallback for direct script execution or different structure
    # This path might need adjustment based on final project structure/PYTHONPATH
    try:
        from tools.toolkit_schema import ToolkitSchema
    except ImportError:
        logging.error("Could not import ToolkitSchema. Ensure tools/toolkit_schema.py is accessible.")
        # Define a dummy class to avoid crashing if import fails during early dev
        class ToolkitSchema: pass


from pydantic import ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ToolRegistry:
    """Manages the loading, validation, and retrieval of toolkit definitions.

    This registry holds validated ToolkitSchema objects in memory, keyed by
    the toolkit name. It provides methods to load definitions from dictionaries
    (typically parsed from JSON) and retrieve them.

    Attributes:
        _registry (Dict[str, ToolkitSchema]): The in-memory dictionary storing
            loaded and validated toolkit schemas.
    """

    def __init__(self):
        """Initializes an empty Tool Registry."""
        self._registry: Dict[str, ToolkitSchema] = {}
        logger.info("Tool Registry initialized.")

    def load_toolkit_from_dict(self, toolkit_data: Dict) -> bool:
        """Loads and validates a toolkit definition from a dictionary.

        Validates the input dictionary against the ToolkitSchema using Pydantic.
        If validation succeeds, the toolkit is added to the registry, potentially
        overwriting an existing entry with the same name.

        Args:
            toolkit_data (Dict): A dictionary representing the toolkit definition,
                expected to conform to the ToolkitSchema.

        Returns:
            bool: True if the toolkit was loaded and validated successfully,
            False otherwise (due to invalid data, validation errors, or
            other exceptions).
        """
        if not isinstance(toolkit_data, dict):
            logger.error("Failed to load toolkit: Input data is not a dictionary.")
            return False

        try:
            # Validate the dictionary against the Pydantic schema
            toolkit = ToolkitSchema.model_validate(toolkit_data)
            toolkit_name = toolkit.name

            if toolkit_name in self._registry:
                # Handle potential version conflicts or updates if needed later
                logger.warning(f"Toolkit '{toolkit_name}' already exists. Overwriting with new definition.")

            self._registry[toolkit_name] = toolkit
            logger.info(f"Successfully loaded and validated toolkit: '{toolkit_name}' version {toolkit.version}")
            return True

        except ValidationError as e:
            # Log validation errors
            toolkit_name_attempt = toolkit_data.get('name', 'Unknown')
            logger.error(f"Validation failed for toolkit '{toolkit_name_attempt}': {e}")
            return False
        except Exception as e:
            # Catch any other unexpected errors during loading
            toolkit_name_attempt = toolkit_data.get('name', 'Unknown')
            logger.error(f"An unexpected error occurred while loading toolkit '{toolkit_name_attempt}': {e}", exc_info=True)
            return False

    def get_toolkit(self, name: str) -> Optional[ToolkitSchema]:
        """Retrieves a loaded toolkit definition by its name.

        Args:
            name (str): The unique name of the toolkit to retrieve.

        Returns:
            Optional[ToolkitSchema]: The validated ToolkitSchema object if found
            in the registry, otherwise None.
        """
        toolkit = self._registry.get(name)
        if not toolkit:
            logger.warning(f"Toolkit '{name}' not found in registry.")
        return toolkit

    def list_toolkits(self) -> List[str]:
        """Lists the names of all currently loaded toolkits.

        Returns:
            List[str]: A list containing the names (keys) of all toolkits
            present in the registry.
        """
        return list(self._registry.keys())

# Example Usage (can be removed or moved to tests later)
if __name__ == '__main__':
    # Create dummy schema if import failed
    if not hasattr(ToolkitSchema, 'model_validate'):
        class ToolDefinition: pass
        class ToolkitRequirements: pass
        class ToolkitLoadingInfo: pass
        class ToolkitSchema:
            name: str = "DummySchema"
            version: str = "0.0"
            description: str = ""
            tools: list = []
            requirements: Optional[ToolkitRequirements] = None
            loading_info: ToolkitLoadingInfo = ToolkitLoadingInfo()
            @classmethod
            def model_validate(cls, data): return cls()


    registry = ToolRegistry()

    # Example valid toolkit data (matches the schema)
    valid_toolkit_data = {
        "name": "ExampleToolkit",
        "version": "1.0.0",
        "description": "An example toolkit.",
        "tools": [
            {
                "name": "ExampleTool",
                "function": "run_example",
                "description": "Runs an example.",
                "inputs": ["input_data:string"],
                "outputs": ["output_data:string"]
            }
        ],
        "requirements": {
            "python_packages": ["requests"]
        },
        "loading_info": {
            "type": "python_module",
            "path": "nova_shift.tools.example.ExampleToolkit"
        }
    }

    # Example invalid toolkit data (missing required 'loading_info')
    invalid_toolkit_data = {
        "name": "InvalidToolkit",
        "version": "1.0",
        "description": "This toolkit is invalid.",
        "tools": []
        # Missing loading_info
    }

    # Load the valid toolkit
    load_success = registry.load_toolkit_from_dict(valid_toolkit_data)
    print(f"Loading valid toolkit successful: {load_success}")

    # Attempt to load the invalid toolkit
    load_fail = registry.load_toolkit_from_dict(invalid_toolkit_data)
    print(f"Loading invalid toolkit successful: {load_fail}")

    # Retrieve the loaded toolkit
    retrieved_toolkit = registry.get_toolkit("ExampleToolkit")
    if retrieved_toolkit:
        print(f"Retrieved toolkit: {retrieved_toolkit.name} v{retrieved_toolkit.version}")
        print(f"Description: {retrieved_toolkit.description}")

    # Attempt to retrieve a non-existent toolkit
    not_found_toolkit = registry.get_toolkit("NonExistentToolkit")
    print(f"Retrieved non-existent toolkit: {not_found_toolkit}")

    # List loaded toolkits
    print(f"Loaded toolkits: {registry.list_toolkits()}")

    # Test loading non-dict data
    load_non_dict = registry.load_toolkit_from_dict("not a dict")
    print(f"Loading non-dict data successful: {load_non_dict}")