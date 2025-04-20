"""
Defines the Pydantic models for the toolkit.json schema.

This schema describes the structure for defining toolkits, including their
metadata, the tools they contain, requirements, and loading information.
"""

from typing import List, Optional, Dict, Literal

from pydantic import BaseModel, Field


class ToolDefinition(BaseModel):
    """
    Defines the structure for a single tool within a toolkit.
    """
    name: str = Field(..., description="Unique tool name within the toolkit.")
    function: str = Field(
        ...,
        description="Name of the Python function/method to call for this tool."
    )
    description: str = Field(
        ...,
        description="Detailed description for LLM reasoning/selection and human understanding."
    )
    inputs: List[str] = Field(
        default_factory=list,
        description="List of input parameters as 'name:type' strings (e.g., 'query:string')."
    )
    outputs: List[str] = Field(
        default_factory=list,
        description="List of output parameters as 'name:type' strings (e.g., 'results:list<string>')."
    )
    dependencies: Optional[List[str]] = Field(
        None,
        description="Optional list of names of other tools this tool depends on."
    )


class ToolkitRequirements(BaseModel):
    """
    Defines optional requirements needed to run the tools in the toolkit.
    """
    python_packages: Optional[List[str]] = Field(
        None,
        description="List of required Python packages (e.g., 'requests>=2.0')."
    )
    api_keys: Optional[List[str]] = Field(
        None,
        description="List of environment variable names holding required API keys."
    )


class ToolkitLoadingInfo(BaseModel):
    """
    Defines how the toolkit's code should be loaded.
    """
    type: Literal["python_module"] = Field(
        ...,
        description="The loading mechanism type. Currently only 'python_module' is supported."
    )
    path: str = Field(
        ...,
        description="The Python import path to the toolkit's main class or module (e.g., 'nova_shift.tools.web_search.WebSearchToolkit')."
    )


class ToolkitSchema(BaseModel):
    """
    Root model defining the structure of a toolkit.json file.
    """
    name: str = Field(..., description="Unique toolkit identifier name.")
    version: str = Field(
        ...,
        description="Semantic version string for the toolkit (e.g., '1.0.0')."
    )
    description: str = Field(
        ...,
        description="Human-readable description of the toolkit's overall purpose."
    )
    tools: List[ToolDefinition] = Field(
        ...,
        description="List of tools provided by this toolkit."
    )
    requirements: Optional[ToolkitRequirements] = Field(
        None,
        description="Optional dependencies required by the toolkit."
    )
    loading_info: ToolkitLoadingInfo = Field(
        ...,
        description="Information on how to load and instantiate the toolkit."
    )

    class Config:
        """Pydantic configuration."""
        extra = 'forbid' # Disallow extra fields not defined in the schema