"""
Unit tests for the Web Search Toolkit.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

# Adjust the import path based on the project structure
# Assumes tests are run from the project root (Desktop/NOVA_SHIFT)
from tools.web_search.web_search_toolkit import perform_web_search, _initialize_tavily_client

# --- Test Cases ---

@pytest.fixture(autouse=True)
def reset_tavily_client():
    """Reset the global client before each test."""
    # Access the global client defined in the module we're testing
    from tools.web_search import web_search_toolkit
    web_search_toolkit.tavily_client = None

@patch('tools.web_search.web_search_toolkit.TavilyClient')
def test_perform_web_search_expected_use(MockTavilyClient): # Removed async
    """Tests successful web search."""
    # Arrange
    mock_client_instance = MockTavilyClient.return_value
    mock_response = {
        "query": "test query",
        "results": [
            {"title": "Result 1", "url": "http://example.com/1", "content": "Content snippet 1..."},
            {"title": "Result 2", "url": "http://example.com/2", "content": "Content snippet 2..."},
        ]
    }
    mock_client_instance.search.return_value = mock_response
    os.environ["TAVILY_API_KEY"] = "fake_key" # Ensure key exists for initialization check

    # Act
    query = "test query"
    result = perform_web_search(query, max_results=2) # Removed await

    # Assert
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"title": "Result 1", "url": "http://example.com/1", "content": "Content snippet 1..."}
    assert result[1] == {"title": "Result 2", "url": "http://example.com/2", "content": "Content snippet 2..."}
    mock_client_instance.search.assert_called_once_with(
        query=query,
        search_depth="basic",
        max_results=2,
        include_answer=False
    )
    # Clean up env var if needed, though mocking should prevent actual use
    del os.environ["TAVILY_API_KEY"]

@patch('tools.web_search.web_search_toolkit.TavilyClient') # Corrected patch target
def test_perform_web_search_no_results(MockTavilyClient): # Removed async
    """Tests web search returning no results."""
    # Arrange
    mock_client_instance = MockTavilyClient.return_value
    mock_response = {"query": "obscure query", "results": []}
    mock_client_instance.search.return_value = mock_response
    os.environ["TAVILY_API_KEY"] = "fake_key"

    # Act
    query = "obscure query"
    result = perform_web_search(query) # Removed await

    # Assert
    assert isinstance(result, list)
    assert len(result) == 0
    mock_client_instance.search.assert_called_once()

    del os.environ["TAVILY_API_KEY"]

@patch('tools.web_search.web_search_toolkit.TavilyClient') # Corrected patch target
def test_perform_web_search_api_error(MockTavilyClient): # Removed async
    """Tests handling of an error during the Tavily API call."""
    # Arrange
    mock_client_instance = MockTavilyClient.return_value
    mock_client_instance.search.side_effect = Exception("Tavily API unavailable")
    os.environ["TAVILY_API_KEY"] = "fake_key"

    # Act
    query = "query causing error"
    result = perform_web_search(query) # Removed await

    # Assert
    assert isinstance(result, str)
    assert "Error: An unexpected error occurred during web search" in result # Reverted error check
    assert "Tavily API unavailable" in result
    mock_client_instance.search.assert_called_once()

    del os.environ["TAVILY_API_KEY"]

def test_perform_web_search_no_api_key(): # Removed async
    """Tests behavior when TAVILY_API_KEY is not set."""
    # Arrange
    if "TAVILY_API_KEY" in os.environ:
        del os.environ["TAVILY_API_KEY"] # Ensure key is not set
    # Reset the global client instance as it might have been initialized by previous tests
    from tools.web_search import web_search_toolkit
    web_search_toolkit.tavily_client = None

    # Act
    result = perform_web_search("test query") # Removed await

    # Assert
    assert isinstance(result, str)
    assert "Error: Tavily client could not be initialized" in result
    assert "Check API key" in result

@patch('tools.web_search.web_search_toolkit.TavilyClient', None) # Corrected patch target
def test_perform_web_search_library_not_installed(): # Removed async
    """Tests behavior if tavily library is not installed (simulated)."""
    # Arrange
    # The patch sets TavilyClient to None, simulating import failure
    if "TAVILY_API_KEY" in os.environ:
         os.environ["TAVILY_API_KEY"] = "fake_key" # Key presence doesn't matter here
    # Reset the global client instance as it might have been initialized by previous tests
    from tools.web_search import web_search_toolkit
    web_search_toolkit.tavily_client = None

    # Act
    result = perform_web_search("test query") # Removed await

    # Assert
    assert isinstance(result, str)
    assert "Error: Tavily client could not be initialized" in result # Should fail initialization
    assert "Check API key and installation" in result # Message includes installation check
def test_perform_web_search_empty_query(): # Removed async
    """Tests passing an empty query string."""
    result = perform_web_search("") # Removed await
    assert isinstance(result, str)
    assert result == "Error: Input query cannot be empty."

def test_perform_web_search_invalid_query_type(): # Removed async
    """Tests passing a non-string query."""
    result = perform_web_search(12345) # type: ignore # Removed await
    assert isinstance(result, str)
    assert result == "Error: Input query must be a string."

# Test initialization separately
@patch('tools.web_search.web_search_toolkit.TavilyClient') # Corrected patch target
def test_initialize_tavily_client_success(MockTavilyClient):
    """Tests successful client initialization."""
    os.environ["TAVILY_API_KEY"] = "fake_key"
    client = _initialize_tavily_client()
    assert client is not None
    MockTavilyClient.assert_called_once_with(api_key="fake_key")
    del os.environ["TAVILY_API_KEY"]

def test_initialize_tavily_client_no_key():
    """Tests client initialization failure due to missing key."""
    if "TAVILY_API_KEY" in os.environ:
        del os.environ["TAVILY_API_KEY"]
    client = _initialize_tavily_client()
    assert client is None