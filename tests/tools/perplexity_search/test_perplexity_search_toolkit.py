import asyncio
import json
import os
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from openai import OpenAIError  # Import the specific error type

# Module to test
from tools.perplexity_search import perplexity_search_toolkit

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio

# --- Test Fixtures ---

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mocks required environment variables for tests."""
    # Only PERPLEXITY_API_KEY is needed now
    monkeypatch.setenv("PERPLEXITY_API_KEY", "fake_perplexity_key")

@pytest.fixture
def mock_openai_client():
    """Provides a mock AsyncOpenAI client."""
    mock_client_instance = AsyncMock()
    # Mock the specific method used in the toolkit
    mock_client_instance.chat.completions.create = AsyncMock()

    # Mock the class instantiation to return our mock instance
    mock_client_class = MagicMock(return_value=mock_client_instance)
    return mock_client_class, mock_client_instance

# --- Test Cases ---

async def test_perplexity_search_success(mock_openai_client):
    """Tests successful search returning expected content."""
    mock_client_class, mock_client_instance = mock_openai_client
    query = "What is Nova SHIFT?"
    expected_content = "A swarm-hive intelligence architecture."
    model_to_use = perplexity_search_toolkit.DEFAULT_MODEL # Use default

    # Mock the response structure from client.chat.completions.create
    mock_response = AsyncMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = expected_content
    mock_client_instance.chat.completions.create.return_value = mock_response

    with patch(
        "tools.perplexity_search.perplexity_search_toolkit.AsyncOpenAI",
        mock_client_class,
    ):
        result_str = await perplexity_search_toolkit.perplexity_search(query)

    # Assertions
    mock_client_class.assert_called_once_with(
        api_key="fake_perplexity_key",
        base_url=perplexity_search_toolkit.PERPLEXITY_API_BASE_URL,
    )
    mock_client_instance.chat.completions.create.assert_awaited_once_with(
        model=model_to_use,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant providing concise and accurate search results.",
            },
            {"role": "user", "content": query},
        ],
    )
    assert isinstance(result_str, str)
    result_dict = json.loads(result_str)
    assert result_dict == {"result": expected_content}

async def test_perplexity_search_success_custom_model(mock_openai_client):
    """Tests successful search using a non-default model."""
    mock_client_class, mock_client_instance = mock_openai_client
    query = "Explain meta-learning."
    custom_model = "sonar-medium-online" # Example custom model
    expected_content = "Learning how to learn."

    mock_response = AsyncMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = expected_content
    mock_client_instance.chat.completions.create.return_value = mock_response

    with patch(
        "tools.perplexity_search.perplexity_search_toolkit.AsyncOpenAI",
        mock_client_class,
    ):
        # Pass the custom model name
        result_str = await perplexity_search_toolkit.perplexity_search(query, model=custom_model)

    # Assertions
    mock_client_instance.chat.completions.create.assert_awaited_once_with(
        model=custom_model, # Verify custom model was used
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant providing concise and accurate search results.",
            },
            {"role": "user", "content": query},
        ],
    )
    assert isinstance(result_str, str)
    result_dict = json.loads(result_str)
    assert result_dict == {"result": expected_content}


async def test_perplexity_search_api_error(mock_openai_client):
    """Tests handling of OpenAIError during API call."""
    mock_client_class, mock_client_instance = mock_openai_client
    query = "Query causing API error"
    error_message = "Invalid API key."
    # Simulate the API call raising an OpenAIError
    mock_client_instance.chat.completions.create.side_effect = OpenAIError(error_message)
    expected_error_dict = {"error": f"Perplexity API Error: {error_message}"}

    with patch(
        "tools.perplexity_search.perplexity_search_toolkit.AsyncOpenAI",
        mock_client_class,
    ):
        result_str = await perplexity_search_toolkit.perplexity_search(query)

    # Assertions
    mock_client_instance.chat.completions.create.assert_awaited_once() # Ensure it was called
    assert isinstance(result_str, str)
    result_dict = json.loads(result_str)
    # Check if the specific error message is included
    assert "error" in result_dict
    assert error_message in result_dict["error"]


async def test_perplexity_search_missing_perplexity_key(monkeypatch):
    """Tests failure when PERPLEXITY_API_KEY is missing."""
    # Remove the key specifically for this test
    monkeypatch.delenv("PERPLEXITY_API_KEY")
    query = "Test query"

    # Expect a KeyError to be raised directly by the function
    with pytest.raises(KeyError, match="PERPLEXITY_API_KEY"):
        await perplexity_search_toolkit.perplexity_search(query)

async def test_perplexity_search_no_choices(mock_openai_client):
    """Tests handling when the API returns no choices."""
    mock_client_class, mock_client_instance = mock_openai_client
    query = "Query returning no choices"

    # Mock the response with an empty choices list
    mock_response = AsyncMock()
    mock_response.choices = [] # Empty list
    mock_client_instance.chat.completions.create.return_value = mock_response
    expected_error_dict = {"error": "Perplexity API returned no choices."}

    with patch(
        "tools.perplexity_search.perplexity_search_toolkit.AsyncOpenAI",
        mock_client_class,
    ):
        result_str = await perplexity_search_toolkit.perplexity_search(query)

    # Assertions
    mock_client_instance.chat.completions.create.assert_awaited_once()
    assert isinstance(result_str, str)
    result_dict = json.loads(result_str)
    assert result_dict == expected_error_dict

async def test_perplexity_search_empty_content(mock_openai_client):
    """Tests handling when the API returns a choice with empty content."""
    mock_client_class, mock_client_instance = mock_openai_client
    query = "Query returning empty content"

    # Mock the response structure but with empty content
    mock_response = AsyncMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "" # Empty string
    mock_client_instance.chat.completions.create.return_value = mock_response
    expected_error_dict = {"error": "Perplexity API returned empty content."}

    with patch(
        "tools.perplexity_search.perplexity_search_toolkit.AsyncOpenAI",
        mock_client_class,
    ):
        result_str = await perplexity_search_toolkit.perplexity_search(query)

    # Assertions
    mock_client_instance.chat.completions.create.assert_awaited_once()
    assert isinstance(result_str, str)
    result_dict = json.loads(result_str)
    assert result_dict == expected_error_dict

async def test_perplexity_search_unexpected_error(mock_openai_client):
    """Tests handling of unexpected errors during the process."""
    mock_client_class, mock_client_instance = mock_openai_client
    query = "Query causing unexpected error"
    error_message = "Something went very wrong."
    # Simulate an unexpected error during the API call
    mock_client_instance.chat.completions.create.side_effect = ValueError(error_message)
    expected_error_dict = {"error": f"Unexpected Error: ValueError - {error_message}"}


    with patch(
        "tools.perplexity_search.perplexity_search_toolkit.AsyncOpenAI",
        mock_client_class,
    ):
        result_str = await perplexity_search_toolkit.perplexity_search(query)

    # Assertions
    mock_client_instance.chat.completions.create.assert_awaited_once()
    assert isinstance(result_str, str)
    result_dict = json.loads(result_str)
    assert result_dict == expected_error_dict