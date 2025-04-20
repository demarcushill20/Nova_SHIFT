import asyncio
import os
from unittest.mock import patch, AsyncMock, MagicMock

import pytest
from pytest_asyncio import fixture

# Module to test
from nova_shift.tools.browser_use_agent.browser_use_agent_toolkit import run_browser_use_gemini_task

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio

# --- Mocks ---

@fixture
async def mock_llm():
    """Pytest fixture for a mocked ChatGoogleGenerativeAI instance."""
    mock = AsyncMock()
    # Mock the ainvoke method to return a mock response object with content
    mock_response = MagicMock()
    mock_response.content = "Mocked LLM response content."
    mock.ainvoke.return_value = mock_response
    return mock

@fixture
async def mock_browser_context():
    """Pytest fixture for a mocked BrowserContext."""
    mock = AsyncMock()
    mock.close = AsyncMock() # Mock the async close method
    return mock

@fixture
async def mock_browser():
    """Pytest fixture for a mocked Browser instance."""
    mock = AsyncMock()
    mock.new_context = AsyncMock(return_value=await fixture('mock_browser_context')()) # Use await fixture
    mock.close = AsyncMock() # Mock the async close method
    return mock

@fixture
async def mock_agent():
    """Pytest fixture for a mocked Agent instance."""
    mock = AsyncMock()
    # Mock the run method to return a mock result object
    mock_result_obj = MagicMock()
    # Simulate the structure expected by _get_result_text
    mock_last_action = MagicMock()
    mock_last_action.extracted_content = "Mocked raw browser agent result."
    mock_result_obj.all_results = [mock_last_action]
    mock.run = AsyncMock(return_value=mock_result_obj)
    return mock

# --- Test Cases ---

@patch('nova_shift.tools.browser_use_agent.browser_use_agent_toolkit.ChatGoogleGenerativeAI')
@patch('nova_shift.tools.browser_use_agent.browser_use_agent_toolkit.Browser')
@patch('nova_shift.tools.browser_use_agent.browser_use_agent_toolkit.Agent')
@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True) # Ensure key exists for this test
async def test_run_browser_use_gemini_task_success(
    mock_agent_cls, mock_browser_cls, mock_llm_cls,
    mock_agent, mock_browser, mock_llm # Use fixtures for instances
):
    """
    Tests successful execution of the tool, returning raw results.
    """
    # Configure mocks
    mock_llm_cls.return_value = mock_llm
    mock_browser_cls.return_value = mock_browser
    mock_agent_cls.return_value = mock_agent

    task_desc = "Find the latest AI news"
    expected_result = "Mocked raw browser agent result."

    result = await run_browser_use_gemini_task(task_desc)

    # Assertions
    mock_llm_cls.assert_called_once() # Check LLM was initialized
    mock_browser_cls.assert_called_once() # Check Browser was initialized
    mock_browser.new_context.assert_awaited_once() # Check context was created
    mock_agent_cls.assert_called_once_with(
        task=task_desc,
        llm=mock_llm,
        browser_context=await mock_browser.new_context() # Check agent was initialized correctly
    )
    mock_agent.run.assert_awaited_once() # Check agent ran
    assert result == expected_result # Check correct result was returned
    mock_browser.close.assert_awaited_once() # Check browser was closed
    (await mock_browser.new_context()).close.assert_awaited_once() # Check context was closed


@patch.dict(os.environ, {}, clear=True) # Ensure GOOGLE_API_KEY is NOT set
async def test_run_browser_use_gemini_task_no_api_key():
    """
    Tests that the function returns an error if GOOGLE_API_KEY is missing.
    """
    task_desc = "This task should fail"
    expected_error_msg = "Error: GOOGLE_API_KEY environment variable not set."

    result = await run_browser_use_gemini_task(task_desc)

    assert result == expected_error_msg

@patch('nova_shift.tools.browser_use_agent.browser_use_agent_toolkit.ChatGoogleGenerativeAI')
@patch('nova_shift.tools.browser_use_agent.browser_use_agent_toolkit.Browser')
@patch('nova_shift.tools.browser_use_agent.browser_use_agent_toolkit.Agent')
@patch('nova_shift.tools.browser_use_agent.browser_use_agent_toolkit._generate_summary', new_callable=AsyncMock) # Mock the helper
@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
async def test_run_browser_use_gemini_task_summarize(
    mock_generate_summary, mock_agent_cls, mock_browser_cls, mock_llm_cls,
    mock_agent, mock_browser, mock_llm
):
    """
    Tests successful execution with summarization enabled.
    """
    # Configure mocks
    mock_llm_cls.return_value = mock_llm
    mock_browser_cls.return_value = mock_browser
    mock_agent_cls.return_value = mock_agent
    mock_generate_summary.return_value = "Mocked summary content."

    task_desc = "Summarize AI news"
    raw_result_text = "Mocked raw browser agent result." # Expected from _get_result_text

    result = await run_browser_use_gemini_task(task_desc, summarize=True)

    # Assertions
    mock_agent.run.assert_awaited_once()
    mock_generate_summary.assert_awaited_once_with(mock_llm, raw_result_text, task_desc)
    assert result == "Mocked summary content."
    mock_browser.close.assert_awaited_once()
    (await mock_browser.new_context()).close.assert_awaited_once()


@patch('nova_shift.tools.browser_use_agent.browser_use_agent_toolkit.ChatGoogleGenerativeAI')
@patch('nova_shift.tools.browser_use_agent.browser_use_agent_toolkit.Browser')
@patch('nova_shift.tools.browser_use_agent.browser_use_agent_toolkit.Agent')
@patch('nova_shift.tools.browser_use_agent.browser_use_agent_toolkit._generate_detailed_report', new_callable=AsyncMock) # Mock the helper
@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True)
async def test_run_browser_use_gemini_task_generate_report(
    mock_generate_report, mock_agent_cls, mock_browser_cls, mock_llm_cls,
    mock_agent, mock_browser, mock_llm
):
    """
    Tests successful execution with report generation enabled.
    """
    # Configure mocks
    mock_llm_cls.return_value = mock_llm
    mock_browser_cls.return_value = mock_browser
    mock_agent_cls.return_value = mock_agent
    mock_generate_report.return_value = "Mocked detailed report content."

    task_desc = "Generate report on AI news"
    raw_result_text = "Mocked raw browser agent result." # Expected from _get_result_text

    result = await run_browser_use_gemini_task(task_desc, generate_report=True)

    # Assertions
    mock_agent.run.assert_awaited_once()
    mock_generate_report.assert_awaited_once_with(mock_llm, raw_result_text, task_desc)
    assert result == "Mocked detailed report content."
    mock_browser.close.assert_awaited_once()
    (await mock_browser.new_context()).close.assert_awaited_once()

# TODO: Add tests for file output functionality (mocking open/makedirs)
# TODO: Add tests for LLM/Browser/Agent initialization errors