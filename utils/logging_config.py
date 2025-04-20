"""
Logging configuration for Nova SHIFT.

Provides structured JSON logging.
Addresses Task P5.T5.1.
"""

import logging
import json
import traceback
import time # Added for latency tracking
from typing import Any, Dict, List, Optional, Union, Sequence # Added for callback types
from uuid import UUID
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage # Added for message handling

# Get the logger for this module
logger = logging.getLogger(__name__)

class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON strings.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, 'extra_data') and isinstance(record.extra_data, dict):
            log_entry.update(record.extra_data)

        # Add exception info if available
        if record.exc_info:
            log_entry['exception'] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        elif record.exc_text:
             log_entry['exception_text'] = record.exc_text


        # Add standard fields if needed (e.g., lineno, pathname)
        # log_entry['lineno'] = record.lineno
        # log_entry['pathname'] = record.pathname

        try:
            return json.dumps(log_entry, ensure_ascii=False)
        except TypeError as e:
            # Fallback for un-serializable data
            log_entry['message'] = f"Original message: {record.getMessage()}. Error serializing log: {e}"
            log_entry['unserializable_data_keys'] = list(getattr(record, 'extra_data', {}).keys())
            return json.dumps({k: str(v) for k, v in log_entry.items()}, ensure_ascii=False)


def setup_logging(level=logging.INFO):
    """
    Configures the root logger to use the JSONFormatter.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers if any (to avoid duplicate logs)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a handler (e.g., StreamHandler to output to console)
    handler = logging.StreamHandler()

    # Create and set the JSON formatter
    formatter = JSONFormatter()
    handler.setFormatter(formatter)

    # Add the handler to the root logger
    root_logger.addHandler(handler)

    logger.info("JSON logging configured.", extra={'extra_data': {'config_level': level}}) # Use module logger

# --- LLM Tracking Callback ---

class LLMTrackingCallback(BaseCallbackHandler):
    """Callback handler to track LLM token usage and latency."""

    def __init__(self):
        self.start_time: Optional[float] = None
        self.total_tokens: int = 0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.successful_requests: int = 0
        self.total_latency: float = 0.0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], *, run_id: UUID, parent_run_id: Optional[UUID] = None, tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self.start_time = time.monotonic()
        # Log the start event if needed (can be verbose)
        # logger.debug("LLM Call Start", extra={'extra_data': {'run_id': str(run_id), 'prompts_count': len(prompts)}})

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        if self.start_time is None:
            return # Should not happen if on_llm_start was called

        latency = time.monotonic() - self.start_time
        self.total_latency += latency
        self.successful_requests += 1

        token_usage = response.llm_output.get("token_usage", {}) if response.llm_output else {}
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", prompt_tokens + completion_tokens) # Calculate if not present

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens

        log_data = {
            'run_id': str(run_id),
            'latency_ms': round(latency * 1000, 2),
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            # 'model_name': response.llm_output.get("model_name") # Requires model to return this
        }
        logger.info("LLM Call End", extra={'extra_data': log_data})
        self.start_time = None # Reset for next call

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
        log_data = {
            'run_id': str(run_id),
            'error_type': type(error).__name__,
            'error_message': str(error),
        }
        logger.error("LLM Call Error", exc_info=error, extra={'extra_data': log_data})
        self.start_time = None # Reset timer on error

    def get_stats(self) -> Dict[str, Any]:
        """Returns the aggregated statistics."""
        return {
            "total_llm_calls": self.successful_requests,
            "total_prompt_tokens": self.prompt_tokens,
            "total_completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "average_latency_ms": round((self.total_latency / self.successful_requests) * 1000, 2) if self.successful_requests > 0 else 0,
        }

# Example usage (optional)
if __name__ == "__main__":
    setup_logging(level=logging.DEBUG)
    test_logger = logging.getLogger("test_logger") # Use different logger name for example

    test_logger.debug("This is a debug message.") # Use test_logger
    test_logger.info("This is an info message.", extra={'extra_data': {'user_id': 123, 'task': 'test'}}) # Use test_logger
    test_logger.warning("This is a warning.") # Use test_logger
    try:
        1 / 0
    except ZeroDivisionError as e:
        test_logger.error("This is an error message with exception info.", exc_info=True, extra={'extra_data': {'calculation': '1/0'}}) # Use test_logger

    # Example Callback Usage (Manual Simulation)
    print("\n--- Testing LLM Callback ---")
    callback = LLMTrackingCallback()
    mock_run_id = UUID('123e4567-e89b-12d3-a456-426614174000')
    mock_response = LLMResult(
        generations=[], # Simplified for example
        llm_output={"token_usage": {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}, "model_name": "gpt-test"}
    )
    callback.on_llm_start(serialized={}, prompts=["Test prompt"], run_id=mock_run_id)
    time.sleep(0.15) # Simulate latency
    callback.on_llm_end(response=mock_response, run_id=mock_run_id)

    mock_run_id_2 = UUID('123e4567-e89b-12d3-a456-426614174001')
    callback.on_llm_start(serialized={}, prompts=["Another prompt"], run_id=mock_run_id_2)
    time.sleep(0.05)
    try:
        raise ValueError("Simulated LLM API error")
    except ValueError as err:
        callback.on_llm_error(error=err, run_id=mock_run_id_2)

    stats = callback.get_stats()
    print("\nLLM Tracking Stats:")
    print(json.dumps(stats, indent=2))