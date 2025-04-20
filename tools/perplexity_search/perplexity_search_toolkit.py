import asyncio
import json
import logging
import os
from typing import Any, Dict, Optional

# Use the official 'openai' library, configured for Perplexity API
# Ensure 'openai' package is installed (pip install openai)
import openai
from openai import AsyncOpenAI, OpenAIError

# Configure logging
# Adheres to logging setup potentially defined elsewhere in the project
logger = logging.getLogger(__name__)
# Basic config if run standalone or if root logger isn't configured
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Constants from Perplexity API documentation and user request
PERPLEXITY_API_BASE_URL = "https://api.perplexity.ai"
DEFAULT_MODEL = "sonar-pro"  # Use the requested flagship model


async def perplexity_search(query: str, model: str = DEFAULT_MODEL) -> str:
    """Performs a search query using the Perplexity API.

    Uses the `openai` Python client configured for the Perplexity API endpoint.
    Retrieves the necessary API key from the PERPLEXITY_API_KEY environment
    variable.

    Args:
        query (str): The search query string.
        model (str): The Perplexity model to use. Defaults to 'sonar-pro'.
                     Other options might include 'sonar-small-online', etc.

    Returns:
        str: A JSON string containing the search result under the key 'result',
             or an error message under the key 'error' if the search fails.

    Raises:
        KeyError: If the 'PERPLEXITY_API_KEY' environment variable is not set.
                  This error is raised to be handled by the calling framework.
    """
    logger.info(f"Initiating Perplexity search for query: '{query}' using model '{model}'")

    try:
        # Retrieve API key from environment - raising KeyError if not found
        # as per the function's docstring.
        perplexity_api_key = os.environ["PERPLEXITY_API_KEY"]
        logger.debug("PERPLEXITY_API_KEY found.")
    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        # Propagate the error as per design
        raise KeyError(
            f"Environment variable {e} not set. Needed for Perplexity search."
        ) from e

    try:
        # Initialize the AsyncOpenAI client pointed to Perplexity's API
        # Ensure proper async client usage
        client = AsyncOpenAI(
            api_key=perplexity_api_key, base_url=PERPLEXITY_API_BASE_URL
        )

        logger.info("Sending request to Perplexity API...")
        # Make the API call using the chat completions endpoint
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant providing concise and accurate search results.",
                },
                {"role": "user", "content": query},
            ],
            # Consider adding other parameters like temperature=0.7 if needed
            # temperature=0.7,
            # max_tokens=500,
        )
        logger.info("Received response from Perplexity API.")
        logger.debug(f"Perplexity API raw response object: {response}")

        # Extract the content from the first choice
        if response.choices and len(response.choices) > 0:
            result_content = response.choices[0].message.content
            if result_content:
                logger.debug(f"Extracted result content (length: {len(result_content)})")
                # Return the content wrapped in a JSON structure for consistency
                return json.dumps({"result": result_content}, indent=2)
            else:
                logger.warning("Perplexity API returned a choice but content was empty.")
                return json.dumps({"error": "Perplexity API returned empty content."}, indent=2)
        else:
            logger.warning("Perplexity API returned no choices in the response.")
            return json.dumps({"error": "Perplexity API returned no choices."}, indent=2)

    except OpenAIError as e:
        # Catch specific API errors from the openai library
        logger.error(f"Perplexity API Error during search: {e}", exc_info=True)
        # Return a JSON string indicating the API error
        return json.dumps({"error": f"Perplexity API Error: {e}"}, indent=2)
    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(f"Unexpected error during Perplexity search: {e}", exc_info=True)
        # Return a JSON string for other unexpected errors
        return json.dumps({"error": f"Unexpected Error: {type(e).__name__} - {e}"}, indent=2)


# Example usage block for standalone testing
async def _main():
    """Runs a test query if the script is executed directly."""
    # Make sure to set PERPLEXITY_API_KEY environment variable before running.
    # Example: export PERPLEXITY_API_KEY='your_pplx_api_key'
    test_query = "What are the latest advancements in quantum computing?"
    print(f"--- Testing Perplexity Search ---")
    print(f"Query: '{test_query}'")
    print(f"Model: '{DEFAULT_MODEL}'")

    if "PERPLEXITY_API_KEY" not in os.environ:
        print("\nERROR: PERPLEXITY_API_KEY environment variable not set.")
        print("Please set it before running the test:")
        print("  export PERPLEXITY_API_KEY='your_pplx_api_key'")
        return

    try:
        results_json = await perplexity_search(test_query)
        print("\n--- Search Results (JSON String) ---")
        print(results_json)

        # Attempt to parse and display the actual result or error
        print("\n--- Parsed Result ---")
        try:
            parsed_results = json.loads(results_json)
            if "result" in parsed_results:
                print(parsed_results["result"])
            elif "error" in parsed_results:
                print(f"API Error: {parsed_results['error']}")
            else:
                print("Unknown JSON structure.")
        except json.JSONDecodeError:
            print("Could not parse the returned JSON string.")

    except KeyError as e:
        # This handles the case where the key is missing, even though checked above.
        print(f"\nError: {e}")
    except Exception as e:
        # Catch any other unexpected error during the _main execution
        print(f"\nAn unexpected error occurred during the test run: {e}")
        logger.exception("Error in _main execution:") # Log stack trace

if __name__ == "__main__":
    # Note: Running this directly requires the PERPLEXITY_API_KEY to be set.
    # Example: export PERPLEXITY_API_KEY='your_pplx_api_key'
    #          python -m NOVA_SHIFT.tools.perplexity_search.perplexity_search_toolkit
    print("Running example usage...")
    asyncio.run(_main())
    print("--- Example usage finished ---")