"""
Implementation of the Web Search Toolkit for Nova SHIFT.

Uses the Tavily API to perform web searches.
Requires the TAVILY_API_KEY environment variable to be set.
"""

import logging
from typing import List, Dict, Any, Optional, Union # Combined imports
import os
# Removed sandbox imports

try:
    from tavily import TavilyClient
except ImportError:
    # This allows the file to be imported even if tavily isn't installed yet,
    # though the function will fail at runtime if called.
    TavilyClient = None
    logging.warning("TavilyClient could not be imported. Install 'tavily-python'.")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global Tavily client instance (initialized when first needed)
tavily_client: Optional[TavilyClient] = None

def _initialize_tavily_client() -> Optional[TavilyClient]:
    """Initializes the Tavily client if not already done."""
    global tavily_client
    if tavily_client is None:
        if TavilyClient is None:
            logger.error("Tavily library is not installed. Cannot initialize client.")
            return None
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            logger.error("TAVILY_API_KEY environment variable not set.")
            return None
        try:
            tavily_client = TavilyClient(api_key=api_key)
            logger.info("Tavily client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Tavily client: {e}", exc_info=True)
            return None
    return tavily_client

# --- Synchronous Tool Function ---
def perform_web_search(query: str, max_results: int = 5) -> Union[List[Dict[str, str]], str]:
    """
    Performs a web search using the Tavily API.

    Args:
        query: The search query string.
        max_results: The maximum number of search results to return.

    Returns:
        A list of dictionaries, where each dictionary represents a search result
        containing 'title', 'url', and 'content' (snippet/summary),
        or an error message string if the search fails.
    """
    if not isinstance(query, str):
        return "Error: Input query must be a string."
    if not query:
        return "Error: Input query cannot be empty."

    client = _initialize_tavily_client()
    if client is None:
        return "Error: Tavily client could not be initialized. Check API key and installation."

    logger.info(f"Performing Tavily web search for query: '{query}' (max_results={max_results})")
    try:
        # Perform the search using Tavily client
        # We request 'include_answer=False' to get standard search results
        response = client.search(
            query=query,
            search_depth="basic", # Use "basic" for standard search results, "advanced" for deeper research report
            max_results=max_results,
            include_answer=False # Set to False to get list of results, True for a summarized answer
        )

        # Extract the results list
        search_results = response.get('results', [])

        # Format results into the desired list of dictionaries
        formatted_results = [
            {"title": result.get("title", ""), "url": result.get("url", ""), "content": result.get("content", "")}
            for result in search_results
        ]

        logger.info(f"Tavily search successful for query '{query}'. Found {len(formatted_results)} results.")
        return formatted_results

    except Exception as e:
        logger.error(f"An error occurred during Tavily search for query '{query}': {e}", exc_info=True)
        return f"Error: An unexpected error occurred during web search: {e}"

# Example Usage (can be removed or moved to tests later)
# Requires TAVILY_API_KEY environment variable to be set
if __name__ == '__main__':
    # Make sure to set the TAVILY_API_KEY environment variable before running this
    # Example: export TAVILY_API_KEY='your_api_key' (Linux/macOS)
    # Example: $env:TAVILY_API_KEY='your_api_key' (PowerShell)

    test_query = "What is the SHIFT architecture in AI?"
    print(f"Performing web search for: '{test_query}'")

    results_or_error = perform_web_search(test_query)

    if isinstance(results_or_error, str):
        print(results_or_error)
    elif isinstance(results_or_error, list):
        print(f"Found {len(results_or_error)} results:")
        for i, result in enumerate(results_or_error):
            print(f"\nResult {i+1}:")
            print(f"  Title: {result.get('title')}")
            print(f"  URL: {result.get('url')}")
            print(f"  Content: {result.get('content', '')[:150]}...") # Print first 150 chars
    else:
        print("Unexpected return type.")

    print("-" * 20)

    # Test empty query
    print("Testing empty query:")
    results_or_error_empty = perform_web_search("")
    print(results_or_error_empty)
    print("-" * 20)

    # Test without API key (if not set)
    # temp_key = os.environ.pop("TAVILY_API_KEY", None) # Temporarily remove key
    # tavily_client = None # Reset client
    # print("Testing without API key (if key was removed):")
    # results_no_key = perform_web_search("test query")
    # print(results_no_key)
    # if temp_key: os.environ["TAVILY_API_KEY"] = temp_key # Restore key