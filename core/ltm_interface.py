"""
Interface for agents to interact with the Long-Term Memory (LTM) system.

Currently implemented using Pinecone Vector Database. Handles embedding generation
and provides methods for storing and retrieving information semantically.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import httpx # For making API calls to MCP Server
# from pinecone import Pinecone, PodSpec, ServerlessSpec, Index # MCP Integration: Pinecone direct access removed
# from langchain_openai import OpenAIEmbeddings # MCP Integration: Embedding handled by MCP

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Load from environment variables for security
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
# Pinecone environment is deprecated, use serverless or pod spec
# PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT") # e.g., "gcp-starter" or specific region
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "nova-shift-ltm")
# Specify pod environment for Pinecone Starter (free tier) or Serverless
PINECONE_POD_ENVIRONMENT = os.environ.get("PINECONE_POD_ENVIRONMENT") # No default, check presence
# Or for Serverless:
PINECONE_SERVERLESS_CLOUD = os.environ.get("PINECONE_SERVERLESS_CLOUD")
PINECONE_SERVERLESS_REGION = os.environ.get("PINECONE_SERVERLESS_REGION")


# Embedding model configuration
# Using OpenAI's text-embedding-3-small as an example
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
# Pinecone index dimension must match the embedding model's output dimension
# text-embedding-3-small default is 1536, text-embedding-3-large is 3072
EMBEDDING_DIMENSION = 1536


class LTMInterface:
    """
    Provides an interface to the Long-Term Memory via Nova Memory MCP.
    Handles communication with the MCP server for storing and retrieving information.
    """

    def __init__(self):
        """Initializes the LTMInterface.

        Sets up the HTTP client for communication with the Nova Memory MCP server.
        """
        self.mcp_base_url: str = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
        mcp_timeout_seconds = float(os.getenv("MCP_TIMEOUT_SECONDS", 30.0)) # From plan's .env
        self.session: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(mcp_timeout_seconds),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20) # Sensible defaults
        )
        logger.info(f"LTMInterface initialized. MCP Server URL: {self.mcp_base_url}, Timeout: {mcp_timeout_seconds}s")

    async def store(self, text: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Stores text and metadata in the LTM via Nova Memory MCP.

        Args:
            text (str): The text content to store.
            metadata (Dict[str, Any]): A dictionary of metadata associated with the text.

        Returns:
            Optional[Dict[str, Any]]: The JSON response from the MCP server if successful, else None.
        """
        payload = {"content": text, "metadata": metadata}
        request_url = f"{self.mcp_base_url}/memory/upsert" # As per plan
        logger.debug(f"Storing document via MCP. URL: {request_url}, Payload snippet: {{'content': '{text[:50]}...'}}")
        try:
            response = await self.session.post(request_url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            logger.info(f"Successfully stored document via MCP. Response status: {response.status_code}")
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"MCP HTTP error storing document: {e.response.status_code} - {e.response.text}", exc_info=True)
        except httpx.RequestError as e:
            logger.error(f"MCP Request error storing document: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"MCP Unexpected error storing document: {e}", exc_info=True)
        return None

    async def retrieve(self, query_text: str, top_k: int = 5) -> Optional[Dict[str, Any]]:
        """Retrieves relevant documents from LTM via Nova Memory MCP.

        Args:
            query_text (str): The text to search for relevant documents.
            top_k (int): The maximum number of documents to retrieve. Defaults to 5.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of results from the MCP server
            if successful, else None. The structure of this dictionary depends on the MCP response
            (e.g., may contain 'vector_results' and 'graph_results').
        """
        payload = {"query": query_text, "top_k": top_k}
        request_url = f"{self.mcp_base_url}/memory/query" # As per plan
        logger.debug(f"Retrieving documents via MCP. URL: {request_url}, Payload: {payload}")
        try:
            response = await self.session.post(request_url, json=payload)
            response.raise_for_status()
            logger.info(f"Successfully retrieved documents via MCP. Response status: {response.status_code}")
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"MCP HTTP error retrieving documents: {e.response.status_code} - {e.response.text}", exc_info=True)
        except httpx.RequestError as e:
            logger.error(f"MCP Request error retrieving documents: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"MCP Unexpected error retrieving documents: {e}", exc_info=True)
        return None

    def format_context_for_llm(self, mcp_results: Optional[Dict[str, Any]]) -> str:
        """Format hybrid MCP results for LLM consumption.

        Args:
            mcp_results (Optional[Dict[str, Any]]): The results dictionary from MCP,
                which might contain 'vector_results', 'graph_results', or 'results'.

        Returns:
            str: A formatted string combining vector and graph results for an LLM.
        """
        if not mcp_results:
            logger.warning("format_context_for_llm received None or empty mcp_results.")
            return "No relevant context found or results were empty."

        context_parts = []

        # Check for Nova Memory MCP format first (results key)
        results = mcp_results.get("results")
        if results and isinstance(results, list):
            context_parts.append("## Relevant Context (from Nova Memory MCP):")
            for i, result in enumerate(results[:10]):  # Limit to top 10 for readability
                if isinstance(result, dict):
                    text_content = result.get('text', result.get('content', 'N/A'))
                    score_content = result.get('normalized_score', result.get('score', 0.0))
                    metadata = result.get('metadata', {})
                    source = metadata.get('category', metadata.get('source', 'unknown'))
                    context_parts.append(f"- [{source}] {text_content} (relevance: {score_content:.2f})")
                else:
                    logger.warning(f"Skipping malformed result item: {result}")
            logger.info(f"Formatted {len(results)} Nova Memory MCP results for LLM context")

        # Fallback to legacy vector results format  
        vector_results = mcp_results.get("vector_results")
        if vector_results and isinstance(vector_results, list) and not results:
            context_parts.append("## Relevant Context (from Vector Search):")
            for result in vector_results:
                if isinstance(result, dict):
                    text_content = result.get('text', result.get('content', 'N/A'))
                    score_content = result.get('score', 0.0)
                    context_parts.append(f"- {text_content} (relevance: {score_content:.2f})")
                else:
                    logger.warning(f"Skipping malformed vector result item: {result}")
        elif vector_results is not None and not results:
            logger.warning(f"Expected list for 'vector_results', got {type(vector_results)}. Data: {vector_results}")


        # Graph results (structured knowledge)
        graph_results = mcp_results.get("graph_results")
        if graph_results and isinstance(graph_results, list):
            context_parts.append("\n## Related Knowledge (from Graph Database):")
            for result in graph_results:
                if isinstance(result, dict):
                    entity = result.get('entity', 'Unknown Entity')
                    properties = result.get('properties', {})
                    description = properties.get('description', 'No description available.') if isinstance(properties, dict) else 'No description available.'
                    context_parts.append(f"- {entity}: {description}")

                    neighbors = result.get('neighbors', [])
                    if isinstance(neighbors, list):
                        for neighbor in neighbors:
                            if isinstance(neighbor, dict):
                                relation = neighbor.get('relation', 'related to')
                                neighbor_entity = neighbor.get('entity', 'Unknown Neighbor')
                                context_parts.append(f"  → {relation}: {neighbor_entity}")
                            else:
                                logger.warning(f"Skipping malformed graph neighbor item: {neighbor}")
                    elif neighbors is not None:
                        logger.warning(f"Expected list for 'neighbors', got {type(neighbors)}. Data: {neighbors}")
                else:
                    logger.warning(f"Skipping malformed graph result item: {result}")
        elif graph_results is not None:
            logger.warning(f"Expected list for 'graph_results', got {type(graph_results)}. Data: {graph_results}")

        if not context_parts:
            return "No processable relevant context found in MCP results."

        return "\n".join(context_parts)

    async def close(self):
        """Closes the HTTPX session. Should be called on application shutdown."""
        if hasattr(self, 'session') and self.session and not self.session.is_closed:
            await self.session.aclose()
            logger.info("LTMInterface HTTP session closed.")
        else:
            logger.debug("LTMInterface HTTP session was already closed or not initialized.")

# Example Usage (Commented out as it's for the old Pinecone implementation and will not work with the new MCP interface without updates)
# async def main():
#     print("Testing LTMInterface...")
#     # This example needs to be updated for the new MCP_SERVER_URL and MCP methods
#     # For example, MCP_SERVER_URL needs to be set in environment variables.
#     # The new LTMInterface does not use PINECONE_API_KEY or OPENAI_API_KEY directly.
#
#     # Example:
#     # if not os.getenv("MCP_SERVER_URL"):
#     #     print("Error: MCP_SERVER_URL environment variable must be set.")
#     #     return
#
#     ltm = LTMInterface()
#     # The new __init__ does not set ltm.index, so this check is invalid.
#     # if not ltm.index:
#     #     print("LTM Initialization failed. Exiting.")
#     #     return
#
#     # Example Store (adjust for new store method and expected MCP interaction)
#     doc_id_1 = "doc_001" # doc_id is not directly passed to new store method's API call
#     text_1 = "The SHIFT architecture uses a swarm of agents coordinated via shared memory."
#     metadata_1 = {"source": "research.md", "topic": "architecture", "version": 1.0, "tags": ["swarm", "memory"]}
#     print(f"\nStoring: {text_1}")
#     # await ltm.store(text_1, metadata_1, doc_id=doc_id_1) # Old signature
#     store_response = await ltm.store(text_1, metadata_1)
#     print(f"Store response: {store_response}")
#
#     text_2 = "Agents can dynamically load tools from a registry using loading_info."
#     metadata_2 = {"source": "architecture.md", "topic": "tools", "important": True}
#     print(f"Storing: {text_2}")
#     store_response_2 = await ltm.store(text_2, metadata_2)
#     print(f"Store response 2: {store_response_2}")
#
#     # MCP server handles indexing, explicit sleep might not be needed or depends on MCP behavior
#     print("MCP server handles indexing.")
#     # await asyncio.sleep(5)
#
#     # Example Retrieve (adjust for new retrieve method)
#     query = "How do agents get new abilities?"
#     print(f"\nRetrieving documents related to: '{query}'")
#     results = await ltm.retrieve(query, top_k=2)
#     if results:
#         # The structure of results depends on the MCP server's response.
#         # The format_context_for_llm method (to be added) will process this.
#         print(f"  Raw results from MCP: {results}")
#     else:
#         print("  No relevant documents found or error in retrieval.")
#
#     # Example Retrieve with Filter (filter_metadata is not in new retrieve method's API call)
#     # query_filter = "What does SHIFT stand for?"
#     # metadata_filter = {"topic": "architecture"}
#     # print(f"\nRetrieving documents related to: '{query_filter}' with filter {metadata_filter}")
#     # results_filtered = await ltm.retrieve(query_filter, top_k=1, filter_metadata=metadata_filter) # Old signature
#     # if results_filtered:
#     #      print(f"  Raw filtered results: {results_filtered}")
#     # else:
#     #     print("  No relevant documents found with the specified filter.")
#
#     await ltm.close() # Close the session when done
#
#
# if __name__ == "__main__":
#     # Requires MCP_SERVER_URL to be set in environment.
#     # Other Pinecone/OpenAI keys are not directly used by LTMInterface anymore.
#     import asyncio # Ensure asyncio is imported if not already at the top
#     # asyncio.run(main())
