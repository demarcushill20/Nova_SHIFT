"""
Interface for agents to interact with the Long-Term Memory (LTM) system.

Currently implemented using Pinecone Vector Database. Handles embedding generation
and provides methods for storing and retrieving information semantically.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone, PodSpec, ServerlessSpec, Index # Import specific classes, Added ServerlessSpec
# The specific classes might be accessed via pinecone.Pinecone, pinecone.Index, pinecone.PodSpec
from langchain_openai import OpenAIEmbeddings # Example embedding model

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
    """Provides an interface to the Long-Term Memory (Pinecone Vector DB).

    Handles initialization of the Pinecone client and embedding model,
    connecting to the specified index (creating it if necessary),
    generating text embeddings, storing documents (text + metadata),
    and retrieving relevant documents via semantic search.

    Attributes:
        pinecone_client: The initialized Pinecone client instance.
        index: The connected Pinecone Index object.
        embeddings_model: The initialized LangChain embedding model instance.
    """

    def __init__(self):
        """Initializes the LTMInterface.

        Sets up the Pinecone client using environment variables for API key and
        index configuration. Initializes the OpenAI embedding model. Checks if the
        target Pinecone index exists and attempts to create it if it doesn't.
        Connects to the target index. Logs errors if initialization fails.
        """
        self.pinecone_client: Optional[Pinecone] = None # Store the client instance
        self.index: Optional[Index] = None # Use imported Index type hint
        self.embeddings_model: Optional[OpenAIEmbeddings] = None

        if not PINECONE_API_KEY:
            logger.error("PINECONE_API_KEY environment variable not set. LTMInterface cannot initialize.")
            raise ValueError("PINECONE_API_KEY environment variable not set.") # Raise exception

        if not os.environ.get("OPENAI_API_KEY"):
             logger.error("OPENAI_API_KEY environment variable not set. Cannot initialize embedding model.")
             raise ValueError("OPENAI_API_KEY environment variable not set.") # Raise exception

        try:
            # Initialize Pinecone client using the imported class
            self.pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
            logger.info("Pinecone client initialized.")

            # Initialize Embedding Model (Task P3.T3.4 - Basic)
            self.embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
            logger.info(f"Initialized embedding model: {EMBEDDING_MODEL_NAME}")

            # Check if index exists, create if not using the client instance
            # list_indexes() returns an IndexList object, get names from its 'indexes' attribute
            index_list = self.pinecone_client.list_indexes()
            existing_index_names = [index_info['name'] for index_info in index_list.indexes] if index_list and hasattr(index_list, 'indexes') else []
            if PINECONE_INDEX_NAME not in existing_index_names:
                logger.warning(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Attempting to create...")
                logger.warning(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Attempting to create...")
                try:
                    # Define spec based on environment variables
                    if PINECONE_SERVERLESS_CLOUD and PINECONE_SERVERLESS_REGION:
                        spec = ServerlessSpec(cloud=PINECONE_SERVERLESS_CLOUD, region=PINECONE_SERVERLESS_REGION)
                        logger.info(f"Using ServerlessSpec with cloud={PINECONE_SERVERLESS_CLOUD}, region={PINECONE_SERVERLESS_REGION}")
                    elif PINECONE_POD_ENVIRONMENT:
                        spec = PodSpec(environment=PINECONE_POD_ENVIRONMENT) # Use imported PodSpec
                        logger.info(f"Using PodSpec with environment: {PINECONE_POD_ENVIRONMENT}")
                    else:
                        logger.error("Pinecone environment configuration missing (require PINECONE_POD_ENVIRONMENT or both PINECONE_SERVERLESS_CLOUD and PINECONE_SERVERLESS_REGION). Cannot create index.")
                        raise ValueError("Missing Pinecone environment configuration for index creation.") # Raise error instead of return

                    # Use the client instance to create the index
                    self.pinecone_client.create_index(
                        name=PINECONE_INDEX_NAME,
                        dimension=EMBEDDING_DIMENSION,
                        metric="cosine", # Common metric for semantic similarity
                        spec=spec
                    )
                    logger.info(f"Successfully created Pinecone index '{PINECONE_INDEX_NAME}'.")
                    # Allow time for index to become ready after creation
                    import time
                    time.sleep(10) # Wait 10 seconds for index readiness
                except Exception as create_err:
                    logger.error(f"Failed to create Pinecone index '{PINECONE_INDEX_NAME}': {create_err}", exc_info=True)
                    return # Cannot proceed if index creation fails

            # Connect to the index using the client instance
            self.index = self.pinecone_client.Index(PINECONE_INDEX_NAME)
            logger.info(f"Connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
            # Optional: Log index stats on init
            try:
                 stats = self.index.describe_index_stats()
                 logger.info(f"Index stats: {stats}")
            except Exception as stats_err:
                 logger.warning(f"Could not retrieve index stats: {stats_err}")


        except Exception as e:
            logger.error(f"Failed to initialize LTMInterface: {e}", exc_info=True)
            self.index = None # Ensure index is None on failure
            self.embeddings_model = None


    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Generates an embedding for the given text.

        Uses the initialized `embeddings_model` to convert text into a vector.
        Handles both async (`aembed_query`) and sync (`embed_query`) methods
        of the embedding model.

        Args:
            text (str): The text to embed.

        Returns:
            Optional[List[float]]: The embedding vector as a list of floats,
            or None if the embedding model is not initialized or embedding fails.
        """
        if not self.embeddings_model:
            logger.error("Embedding model not initialized.")
            return None
        try:
            # Use aembed_query for async compatibility if available, else embed_query
            if hasattr(self.embeddings_model, 'aembed_query'):
                 embedding = await self.embeddings_model.aembed_query(text)
            else:
                 # Fallback to synchronous if async not available (less ideal in async context)
                 logger.warning("Using synchronous embed_query as aembed_query is not available.")
                 embedding = self.embeddings_model.embed_query(text)

            logger.debug(f"Generated embedding for text snippet: '{text[:50]}...'")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            return None

    async def store(self, text: str, metadata: Dict[str, Any], doc_id: Optional[str] = None):
        """Stores text and metadata in the LTM (Pinecone).

        Generates an embedding for the input text. Prepares metadata compatible
        with Pinecone (string, number, boolean, or list of strings), storing the
        original text within the metadata. Upserts the vector (ID, embedding,
        metadata) into the configured Pinecone index.

        Args:
            text (str): The text content to store.
            metadata (Dict[str, Any]): A dictionary of metadata associated with the
                text. Values will be converted or serialized to fit Pinecone limits.
            doc_id (Optional[str]): Optional unique ID for the document. If None,
                a random hex ID is generated.
        """
        if not self.index or not self.embeddings_model:
            logger.error("LTM Interface not properly initialized. Cannot store document.")
            return

        logger.debug(f"Attempting to store document (ID: {doc_id or 'Auto'}). Metadata: {metadata}")

        # 1. Generate Embedding
        embedding = await self._get_embedding(text)
        if embedding is None:
            logger.error("Failed to store document because embedding generation failed.")
            return

        # 2. Prepare vector for Pinecone upsert
        # Pinecone metadata values must be string, number, boolean, or list of strings.
        # Ensure metadata conforms to this. We also store the original text.
        pinecone_metadata = {"original_text": text}
        for key, value in metadata.items():
             if isinstance(value, (str, int, float, bool)):
                 pinecone_metadata[key] = value
             elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                 pinecone_metadata[key] = value
             else:
                  # Attempt to serialize other types as strings, log warning
                  try:
                      pinecone_metadata[key] = json.dumps(value)
                      logger.warning(f"Metadata key '{key}' was serialized to JSON string: {pinecone_metadata[key][:100]}...")
                  except TypeError:
                       logger.error(f"Metadata key '{key}' has unserializable type '{type(value)}'. Skipping.")


        vector_id = doc_id if doc_id else os.urandom(16).hex() # Generate random ID if none provided

        vector_to_upsert = (vector_id, embedding, pinecone_metadata)

        # 3. Upsert to Pinecone
        try:
            upsert_response = self.index.upsert(vectors=[vector_to_upsert])
            logger.info(f"Successfully stored document ID '{vector_id}'. Response: {upsert_response}")
        except Exception as e:
            logger.error(f"Failed to upsert document ID '{vector_id}' to Pinecone: {e}", exc_info=True)


    async def retrieve(self, query_text: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Retrieves relevant documents from LTM via semantic search.

        Generates an embedding for the `query_text`. Queries the Pinecone index
        for the `top_k` most similar vectors, optionally applying metadata filters.
        Formats the results, attempting to deserialize any JSON strings stored
        in the metadata.

        Args:
            query_text (str): The text to search for relevant documents.
            top_k (int): The maximum number of documents to retrieve. Defaults to 5.
            filter_metadata (Optional[Dict[str, Any]]): Optional dictionary for
                metadata filtering according to Pinecone's filtering syntax.
                Example: `{"source": "research.md"}`.

        Returns:
            List[Dict[str, Any]]: A list of result dictionaries. Each dictionary
            contains 'id' (str), 'score' (float), and 'metadata' (Dict).
            The 'metadata' dictionary includes the 'original_text' and any other
            stored metadata. Returns an empty list if retrieval fails or no
            documents match.
        """
        if not self.index or not self.embeddings_model:
            logger.error("LTM Interface not properly initialized. Cannot retrieve documents.")
            return []

        logger.debug(f"Attempting to retrieve top {top_k} documents for query: '{query_text[:100]}...'")

        # 1. Generate Query Embedding
        query_embedding = await self._get_embedding(query_text)
        if query_embedding is None:
            logger.error("Failed to retrieve documents because query embedding generation failed.")
            return []

        # 2. Query Pinecone
        try:
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_metadata # Pass the filter dictionary directly
            )
            logger.info(f"Retrieved {len(query_response.get('matches', []))} documents from Pinecone.")

            # 3. Format Results
            results = []
            for match in query_response.get('matches', []):
                metadata = match.get('metadata', {})
                # Attempt to deserialize metadata fields that were stringified JSON
                formatted_metadata = {}
                for key, value in metadata.items():
                     if isinstance(value, str):
                          try:
                               # Check if it looks like serialized JSON
                               if (value.startswith('{') and value.endswith('}')) or \
                                  (value.startswith('[') and value.endswith(']')):
                                    formatted_metadata[key] = json.loads(value)
                               else:
                                    formatted_metadata[key] = value # Keep as string
                          except json.JSONDecodeError:
                               formatted_metadata[key] = value # Keep as string if decode fails
                     else:
                          formatted_metadata[key] = value

                results.append({
                    "id": match.get('id'),
                    "score": match.get('score'),
                    "metadata": formatted_metadata # Use potentially deserialized metadata
                })
            return results

        except Exception as e:
            logger.error(f"Failed to query Pinecone: {e}", exc_info=True)
            return []

# Example Usage
async def main():
    print("Testing LTMInterface...")
    if not PINECONE_API_KEY or not os.environ.get("OPENAI_API_KEY"):
        print("Error: PINECONE_API_KEY and OPENAI_API_KEY environment variables must be set.")
        return

    ltm = LTMInterface()
    if not ltm.index:
        print("LTM Initialization failed. Exiting.")
        return

    # Example Store
    doc_id_1 = "doc_001"
    text_1 = "The SHIFT architecture uses a swarm of agents coordinated via shared memory."
    metadata_1 = {"source": "research.md", "topic": "architecture", "version": 1.0, "tags": ["swarm", "memory"]}
    print(f"\nStoring: {text_1}")
    await ltm.store(text_1, metadata_1, doc_id=doc_id_1)

    doc_id_2 = "doc_002"
    text_2 = "Agents can dynamically load tools from a registry using loading_info."
    metadata_2 = {"source": "architecture.md", "topic": "tools", "important": True}
    print(f"Storing: {text_2}")
    await ltm.store(text_2, metadata_2, doc_id=doc_id_2)

    # Allow time for upserts to process in Pinecone
    print("Waiting for Pinecone index to update...")
    await asyncio.sleep(5)

    # Example Retrieve
    query = "How do agents get new abilities?"
    print(f"\nRetrieving documents related to: '{query}'")
    results = await ltm.retrieve(query, top_k=2)
    if results:
        for i, res in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    ID: {res.get('id')}")
            print(f"    Score: {res.get('score'):.4f}")
            print(f"    Metadata: {res.get('metadata')}")
            # print(f"    Text: {res.get('metadata', {}).get('original_text')}") # Included in metadata now
    else:
        print("  No relevant documents found.")

    # Example Retrieve with Filter
    query_filter = "What does SHIFT stand for?"
    metadata_filter = {"topic": "architecture"}
    print(f"\nRetrieving documents related to: '{query_filter}' with filter {metadata_filter}")
    results_filtered = await ltm.retrieve(query_filter, top_k=1, filter_metadata=metadata_filter)
    if results_filtered:
         for res in results_filtered:
            print(f"  Filtered Result:")
            print(f"    ID: {res.get('id')}")
            print(f"    Score: {res.get('score'):.4f}")
            print(f"    Metadata: {res.get('metadata')}")
    else:
        print("  No relevant documents found with the specified filter.")


if __name__ == "__main__":
    # Requires PINECONE_API_KEY, OPENAI_API_KEY, and potentially PINECONE_POD_ENVIRONMENT
    # Also requires a running Pinecone index (or it will attempt to create one)
    import asyncio # Need this import here for the main block
    asyncio.run(main())
