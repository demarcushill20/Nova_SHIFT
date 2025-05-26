"""
Collective Learning Engine: Responsible for processing task outcomes
and updating system parameters like tool scores.
"""

import logging
import asyncio
import redis.asyncio as redis
from typing import Dict, List, Any, Optional # Added List, Any
from datetime import datetime # Added for timestamping

# Assuming LTMInterface will be imported if not already
# from .ltm_interface import LTMInterface # Example import path
LTMInterface = Any # Placeholder if direct import is complex here

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: Move Redis connection details to configuration (Phase 5)
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0 # Use the main DB, or a separate one for scores? Let's use main for now.

class CollectiveLearningEngine:
    """Analyzes logged tool usage data from Redis and updates tool scores.

    This engine periodically scans Redis for tool usage counters logged by agents,
    calculates scores (e.g., success rate), and stores these scores back into
    Redis for agents to use during tool selection.

    Attributes:
        _pool: Redis connection pool.
        _client: Async Redis client instance.
    """

    def __init__(self,
                 ltm_interface: LTMInterface, # Added LTMInterface dependency
                 host: str = REDIS_HOST,
                 port: int = REDIS_PORT,
                 db: int = REDIS_DB):
        """Initializes the CollectiveLearningEngine.

        Sets up the Redis connection pool and stores the LTM interface.

        Args:
            ltm_interface (LTMInterface): Interface for long-term memory access.
            host (str): Redis server host. Defaults to REDIS_HOST.
            port (int): Redis server port. Defaults to REDIS_PORT.
            db (int): Redis database number. Defaults to REDIS_DB.
        """
        self._ltm_interface = ltm_interface # Store LTMInterface
        self._pool = redis.ConnectionPool(host=host, port=port, db=db, decode_responses=True)
        self._client: Optional[redis.Redis] = None
        logger.info(f"CollectiveLearningEngine initialized (pointing to {host}:{port}/{db}, LTMInterface provided).")

    async def _get_client(self) -> redis.Redis:
        """Gets or creates an async Redis client from the connection pool.

        Includes a basic ping check to verify connectivity.

        Returns:
            redis.Redis: An active asynchronous Redis client instance.

        Raises:
            redis.ConnectionError: If the connection ping fails.
        """
        if self._client is None:
            self._client = redis.Redis(connection_pool=self._pool)
        try:
            await self._client.ping()
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            self._client = redis.Redis(connection_pool=self._pool) # Try recreating
            raise # Re-raise the exception after logging
        return self._client

    async def calculate_and_store_tool_scores(self) -> Dict[str, float]:
        """Calculates and stores tool scores based on usage data in Redis.

        Scans Redis for keys matching 'tool:*:usage_count'. For each tool found,
        it reads the usage, success, failure, and duration counters. It calculates
        a simple success rate score (success_count / usage_count) and stores it
        in Redis under the key 'tool:<tool_name>:score'.

        (Implements P3.T4.3 and P3.T4.4)

        Returns:
            Dict[str, float]: A dictionary mapping tool names to their newly
            calculated and stored scores. Returns an empty dictionary if errors occur
            during scanning or processing.
        """
        client = await self._get_client()
        calculated_scores: Dict[str, float] = {}
        tool_keys = []
        try:
            # Scan for keys matching the tool usage pattern
            async for key in client.scan_iter("tool:*:usage_count"):
                tool_keys.append(key)
        except redis.RedisError as e:
            logger.error(f"Error scanning Redis keys: {e}")
            return calculated_scores # Return empty if scan fails

        logger.info(f"Found {len(tool_keys)} tool usage keys to process.")

        for usage_key in tool_keys:
            try:
                # Extract tool name from the key (e.g., "tool:calculate:usage_count" -> "calculate")
                parts = usage_key.split(':')
                if len(parts) != 3 or parts[0] != 'tool' or parts[2] != 'usage_count':
                    logger.warning(f"Skipping malformed tool key: {usage_key}")
                    continue
                tool_name = parts[1]
                base_key = f"tool:{tool_name}"

                # Read counters using mget for efficiency
                keys_to_get = [
                    f"{base_key}:usage_count",
                    f"{base_key}:success_count",
                    f"{base_key}:failure_count",
                    f"{base_key}:total_duration"
                ]
                values = await client.mget(keys_to_get)

                # Safely convert values to appropriate types (int or float)
                usage_count = int(values[0]) if values[0] is not None else 0
                success_count = int(values[1]) if values[1] is not None else 0
                # failure_count = int(values[2]) if values[2] is not None else 0 # Not used in simple score
                # total_duration = float(values[3]) if values[3] is not None else 0.0 # Not used in simple score

                if usage_count > 0:
                    # Calculate simple success rate score
                    score = success_count / usage_count
                    calculated_scores[tool_name] = score

                    # Store the calculated score back into Redis (P3.T4.4)
                    score_key = f"{base_key}:score"
                    await client.set(score_key, score)
                    logger.info(f"Calculated and stored score for '{tool_name}': {score:.4f} (Success: {success_count}/{usage_count})")
                else:
                    logger.warning(f"Tool '{tool_name}' has usage_count=0. Skipping score calculation.")
                    # Optionally set score to a default value, e.g., 0.5 or remove existing score
                    await client.delete(f"{base_key}:score")


            except (TypeError, ValueError) as e:
                 logger.error(f"Error processing data for key '{usage_key}': Invalid data type in Redis. {e}")
            except redis.RedisError as e:
                 logger.error(f"Redis error processing key '{usage_key}': {e}")
            except Exception as e:
                 logger.error(f"Unexpected error processing key '{usage_key}': {e}", exc_info=True)

        return calculated_scores

    async def analyze_task_outcome(self, task_id: str, success: bool, tools_used: List[str]):
        """Analyzes task outcome, stores learning in LTM, and updates tool scores.
        As per integration plan 3.3.1.
        """
        logger.info(f"Analyzing outcome for task_id='{task_id}', success={success}, tools_used={tools_used}")
        # Store learning in hybrid memory (LTM via MCP)
        learning_summary = f"Task {task_id}: {'Success' if success else 'Failure'} using {tools_used}"
        metadata = {
            "category": "collective_learning",
            "task_id": task_id,
            "success": success,
            "tools": tools_used, # Storing as a list
            "timestamp": datetime.now().isoformat()
        }
        try:
            store_response = await self._ltm_interface.store(text=learning_summary, metadata=metadata)
            if store_response:
                logger.info(f"Stored learning summary for task '{task_id}' to LTM. Response: {store_response}")
            else:
                logger.warning(f"Failed to store learning summary for task '{task_id}' to LTM (no response or error).")
        except Exception as e:
            logger.error(f"Error storing learning summary for task '{task_id}' to LTM: {e}", exc_info=True)

        # Query related learning experiences
        related_learning_query = f"collective learning {' '.join(tools_used)}"
        related_learning_results: Optional[Dict[str, Any]] = None
        try:
            related_learning_results = await self._ltm_interface.retrieve(query_text=related_learning_query)
            if related_learning_results:
                logger.info(f"Retrieved related learning experiences for tools {tools_used} from LTM.")
                # formatted_related_context = self._ltm_interface.format_context_for_llm(related_learning_results) # If needed
                # logger.debug(f"Formatted related learning context:\n{formatted_related_context}")
            else:
                logger.info(f"No related learning experiences found for tools {tools_used} in LTM.")
        except Exception as e:
            logger.error(f"Error retrieving related learning for tools {tools_used} from LTM: {e}", exc_info=True)

        # Update tool scores based on hybrid insights
        # This method needs to be implemented based on how scores are managed and how hybrid insights influence them.
        await self.update_tool_scores_with_context(tools_used, success, related_learning_results)

    async def update_tool_scores_with_context(self, tools_used: List[str], success: bool, related_learning: Optional[Dict[str, Any]]):
        """
        Placeholder for updating tool scores based on task success and related learning context from LTM.
        The actual scoring logic (how `related_learning` influences scores) needs to be defined.
        This might involve more sophisticated logic than simple success/failure counts if hybrid insights
        from LTM (vector + graph) are to be deeply integrated into scoring.
        For now, it can call the existing Redis-based scoring or be extended.
        """
        logger.info(f"Placeholder: update_tool_scores_with_context called for tools={tools_used}, success={success}.")
        if related_learning:
            logger.debug(f"Related learning context received (first 100 chars): {str(related_learning)[:100]}...")
        
        # Option 1: Delegate to existing Redis-based scoring for now
        # This doesn't use `related_learning` yet but keeps current functionality.
        # For each tool used, one might log its individual success/failure to Redis
        # if the `CollectiveLearningEngine`'s Redis methods are still the primary mechanism.
        # Or, this method could directly update scores in Redis based on new logic.
        
        # Example: If we want to maintain the Redis-based success/failure counts
        # This part would need access to the Redis client, similar to `calculate_and_store_tool_scores`
        # client = await self._get_client()
        # for tool_name in tools_used:
        #     base_key = f"tool:{tool_name}"
        #     await client.incr(f"{base_key}:usage_count")
        #     if success:
        #         await client.incr(f"{base_key}:success_count")
        #     else:
        #         await client.incr(f"{base_key}:failure_count")
        # logger.info(f"Incremented Redis counters for tools: {tools_used} based on task success: {success}")

        # Option 2: Implement new scoring logic here that uses `related_learning`.
        # This is where the "hybrid insights" part of the plan would come in.
        # For example, if related_learning shows similar past failures with these tools for similar tasks,
        # the score update might be more nuanced.
        
        # For now, let's just log and perhaps call the existing score calculation
        # to ensure scores are generally updated.
        logger.info("Calling calculate_and_store_tool_scores to refresh overall scores based on Redis data.")
        await self.calculate_and_store_tool_scores() # Recalculate all scores

    async def close(self):
        """Closes the Redis client and connection pool gracefully."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        # Also close LTM interface if it has a close method
        if hasattr(self._ltm_interface, 'close') and asyncio.iscoroutinefunction(self._ltm_interface.close):
            await self._ltm_interface.close()
            logger.info("LTMInterface connection closed by CollectiveLearningEngine.")
        logger.info("CollectiveLearningEngine connections closed.")

# Example Usage
async def main():
    print("Running CollectiveLearningEngine example...")

    # Mock LTMInterface for the example
    class MockLTMInterface:
        async def store(self, text: str, metadata: Dict):
            print(f"LTM_STORE (Mock): Text='{text[:50]}...', Metadata={metadata}")
            return {"status": "success", "id": "mock_ltm_id_" + metadata.get("task_id", "unknown")}
        
        async def retrieve(self, query_text: str, top_k: int = 5):
            print(f"LTM_RETRIEVE (Mock): Query='{query_text[:50]}...', top_k={top_k}")
            if "collective learning" in query_text:
                return { # Simulating MCP-like response structure
                    "vector_results": [
                        {"text": "Previously, tool_A and tool_B failed on similar task_X.", "score": 0.8},
                        {"text": "Tool_A was successful with tool_C on task_Y.", "score": 0.7}
                    ],
                    "graph_results": []
                }
            return None
        
        async def close(self):
            print("LTM_CLOSE (Mock): Called.")

    mock_ltm = MockLTMInterface()
    engine = CollectiveLearningEngine(ltm_interface=mock_ltm, db=1) # Pass mock LTM

    # Simulate some tool usage data (this part might be less relevant if analyze_task_outcome updates Redis directly)
    print("Simulating initial tool usage data in Redis DB 1 (for calculate_and_store_tool_scores)...")
    client = await engine._get_client()
    await client.set("tool:tool_A:usage_count", 10)
    await client.set("tool:tool_A:success_count", 8)
    await client.set("tool:tool_B:usage_count", 5)
    await client.set("tool:tool_B:success_count", 2)
    print("Simulated data written.")

    # Test analyze_task_outcome
    print("\nTesting analyze_task_outcome (Success)...")
    await engine.analyze_task_outcome(task_id="task123", success=True, tools_used=["tool_A", "tool_C"])
    
    print("\nTesting analyze_task_outcome (Failure)...")
    await engine.analyze_task_outcome(task_id="task456", success=False, tools_used=["tool_A", "tool_B"])

    # Calculate and store scores (this will reflect any updates made by analyze_task_outcome if it modifies Redis directly)
    print("\nCalculating and storing overall scores (after analyze_task_outcome calls)...")
    scores = await engine.calculate_and_store_tool_scores()
    print(f"\nCalculated Scores: {scores}")

    # Verify scores stored in Redis
    print("\nVerifying scores in Redis DB 1:")
    tool_A_score = await client.get("tool:tool_A:score")
    tool_B_score = await client.get("tool:tool_B:score")
    tool_C_score = await client.get("tool:tool_C:score") # May or may not exist depending on analyze_task_outcome's Redis interaction
    print(f"  tool:tool_A:score = {tool_A_score}")
    print(f"  tool:tool_B:score = {tool_B_score}")
    print(f"  tool:tool_C:score = {tool_C_score}")


    # Clean up test data and close connection
    await client.flushdb()
    print("\nFlushed Redis DB 1.")
    await engine.close()

if __name__ == "__main__":
    # Requires a running Redis server
    asyncio.run(main())