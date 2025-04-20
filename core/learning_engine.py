"""
Collective Learning Engine: Responsible for processing task outcomes
and updating system parameters like tool scores.
"""

import logging
import asyncio
import redis.asyncio as redis
from typing import Dict, Optional

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

    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, db: int = REDIS_DB):
        """Initializes the CollectiveLearningEngine.

        Sets up the Redis connection pool used to read usage logs and write scores.

        Args:
            host (str): Redis server host. Defaults to REDIS_HOST.
            port (int): Redis server port. Defaults to REDIS_PORT.
            db (int): Redis database number. Defaults to REDIS_DB.
        """
        self._pool = redis.ConnectionPool(host=host, port=port, db=db, decode_responses=True)
        self._client: Optional[redis.Redis] = None
        logger.info(f"CollectiveLearningEngine initialized (pointing to {host}:{port}/{db}).")

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

    async def close(self):
        """Closes the Redis client and connection pool gracefully."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        logger.info("CollectiveLearningEngine connection pool closed.")

# Example Usage
async def main():
    print("Running CollectiveLearningEngine example...")
    engine = CollectiveLearningEngine(db=1) # Use test DB if needed

    # Simulate some tool usage data (replace with actual agent logging in practice)
    print("Simulating tool usage data in Redis DB 1...")
    client = await engine._get_client()
    await client.set("tool:calculate:usage_count", 10)
    await client.set("tool:calculate:success_count", 8)
    await client.set("tool:calculate:failure_count", 2)
    await client.set("tool:calculate:total_duration", 5.5)

    await client.set("tool:search_internet:usage_count", 5)
    await client.set("tool:search_internet:success_count", 5)
    await client.set("tool:search_internet:failure_count", 0)
    await client.set("tool:search_internet:total_duration", 12.1)

    await client.set("tool:read_file:usage_count", 2)
    await client.set("tool:read_file:success_count", 1)
    await client.set("tool:read_file:failure_count", 1)
    await client.set("tool:read_file:total_duration", 0.8)
    print("Simulated data written.")

    # Calculate and store scores
    print("\nCalculating and storing scores...")
    scores = await engine.calculate_and_store_tool_scores()
    print(f"\nCalculated Scores: {scores}")

    # Verify scores stored in Redis
    print("\nVerifying scores in Redis DB 1:")
    calc_score = await client.get("tool:calculate:score")
    search_score = await client.get("tool:search_internet:score")
    read_score = await client.get("tool:read_file:score")
    print(f"  tool:calculate:score = {calc_score}")
    print(f"  tool:search_internet:score = {search_score}")
    print(f"  tool:read_file:score = {read_score}")

    # Clean up test data and close connection
    await client.flushdb()
    print("\nFlushed Redis DB 1.")
    await engine.close()

if __name__ == "__main__":
    # Requires a running Redis server
    asyncio.run(main())