"""
Interface for agents to interact with the shared memory system (Redis).

Provides methods for reading, writing, deleting key-value data and publishing messages.
"""

import redis.asyncio as redis
import logging
import json # Using JSON for serialization of complex types
from typing import Any, Optional, Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TODO: Move Redis connection details to configuration (Phase 5)
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

class SharedMemoryInterface:
    """
    Provides an asynchronous interface to Redis for shared memory operations.

    Handles serialization/deserialization for storing Python objects.
    """

    def __init__(self, host: str = REDIS_HOST, port: int = REDIS_PORT, db: int = REDIS_DB):
        """
        Initializes the Redis connection pool.

        Args:
            host: Redis server host.
            port: Redis server port.
            db: Redis database number.
        """
        self._pool = redis.ConnectionPool(host=host, port=port, db=db, decode_responses=True)
        self._client = None
        logger.info(f"SharedMemoryInterface initialized (pointing to {host}:{port}/{db}).")

    async def _get_client(self) -> redis.Redis:
        """Gets an async Redis client from the connection pool."""
        if self._client is None:
             # Create client on first use within the async context
            self._client = redis.Redis(connection_pool=self._pool)
        # Verify connection (optional, adds overhead but ensures connectivity)
        try:
            await self._client.ping()
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            # Attempt to recreate client? Or raise? For now, log and proceed.
            self._client = redis.Redis(connection_pool=self._pool) # Try recreating
            raise # Re-raise the exception after logging
        return self._client

    async def write(self, key: str, value: Any, expiry_seconds: Optional[int] = None):
        """
        Writes a value to Redis, serializing complex types to JSON.

        Args:
            key: The key to store the value under.
            value: The value to store (can be complex type).
            expiry_seconds: Optional time-to-live for the key in seconds.
        """
        try:
            client = await self._get_client()
            # Serialize non-primitive types to JSON
            if not isinstance(value, (str, int, float, bytes)):
                serialized_value = json.dumps(value)
            else:
                serialized_value = value

            await client.set(key, serialized_value, ex=expiry_seconds)
            logger.debug(f"Wrote key '{key}' (Expiry: {expiry_seconds}s)")
        except redis.RedisError as e:
            logger.error(f"Error writing key '{key}' to Redis: {e}")
        except TypeError as e:
             logger.error(f"Error serializing value for key '{key}': {e}. Value: {value}")
        except Exception as e:
            logger.error(f"Unexpected error writing key '{key}': {e}")


    async def read(self, key: str) -> Any:
        """
        Reads a value from Redis, attempting to deserialize JSON strings.

        Args:
            key: The key to read.

        Returns:
            The deserialized value, or None if the key doesn't exist or an error occurs.
        """
        try:
            client = await self._get_client()
            value = await client.get(key)
            if value is None:
                logger.debug(f"Key '{key}' not found in Redis.")
                return None

            # Attempt to deserialize if it looks like JSON
            try:
                # Check if it's likely a JSON string (starts with { or [)
                if isinstance(value, str) and (value.startswith('{') and value.endswith('}')) or \
                   (value.startswith('[') and value.endswith(']')):
                    deserialized_value = json.loads(value)
                    logger.debug(f"Read and deserialized key '{key}'")
                    return deserialized_value
                else:
                     logger.debug(f"Read key '{key}' (as string/primitive)")
                     return value # Return as is if not clearly JSON
            except json.JSONDecodeError:
                 logger.warning(f"Value for key '{key}' looks like JSON but failed to decode. Returning as string.")
                 return value # Return raw string if decoding fails
        except redis.RedisError as e:
            logger.error(f"Error reading key '{key}' from Redis: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading key '{key}': {e}")
            return None

    async def read_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Reads multiple keys from Redis, attempting to deserialize JSON strings.

        Args:
            keys: A list of keys to read.

        Returns:
            A dictionary mapping found keys to their deserialized values.
            Keys not found will be omitted from the dictionary.
            Returns an empty dict if keys list is empty or an error occurs.
        """
        if not keys:
            return {}
        results = {}
        try:
            client = await self._get_client()
            values = await client.mget(keys)
            for key, value in zip(keys, values):
                if value is None:
                    logger.debug(f"Key '{key}' not found in Redis during MGET.")
                    continue

                # Attempt to deserialize if it looks like JSON
                try:
                    if isinstance(value, str) and (value.startswith('{') and value.endswith('}')) or \
                       (value.startswith('[') and value.endswith(']')):
                        results[key] = json.loads(value)
                        logger.debug(f"Read and deserialized key '{key}' during MGET")
                    else:
                        results[key] = value # Return as is if not clearly JSON
                        logger.debug(f"Read key '{key}' (as string/primitive) during MGET")
                except json.JSONDecodeError:
                    logger.warning(f"Value for key '{key}' looks like JSON but failed to decode during MGET. Storing as string.")
                    results[key] = value # Store raw string if decoding fails
        except redis.RedisError as e:
            logger.error(f"Error reading multiple keys {keys} from Redis: {e}")
            return {} # Return empty dict on error
        except Exception as e:
            logger.error(f"Unexpected error reading multiple keys {keys}: {e}")
            return {}
        return results

    async def delete(self, key: str) -> bool:
        """
        Deletes a key from Redis.

        Args:
            key: The key to delete.

        Returns:
            True if the key was deleted, False otherwise.
        """
        try:
            client = await self._get_client()
            result = await client.delete(key)
            deleted = result > 0
            if deleted:
                logger.debug(f"Deleted key '{key}'")
            else:
                 logger.debug(f"Key '{key}' not found for deletion.")
            return deleted
        except redis.RedisError as e:
            logger.error(f"Error deleting key '{key}' from Redis: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting key '{key}': {e}")
            return False

    async def write_many(self, data: Dict[str, Any], common_expiry_seconds: Optional[int] = None):
        """
        Writes multiple key-value pairs to Redis using a pipeline for efficiency.
        Handles serialization of complex types to JSON.

        Note: Using a pipeline instead of MSET allows for potentially different
              data types and setting expiry more easily if needed per key later,
              though this implementation uses a common expiry for simplicity.

        Args:
            data: A dictionary where keys are Redis keys and values are the values to store.
            common_expiry_seconds: Optional common time-to-live for all keys in seconds.
        """
        if not data:
            return
        try:
            client = await self._get_client()
            async with client.pipeline(transaction=False) as pipe: # Use transaction=False for performance unless atomicity is critical
                for key, value in data.items():
                    # Serialize non-primitive types to JSON
                    if not isinstance(value, (str, int, float, bytes)):
                        try:
                            serialized_value = json.dumps(value)
                        except TypeError as e:
                            logger.error(f"Error serializing value for key '{key}' during MSET: {e}. Skipping key.")
                            continue # Skip this key-value pair
                    else:
                        serialized_value = value
                    pipe.set(key, serialized_value, ex=common_expiry_seconds)
                await pipe.execute()
                logger.debug(f"Wrote {len(data)} keys using pipeline (Expiry: {common_expiry_seconds}s)")
        except redis.RedisError as e:
            logger.error(f"Error writing multiple keys to Redis using pipeline: {e}")
        except Exception as e:
            logger.error(f"Unexpected error writing multiple keys using pipeline: {e}")


    async def delete_many(self, keys: List[str]) -> int:
        """
        Deletes multiple keys from Redis.

        Args:
            keys: A list of keys to delete.

        Returns:
            The number of keys that were deleted. Returns 0 if keys list is empty or on error.
        """
        if not keys:
            return 0
        try:
            client = await self._get_client()
            num_deleted = await client.delete(*keys) # Unpack list into arguments
            logger.debug(f"Attempted to delete {len(keys)} keys. Deleted: {num_deleted}")
            return num_deleted
        except redis.RedisError as e:
            logger.error(f"Error deleting multiple keys {keys} from Redis: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error deleting multiple keys {keys}: {e}")
            return 0

    async def increment_counter(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Atomically increments an integer counter in Redis using INCRBY.
        Initializes the key to 0 if it doesn't exist before incrementing.

        Args:
            key: The key of the counter.
            amount: The integer amount to increment by (default: 1).

        Returns:
            The value of the counter after the increment, or None on error.
        """
        try:
            client = await self._get_client()
            new_value = await client.incrby(key, amount)
            logger.debug(f"Incremented counter '{key}' by {amount}. New value: {new_value}")
            return new_value
        except redis.ResponseError as e:
             # Handle cases where the key holds the wrong type
             logger.error(f"Error incrementing counter '{key}' (wrong type?): {e}")
             return None
        except redis.RedisError as e:
            logger.error(f"Error incrementing counter '{key}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error incrementing counter '{key}': {e}")
            return None

    async def increment_float(self, key: str, amount: float = 1.0) -> Optional[float]:
        """
        Atomically increments a float counter in Redis using INCRBYFLOAT.
        Initializes the key to 0.0 if it doesn't exist before incrementing.

        Args:
            key: The key of the float counter.
            amount: The float amount to increment by (default: 1.0).

        Returns:
            The value of the counter after the increment (as a float), or None on error.
        """
        try:
            client = await self._get_client()
            # INCRBYFLOAT returns bytes, need to decode and convert to float
            new_value_bytes = await client.incrbyfloat(key, amount)
            new_value = float(new_value_bytes) # type: ignore # redis-py types might not reflect float return accurately
            logger.debug(f"Incremented float '{key}' by {amount}. New value: {new_value}")
            return new_value
        except redis.ResponseError as e:
             # Handle cases where the key holds the wrong type
             logger.error(f"Error incrementing float '{key}' (wrong type?): {e}")
             return None
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(f"Error incrementing or parsing float '{key}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error incrementing float '{key}': {e}")
            return None

    async def publish(self, channel: str, message: Any):
        """
        Publishes a message to a Redis channel, serializing complex types to JSON.

        Args:
            channel: The channel to publish to.
            message: The message to publish (can be complex type).
        """
        try:
            client = await self._get_client()
            # Serialize non-primitive types to JSON
            if not isinstance(message, (str, int, float, bytes)):
                serialized_message = json.dumps(message)
            else:
                serialized_message = message

            await client.publish(channel, serialized_message)
            logger.debug(f"Published message to channel '{channel}'")
        except redis.RedisError as e:
            logger.error(f"Error publishing to channel '{channel}': {e}")
        except TypeError as e:
             logger.error(f"Error serializing message for channel '{channel}': {e}. Message: {message}")
        except Exception as e:
            logger.error(f"Unexpected error publishing to channel '{channel}': {e}")

    async def read_score(self, tool_name: str, default_score: float = 0.5) -> float:
        """
        Reads the calculated score for a specific tool from Redis.

        Args:
            tool_name: The name of the tool (e.g., 'calculate', 'search_internet').
            default_score: The score to return if the tool has no score stored yet.

        Returns:
            The score as a float, or the default_score if not found or on error.
        """
        score_key = f"tool:{tool_name}:score"
        try:
            client = await self._get_client()
            score_str = await client.get(score_key)
            if score_str is not None:
                return float(score_str)
            else:
                logger.debug(f"Score not found for tool '{tool_name}', returning default {default_score}.")
                return default_score
        except (redis.RedisError, ValueError, TypeError) as e:
            logger.error(f"Error reading or parsing score for tool '{tool_name}' from key '{score_key}': {e}. Returning default {default_score}.")
            return default_score
        except Exception as e:
            logger.error(f"Unexpected error reading score for tool '{tool_name}': {e}. Returning default {default_score}.")
            return default_score

    async def close(self):
        """Closes the Redis connection pool."""
        if self._client:
            await self._client.close()
        await self._pool.disconnect()
        logger.info("SharedMemoryInterface connection pool closed.")

# Example Usage (for testing purposes)
async def example_usage():
    shared_memory = SharedMemoryInterface(db=1) # Use DB 1 for example
    try:
        # Clear DB first
        client = await shared_memory._get_client()
        await client.flushdb()
        print("Flushed Redis DB 1 for example.")

        print("\n--- Single Operations ---")
        # Write examples
        await shared_memory.write("task:123:status", "running")
        await shared_memory.write("task:123:result", {"data": [1, 2, 3], "source": "agent_001"}, expiry_seconds=60)
        await shared_memory.write("agent:001:heartbeat", time.time(), expiry_seconds=30)
        await shared_memory.write("tool:calculator:score", 0.95)

        # Read examples
        status = await shared_memory.read("task:123:status")
        print(f"Task 123 Status: {status}")
        result = await shared_memory.read("task:123:result")
        print(f"Task 123 Result: {result}")
        score = await shared_memory.read_score("calculator")
        print(f"Calculator Score: {score}")
        non_existent = await shared_memory.read("non_existent_key")
        print(f"Non-existent key: {non_existent}")

        # Increment examples
        usage_count = await shared_memory.increment_counter("tool:calculator:usage")
        print(f"Calculator Usage Count after 1st increment: {usage_count}")
        usage_count = await shared_memory.increment_counter("tool:calculator:usage", amount=5)
        print(f"Calculator Usage Count after 2nd increment (+5): {usage_count}")
        total_duration = await shared_memory.increment_float("tool:calculator:duration", 0.123)
        print(f"Calculator Duration after 1st increment: {total_duration}")
        total_duration = await shared_memory.increment_float("tool:calculator:duration", 0.456)
        print(f"Calculator Duration after 2nd increment: {total_duration}")

        # Publish example
        await shared_memory.publish("agent_notifications", {"agent_id": "agent_002", "event": "task_completed", "task_id": "task_XYZ"})
        print("Published notification.")

        # Delete example
        deleted = await shared_memory.delete("agent:001:heartbeat")
        print(f"Deleted heartbeat key: {deleted}")
        heartbeat_after_delete = await shared_memory.read("agent:001:heartbeat")
        print(f"Heartbeat after delete: {heartbeat_after_delete}")

        print("\n--- Batch Operations ---")
        # Write Many Example
        batch_data = {
            "agent:002:status": "idle",
            "agent:003:status": "busy",
            "task:456:data": {"value": 100, "processed": False}
        }
        await shared_memory.write_many(batch_data, common_expiry_seconds=120)
        print(f"Wrote {len(batch_data)} keys using write_many.")

        # Read Many Example
        keys_to_read = ["task:123:status", "agent:002:status", "agent:003:status", "task:456:data", "non_existent_batch"]
        batch_results = await shared_memory.read_many(keys_to_read)
        print("Read many results:")
        for k, v in batch_results.items():
            print(f"  {k}: {v}")
        print(f"(Total keys found: {len(batch_results)})")


        # Delete Many Example
        keys_to_delete = ["agent:002:status", "agent:003:status", "task:123:result"]
        num_deleted = await shared_memory.delete_many(keys_to_delete)
        print(f"Attempted to delete {len(keys_to_delete)} keys. Actually deleted: {num_deleted}")
        # Verify deletion
        verify_results = await shared_memory.read_many(keys_to_delete)
        print(f"Verification read after delete_many (should be empty): {verify_results}")


    finally:
        await shared_memory.close()

if __name__ == "__main__":
    import asyncio
    # Requires a running Redis server on localhost:6379
    print("Running SharedMemoryInterface example...")
    try:
        asyncio.run(example_usage())
    except redis.exceptions.ConnectionError as e:
        print(f"\n*** Redis connection error: {e}. Please ensure Redis is running on {REDIS_HOST}:{REDIS_PORT}. ***")
