"""
Unit tests for the SharedMemoryInterface using fakeredis.
"""

import pytest
import pytest_asyncio
import fakeredis.aioredis
import json
from typing import Any, Dict

# Assuming SharedMemoryInterface is in nova_shift.core
# Adjust the import path as necessary based on your project structure
from core.shared_memory_interface import SharedMemoryInterface

# Fixture to provide a SharedMemoryInterface instance backed by FakeRedis
@pytest_asyncio.fixture
async def memory_interface() -> SharedMemoryInterface:
    """Provides a SharedMemoryInterface instance using FakeRedis for testing."""
    # Use FakeRedis for async
    fake_redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)

    # Create the interface instance
    interface = SharedMemoryInterface(host='fake', port=1234, db=0) # Host/port don't matter for fake

    # Monkeypatch the _get_client method to return the fake client
    async def get_fake_client():
        return fake_redis_client
    interface._get_client = get_fake_client # type: ignore

    yield interface

    # Cleanup: Clear the fake redis instance after test
    await fake_redis_client.flushall()
    await interface.close() # Close the interface (though fake client doesn't need explicit closing)

@pytest.mark.asyncio
async def test_write_many_simple(memory_interface: SharedMemoryInterface):
    """Test writing multiple simple string values."""
    data = {"key1": "value1", "key2": "value2"}
    await memory_interface.write_many(data)
    assert await memory_interface.read("key1") == "value1"
    assert await memory_interface.read("key2") == "value2"

@pytest.mark.asyncio
async def test_write_many_complex(memory_interface: SharedMemoryInterface):
    """Test writing multiple complex dictionary values (JSON serialization)."""
    data = {
        "complex1": {"a": 1, "b": [True, None]},
        "complex2": {"c": "hello"}
    }
    await memory_interface.write_many(data)
    read_complex1 = await memory_interface.read("complex1")
    read_complex2 = await memory_interface.read("complex2")
    assert read_complex1 == {"a": 1, "b": [True, None]}
    assert read_complex2 == {"c": "hello"}

@pytest.mark.asyncio
async def test_write_many_mixed_types(memory_interface: SharedMemoryInterface):
    """Test writing a mix of primitive and complex types."""
    data = {
        "mix_str": "simple",
        "mix_int": 123,
        "mix_float": 4.56,
        "mix_dict": {"nested": True},
        "mix_list": [1, "two", 3.0]
    }
    await memory_interface.write_many(data)
    assert await memory_interface.read("mix_str") == "simple"
    # Note: fakeredis might store numbers as strings, adjust assertion if needed
    # Let's check the type after reading
    read_int = await memory_interface.read("mix_int")
    assert read_int == "123" or read_int == 123 # Allow string or int from fakeredis
    read_float = await memory_interface.read("mix_float")
    assert read_float == "4.56" or read_float == 4.56 # Allow string or float
    assert await memory_interface.read("mix_dict") == {"nested": True}
    assert await memory_interface.read("mix_list") == [1, "two", 3.0]

@pytest.mark.asyncio
async def test_write_many_with_expiry(memory_interface: SharedMemoryInterface):
    """Test writing multiple values with a common expiry."""
    data = {"exp_key1": "temp1", "exp_key2": {"temp": 2}}
    # Using a very short expiry for testing might be flaky, use 1 second
    await memory_interface.write_many(data, common_expiry_seconds=1)
    assert await memory_interface.read("exp_key1") == "temp1"
    assert await memory_interface.read("exp_key2") == {"temp": 2}

    # Wait for expiry (adjust sleep time if needed, fakeredis might handle time differently)
    # For fakeredis, direct time manipulation might be better if available,
    # but simple sleep works for basic check.
    await asyncio.sleep(1.1)

    assert await memory_interface.read("exp_key1") is None
    assert await memory_interface.read("exp_key2") is None

@pytest.mark.asyncio
async def test_write_many_empty(memory_interface: SharedMemoryInterface):
    """Test writing an empty dictionary."""
    await memory_interface.write_many({})
    # No error should occur, and no keys should be written (verify indirectly)
    # Let's write something else to ensure the connection works
    await memory_interface.write("test_key", "test_value")
    assert await memory_interface.read("test_key") == "test_value"

@pytest.mark.asyncio
async def test_read_many_simple(memory_interface: SharedMemoryInterface):
    """Test reading multiple simple string values."""
    await memory_interface.write("r_key1", "val1")
    await memory_interface.write("r_key2", "val2")
    await memory_interface.write("r_key3", "val3")
    keys_to_read = ["r_key1", "r_key2", "non_existent"]
    results = await memory_interface.read_many(keys_to_read)
    assert results == {"r_key1": "val1", "r_key2": "val2"}

@pytest.mark.asyncio
async def test_read_many_complex(memory_interface: SharedMemoryInterface):
    """Test reading multiple complex dictionary values."""
    await memory_interface.write("r_complex1", {"x": 10})
    await memory_interface.write("r_complex2", [5, 6])
    keys_to_read = ["r_complex1", "r_complex2", "non_existent"]
    results = await memory_interface.read_many(keys_to_read)
    assert results == {"r_complex1": {"x": 10}, "r_complex2": [5, 6]}

@pytest.mark.asyncio
async def test_read_many_mixed_types(memory_interface: SharedMemoryInterface):
    """Test reading a mix of types, including non-JSON strings."""
    await memory_interface.write("rm_str", "plain string")
    await memory_interface.write("rm_json", {"is_json": True})
    await memory_interface.write("rm_num_str", "12345") # String that isn't JSON
    keys_to_read = ["rm_str", "rm_json", "rm_num_str", "non_existent"]
    results = await memory_interface.read_many(keys_to_read)
    assert results == {
        "rm_str": "plain string",
        "rm_json": {"is_json": True},
        "rm_num_str": "12345"
    }

@pytest.mark.asyncio
async def test_read_many_empty_keys(memory_interface: SharedMemoryInterface):
    """Test reading with an empty list of keys."""
    results = await memory_interface.read_many([])
    assert results == {}

@pytest.mark.asyncio
async def test_read_many_all_non_existent(memory_interface: SharedMemoryInterface):
    """Test reading only keys that do not exist."""
    keys_to_read = ["phantom1", "phantom2"]
    results = await memory_interface.read_many(keys_to_read)
    assert results == {}

@pytest.mark.asyncio
async def test_delete_many_existing(memory_interface: SharedMemoryInterface):
    """Test deleting multiple existing keys."""
    await memory_interface.write("del_key1", "a")
    await memory_interface.write("del_key2", "b")
    await memory_interface.write("keep_key", "c")
    keys_to_delete = ["del_key1", "del_key2"]
    num_deleted = await memory_interface.delete_many(keys_to_delete)
    assert num_deleted == 2
    assert await memory_interface.read("del_key1") is None
    assert await memory_interface.read("del_key2") is None
    assert await memory_interface.read("keep_key") == "c"

@pytest.mark.asyncio
async def test_delete_many_some_non_existent(memory_interface: SharedMemoryInterface):
    """Test deleting a mix of existing and non-existent keys."""
    await memory_interface.write("del_key_exist", "exists")
    keys_to_delete = ["del_key_exist", "del_key_ghost"]
    num_deleted = await memory_interface.delete_many(keys_to_delete)
    assert num_deleted == 1 # Only the existing key is counted
    assert await memory_interface.read("del_key_exist") is None

@pytest.mark.asyncio
async def test_delete_many_all_non_existent(memory_interface: SharedMemoryInterface):
    """Test deleting only keys that do not exist."""
    keys_to_delete = ["ghost1", "ghost2", "ghost3"]
    num_deleted = await memory_interface.delete_many(keys_to_delete)
    assert num_deleted == 0

@pytest.mark.asyncio
async def test_delete_many_empty_list(memory_interface: SharedMemoryInterface):
    """Test deleting with an empty list of keys."""
    num_deleted = await memory_interface.delete_many([])
    assert num_deleted == 0
    # Ensure other keys are unaffected
    await memory_interface.write("check_key", "still_here")
    assert await memory_interface.read("check_key") == "still_here"

# Need asyncio for sleep in expiry test
import asyncio