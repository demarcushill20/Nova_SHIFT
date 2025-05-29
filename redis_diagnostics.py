#!/usr/bin/env python3
"""
Redis Connection Diagnostics for Nova-SHIFT
Tests both sync and async connections to identify the issue
"""

import redis
import redis.asyncio as redis_async
import asyncio
import time

def test_sync_redis():
    """Test synchronous Redis connection"""
    print("Testing synchronous Redis connection...")
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        result = r.ping()
        print(f"✓ Sync Redis connection: {result}")
        
        # Test basic operations
        r.set("test_key", "test_value")
        value = r.get("test_key")
        print(f"✓ Sync Redis operations: {value}")
        r.delete("test_key")
        return True
    except Exception as e:
        print(f"✗ Sync Redis error: {e}")
        return False

async def test_async_redis_with_timeout():
    """Test async Redis with explicit timeout"""
    print("Testing async Redis connection with timeout...")
    try:
        # Use explicit timeout settings
        pool = redis_async.ConnectionPool(
            host='localhost', 
            port=6379, 
            db=0, 
            decode_responses=True,
            socket_timeout=5,  # 5 second timeout
            socket_connect_timeout=5,
            retry_on_timeout=True
        )
        
        client = redis_async.Redis(connection_pool=pool)
        
        # Test with asyncio timeout
        result = await asyncio.wait_for(client.ping(), timeout=5.0)
        print(f"✓ Async Redis connection: {result}")
        
        # Test basic operations
        await client.set("test_async_key", "test_async_value")
        value = await client.get("test_async_key")
        print(f"✓ Async Redis operations: {value}")
        await client.delete("test_async_key")
        
        await client.close()
        return True
        
    except asyncio.TimeoutError:
        print("✗ Async Redis timeout - this is the issue!")
        return False
    except Exception as e:
        print(f"✗ Async Redis error: {e}")
        return False

async def test_async_redis_alternative():
    """Test async Redis with alternative configuration"""
    print("Testing async Redis with alternative configuration...")
    try:
        # Try without connection pool
        client = redis_async.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=True,
            socket_timeout=3,
            socket_connect_timeout=3,
            health_check_interval=30
        )
        
        result = await asyncio.wait_for(client.ping(), timeout=3.0)
        print(f"✓ Alt async Redis connection: {result}")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"✗ Alt async Redis error: {e}")
        return False

async def main():
    print("="*60)
    print("REDIS CONNECTION DIAGNOSTICS FOR NOVA-SHIFT")
    print("="*60)
    
    # Test sync connection
    sync_works = test_sync_redis()
    
    print()
    
    # Test async connection with timeout
    async_works = await test_async_redis_with_timeout()
    
    print()
    
    # Test alternative async config
    alt_async_works = await test_async_redis_alternative()
    
    print()
    print("="*60)
    print("DIAGNOSIS RESULTS:")
    print("="*60)
    print(f"Synchronous Redis: {'✓ Working' if sync_works else '✗ Failed'}")
    print(f"Async Redis (pool): {'✓ Working' if async_works else '✗ Failed'}")
    print(f"Async Redis (direct): {'✓ Working' if alt_async_works else '✗ Failed'}")
    
    if sync_works and not async_works:
        print("\n🔍 ISSUE IDENTIFIED:")
        print("Docker Redis works with sync clients but not async clients")
        print("This is a common Docker networking + async Python issue")
        
        print("\n💡 SOLUTIONS:")
        print("1. Add explicit timeouts to Nova-SHIFT SharedMemoryInterface")
        print("2. Use direct Redis client instead of connection pool")
        print("3. Configure Docker networking for better async support")
        print("4. Use Redis container with specific network settings")
    
    return sync_works, async_works, alt_async_works

if __name__ == "__main__":
    asyncio.run(main())
