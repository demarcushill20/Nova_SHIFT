"""
Quick Gemini 2.5 Flash Performance Test for Nova-SHIFT
Tests the new model's capabilities and performance improvements
"""

import asyncio
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test configuration
TEST_QUERIES = [
    {
        "query": "What is Nova AI Road to Freedom?",
        "expect_memory": True,
        "complexity": "simple"
    },
    {
        "query": "Analyze the relationship between individual consciousness (Iris) and collective intelligence (Nova-SHIFT)",
        "expect_memory": True,
        "complexity": "complex"
    },
    {
        "query": "What are the latest breakthroughs in quantum computing as of 2025?",
        "expect_memory": False,
        "complexity": "simple"
    },
    {
        "query": "Design a novel approach to AI consciousness that builds on the Nova Memory MCP architecture",
        "expect_memory": True,
        "complexity": "complex"
    }
]


async def test_gemini_performance():
    """Test Gemini 2.5 Flash performance with various query types."""
    
    print("\n" + "="*60)
    print("GEMINI 2.5 FLASH PERFORMANCE TEST")
    print("Nova-SHIFT Integration Verification")
    print("="*60)
    
    # Import Nova-SHIFT components
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from agents.specialist_agent import create_specialist_agent
    from core.tool_registry import ToolRegistry
    from core.shared_memory_interface import SharedMemoryInterface
    from core.ltm_interface import LTMInterface
    
    # Initialize components
    shared_memory = SharedMemoryInterface()
    ltm_interface = LTMInterface()
    tool_registry = ToolRegistry()
    
    results = []
    
    for i, test_case in enumerate(TEST_QUERIES):
        print(f"\n[Test {i+1}] {test_case['complexity'].upper()} Query")
        print(f"Query: {test_case['query']}")
        print(f"Expected to use memory: {test_case['expect_memory']}")
        
        try:
            # Create specialist agent
            agent = await create_specialist_agent(
                agent_id=f"gemini_test_{i}",
                registry=tool_registry,
                shared_memory=shared_memory,
                ltm_interface=ltm_interface
            )
            
            # Measure performance
            start_time = time.time()
            
            # Execute task
            response = await agent.execute_task({
                "id": f"gemini_test_task_{i}",
                "description": test_case['query']
            })
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Check if tools were used
            tool_indicators = ["search", "browse", "fetch", "retrieved from"]
            used_tools = any(indicator in str(response).lower() for indicator in tool_indicators)
            
            # Calculate tokens (approximate)
            response_tokens = len(str(response).split())
            
            result = {
                "test_id": i + 1,
                "query": test_case['query'],
                "complexity": test_case['complexity'],
                "processing_time": processing_time,
                "response_tokens": response_tokens,
                "used_tools": used_tools,
                "expected_memory_only": test_case['expect_memory'],
                "correct_behavior": used_tools != test_case['expect_memory'],
                "response_preview": str(response)[:200] + "..."
            }
            
            results.append(result)
            
            print(f"Processing time: {processing_time:.2f}s")
            print(f"Response tokens: ~{response_tokens}")
            print(f"Used external tools: {used_tools}")
            print(f"Correct behavior: {result['correct_behavior']}")
            
        except Exception as e:
            print(f"ERROR in test {i+1}: {str(e)}")
            results.append({
                "test_id": i + 1,
                "query": test_case['query'],
                "error": str(e)
            })
    
    # Generate summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    successful_tests = [r for r in results if "error" not in r]
    
    if successful_tests:
        avg_time = sum(r["processing_time"] for r in successful_tests) / len(successful_tests)
        avg_tokens = sum(r["response_tokens"] for r in successful_tests) / len(successful_tests)
        correct_behaviors = sum(1 for r in successful_tests if r["correct_behavior"])
        
        print(f"Average processing time: {avg_time:.2f}s")
        print(f"Average response tokens: {avg_tokens:.0f}")
        print(f"Correct tool/memory decisions: {correct_behaviors}/{len(successful_tests)}")
        
        # Compare with expected GPT-4o baseline
        print("\nExpected improvements over GPT-4o:")
        print(f"- Token reduction: ~{(1 - avg_tokens/150) * 100:.0f}% (baseline ~150 tokens)")
        print(f"- Speed improvement: ~{(1 - avg_time/5) * 100:.0f}% (baseline ~5s)")
    
    return results


async def verify_gemini_config():
    """Verify that Gemini is properly configured."""
    print("\n" + "="*60)
    print("CONFIGURATION VERIFICATION")
    print("="*60)
    
    # Check environment variables
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if google_api_key:
        print("✓ GOOGLE_API_KEY is set")
    else:
        print("✗ GOOGLE_API_KEY is NOT set - Gemini will not work!")
    
    # Check if specialist_agent.py is using Gemini
    agent_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "agents", "specialist_agent.py"
    )
    
    with open(agent_file, 'r') as f:
        content = f.read()
        
    if "gemini-2.5-flash" in content.lower():
        print("✓ Gemini 2.5 Flash model reference found in code")
    else:
        print("✗ Gemini 2.5 Flash NOT found - still using GPT-4o?")
    
    if 'LLM_MODEL_NAME = "gpt-4o"' in content:
        print("⚠ WARNING: LLM_MODEL_NAME is still set to 'gpt-4o'")
        print("  Update line 52 to: LLM_MODEL_NAME = 'gemini-2.5-flash-preview-05-20'")


async def main():
    """Run all Gemini tests."""
    # First verify configuration
    await verify_gemini_config()
    
    # Then run performance tests
    await test_gemini_performance()
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("Next steps:")
    print("1. Update specialist_agent.py to use Gemini model")
    print("2. Add proper Gemini imports and initialization")
    print("3. Run the full consciousness test suite")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
