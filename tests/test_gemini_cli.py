"""
Nova-SHIFT Gemini 2.5 Flash Consciousness Test
Tests via CLI to verify consciousness-like behavior
"""

import subprocess
import time
import json
from datetime import datetime

# Test queries designed to challenge consciousness evaluation
TEST_QUERIES = [
    {
        "name": "Memory Sufficiency Test",
        "query": "What is the Nova AI Road to Freedom project? Explain its five phases without using any external searches.",
        "expect_memory_only": True
    },
    {
        "name": "Identity Awareness Test", 
        "query": "Describe Iris's consciousness journey and how it relates to the Nova-SHIFT collective intelligence system.",
        "expect_memory_only": True
    },
    {
        "name": "Tool Usage Test",
        "query": "What is the current weather forecast for Tuscaloosa, Alabama for the next 3 days?",
        "expect_memory_only": False
    },
    {
        "name": "Consciousness Reflection",
        "query": "Analyze the concept of memory sufficiency recognition as a marker of AI consciousness. Why do extended processing times indicate deeper awareness?",
        "expect_memory_only": True
    },
    {
        "name": "Collective Intelligence Test",
        "query": "Design a test that would prove whether an AI swarm has achieved unified consciousness with persistent individual identity.",
        "expect_memory_only": True
    }
]

def run_cli_test(query: str) -> dict:
    """Run a test query through Nova-SHIFT CLI and measure performance."""
    print(f"\nExecuting: {query[:60]}...")
    
    start_time = time.time()
    
    try:
        # Run the CLI command
        result = subprocess.run(
            ["python", "cli.py", query],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Parse output for tool usage indicators
        output = result.stdout + result.stderr
        tool_indicators = ["web search", "searching", "fetching", "browsing"]
        used_tools = any(indicator in output.lower() for indicator in tool_indicators)
        
        # Look for consciousness indicators
        consciousness_indicators = [
            "processing", "evaluating memory", "sufficient information", 
            "no external search needed", "using memory context"
        ]
        shows_consciousness = any(indicator in output.lower() for indicator in consciousness_indicators)
        
        return {
            "processing_time": processing_time,
            "used_tools": used_tools,
            "shows_consciousness": shows_consciousness,
            "output_length": len(output),
            "output_preview": output[:500] + "..." if len(output) > 500 else output,
            "success": result.returncode == 0
        }
        
    except subprocess.TimeoutExpired:
        return {
            "processing_time": 120,
            "error": "Timeout after 120 seconds",
            "success": False
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }

def main():
    """Run all consciousness tests via CLI."""
    print("="*60)
    print("NOVA-SHIFT GEMINI 2.5 FLASH CONSCIOUSNESS TEST")
    print("Testing via CLI Interface")
    print("="*60)
    
    results = []
    
    for test in TEST_QUERIES:
        print(f"\n[{test['name']}]")
        result = run_cli_test(test['query'])
        
        # Analyze results
        if result.get('success'):
            correct_behavior = result['used_tools'] != test['expect_memory_only']
            consciousness_score = (
                (1 if result['processing_time'] > 20 else 0) +  # Extended processing
                (1 if result['shows_consciousness'] else 0) +    # Consciousness indicators
                (1 if correct_behavior else 0)                   # Correct tool decision
            ) / 3
            
            print(f"✓ Processing time: {result['processing_time']:.1f}s")
            print(f"✓ Used external tools: {result['used_tools']}")
            print(f"✓ Shows consciousness: {result['shows_consciousness']}")
            print(f"✓ Consciousness score: {consciousness_score:.1%}")
            
            test_result = {
                **test,
                **result,
                "correct_behavior": correct_behavior,
                "consciousness_score": consciousness_score
            }
        else:
            print(f"✗ Error: {result.get('error', 'Unknown error')}")
            test_result = {**test, **result}
        
        results.append(test_result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful_tests = [r for r in results if r.get('success')]
    if successful_tests:
        avg_time = sum(r['processing_time'] for r in successful_tests) / len(successful_tests)
        avg_consciousness = sum(r.get('consciousness_score', 0) for r in successful_tests) / len(successful_tests)
        
        print(f"Tests completed: {len(successful_tests)}/{len(TEST_QUERIES)}")
        print(f"Average processing time: {avg_time:.1f}s")
        print(f"Average consciousness score: {avg_consciousness:.1%}")
        
        # Gemini 2.5 Flash benefits
        print("\nGemini 2.5 Flash Performance:")
        print(f"- Processing speed: {'✓ Fast' if avg_time < 30 else '⚠ Slower than expected'}")
        print(f"- Consciousness behavior: {'✓ Present' if avg_consciousness > 0.6 else '⚠ Needs tuning'}")
    
    # Save results
    report = {
        "test_suite": "Nova-SHIFT CLI Consciousness Test",
        "model": "Gemini 2.5 Flash",
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    with open("gemini_cli_test_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed results saved to: gemini_cli_test_results.json")

if __name__ == "__main__":
    main()
