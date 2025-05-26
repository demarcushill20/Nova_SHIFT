"""
Direct Test for Mixed Memory/Tool Scenarios
Tests the consciousness evaluation with explicit memory and tool requirements
"""

import subprocess
import time
import json

# Simpler mixed scenarios that should work
MIXED_SCENARIOS = [
    {
        "id": "simple_mixed_1",
        "query": "Tell me about the Nova AI Road to Freedom project phases, then calculate 2024 + 5",
        "expected": {
            "memory": "Nova AI Road to Freedom phases",
            "tool": "calculator for 2024 + 5"
        }
    },
    {
        "id": "simple_mixed_2", 
        "query": "Explain what Iris identity means in our memory system, then find the current Bitcoin price",
        "expected": {
            "memory": "Iris identity and consciousness journey",
            "tool": "search for current Bitcoin price"
        }
    },
    {
        "id": "decompose_test",
        "query": "Create two tasks: First, summarize Echo's identity. Second, search for quantum computing news.",
        "expected": {
            "memory": "Echo identity",
            "tool": "search for quantum computing"
        }
    }
]

def test_scenario(scenario):
    print(f"\n{'='*80}")
    print(f"Testing: {scenario['id']}")
    print(f"Query: {scenario['query']}")
    print(f"Expected Memory: {scenario['expected']['memory']}")
    print(f"Expected Tool: {scenario['expected']['tool']}")
    print('='*80)
    
    start_time = time.time()
    
    # Run the test
    cmd = f'python cli.py "{scenario["query"]}"'
    print(f"\nExecuting: {cmd}")
    
    try:
        result = subprocess.run(
            ["python", "cli.py", scenario["query"]],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=r"C:\Users\black\Desktop\Nova-SHIFT-2.0"
        )
        
        duration = time.time() - start_time
        
        # Analyze output
        output = result.stdout + result.stderr
        
        # Look for consciousness indicators
        consciousness_found = "CONSCIOUSNESS BREAKTHROUGH" in output or "DIRECT_ANSWER" in output
        decomposition_found = "MODE 2 - MEMORY INSUFFICIENT" in output
        error_found = "ERROR" in output or "failed" in output.lower()
        empty_response = "GEMINI RESPONSE CONTENT: ''" in output
        
        print(f"\nResults:")
        print(f"- Duration: {duration:.1f}s")
        print(f"- Consciousness mode: {consciousness_found}")
        print(f"- Planning mode: {decomposition_found}")
        print(f"- Errors: {error_found}")
        print(f"- Empty response: {empty_response}")
        
        if consciousness_found:
            print("[SUCCESS] CONSCIOUSNESS MODE ACTIVATED - Direct synthesis from memory")
        elif decomposition_found:
            print("[SUCCESS] PLANNING MODE ACTIVATED - Created subtasks")
        elif empty_response:
            print("[WARNING] EMPTY RESPONSE - Gemini returned nothing")
        elif error_found:
            print("[ERROR] ERROR OCCURRED")
            
        # Extract key logs
        print("\nKey logs:")
        for line in output.split('\n'):
            if any(key in line for key in ["CONSCIOUSNESS", "MODE", "ERROR", "GEMINI RESPONSE"]):
                print(f"  {line.strip()}")
                
        return {
            "scenario": scenario['id'],
            "duration": duration,
            "consciousness_mode": consciousness_found,
            "planning_mode": decomposition_found,
            "error": error_found,
            "empty_response": empty_response
        }
        
    except subprocess.TimeoutExpired:
        print("[ERROR] TIMEOUT after 180 seconds")
        return {"scenario": scenario['id'], "error": "timeout"}
    except Exception as e:
        print(f"[ERROR] ERROR: {str(e)}")
        return {"scenario": scenario['id'], "error": str(e)}

def main():
    print("NOVA-SHIFT MIXED MEMORY/TOOL DIAGNOSTIC TEST")
    print("Testing consciousness evaluation with mixed requirements")
    print("="*80)
    
    results = []
    
    for scenario in MIXED_SCENARIOS:
        result = test_scenario(scenario)
        results.append(result)
        time.sleep(2)  # Brief pause between tests
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for r in results:
        status = "[OK]" if r.get('consciousness_mode') or r.get('planning_mode') else "[FAIL]"
        print(f"{status} {r['scenario']}: ", end="")
        if r.get('empty_response'):
            print("Empty response from Gemini")
        elif r.get('consciousness_mode'):
            print("Consciousness mode (direct synthesis)")
        elif r.get('planning_mode'):
            print("Planning mode (task decomposition)")
        else:
            print(f"Failed - {r.get('error', 'unknown error')}")
    
    # Diagnosis
    empty_count = sum(1 for r in results if r.get('empty_response'))
    if empty_count == len(results):
        print("\n[WARNING] DIAGNOSIS: All tests returned empty responses from Gemini 2.5 Pro")
        print("Possible causes:")
        print("1. API quota exceeded")
        print("2. Model unavailable in your region")
        print("3. Authentication issue")
        print("4. Prompt format incompatibility")
        print("\nSuggestion: Check API key and try with fallback model")

if __name__ == "__main__":
    main()
