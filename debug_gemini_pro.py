#!/usr/bin/env python3
"""
Gemini Pro Direct Test - Isolate the LLM Issue
Test Gemini 2.5 Pro with simple vs complex prompts
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

async def test_gemini_pro_responses():
    """Test Gemini Pro with different prompt complexities"""
    
    print("="*60)
    print("GEMINI 2.5 PRO DIRECT TESTING")
    print("="*60)
    
    # Initialize Gemini Pro (same as Nova-SHIFT)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("No Google API key found!")
        return
    
    gemini_pro = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-preview-05-06",
        temperature=0,
        api_key=google_api_key,
        max_tokens=2048,
        max_retries=3
    )
    
    # Test 1: Very Simple Prompt
    print("\n[TEST 1] Simple Prompt:")
    simple_prompt = "Hello! Please respond with a simple greeting."
    
    try:
        response = await gemini_pro.ainvoke(simple_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        print(f"Response: '{content}'")
        print(f"Length: {len(content)}")
        print(f"Type: {type(content)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: JSON Format Request
    print("\n[TEST 2] JSON Format Request:")
    json_prompt = """
    Please create a simple JSON array with one task:
    [{"subtask_id": "test_task", "description": "Test task", "suggested_tool": "None", "depends_on": []}]
    
    Response:
    """
    
    try:
        response = await gemini_pro.ainvoke(json_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        print(f"Response: '{content}'")
        print(f"Length: {len(content)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Dual-Mode Format (Similar to Nova-SHIFT)
    print("\n[TEST 3] Dual-Mode Format:")
    dual_prompt = """
    GOAL: What is artificial intelligence?
    
    MODE 1 - DIRECT ANSWER:
    If you know about this topic, start with "DIRECT_ANSWER:" and explain.
    
    MODE 2 - JSON TASKS:
    If you need research, provide JSON array of tasks.
    
    Response:
    """
    
    try:
        response = await gemini_pro.ainvoke(dual_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        print(f"Response: '{content}'")
        print(f"Length: {len(content)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Long Context (Similar to Nova-SHIFT with MCP results)
    print("\n[TEST 4] Long Context Test:")
    long_prompt = f"""
    GOAL: Research neuromorphic computing
    
    MEMORY CONTEXT:
    --- START MEMORY CONTEXT ---
    {'Sample context line. ' * 200}  # Simulate long MCP context
    --- END MEMORY CONTEXT ---
    
    INSTRUCTIONS: Either respond with "DIRECT_ANSWER:" or create JSON tasks.
    
    Response:
    """
    
    try:
        response = await gemini_pro.ainvoke(long_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        print(f"Response: '{content}'")
        print(f"Length: {len(content)}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_gemini_pro_responses())
