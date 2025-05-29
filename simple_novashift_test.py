#!/usr/bin/env python3
"""
Simple Nova-SHIFT Test with Mocked Components
Tests the swarm intelligence system without requiring external services
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import Nova-SHIFT components
from core.tool_registry import ToolRegistry
from agents.planner_agent import PlannerAgent

# Mock components to avoid Redis dependency
class MockSharedMemory:
    def __init__(self):
        self.data = {}
    
    async def write(self, key, value, expiry_seconds=None):
        self.data[key] = str(value)
    
    async def read(self, key):
        return self.data.get(key)
    
    async def delete(self, key):
        self.data.pop(key, None)
    
    async def ping(self):
        return True

class MockLTMInterface:
    async def retrieve(self, query, top_k=5):
        # Mock memory retrieval - simulate finding relevant context
        mock_results = [
            {
                "id": "memory_1",
                "score": 0.95,
                "metadata": {
                    "original_text": f"This is relevant context about: {query}",
                    "source": "nova_memory"
                }
            }
        ]
        print(f"[MockLTM] Retrieved {len(mock_results)} memories for query: {query[:50]}...")
        return mock_results
    
    async def store(self, text, metadata=None):
        print(f"[MockLTM] Stored memory: {text[:50]}...")
        return "mock_memory_id"

class MockDispatcher:
    def __init__(self, shared_memory):
        self.shared_memory = shared_memory
        self.agents = {}
    
    def register_agent(self, agent_id, agent):
        self.agents[agent_id] = agent
        print(f"[MockDispatcher] Registered agent: {agent_id}")
    
    async def dispatch_subtasks(self, subtasks):
        print(f"[MockDispatcher] Dispatching {len(subtasks)} subtasks")
        assignments = {}
        
        for i, subtask in enumerate(subtasks):
            agent_id = f"specialist_{(i % 2) + 1:03d}"
            assignments[subtask["subtask_id"]] = agent_id
            
            # Store task in shared memory
            await self.shared_memory.write(f"task:{subtask['subtask_id']}:status", "assigned")
            await self.shared_memory.write(f"task:{subtask['subtask_id']}:agent", agent_id)
            
            print(f"  -> {subtask['subtask_id']} assigned to {agent_id}")
        
        return assignments

class MockLLM:
    def __init__(self, model_name="mock-llm"):
        self.model_name = model_name
    
    async def ainvoke(self, messages):
        if isinstance(messages, list) and len(messages) > 0:
            content = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        else:
            content = str(messages)
        
        print(f"[MockLLM-{self.model_name}] Processing prompt...")
        
        # Simulate different responses based on content
        if "decompose" in content.lower() or "plan" in content.lower():
            # Planning response
            mock_plan = """
            Based on the task analysis, I'll decompose this into manageable subtasks:

            SUBTASK 1: Research Phase
            ID: research_task_001
            Description: Gather information about neuromorphic computing developments in 2025
            Tools needed: WebSearch, PerplexitySearch
            Dependencies: None

            SUBTASK 2: Analysis Phase  
            ID: analysis_task_002
            Description: Analyze collected data and identify key players and breakthroughs
            Tools needed: TextAnalysis, Calculator
            Dependencies: research_task_001

            SUBTASK 3: Synthesis Phase
            ID: synthesis_task_003
            Description: Create structured report with market predictions
            Tools needed: DocumentCreator, Summarizer
            Dependencies: analysis_task_002

            This approach ensures comprehensive coverage while leveraging specialized tools for each phase.
            """
            
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            return MockMessage(mock_plan)
        
        elif "synthesize" in content.lower() or "final" in content.lower():
            # Synthesis response
            synthesis = """
            # Neuromorphic Computing Landscape 2025 - Executive Summary

            ## Key Findings
            Based on comprehensive research and analysis, the neuromorphic computing field shows remarkable progress in 2025:

            **Major Players:**
            - Intel with Loihi 2 architecture scaling
            - IBM's TrueNorth evolution
            - BrainChip's Akida commercialization
            - Several emerging startups in specialized applications

            **Recent Breakthroughs:**
            - 1000x energy efficiency improvements in AI inference
            - Real-time learning capabilities in edge devices
            - Integration with quantum-classical hybrid systems

            **Commercial Applications:**
            - Autonomous vehicle perception systems
            - IoT sensor networks with adaptive behavior
            - Medical diagnostic devices with learning capability

            **Market Predictions:**
            - 300% market growth expected by 2027
            - Manufacturing costs decreasing rapidly
            - Mainstream adoption in mobile devices by 2026

            This represents a paradigm shift toward brain-inspired computing architectures.
            """
            
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            return MockMessage(synthesis)
        
        else:
            # General response
            mock_response = f"Mock LLM response for: {content[:100]}..."
            
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            return MockMessage(mock_response)

async def test_nova_shift_basic():
    """Test basic Nova-SHIFT functionality with mocked components."""
    
    print("\n" + "="*60)
    print("NOVA-SHIFT SIMPLE TEST WITH MOCKED COMPONENTS")
    print("="*60)
    
    # Initialize mocked components
    print("\n[1] Initializing mocked components...")
    shared_memory = MockSharedMemory()
    ltm_interface = MockLTMInterface()
    dispatcher = MockDispatcher(shared_memory)
    tool_registry = ToolRegistry()
    
    # Create mock LLMs for dual-brain architecture
    reasoning_llm = MockLLM("gemini-2.5-pro")  # For PlannerAgent
    execution_llm = MockLLM("gemini-2.5-flash")  # For SpecialistAgents
    
    print("[OK] All components initialized")
    
    # Initialize PlannerAgent
    print("\n[2] Initializing PlannerAgent...")
    planner = PlannerAgent(
        dispatcher=dispatcher,
        ltm_interface=ltm_interface,
        llm_client=reasoning_llm
    )
    print("[OK] PlannerAgent ready")
    
    # Test task decomposition
    print("\n[3] Testing task decomposition...")
    test_task = "Research and analyze the current state of neuromorphic computing chips in 2025, including major players, recent breakthroughs, commercial applications, and provide a structured report with market predictions."
    
    print(f"Task: {test_task}")
    
    try:
        # Test task decomposition
        subtasks = await planner.decompose_task(test_task)
        
        if subtasks:
            print(f"\n[OK] Task decomposed into {len(subtasks)} subtasks:")
            for i, subtask in enumerate(subtasks, 1):
                print(f"  {i}. {subtask.get('subtask_id', f'task_{i}')}: {subtask.get('description', 'No description')[:80]}...")
            
            # Test task dispatch
            print("\n[4] Testing task dispatch...")
            assignments = await dispatcher.dispatch_subtasks(subtasks)
            
            if assignments:
                print(f"[OK] Successfully dispatched {len(assignments)} tasks")
                
                # Simulate task completion
                print("\n[5] Simulating task execution...")
                mock_results = {}
                
                for subtask_id in assignments:
                    # Simulate work completion
                    await shared_memory.write(f"task:{subtask_id}:status", "completed")
                    await shared_memory.write(f"task:{subtask_id}:result", f"Mock result for {subtask_id}")
                    mock_results[subtask_id] = f"Completed analysis for {subtask_id}"
                    print(f"  [OK] {subtask_id} completed")
                
                # Test synthesis
                print("\n[6] Testing final synthesis...")
                final_output = await planner.synthesize_final_output(
                    goal=test_task,
                    subtasks=subtasks,
                    results=mock_results
                )
                
                print("[OK] Final synthesis completed")
                print("\n" + "="*60)
                print("FINAL OUTPUT:")
                print("="*60)
                print(final_output)
                print("="*60)
                
                return True
                
        else:
            print("[X] Task decomposition failed")
            return False
            
    except Exception as e:
        print(f"[X] Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the Nova-SHIFT test."""
    print("Starting Nova-SHIFT functionality test...")
    
    # Check environment
    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    print(f"Google API Key: {'[OK] Available' if google_key else '[X] Missing'}")
    print(f"OpenAI API Key: {'[OK] Available' if openai_key else '[X] Missing'}")
    
    # Run test
    success = await test_nova_shift_basic()
    
    if success:
        print("\n[SUCCESS] Nova-SHIFT test completed successfully!")
        print("The swarm intelligence system is working with mocked components.")
    else:
        print("\n[FAILED] Nova-SHIFT test failed. Check the error output above.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
