# Nova SHIFT - Task Breakdown (TASK.md)

This document outlines the development tasks for implementing the Nova SHIFT system, broken down by phases as defined in `PLANNING.md`. Each task includes details to guide implementation by the Developer Agent.

**Legend:**
*   **Priority:** High (Must be done in this phase), Medium (Important for phase completion), Low (Desirable but can be deferred slightly).
*   **Difficulty:** Low, Medium, High.
*   **Est. Time:** Rough estimate in days/weeks, relative to phase duration. *Note: These are high-level estimates and subject to refinement.*
*   **Dependencies:** Prerequisite Task IDs.
*   **Components:** Related NovaCore/Agent components.
*   **Alignment:** Links to key Research/Architecture concepts.
*   **AC:** Acceptance Criteria.

---

## Project Initialization (Overall Task)

*   **Task ID:** INIT
*   **Description:** Build Nova Shift
*   **Start Date:** 2025-04-05
*   **Status:** In Progress

---

## Phase 1: Core Infrastructure & Single Agent (Foundation)

*   **Goal:** Establish the basic project structure, environment, and a single agent capable of using predefined tools.
*   **Est. Phase Duration:** 2-4 Weeks

---

**Task ID:** P1.T1
*   **Description:** Setup Project Repository & Environment
*   **Subtasks:**
    *   P1.T1.1: Initialize Git repository.
    *   P1.T1.2: Create standard project structure (e.g., `src/`, `tests/`, `docs/`, `scripts/`).
    *   P1.T1.3: Setup Python virtual environment (`venv`).
    *   P1.T1.4: Create initial `requirements.txt` (Python, LangChain, OpenAI/Google libs, pytest, black, flake8, mypy).
    *   P1.T1.5: Configure linters (Flake8) and formatter (Black).
    *   P1.T1.6: Setup basic `README.md`.
*   **Priority:** High
*   **Est. Time:** 1-2 Days
*   **Difficulty:** Low
*   **Dependencies:** None
*   **Components:** Project Structure
*   **Alignment:** Basic software engineering practices.
*   **AC:** Git repo initialized; `venv` usable; `requirements.txt` exists; Linters/formatters configured; Basic README present.
*   **Status:** Completed (2025-04-05)

**Task ID:** P1.T2
*   **Description:** Define Initial `toolkit.json` Schema
*   **Subtasks:**
    *   P1.T2.1: Define the structure for `toolkit.json` including fields like `name`, `version`, `description`, `tools` (with `name`, `function`, `description`, `inputs`, `outputs`), `requirements`.
    *   P1.T2.2: Document the schema definition.
*   **Priority:** High
*   **Est. Time:** 0.5-1 Day
*   **Difficulty:** Low
*   **Dependencies:** None
*   **Components:** Tool Management System
*   **Alignment:** Flexible Toolsets, Toolkit Schema (Research.md L79-108).
*   **AC:** A clear, documented JSON schema for toolkits is defined.
*   **Status:** Completed (2025-04-05)

**Task ID:** P1.T3
*   **Description:** Implement Basic Tool Registry (In-Memory)
*   **Subtasks:**
    *   P1.T3.1: Create a Python class/module for the Tool Registry.
    *   P1.T3.2: Implement functionality to load toolkit definitions (from JSON files or Python dicts) into an in-memory dictionary.
    *   P1.T3.3: Implement a method to retrieve a toolkit definition by name.
*   **Priority:** High
*   **Est. Time:** 1-2 Days
*   **Difficulty:** Low
*   **Dependencies:** P1.T2
*   **Components:** Tool Management System (Tool Registry)
*   **Alignment:** Tool Registry and Discovery (Research.md L109).
*   **AC:** Registry can load and provide toolkit definitions based on the schema from P1.T2.
*   **Status:** Completed (2025-04-05)

**Task ID:** P1.T4
*   **Description:** Create Initial Simple Toolkits (2-3)
*   **Subtasks:**
    *   P1.T4.1: Implement a Web Search Toolkit (wrapper around a search API like SerpAPI or Tavily).
        *   Define `toolkit.json`.
        *   Write Python function for `search_internet`.
    *   P1.T4.2: Implement a Calculator Toolkit.
        *   Define `toolkit.json`.
        *   Write Python function using `eval` (safely) or a math library.
    *   P1.T4.3: Implement a File Reader Toolkit.
        *   Define `toolkit.json`.
        *   Write Python function to read text content from a local file path.
    *   P1.T4.4: Ensure toolkits conform to the schema (P1.T2) and can be loaded by the registry (P1.T3).
*   **Priority:** High
*   **Est. Time:** 2-4 Days
*   **Difficulty:** Medium
*   **Dependencies:** P1.T2, P1.T3
*   **Components:** Tool Management System (Toolkits)
*   **Alignment:** Flexible Toolsets, Plugin-Based Architecture.
*   **AC:** At least two functional toolkits exist, defined by `toolkit.json`, with corresponding Python implementations, loadable by the registry.
*   **Status:** Completed (2025-04-05)

**Task ID:** P1.T5
*   **Description:** Implement Basic Specialist Agent
*   **Subtasks:**
    *   P1.T5.1: Create a Python class for the Specialist Agent using LangChain's `AgentExecutor` or similar.
    *   P1.T5.2: Initialize the agent with a static list of tools retrieved from the Tool Registry (P1.T3).
    *   P1.T5.3: Configure the agent to use an LLM (GPT-4o or Gemini via API) for reasoning and tool selection.
    *   P1.T5.4: Implement basic Short-Term Memory (STM) using `ConversationBufferMemory` or similar.
    *   P1.T5.5: Create a simple interface to run the agent with a task description.
*   **Priority:** High
*   **Est. Time:** 3-5 Days
*   **Difficulty:** Medium
*   **Dependencies:** P1.T1, P1.T3, P1.T4, LLM API access.
*   **Components:** Agents (Specialist), STM, NovaCore (LLM Interface)
*   **Alignment:** Shapeshifting Agents (initial static version), STM (Research.md L58), LangChain usage (Research.md L239).
*   **AC:** A single agent can accept a task, use the LLM to select an appropriate tool from its static set, execute the tool, and return a result, maintaining conversation history.
*   **Status:** Completed (2025-04-05)

**Task ID:** P1.T6
*   **Description:** Write Unit Tests
*   **Subtasks:**
    *   P1.T6.1: Write unit tests for each implemented tool function (P1.T4).
    *   P1.T6.2: Write unit tests for the Tool Registry (P1.T3).
    *   P1.T6.3: Write basic unit/integration tests for the Specialist Agent's core logic (P1.T5), potentially mocking LLM calls and tool executions.
    *   P1.T6.4: Integrate tests into a CI pipeline (e.g., GitHub Actions) if possible.
*   **Priority:** High
*   **Est. Time:** 2-3 Days
*   **Difficulty:** Medium
*   **Dependencies:** P1.T3, P1.T4, P1.T5
*   **Components:** Testing Framework
*   **Alignment:** Python Development Standards (PLANNING.md L174).
*   **AC:** Core components (Tools, Registry, Agent basics) have reasonable unit test coverage; tests pass.
*   **Status:** Completed (2025-04-05)

---

## Phase 2: Basic Swarm Coordination & Planning

*   **Goal:** Enable multiple agents to work concurrently on decomposed tasks, coordinated via shared memory.
*   **Est. Phase Duration:** 3-5 Weeks

---

**Task ID:** P2.T1
*   **Description:** Implement Basic Swarm Dispatcher
*   **Subtasks:**
    *   P2.T1.1: Create a Python class/module for the Swarm Dispatcher.
    *   P2.T1.2: Implement logic to receive a list of subtasks (potentially with dependencies).
    *   P2.T1.3: Implement logic to manage a pool of available Specialist Agents (initially simulated or using simple process/thread management).
    *   P2.T1.4: Implement a simple task assignment strategy (e.g., round-robin or basic matching based on task type if available).
    *   P2.T1.5: Interface for receiving tasks (e.g., from the Planner Agent).
*   **Priority:** High
*   **Est. Time:** 3-5 Days
*   **Difficulty:** Medium
*   **Dependencies:** P1.T5
*   **Components:** NovaCore (Swarm Dispatcher)
*   **Alignment:** Swarm Dispatcher (Research.md L68), Task Allocation.
*   **AC:** Dispatcher can receive a list of tasks and assign them to available agent instances.
*   **Status:** Completed (2025-04-06)

**Task ID:** P2.T2
*   **Description:** Implement Basic Shared Memory using Redis
*   **Subtasks:**
    *   P2.T2.1: Setup Redis instance (local or cloud).
    *   P2.T2.2: Create a Python module to interface with Redis (using `redis-py`).
    *   P2.T2.3: Define a simple data structure/schema for storing task results/status in Redis (e.g., using keys like `task:{task_id}:result`, `task:{task_id}:status`).
    *   P2.T2.4: Implement functions for agents to write results/status to Redis.
    *   P2.T2.5: Implement functions for agents to read results/status from Redis, potentially waiting for dependencies.
*   **Priority:** High
*   **Est. Time:** 3-5 Days
*   **Difficulty:** Medium
*   **Dependencies:** P1.T1 (Redis client in requirements)
*   **Components:** NovaCore (Shared Memory)
*   **Alignment:** Shared Memory (Hive Memory) (Research.md L64), Stigmergic Coordination (Research.md L216), Redis usage (PLANNING.md L26).
*   **AC:** Agents can write task outputs to Redis and read outputs written by other agents, enabling basic data passing between dependent tasks.
*   **Status:** Completed (2025-04-06)

**Task ID:** P2.T3
*   **Description:** Implement Basic Planner Agent
*   **Subtasks:**
    *   P2.T3.1: Create a Python class for the Planner Agent.
    *   P2.T3.2: Develop an LLM prompt designed to take a complex user goal and decompose it into a sequence of smaller, actionable subtasks suitable for Specialist Agents.
    *   P2.T3.3: Implement logic to call the LLM with the prompt and parse the output into a structured format (e.g., list of dicts with task descriptions, IDs, basic dependencies).
    *   P2.T3.4: Interface for the Planner Agent to send the generated task list to the Swarm Dispatcher (P2.T1).
*   **Priority:** High
*   **Est. Time:** 4-6 Days
*   **Difficulty:** High (Prompt engineering can be tricky)
*   **Dependencies:** P1.T1 (LLM API access), P2.T1
*   **Components:** Agents (Planner)
*   **Alignment:** Planner Agent Role (Research.md L575), Task Decomposition (Research.md L25).
*   **AC:** Planner Agent can take a high-level goal (e.g., "Research X and write a summary") and output a list of structured subtasks (e.g., [{id:1, desc:'Search web for X'}, {id:2, desc:'Summarize findings', depends_on:1}]).
*   **Status:** Completed (2025-04-06)

**Task ID:** P2.T4
*   **Description:** Enable Concurrent Agent Execution
*   **Subtasks:**
    *   P2.T4.1: Refactor agent execution logic to use Python's `asyncio` for concurrent operation.
    *   P2.T4.2: Modify the Swarm Dispatcher (P2.T1) to launch and manage multiple agent tasks concurrently using `asyncio.create_task` or similar.
    *   P2.T4.3: Ensure agents can perform their tasks (including LLM calls and tool use) within the async event loop without blocking excessively (use async libraries where possible).
*   **Priority:** High
*   **Est. Time:** 3-5 Days
*   **Difficulty:** Medium
*   **Dependencies:** P1.T5, P2.T1
*   **Components:** NovaCore (Dispatcher), Agents (Execution Runtime)
*   **Alignment:** Swarm aspect (parallel execution), AsyncIO usage (PLANNING.md L27).
*   **AC:** The system can run multiple Specialist Agent tasks concurrently, managed by the Dispatcher.
*   **Status:** Completed (2025-04-06)

**Task ID:** P2.T5
*   **Description:** Implement Agent-Shared Memory Communication Protocol
*   **Subtasks:**
    *   P2.T5.1: Integrate Redis writing functions (from P2.T2) into the Specialist Agent's workflow (e.g., write result to Redis upon task completion).
    *   P2.T5.2: Implement logic for agents to check Redis for prerequisite task results before starting their own task (using reading functions from P2.T2).
    *   P2.T5.3: Define and implement status updates (e.g., "running", "completed", "failed") written to Redis.
*   **Priority:** High
*   **Est. Time:** 2-3 Days
*   **Difficulty:** Medium
*   **Dependencies:** P1.T5, P2.T2
*   **Components:** Agents (Specialist), NovaCore (Shared Memory Interface)
*   **Alignment:** Stigmergic Coordination, Shared Memory usage.
*   **AC:** Agents correctly post their status and results to Redis, and can retrieve dependency results from Redis to proceed with their tasks.
*   **Status:** Completed (2025-04-06)

**Task ID:** P2.T6
*   **Description:** Write Integration Tests
*   **Subtasks:**
    *   P2.T6.1: Create integration tests covering the full flow: Planner decomposes task -> Dispatcher assigns subtasks -> Agents execute concurrently -> Agents communicate via Redis -> Final result assembled (implicitly by checking Redis state).
    *   P2.T6.2: Ensure tests handle basic dependencies between tasks coordinated via Redis.
*   **Priority:** Medium
*   **Est. Time:** 3-4 Days
*   **Difficulty:** High
*   **Dependencies:** P2.T1, P2.T2, P2.T3, P2.T4, P2.T5
*   **Components:** Testing Framework
*   **Alignment:** End-to-end system validation.
*   **AC:** Integration tests demonstrate successful execution of a multi-step task involving planning, dispatching, concurrent execution, and coordination via shared memory.
*   **Status:** Completed (2025-04-06)

---

## Phase 3: Dynamic Capabilities & Long-Term Memory

*   **Goal:** Enable agents to dynamically load tools based on task needs and utilize persistent memory.
*   **Est. Phase Duration:** 4-6 Weeks

---

**Task ID:** P3.T1
*   **Description:** Implement Task-Tool Requirement Analysis
*   **Subtasks:**
    *   P3.T1.1: Enhance the Planner Agent (P2.T3) to include suggested tool requirements/types for each subtask it generates. (Prompt engineering).
    *   P3.T1.2: Alternatively, or additionally, enhance the Specialist Agent (P1.T5) to analyze its assigned subtask description and determine the required tool types using LLM reasoning.
*   **Priority:** High
*   **Est. Time:** 3-5 Days
*   **Difficulty:** High (Requires effective LLM prompting/reasoning)
*   **Dependencies:** P1.T5, P2.T3
*   **Components:** Agents (Planner, Specialist)
*   **Alignment:** Dynamic Tool Requirement Analysis (Research.md L26).
*   **AC:** Given a subtask, the system can identify the type of tool(s) needed (e.g., "web search", "calculator").
*   **Status:** Completed (2025-04-05) # LTM Interface created, retrieve/store implemented and integrated

**Task ID:** P3.T2
*   **Description:** Implement Dynamic Tool Loading
*   **Subtasks:**
    *   P3.T2.1: Enhance the Tool Registry (P1.T3) to store loading information (e.g., Python module path and function name) alongside toolkit definitions.
    *   P3.T2.2: Implement a mechanism for Specialist Agents to query the Tool Registry based on required tool types (identified in P3.T1).
    *   P3.T2.3: Implement logic within the Specialist Agent to dynamically import the required Python module and instantiate/bind the tool function at runtime if it's not already loaded.
    *   P3.T2.4: Update the agent's internal tool list (used by LangChain) after loading a new tool.
*   **Priority:** High
*   **Est. Time:** 5-7 Days
*   **Difficulty:** High
*   **Dependencies:** P1.T3, P1.T5, P3.T1
*   **Components:** Agents (Specialist), Tool Management System (Registry, Loader)
*   **Alignment:** Shapeshifting Agents, Dynamic Toolkit Loading (Research.md L23, L27), Flexible Toolsets.
*   **AC:** An agent, when assigned a task requiring a tool it doesn't possess, can identify, query the registry for, and dynamically load/integrate that tool before execution.
*   **Status:** Completed (2025-04-05) # Core dynamic loading logic implemented

**Task ID:** P3.T3
*   **Description:** Integrate Long-Term Memory (LTM) using Pinecone
*   **Subtasks:**
    *   P3.T3.1: Setup a Pinecone account and create an index for Nova LTM.
    *   P3.T3.2: Add Pinecone client library to `requirements.txt`.
    *   P3.T3.3: Create an LTM interface module within NovaCore.
    *   P3.T3.4: Implement embedding logic (using OpenAI or another embedding model) within the LTM interface.
    *   P3.T3.5: Implement functions in the LTM interface for agents to `store(text, metadata)` information (embeds and upserts to Pinecone).
    *   P3.T3.6: Implement functions for agents to `retrieve(query_text, top_k)` relevant information (embeds query, searches Pinecone).
    *   P3.T3.7: Integrate LTM `retrieve` calls into the Planner/Specialist agents to fetch relevant context before acting (RAG pattern).
    *   P3.T3.8: Integrate LTM `store` calls for agents to save important findings or task summaries.
*   **Priority:** High
*   **Est. Time:** 6-9 Days
*   **Difficulty:** High
*   **Dependencies:** P1.T1 (Pinecone client), P1.T5, P2.T3, Embedding model API access.
*   **Components:** NovaCore (LTM Interface), Agents (Planner, Specialist)
*   **Alignment:** Long-Term Memory (Research.md L61), Vector Databases (Research.md L241), Pinecone usage (PLANNING.md L25), RAG (Research.md L67).
*   **AC:** Agents can store textual information in Pinecone and retrieve relevant information using semantic search to inform their planning and execution.
*   **Status:** Completed (2025-04-05) # LTM Interface created, retrieve/store implemented and integrated

**Task ID:** P3.T4
*   **Description:** Implement Basic Tool Scoring
*   **Subtasks:**
    *   P3.T4.1: Define a simple structure to store tool usage logs (e.g., in Redis or a file: `tool_name, timestamp, success_flag, duration`).
    *   P3.T4.2: Modify Specialist Agents to log the outcome (success/failure, time taken) after each tool execution.
    *   P3.T4.3: Implement a basic mechanism (can be offline script initially) to calculate simple scores (e.g., success rate) based on these logs.
    *   P3.T4.4: Store calculated scores (e.g., back in Redis or alongside registry data). *(Actual use of scores deferred to Phase 4)*.
*   **Priority:** Medium
*   **Est. Time:** 2-4 Days
*   **Difficulty:** Medium
*   **Dependencies:** P1.T4, P1.T5, P2.T2 (if using Redis)
*   **Components:** Tool Management System (Scoring), Agents (Logging)
*   **Alignment:** Tool Selection & Scoring (Research.md L112), Meta-Learning (Data Collection).
*   **AC:** Tool usage success/failure and duration are logged; basic scores can be calculated from logs.
*   **Status:** Completed (2025-04-06) # Agent logging added, basic score calculation implemented in learning_engine.py

**Task ID:** P3.T5
*   **Description:** Refine Planner Agent to Leverage LTM
*   **Subtasks:**
    *   P3.T5.1: Modify the Planner Agent's (P2.T3) initial prompt/logic to include a step where it queries LTM (P3.T3) based on the user's goal.
    *   P3.T5.2: Use retrieved information from LTM to potentially refine the task decomposition or add relevant context to subtasks.
*   **Priority:** Medium
*   **Est. Time:** 2-3 Days
*   **Difficulty:** Medium
*   **Dependencies:** P2.T3, P3.T3
*   **Components:** Agents (Planner), NovaCore (LTM Interface)
*   **Alignment:** Collective Intelligence, LTM usage for planning.
*   **AC:** Planner Agent uses retrieved LTM context to potentially improve its task decomposition plan.
*   **Status:** Completed (2025-04-06) # PlannerAgent code includes LTM retrieval and prompt integration.

**Task ID:** P3.T6
*   **Description:** Write Tests for Dynamic Loading & LTM
*   **Subtasks:**
    *   P3.T6.1: Write tests to verify that agents can dynamically load tools they don't initially possess (P3.T2).
    *   P3.T6.2: Write integration tests for storing and retrieving information from LTM (P3.T3), potentially mocking Pinecone API calls.
    *   P3.T6.3: Test the RAG pattern integration in agents (P3.T3.7).
*   **Priority:** Medium
*   **Est. Time:** 3-4 Days
*   **Difficulty:** Medium
*   **Dependencies:** P3.T2, P3.T3
*   **Components:** Testing Framework
*   **Alignment:** System validation.
*   **AC:** Tests confirm dynamic tool loading works as expected and LTM integration allows storing/retrieving information correctly.
*   **Status:** Completed (2025-04-06) # Created tests/agents/test_specialist_agent_dynamic_ltm.py
*   **AC:** Tests confirm dynamic tool loading works as expected and LTM integration allows storing/retrieving information correctly.

---

## Phase 4: Meta-Learning & Advanced Roles

*   **Goal:** Introduce self-improvement capabilities and specialized Architect/Developer agents.
*   **Est. Phase Duration:** 5-8 Weeks

---

**Task ID:** P4.T1
*   **Description:** Implement Collective Learning Engine (Tool Score Usage)
*   **Subtasks:**
    *   P4.T1.1: Create the Collective Learning Engine module within NovaCore.
    *   P4.T1.2: Implement logic for the engine to periodically read tool performance logs (from P3.T4) and update tool scores in the Tool Registry/Redis.
    *   P4.T1.3: Modify Specialist Agents' tool selection logic (potentially by adjusting LLM prompts or internal logic) to consider the tool scores when multiple tools could fulfill a requirement. Prefer higher-scoring tools.
*   **Priority:** High
*   **Est. Time:** 4-6 Days
*   **Difficulty:** High
*   **Dependencies:** P1.T3, P1.T5, P3.T4
*   **Components:** NovaCore (Collective Learning Engine), Agents (Specialist), Tool Management System (Scoring)
*   **Alignment:** Meta-Learning (Research.md L171), Collective Learning Engine (Research.md L72), Tool Scoring usage.
*   **AC:** Agents demonstrably favor tools with higher performance scores when selecting among alternatives; scores are updated based on usage logs.
*   **Status:** Completed (2025-04-06) # learning_engine.py created, SpecialistAgent modified to include scores in prompt.
*   **Components:** NovaCore (Collective Learning Engine), Agents (Specialist), Tool Management System (Scoring)
*   **Alignment:** Meta-Learning (Research.md L171), Collective Learning Engine (Research.md L72), Tool Scoring usage.
*   **AC:** Agents demonstrably favor tools with higher performance scores when selecting among alternatives; scores are updated based on usage logs.

**Task ID:** P4.T2
*   **Description:** Develop Architect Agent Prototype
*   **Subtasks:**
    *   P4.T2.1: Create a Python class for the Architect Agent.
    *   P4.T2.2: Design LLM prompts for the Architect role, focusing on taking high-level goals or complex/novel problems and outputting strategic plans, workflow designs, or suggestions for new tool combinations.
    *   P4.T2.3: Implement basic invocation logic for the Architect Agent (e.g., triggered by Planner for specific types of tasks).
    *   P4.T2.4: Define how the Architect's output integrates back into the system (e.g., providing a refined plan to the Planner).
*   **Priority:** Medium
*   **Est. Time:** 5-8 Days
*   **Difficulty:** High (Defining role and prompts is complex)
*   **Dependencies:** P1.T1 (LLM API access)
*   **Components:** Agents (Architect)
*   **Alignment:** Architect Agent Role (Research.md L585).
*   **AC:** Architect agent prototype can accept a complex goal and produce a high-level strategic plan or workflow suggestion using LLM reasoning.
*   **Status:** Completed (2025-04-06) # Created agents/architect_agent.py with basic class structure and prompt.
*   **Priority:** Medium
*   **Est. Time:** 5-8 Days
*   **Difficulty:** High (Defining role and prompts is complex)
*   **Dependencies:** P1.T1 (LLM API access)
*   **Components:** Agents (Architect)
*   **Alignment:** Architect Agent Role (Research.md L585).
*   **AC:** Architect agent prototype can accept a complex goal and produce a high-level strategic plan or workflow suggestion using LLM reasoning.

**Task ID:** P4.T3
*   **Description:** Develop Developer Agent Prototype
*   **Subtasks:**
    *   P4.T3.1: Create a Python class for the Developer Agent.
    *   P4.T3.2: Design LLM prompts focused on code generation: take a natural language specification for a simple tool and generate the Python function code.
    *   P4.T3.3: Implement logic to extract the generated Python code.
    *   P4.T3.4: Implement logic to generate a basic `toolkit.json` file based on the specification and generated function.
    *   P4.T3.5: (Optional Stretch Goal) Implement a very basic sandbox environment (e.g., using `exec` carefully or `restrictedpython`) to test the generated code with sample inputs.
    *   P4.T3.6: Define workflow for how a new tool requirement triggers the Developer Agent and how its output (code + JSON) gets added to the Tool Registry (manual review step likely needed initially).
*   **Priority:** Medium
*   **Est. Time:** 6-10 Days
*   **Difficulty:** High (Code generation, testing, and integration are complex)
*   **Dependencies:** P1.T1 (LLM API access), P1.T2, P1.T3
*   **Components:** Agents (Developer), Tool Management System
*   **Alignment:** Developer Agent Role (Research.md L593), Tool Evolution (Research.md L177).
*   **AC:** Developer agent prototype can take a simple tool specification (e.g., "a tool to reverse a string"), generate Python code and a `toolkit.json`, allowing it (potentially after review) to be added to the registry.
*   **Status:** Completed (2025-04-06) # Created agents/developer_agent.py with basic class structure and prompts for code/JSON generation. Sandbox testing is optional/basic.

**Task ID:** P4.T4
*   **Description:** Implement More Robust Coordination & Fault Tolerance
*   **Subtasks:**
    *   P4.T4.1: Implement an agent heartbeat mechanism: agents periodically update their status/timestamp in Redis.
    *   P4.T4.2: Implement monitoring logic (e.g., in the Dispatcher or a separate monitor process) to detect agents that miss heartbeats.
    *   P4.T4.3: Enhance the Swarm Dispatcher (P2.T1) to handle detected agent failures by reassigning their pending tasks to other available agents.
    *   P4.T4.4: (Optional) Implement basic task state checkpointing to Shared Memory so a new agent can potentially resume a failed task.
*   **Priority:** High
*   **Est. Time:** 4-7 Days
*   **Difficulty:** High
*   **Dependencies:** P2.T1, P2.T2, P2.T4
*   **Components:** NovaCore (Dispatcher, Shared Memory), Agents (Heartbeat)
*   **Alignment:** Fault Tolerance and Redundancy (Research.md L16), Failure Recovery (Research.md L219), Hive Sync (Research.md L70).
*   **AC:** The system can detect agent failures via heartbeats and reassign their tasks to ensure progress; basic task reallocation upon failure is functional.
*   **Status:** Completed (2025-04-06) # Heartbeat added to SpecialistAgent, Dispatcher updated with monitoring and basic reallocation logic.
*   **Priority:** High
*   **Est. Time:** 4-7 Days
*   **Difficulty:** High
*   **Dependencies:** P2.T1, P2.T2, P2.T4
*   **Components:** NovaCore (Dispatcher, Shared Memory), Agents (Heartbeat)
*   **Alignment:** Fault Tolerance and Redundancy (Research.md L16), Failure Recovery (Research.md L219), Hive Sync (Research.md L70).
*   **AC:** The system can detect agent failures via heartbeats and reassign their tasks to ensure progress; basic task reallocation upon failure is functional.

**Task ID:** P4.T5
*   **Description:** Write Tests for Meta-Learning & Advanced Roles
*   **Subtasks:**
    *   P4.T5.1: Write tests to verify that tool scores are updated correctly by the Collective Learning Engine (P4.T1).
    *   P4.T5.2: Write tests to verify that agents' tool selection is influenced by updated scores (P4.T1).
    *   P4.T5.3: Write basic tests for the Architect (P4.T2) and Developer (P4.T3) agent prototypes (testing input/output based on prompts).
    *   P4.T5.4: Write tests for the fault tolerance mechanism (P4.T4), simulating agent failures and verifying task reallocation.
*   **Priority:** Medium
*   **Est. Time:** 4-6 Days
*   **Difficulty:** High
*   **Dependencies:** P4.T1, P4.T2, P4.T3, P4.T4
*   **Components:** Testing Framework
*   **Alignment:** System validation for advanced features.
*   **AC:** Tests confirm meta-learning updates scores and influences behavior; advanced agent prototypes function at a basic level; fault tolerance handles simulated failures.
*   **Status:** Completed (2025-04-06) # Created tests for fault tolerance (P4.T5.4) and meta-learning scoring/usage (P4.T5.1, P4.T5.2). P4.T5.3 deferred as P4.T2/P4.T3 are not yet implemented.
*   **AC:** Tests confirm meta-learning updates scores and influences behavior; advanced agent prototypes function at a basic level; fault tolerance handles simulated failures.
*   **Status:** Completed (2025-04-06) # Created tests for fault tolerance (P4.T5.4) and meta-learning scoring/usage (P4.T5.1, P4.T5.2). P4.T5.3 deferred as P4.T2/P4.T3 are not yet implemented.

---

**Task ID:** MISC.T1
*   **Description:** Create `.env.example` file
*   **Subtasks:**
    *   MISC.T1.1: Identify necessary environment variables based on RESEARCH.md and PLANNING.md (LLM keys, Pinecone, Redis, potential tool keys).
    *   MISC.T1.2: Create the `.env.example` file with placeholders and comments.
*   **Priority:** High
*   **Est. Time:** 0.5 Days
*   **Difficulty:** Low
*   **Dependencies:** PLANNING.md (Tech Stack definition)
*   **Components:** Configuration
*   **Alignment:** Project Setup, Security Best Practices (API keys).
*   **AC:** `.env.example` file exists in the project root with necessary variables for core components (LLMs, Pinecone, Redis) and placeholders for tool keys.
*   **Status:** Completed (2025-04-06)

---

**Task ID:** MISC.T2
*   **Description:** Update `.env.example` based on project file scan
*   **Subtasks:**
    *   MISC.T2.1: Scan project files (`.py`, `toolkit.json`) for potential environment variable requirements (API keys, connection strings).
    *   MISC.T2.2: Update `.env.example` to include all identified variables with placeholders and comments.
*   **Priority:** Medium
*   **Est. Time:** 1 Day
*   **Difficulty:** Medium
*   **Dependencies:** Project files, MISC.T1
*   **Components:** Configuration
*   **Alignment:** Project Setup, Security Best Practices.
*   **AC:** `.env.example` file accurately reflects all environment variables potentially needed by the current codebase.
*   **Status:** Completed (2025-04-06)

---

## Phase 5: Scaling, Optimization & Production Readiness

*   **Goal:** Enhance performance, scalability, security, and monitoring for potential deployment.
*   **Est. Phase Duration:** 4-6 Weeks (Initial Setup) + Ongoing Optimization

---

**Task ID:** P5.T1
*   **Description:** Optimize Communication Overhead
*   **Subtasks:**
*       P5.T1.1: Analyze Redis usage patterns and identify potential bottlenecks.
*       P5.T1.2: Implement message batching where appropriate.
*       P5.T1.3: Refine communication protocols to reduce unnecessary broadcasts (e.g., use more targeted messages or smarter subscriptions).
*   **Priority:** Medium
*   **Est. Time:** 3-5 Days
*   **Difficulty:** Medium
*   **Dependencies:** Phase 2-4 components.
*   **Components:** NovaCore (Shared Memory Interface), Agents (Communication Logic)
*   **Alignment:** Communication Overhead and Efficiency (Research.md L223).
*   **AC:** Measurable reduction in communication volume or latency under load.
*   **Status:** Completed (2025-04-06) # Optimized tool usage logging with atomic increments (P5.T1.2); Added Pub/Sub notifications (P5.T1.3). Further analysis (P5.T1.1) deferred.
*   **Subtasks:**
    *   P5.T1.1: Analyze Redis usage patterns and identify potential bottlenecks. (Deferred)
    *   P5.T1.2: Implement message batching where appropriate. (Completed via atomic increments)
    *   P5.T1.3: Refine communication protocols to reduce unnecessary broadcasts (e.g., use more targeted messages or smarter subscriptions). (Partially completed via Pub/Sub)
*   **Priority:** Medium
*   **Est. Time:** 3-5 Days
*   **Difficulty:** Medium
*   **Dependencies:** Phase 2-4 components.
*   **Components:** NovaCore (Shared Memory Interface), Agents (Communication Logic)
*   **Alignment:** Communication Overhead and Efficiency (Research.md L223).
*   **AC:** Tool usage logging uses atomic Redis operations; Task status updates are published via Pub/Sub.

**Task ID:** P5.T2
*   **Description:** Implement Security Sandboxing for Tool Execution
*   **Subtasks:**
    *   P5.T2.1: Research and choose a sandboxing approach (e.g., `restrictedpython`, subprocesses with limited permissions, Docker containers per tool execution).
    *   P5.T2.2: Integrate the chosen sandboxing mechanism into the Specialist Agent's tool execution flow.
    *   P5.T2.3: Define and enforce resource limits (CPU, memory, network access) for sandboxed execution.
*   **Priority:** High
*   **Est. Time:** 5-8 Days
*   **Difficulty:** High
*   **Dependencies:** P1.T5
*   **Components:** Agents (Tool Execution Runtime)
*   **Alignment:** Plugin Security and Sandboxing (Research.md L169), Risk Mitigation (PLANNING.md L183).
*   **AC:** Tool code execution occurs within a restricted environment, mitigating security risks from potentially untrusted tool code.
*   **Status:** Completed (2025-04-06) # Implemented memory limit monitoring via psutil in utils/sandbox.py. CPU/Network limits not implemented.
*   **Difficulty:** High
*   **Dependencies:** P1.T5
*   **Components:** Agents (Tool Execution Runtime)
*   **Alignment:** Plugin Security and Sandboxing (Research.md L169), Risk Mitigation (PLANNING.md L183).
*   **AC:** Tool code execution occurs within a restricted environment; memory usage is monitored and limited.

**Task ID:** P5.T3
*   **Description:** Explore Scaling Agent Execution (Ray/Containers)
*   **Subtasks:**
    *   P5.T3.1: Evaluate Ray for distributed agent execution (replace/augment `asyncio`). Refactor agents as Ray actors if chosen.
    *   P5.T3.2: Alternatively, containerize agents (Docker) and explore orchestration with Docker Compose or Kubernetes.
    *   P5.T3.3: Adapt the Swarm Dispatcher and communication mechanisms for the chosen scaling solution.
*   **Priority:** Medium (Depends on expected load)
*   **Est. Time:** 6-10 Days
*   **Difficulty:** High
*   **Dependencies:** Phase 2-4 components.
*   **Components:** Deployment Architecture, NovaCore (Dispatcher), Agents
*   **Alignment:** Parallelism and Orchestration (Research.md L255), Scaling (PLANNING.md L27, L137).
*   **AC:** Proof-of-concept demonstrating agent execution scaled beyond a single process using Ray or container orchestration.

**Task ID:** P5.T4
*   **Description:** Refine Meta-Learning Algorithms
*   **Subtasks:**
    *   P5.T4.1: Explore more sophisticated reward functions for tool usage beyond simple success/failure.
    *   P5.T4.2: Potentially implement basic reinforcement learning updates for tool selection policies instead of just score weighting.
    *   P5.T4.3: Implement mechanisms for pruning or flagging underperforming/obsolete tools based on learning data.
*   **Priority:** Low (Research-oriented)
*   **Est. Time:** 5-10 Days
*   **Difficulty:** High
*   **Dependencies:** P4.T1
*   **Components:** NovaCore (Collective Learning Engine)
*   **Alignment:** Meta-Learning Loop (Research.md L179), Tool Evolution.
*   **AC:** Implementation of at least one refinement to the meta-learning process (e.g., improved reward signal, basic RL update).

**Task ID:** P5.T5
*   **Description:** Implement Comprehensive Monitoring and Logging (LLMOps)
*   **Subtasks:**
    *   P5.T5.1: Integrate structured logging throughout the system (agent decisions, tool calls, errors, timings). (Completed)
    *   P5.T5.2: Track LLM API calls (tokens used, cost, latency). (Completed via Callback)
    *   P5.T5.3: Setup dashboards (e.g., using Grafana, Kibana, or a dedicated LLMOps platform like LangSmith, Weights & Biases) to visualize system health, performance, and costs. (Deferred - Requires external setup)
    *   P5.T5.4: Implement tracing to follow a task through planning, dispatch, and execution steps. (Deferred - Requires external setup/integration)
*   **Priority:** High
*   **Est. Time:** 5-8 Days
*   **Difficulty:** Medium
*   **Dependencies:** Phase 1-4 components.
*   **Components:** Monitoring & Logging Infrastructure
*   **Alignment:** Governance and Monitoring (Research.md L326), LLMOps (PLANNING.md L139).
*   **AC:** Structured JSON logs are produced; LLM call metrics (tokens, latency) are logged via callback. Dashboard and tracing setup deferred.
*   **Status:** Partially Completed (2025-04-06) # P5.T5.1 & P5.T5.2 implemented. P5.T5.3 & P5.T5.4 deferred.

**Task ID:** P5.T6
*   **Description:** Conduct Stress Testing & Performance Benchmarking
*   **Subtasks:**
    *   P5.T6.1: Define key performance indicators (KPIs) (e.g., task completion rate, average task latency, resource utilization).
    *   P5.T6.2: Develop scripts or use tools to simulate high load (many concurrent tasks, complex requests).
    *   P5.T6.3: Run benchmarks, measure KPIs, identify bottlenecks.
*   **Priority:** Medium
*   **Est. Time:** 3-5 Days
*   **Difficulty:** Medium
*   **Dependencies:** P5.T3 (if scaling implemented), P5.T5 (for measurement)
*   **Components:** Testing Framework, Monitoring Infrastructure
*   **Alignment:** Performance Optimization.
*   **AC:** Performance under simulated load is measured and bottlenecks are identified.

**Task ID:** P5.T7
*   **Description:** Improve Documentation
*   **Subtasks:**
    *   P5.T7.1: Update `README.md` with setup, configuration, and running instructions. (Completed)
    *   P5.T7.2: Add architecture diagrams and workflow descriptions to `docs/`. (Completed - Diagram added to ARCHITECTURE.md)
    *   P5.T7.3: Ensure code has adequate docstrings (Google style). (Completed for core/agents)
    *   P5.T7.4: Create basic deployment and maintenance guides. (Deferred)
*   **Priority:** High
*   **Est. Time:** 3-5 Days
*   **Difficulty:** Medium
*   **Dependencies:** All previous phases.
*   **Components:** Documentation
*   **Alignment:** Maintainability, Python Development Standards (PLANNING.md L175).
*   **AC:** README updated; Architecture diagram added; Core/Agent code has Google-style docstrings. Deployment guides deferred.
*   **Status:** Partially Completed (2025-04-06) # P5.T7.1, P5.T7.2, P5.T7.3 completed. P5.T7.4 deferred.

---

## Tool Integrations

---

**Task ID:** TOOL.T1
*   **Description:** Integrate Brave Search MCP Toolkit
*   **Subtasks:**
*       TOOL.T1.1: Create `tools/brave_search/toolkit.json` defining Brave Search tools.
*       TOOL.T1.2: Implement `tools/brave_search/brave_search_toolkit.py` with functions wrapping MCP calls.
*       TOOL.T1.3: Add `smithery` and `mcp` to `requirements.txt`.
*       TOOL.T1.4: Add `BRAVE_API_KEY` and `SMITHERY_API_KEY` to `.env`.
*       TOOL.T1.5: Verify toolkit is loaded automatically by `load_toolkits_from_directory`.
*       TOOL.T1.6: Create unit tests in `tests/tools/brave_search/test_brave_search_toolkit.py`.
*       TOOL.T1.7: Update `README.md` with new dependencies and environment variables.
*   **Priority:** High
*   **Est. Time:** 1-2 Days
*   **Difficulty:** Medium
*   **Dependencies:** Phase 1 components (Registry, Schema), Phase 3 (Dynamic Loading understanding).
*   **Components:** Tool Management System, Brave Search Toolkit, Configuration, Testing.
*   **Alignment:** Flexible Toolsets, Extensibility.
*   **AC:** Brave Search tools (`brave_web_search`, `brave_local_search`) are available to agents, dependencies are documented, tests pass.
*   **Status:** Completed (2025-04-07)

---

**Task ID:** FEAT.T1
*   **Description:** Create CLI for Natural Language Task Input
*   **Subtasks:**
    *   FEAT.T1.1: Create `cli.py` script with argument parsing for task description.
    *   FEAT.T1.2: Integrate NovaCore initialization logic (Dispatcher, Memory, LTM, Registry).
    *   FEAT.T1.3: Initialize Planner and Specialist agents.
    *   FEAT.T1.4: Implement workflow: Get input -> Plan -> Dispatch -> Monitor -> Display Result.
    *   FEAT.T1.5: Add basic progress monitoring by checking Redis status.
    *   FEAT.T1.6: Update README.md with CLI usage instructions.
*   **Priority:** High
*   **Est. Time:** 1-2 Days
*   **Difficulty:** Medium
*   **Dependencies:** Phase 1-4 components (especially Planner Agent P2.T3).
*   **Components:** CLI Interface, Planner Agent, Swarm Dispatcher, Shared Memory.
*   **Alignment:** User Interaction, System Entry Point.
*   **AC:** User can run `python cli.py "Your task description"` and receive the final result after agent execution.
*   **Status:** Completed (2025-04-07)

---

**Task ID:** TOOL.T2
*   **Description:** Integrate Perplexity Search MCP Toolkit
*   **Subtasks:**
*       TOOL.T2.1: Create `tools/perplexity_search/toolkit.json` defining Perplexity search tool(s).
*       TOOL.T2.2: Implement `tools/perplexity_search/perplexity_search_toolkit.py` with functions wrapping MCP calls using Smithery SDK.
*       TOOL.T2.3: Add `smithery-mcp` (or relevant package) to `requirements.txt`.
*       TOOL.T2.4: Add `PERPLEXITY_API_KEY` to `.env.example`. (Confirm if `SMITHERY_API_KEY` is needed globally or per-tool).
*       TOOL.T2.5: Verify toolkit is loaded automatically by `load_toolkits_from_directory`.
*       TOOL.T2.6: Create unit tests in `tests/tools/perplexity_search/test_perplexity_search_toolkit.py`.
*       TOOL.T2.7: Update `README.md` with new dependencies and environment variables.
*   **Priority:** High
*   **Est. Time:** 1-2 Days
*   **Difficulty:** Medium
*   **Dependencies:** Phase 1 components, Phase 3 (Dynamic Loading), Smithery SDK understanding.
*   **Components:** Tool Management System, Perplexity Search Toolkit, Configuration, Testing.
*   **Alignment:** Flexible Toolsets, Extensibility, MCP Integration.
*   **AC:** Perplexity Search tool is available to agents, dependencies are documented, tests pass.
*   **Status:** Completed (2025-04-08) - Re-implemented to use direct Perplexity API via openai library instead of MCP.

---

**Task ID:** TOOL.T3
*   **Description:** Integrate BrowserUseAgent Toolkit
*   **Subtasks:**
    *   TOOL.T3.1: Create `tools/browser_use_agent/browser_use_agent_toolkit.py` with refactored logic.
    *   TOOL.T3.2: Create `tools/browser_use_agent/toolkit.json` defining the `run_browser_use_gemini_task` tool.
    *   TOOL.T3.3: Add `browser-use`, `playwright` to `requirements.txt`.
    *   TOOL.T3.4: Ensure `GOOGLE_API_KEY` is documented in `.env.example`.
    *   TOOL.T3.5: Verify toolkit is loaded automatically by `load_toolkits_from_directory`.
    *   TOOL.T3.6: Create unit tests in `tests/tools/browser_use_agent/test_browser_use_agent_toolkit.py`.
    *   TOOL.T3.7: Update `README.md` with new dependencies, environment variables, and `playwright install` instruction.
*   **Priority:** High
*   **Est. Time:** 1-2 Days
*   **Difficulty:** Medium
*   **Dependencies:** Phase 1 components, Phase 3 (Dynamic Loading).
*   **Components:** Tool Management System, BrowserUseAgent Toolkit, Configuration, Testing, Documentation.
*   **Alignment:** Flexible Toolsets, Extensibility.
*   **AC:** BrowserUseAgent tool is available to agents, dependencies are documented, tests pass, README updated.
*   **Status:** Completed (2025-04-08)

---