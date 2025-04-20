# Nova SHIFT - System Architecture

## 1. Introduction & Vision

### 1.1. Overview
Nova SHIFT (Swarm-Hive Intelligence with Flexible Toolsets) is the core intelligence architecture powering the Nova system. It represents a paradigm shift from monolithic AI models to a decentralized, collaborative ecosystem of intelligent agents. Nova aims to function as a unified, self-improving AI capable of tackling complex, multi-domain problems by leveraging a "hive mind" of adaptable agents that dynamically acquire and utilize specialized tools.

### 1.2. Architectural Vision
The vision is to create a robust, scalable, and adaptable AI system based on the principles of swarm intelligence and modularity. Key architectural goals include:
*   **Adaptability:** Agents dynamically load tools ("shapeshift") based on task requirements.
*   **Collaboration:** Agents coordinate effectively through shared memory and communication protocols.
*   **Scalability:** The architecture should support a growing number of agents and tools, handling increasing complexity.
*   **Resilience:** The system should tolerate individual agent or tool failures gracefully.
*   **Extensibility:** New tools, agents, and capabilities can be added easily via a plugin-based system.
*   **Continuous Improvement:** The system incorporates meta-learning to refine its strategies and tool usage over time.

### 1.3. Scope
This document details the high-level and component-level architecture of the Nova SHIFT system, including its core engine (NovaCore), agent types, memory systems, tool management, communication protocols, and the underlying technology stack as planned for the initial phases of development.

## 2. System Analysis & Requirements Synthesis

### 2.1. Core Concepts (from Research & Planning)
*   **Swarm-Hive Duality:** Agents operate in parallel (Swarm) but synchronize via shared state (Hive).
*   **Shapeshifting Agents:** Agents load toolkits dynamically based on task needs. Roles include Planner, Architect, Developer, and Specialists.
*   **NovaCore Engine:** Central infrastructure managing agents, memory, tools, and learning.
*   **Multi-Layered Memory:** Short-Term (STM - local context), Long-Term (LTM - persistent knowledge via Vector DB like Pinecone), Shared (coordination blackboard via Redis).
*   **Flexible Toolsets:** Modular tools/plugins defined by a schema (`toolkit.json`), managed via a Tool Registry, loaded dynamically, and scored based on performance.
*   **Meta-Learning:** Continuous improvement based on task outcomes and performance feedback.
*   **Communication:** Direct messaging, broadcasts, and indirect coordination (stigmergy) via Shared Memory.

### 2.2. Key Functional Requirements
*   Accept complex user goals/tasks.
*   Decompose tasks into manageable subtasks.
*   Identify and dynamically load necessary tools for subtasks.
*   Execute subtasks using appropriate tools via agents.
*   Coordinate execution among multiple agents.
*   Store and retrieve persistent knowledge relevant to tasks.
*   Synthesize subtask results into a coherent final output.
*   Learn from task execution to improve future performance.
*   (Future) Allow for the creation of new tools via a Developer agent.
*   (Future) Allow for high-level workflow design via an Architect agent.

### 2.3. Key Non-Functional Requirements
*   **Performance:** Efficient task execution, acceptable latency for user interaction.
*   **Scalability:** Ability to handle increasing numbers of agents, tools, and concurrent tasks.
*   **Reliability:** Fault tolerance for agent/tool failures, consistent operation.
*   **Security:** Secure execution of tools, protection of sensitive data (e.g., API keys).
*   **Maintainability:** Modular design, clear interfaces, adherence to coding standards.
*   **Extensibility:** Ease of adding new tools and agent capabilities.
*   **Observability:** Adequate logging and monitoring for debugging and performance analysis.

### 2.4. Constraints & Assumptions
*   Reliance on external LLM APIs (OpenAI, Google) for core reasoning.
*   Dependency on external services (Pinecone, potentially others via tools).
*   Initial development focuses on core functionality as per the phased plan.
*   Security of dynamically loaded code requires careful sandboxing.
*   Network latency can impact performance, especially for external API calls.

## 3. High-Level Architecture Design

### 3.1. Architectural Style
Nova SHIFT employs a **Distributed, Agent-Based Architecture** combined with elements of **Microservices** (for potentially scalable components like the Tool Registry or specialized tools) and a **Blackboard System** (for Shared Memory coordination).

### 3.2. Core Components
The system is primarily composed of two major parts:
1.  **NovaCore Engine:** The central infrastructure providing shared services and orchestration.
2.  **Agents:** Independent (but coordinated) processes or tasks performing reasoning and execution.

### 3.3. System Context Diagram (Conceptual C4 Level 1)

```mermaid
graph TD
    User[User] -- Interacts via Interface --> Planner(Planner Agent);
    Planner -- Decomposes Task --> NovaCore(NovaCore Engine);
    NovaCore -- Dispatches Subtasks --> Agents(Agent Swarm);
    Agents -- Execute Subtasks --> Tools(External Tools/APIs);
    Agents -- Coordinate & Share Knowledge --> NovaCore;
    NovaCore -- Stores/Retrieves Knowledge --> LTM[Long-Term Memory (Pinecone)];
    NovaCore -- Manages Tools --> ToolRegistry[Tool Registry];
    NovaCore -- Facilitates Coordination --> SharedMemory[Shared Memory (Redis)];

    subgraph Agent Swarm
        direction LR
        Specialist1[Specialist Agent];
        Specialist2[Specialist Agent];
        Architect[Architect Agent (Phase 4+)];
        Developer[Developer Agent (Phase 4+)];
    end

    subgraph NovaCore Engine
        direction TB
        Dispatcher[Swarm Dispatcher];
        LTM_Interface[LTM Interface];
        SharedMemory_Interface[Shared Memory Interface];
        ToolRegistry_Interface[Tool Registry Interface];
        LearningEngine[Collective Learning Engine];
    end

    style NovaCore fill:#f9f,stroke:#333,stroke-width:2px
    style Agents fill:#ccf,stroke:#333,stroke-width:2px
```

### 3.4. Deployment Strategy (Initial Phases)
*   **Development:** Run as Python processes/async tasks on a single machine. Redis and Pinecone accessed via network/API.
*   **Testing/Staging:** Containerize components (NovaCore services, Agent base image) using Docker and orchestrate with Docker Compose.
*   **Production (Future):** Potentially scale using Kubernetes or a serverless platform (e.g., Ray Serve) for agents and core services. Redis and Pinecone remain managed services or deployed as scalable clusters.

## 4. Component Design (Conceptual C4 Level 2/3)

### 4.0. Component Interaction Diagram

This diagram shows the primary interactions between a Specialist Agent and the NovaCore Engine components during task execution.

```mermaid
graph TD
    subgraph Specialist Agent (agent_id)
        direction TB
        SA_Logic[Execution Logic (execute_task)];
        SA_Toolkit[Dynamic Toolkit (self.tools)];
        SA_Memory[STM (self.memory)];
        SA_LLM[LLM Client (self.llm)];
        SA_Callback[LLM Callback (self.llm_callback_handler)];
    end

    subgraph NovaCore Engine
        direction TB
        Dispatcher[Swarm Dispatcher];
        SharedMem[Shared Memory Interface];
        LTM[LTM Interface];
        Registry[Tool Registry Interface];
        Learning[Collective Learning Engine];
    end

    subgraph External Services
        direction TB
        LLM_API[LLM API (OpenAI/Google)];
        Pinecone_API[Pinecone API];
        Redis_Service[Redis Server];
        Tool_APIs[External Tool APIs (e.g., Search)];
    end

    %% Agent Interactions with NovaCore
    SA_Logic -- Receives Task --> Dispatcher;
    SA_Logic -- Updates Status/Results --> SharedMem;
    SA_Logic -- Reads Dependencies --> SharedMem;
    SA_Logic -- Retrieves Context --> LTM;
    SA_Logic -- Stores Results --> LTM;
    SA_Logic -- Finds/Loads Tools --> Registry;
    SA_Logic -- Logs Usage --> Learning;
    SA_Logic -- Notifies Completion --> Dispatcher;

    %% Agent Internal Interactions
    SA_Logic -- Uses --> SA_Toolkit;
    SA_Logic -- Uses --> SA_Memory;
    SA_Logic -- Calls --> SA_LLM;
    SA_LLM -- Triggers --> SA_Callback;

    %% NovaCore Interactions with External Services
    SharedMem -- Reads/Writes --> Redis_Service;
    LTM -- Reads/Writes --> Pinecone_API;
    Learning -- Reads Logs --> Redis_Service; % Assuming logs stored in Redis initially
    Learning -- Updates Scores --> Registry; % Updates internal state or Redis

    %% Agent/Tool Interactions with External Services
    SA_LLM -- Calls --> LLM_API;
    SA_Toolkit -- Executes Tool --> Tool_APIs;

    style Specialist Agent fill:#ccf,stroke:#333,stroke-width:2px
    style NovaCore Engine fill:#f9f,stroke:#333,stroke-width:2px
    style External Services fill:#9cf,stroke:#333,stroke-width:2px

```

### 4.1. NovaCore Engine

#### 4.1.1. Swarm Dispatcher
*   **Responsibility:** Receives task graphs (from Planner), assigns subtasks to available Specialist Agents, tracks task status, handles basic failure detection and reallocation (Phase 4+).
*   **Interface (Conceptual Python):**
    ```python
    class SwarmDispatcher:
        def dispatch_subtasks(self, task_graph: Dict) -> Dict[str, str]: # Returns {subtask_id: agent_id}
            pass
        def update_task_status(self, subtask_id: str, status: str, result: Any = None):
            pass
        def get_agent_status(self) -> Dict[str, str]: # Returns {agent_id: status}
            pass
        def reallocate_task(self, subtask_id: str) -> Optional[str]: # Returns new agent_id or None
             pass
    ```
*   **Technology:** Python (AsyncIO), potentially integrated with Redis for agent availability tracking.

#### 4.1.2. Shared Memory Interface
*   **Responsibility:** Provides a unified interface for agents to read/write coordination data (task results, status updates, intermediate data) to the shared blackboard. Abstracts the underlying Redis implementation.
*   **Interface (Conceptual Python):**
    ```python
    class SharedMemoryInterface:
        def write(self, key: str, value: Any, expiry_seconds: Optional[int] = None):
            pass
        def read(self, key: str) -> Any:
            pass
        def delete(self, key: str):
            pass
        def publish(self, channel: str, message: Any): # For broadcasts
            pass
        # subscribe method would likely be handled within agent logic
    ```
*   **Technology:** Python wrapper around `redis-py` library (using Redis Pub/Sub and Key-Value).

#### 4.1.3. Long-Term Memory (LTM) Interface
*   **Responsibility:** Provides an interface for agents to store and retrieve information semantically from the persistent knowledge base (Pinecone). Handles embedding generation.
*   **Interface (Conceptual Python):**
    ```python
    class LTMInterface:
        def store(self, text: str, metadata: Dict):
            # Handles embedding and upserting to Pinecone
            pass
        def retrieve(self, query_text: str, top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict]:
            # Handles embedding query and searching Pinecone
            pass
    ```
*   **Technology:** Python wrapper around `pinecone-client` and an embedding model API (e.g., OpenAI Embeddings).

#### 4.1.4. Tool Registry Interface
*   **Responsibility:** Manages the catalog of available toolkits. Allows agents to discover tools and retrieve information needed for dynamic loading (metadata, location/path). Manages tool scores (updated by Learning Engine).
*   **Interface (Conceptual Python):**
    ```python
    class ToolRegistryInterface:
        def register_toolkit(self, toolkit_config: Dict):
            pass
        def find_toolkits(self, capability_query: str) -> List[Dict]: # Returns list of toolkit configs
            pass
        def get_toolkit_config(self, toolkit_name: str) -> Optional[Dict]:
            pass
        def get_tool_score(self, tool_name: str) -> float:
            pass
        def update_tool_score(self, tool_name: str, new_score: float):
            pass
    ```
*   **Technology:** Python. Initial implementation uses an in-memory dictionary, potentially backed by Redis or a simple database later.

#### 4.1.5. Collective Learning Engine
*   **Responsibility:** Collects feedback (task success, tool performance logs) from agents, analyzes it, updates tool scores in the Registry, and potentially refines system-wide strategies (future).
*   **Interface (Conceptual Python):**
    ```python
    class CollectiveLearningEngine:
        def log_task_outcome(self, task_id: str, success: bool, used_tools: List[str], metrics: Dict):
            pass
        def process_logs_and_update_scores(self):
            # Reads logs, calculates new scores, calls registry.update_tool_score()
            pass
    ```
*   **Technology:** Python. Initial implementation involves simple logging (to Redis or file) and periodic score updates based on success rates.

### 4.2. Agents

#### 4.2.1. Base Agent Structure
*   **Core:** Python class/process/async task.
*   **Components:**
    *   LLM Client (for reasoning).
    *   Short-Term Memory (STM) instance (e.g., LangChain buffer).
    *   Interfaces to NovaCore services (Shared Memory, LTM, Tool Registry).
    *   Dynamic Toolkit: A dictionary holding currently loaded tool functions/objects.
    *   Main execution loop/handler.
*   **Technology:** Python, LangChain/LangGraph, AsyncIO.

#### 4.2.2. Planner Agent
*   **Responsibility:** Receives user goal, interacts with LTM for context, uses LLM to decompose goal into a structured task graph (nodes=subtasks, edges=dependencies), sends graph to Dispatcher.
*   **Key Interactions:** User Input, LTM Interface, Swarm Dispatcher.

#### 4.2.3. Specialist Agent (Shapeshifter)
*   **Responsibility:** Receives assigned subtask from Dispatcher. Analyzes subtask requirements. Queries Tool Registry for needed tools. Dynamically loads/imports tools into its toolkit if not present. Executes subtask using tools. Communicates results/status via Shared Memory. Provides feedback to Learning Engine.
*   **Key Interactions:** Swarm Dispatcher, Tool Registry Interface, Shared Memory Interface, LTM Interface (for context/RAG), Collective Learning Engine.

#### 4.2.4. Architect Agent (Phase 4+)
*   **Responsibility:** Handles complex, novel, or ambiguous goals requiring high-level design. May design new multi-step workflows, suggest combinations of existing tools, or identify the need for entirely new capabilities (triggering the Developer Agent).
*   **Key Interactions:** Planner Agent (escalation), LTM Interface, Tool Registry Interface, potentially Developer Agent.

#### 4.2.5. Developer Agent (Phase 4+)
*   **Responsibility:** Receives specifications for a new tool. Uses an LLM to generate the tool's Python code and `toolkit.json` definition. Performs basic testing (e.g., syntax check, sandbox execution). Registers the new toolkit with the Tool Registry.
*   **Key Interactions:** Architect Agent (requests), Tool Registry Interface, Code Execution Sandbox.

## 5. Data Architecture

### 5.1. Memory Layers
*   **Short-Term Memory (STM):**
    *   **Purpose:** Local agent context, conversation history, intermediate reasoning steps.
    *   **Implementation:** In-memory data structures within each agent process (e.g., Python lists, dictionaries, LangChain Memory modules). Transient.
*   **Long-Term Memory (LTM):**
    *   **Purpose:** Persistent, semantic knowledge base. Stores facts, past task summaries, learned insights. Enables RAG.
    *   **Implementation:** Pinecone Vector Database. Data stored as text embeddings with associated metadata. Accessed via LTM Interface.
*   **Shared Memory:**
    *   **Purpose:** Real-time coordination blackboard. Stores task status, intermediate results for dependent tasks, broadcast messages, agent availability.
    *   **Implementation:** Redis (Key-Value store for state, Pub/Sub for broadcasts). Accessed via Shared Memory Interface.

### 5.2. Key Data Models / Schemas

#### 5.2.1. `toolkit.json` Schema (v1)
```json
{
  "name": "string (Unique toolkit identifier)",
  "version": "string (Semantic versioning, e.g., 1.0.0)",
  "description": "string (Human-readable description of the toolkit's purpose)",
  "tools": [
    {
      "name": "string (Unique tool name within the toolkit)",
      "function": "string (Name of the Python function/method to call)",
      "description": "string (Detailed description for LLM reasoning/selection)",
      "inputs": [ "string (Parameter name:type, e.g., 'query:string')" ],
      "outputs": [ "string (Output name:type, e.g., 'results:list<string>')" ],
      "dependencies": [ "string (Names of other tools this depends on, if any)" ] // Optional
    }
    // More tools...
  ],
  "requirements": { // Optional: Dependencies needed to run tools
    "python_packages": [ "string (e.g., 'requests>=2.0')" ],
    "api_keys": [ "string (Name of environment variable holding required API key)" ]
  },
  "loading_info": { // How to load the code
      "type": "python_module", // Or "api_endpoint", "docker_image" etc.
      "path": "string (e.g., 'nova.tools.websearch.WebSearchToolkit')"
  }
}
```

#### 5.2.2. Task Graph Structure (Conceptual)
A dictionary or graph structure representing tasks and dependencies.
```json
{
  "graph_id": "unique_task_graph_id",
  "root_task_id": "user_goal_id",
  "nodes": {
    "subtask_1": {
      "description": "Subtask description for agent",
      "status": "pending | assigned | running | completed | failed",
      "assigned_agent": "agent_id | null",
      "result": "null | <result_data>",
      "required_tools": ["ToolName1", "ToolName2"], // Identified by Planner/Agent
      "depends_on": ["root_task_id"]
    },
    "subtask_2": {
      // ...
      "depends_on": ["subtask_1"] // Example dependency
    }
    // ... more nodes
  }
}
```
This structure might be stored partially in Shared Memory during execution.

### 5.3. Data Flow (Typical Task)

1.  **User Goal** -> Planner Agent
2.  Planner -> **Task Graph** -> Swarm Dispatcher
3.  Dispatcher -> **Subtask Assignment** -> Specialist Agent(s)
4.  Agent -> **Tool Requirement Query** -> Tool Registry
5.  Registry -> **Toolkit Config/Loading Info** -> Agent
6.  Agent -> **(Dynamic Tool Loading)**
7.  Agent -> **LTM Query (Context)** -> LTM Interface -> Pinecone
8.  Pinecone -> **Retrieved Knowledge** -> Agent
9.  Agent -> **Tool Execution** -> (Internal Function / External API)
10. Agent -> **Subtask Result/Status** -> Shared Memory Interface -> Redis
11. Agent -> **(Reads dependent results from Redis)**
12. Agent -> **Task Feedback** -> Collective Learning Engine
13. Learning Engine -> **Score Update** -> Tool Registry Interface
14. (Final Agent/Planner) -> **Synthesized Result** -> User

## 6. Integration & Communication Architecture

### 6.1. Internal Communication
*   **Agent <-> NovaCore:** Agents interact with NovaCore interfaces (Python method calls) to access shared services (Memory, Registry, Dispatcher, Learning).
*   **Agent <-> Agent (Coordination):** Primarily indirect via Shared Memory (Redis). Agents write status/results; other agents read/subscribe. Direct messaging via Redis Pub/Sub for specific needs (e.g., help requests, broadcasts).
*   **Dispatcher -> Agent:** Task assignments, potentially via Redis Pub/Sub channel specific to each agent or direct function calls if co-located.

### 6.2. External Communication
*   **Agents -> External Tools/APIs:** Agents execute tools which may involve making HTTP requests to external APIs (e.g., Web Search API, LLM API, Pinecone API). Handled by the tool's implementation code. API keys managed securely (e.g., environment variables, vault service).
*   **User -> System:** Initial interaction likely via a CLI or simple Web UI that communicates with the Planner Agent (details TBD, outside core architecture for now).

### 6.3. Communication Protocols
*   **Shared Memory:** Key-value writes/reads for state, Pub/Sub for events/broadcasts (Redis protocol).
*   **LTM:** HTTPS requests to Pinecone API.
*   **LLM:** HTTPS requests to OpenAI/Google APIs.
*   **Internal (if distributed):** Potentially gRPC or REST APIs between NovaCore services and Agents if deployed as separate microservices/containers.

## 7. Python-Specific Architectural Considerations

### 7.1. Module Organization (Suggested Structure)
```
nova_shift/
├── agents/                 # Agent implementations
│   ├── __init__.py
│   ├── base_agent.py
│   ├── planner_agent.py
│   ├── specialist_agent.py
│   ├── architect_agent.py  # Phase 4+
│   └── developer_agent.py  # Phase 4+
├── core/                   # NovaCore engine components
│   ├── __init__.py
│   ├── dispatcher.py
│   ├── ltm_interface.py
│   ├── shared_memory_interface.py
│   ├── tool_registry.py
│   └── learning_engine.py
├── tools/                  # Toolkit implementations
│   ├── __init__.py
│   ├── toolkit_schema.py   # Pydantic model for toolkit.json
│   ├── web_search/
│   │   ├── __init__.py
│   │   ├── toolkit.json
│   │   └── web_search_toolkit.py
│   └── calculator/
│       ├── __init__.py
│       ├── toolkit.json
│       └── calculator_toolkit.py
├── config/                 # Configuration files (API keys, settings)
│   └── settings.py         # Using Pydantic BaseSettings or similar
├── utils/                  # Shared utility functions
│   └── sandbox.py          # Tool execution sandboxing
├── tests/                  # Unit and integration tests
│   ├── agents/
│   ├── core/
│   └── tools/
├── main.py                 # Entry point for running the system
├── requirements.txt
├── pyproject.toml          # For build system, black, flake8, mypy config
└── README.md
└── ARCHITECTURE.md         # This file
└── PLANNING.md
└── RESEARCH.md
```

### 7.2. Dependency Management
*   Use `venv` for environment isolation.
*   Pin dependencies in `requirements.txt` using `pip freeze > requirements.txt` or tools like `pip-tools`.
*   Specify Python version (3.10+).

### 7.3. Concurrency & Parallelism
*   **Initial:** Use Python's `asyncio` for concurrent I/O operations (API calls) and managing multiple agents within a single process or few processes.
*   **Scaling:** Explore `Ray` for distributing agents across multiple cores/machines as actors. Ray's object store could potentially supplement Redis for certain shared state.

### 7.4. Error Handling & Resilience
*   Implement robust error handling within agents and tools (try-except blocks).
*   Agents should report failures to the Dispatcher/Shared Memory.
*   Dispatcher implements retry logic or task reallocation.
*   Use heartbeat mechanisms (Phase 4+) to detect unresponsive agents.
*   Tools should handle external API errors gracefully (timeouts, rate limits).

### 7.5. Configuration Management
*   Use environment variables or configuration files (`.env`, YAML/JSON) for settings like API keys, database URLs, model names.
*   Libraries like `python-dotenv` and `Pydantic`'s `BaseSettings` can manage configuration loading.

### 7.6. Testing Strategy
*   **Unit Tests (`pytest`):** Test individual tools, NovaCore component logic, agent helper functions. Mock external dependencies (APIs, databases).
*   **Integration Tests (`pytest`):** Test interactions between components (e.g., Planner -> Dispatcher -> Agent -> Shared Memory). May require running Redis/mocked Pinecone.
*   **End-to-End Tests:** Simulate user requests and verify the final output and intermediate coordination steps.

### 7.7. Security Considerations
*   **Tool Execution Sandboxing:** CRITICAL. Dynamically loaded code must run in a restricted environment. Options:
    *   `restrictedpython` library.
    *   Running tools in separate subprocesses with limited permissions.
    *   Running tools in Docker containers with resource limits and network policies.
*   **API Key Management:** Store keys securely (environment variables, secrets manager), never hardcode. Provide keys to agents/tools on a need-to-know basis.
*   **Input Sanitization:** Sanitize user inputs to prevent prompt injection attacks against LLMs.
*   **Dependency Security:** Regularly scan dependencies for vulnerabilities (`pip-audit`, `safety`).

## 8. Architecture Validation & Risks

### 8.1. Validation Approach
*   **Prototyping:** Implement core components iteratively as per the phased plan.
*   **Testing:** Rigorous unit, integration, and E2E testing.
*   **Benchmarking (Phase 5):** Measure performance and scalability under load.
*   **Code Reviews:** Ensure adherence to design and standards.

### 8.2. Key Risks & Mitigations (Summary from Planning)
*   **Complexity:** Mitigate with iterative development, frameworks, testing.
*   **LLM Reliability/Cost:** Mitigate with verification steps, cost monitoring, caching.
*   **Tool Security:** Mitigate with strict sandboxing, vetting.
*   **Scalability Bottlenecks:** Mitigate with scalable tech (Redis Cluster, Ray), communication optimization.
*   **Meta-Learning Instability:** Mitigate with simple initial methods, monitoring, safeguards.

## 9. Architectural Decision Records (ADRs)
*ADRs should be maintained separately to document significant architectural decisions and their rationale.*
*   **ADR-001:** Choice of Technology Stack (Python, LangChain, Pinecone, Redis). *Rationale: Leverage mature Python AI ecosystem, existing frameworks, and managed services for key components.*
*   **ADR-002:** Shared Memory Implementation (Redis). *Rationale: Provides both Key-Value store and Pub/Sub needed for coordination, mature and performant.*
*   **ADR-003:** LTM Implementation (Pinecone). *Rationale: Managed service simplifies vector DB operations, scalable.*
*   *(Future ADRs will document decisions on sandboxing, scaling frameworks, specific learning algorithms, etc.)*

## 10. Limitations & Future Considerations
*   Initial phases focus on core functionality; advanced roles (Architect, Developer) and sophisticated meta-learning are future work.
*   Performance and scalability require dedicated optimization in later phases.
*   Real-world robustness depends heavily on the quality of tools and the effectiveness of error handling and meta-learning.
*   Explainability needs further development beyond basic logging.
*   User Interface is not defined in this core architecture.

## 11. Glossary
*   **SHIFT:** Swarm-Hive Intelligence with Flexible Toolsets (The architecture).
*   **Nova:** The AI system powered by SHIFT.
*   **NovaCore:** The central engine/infrastructure.
*   **Agent:** An autonomous unit performing reasoning and actions.
*   **Toolkit:** A collection of related tools packaged together.
*   **Tool:** A specific capability (function, API call) an agent can use.
*   **STM:** Short-Term Memory.
*   **LTM:** Long-Term Memory.
*   **Shared Memory:** Coordination blackboard.
*   **Tool Registry:** Catalog of available toolkits.
*   **Dispatcher:** Component assigning tasks to agents.
*   **Stigmergy:** Indirect coordination via environmental modification (Shared Memory).
*   **RAG:** Retrieval-Augmented Generation.