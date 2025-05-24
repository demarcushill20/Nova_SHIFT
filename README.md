# Nova SHIFT

This repository contains the implementation of the Nova SHIFT (Swarm-Hive Intelligence with Flexible Toolsets) system.

Nova SHIFT is a decentralized AI architecture designed to power the Nova system. It utilizes a hive-mind of adaptable, "shapeshifting" agents that collaborate and dynamically load specialized toolsets to solve complex problems.

See `RESEARCH.md`, `PLANNING.md`, and `ARCHITECTURE.md` for more details on the design and concepts.

## Prerequisites

*   **Python:** 3.10+
*   **Redis:** A running Redis instance (localhost:6379 by default). You can install Redis locally or use a cloud service.
*   **Environment Variables:** Certain API keys are required for specific tools and models. Create a `.env` file in the project root or set the following environment variables:
    *   `OPENAI_API_KEY`: Your OpenAI API key (for LLM reasoning).
    *   `TAVILY_API_KEY`: Your Tavily Search API key (for the web search tool).
    *   `PINECONE_API_KEY`: Your Pinecone API key (for Long-Term Memory).
    *   `PINECONE_ENVIRONMENT`: Your Pinecone environment name (e.g., `gcp-starter`). *(Note: Pinecone setup might require additional steps like creating an index)*
    *   `BRAVE_API_KEY`: Your Brave Search API key (for the Brave Search toolkit).
    *   `SMITHERY_API_KEY`: Your Smithery API key (for connecting to MCP servers like Brave Search).
    *   `PERPLEXITY_API_KEY`: Your Perplexity API key (for the Perplexity Search toolkit).
    *   `GOOGLE_API_KEY`: Your Google API key (Required for Gemini models, used by BrowserUseAgentToolkit).
## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd NOVA_SHIFT
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    *   Windows: `.\venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` includes dependencies like `langchain`, `openai`, `redis`, `pinecone-client`, `tavily-python`, `smithery`, `mcp`, `browser-use`, `playwright`, `pytest`, etc. Ensure `psutil` is included if using memory limits in the sandbox.)*

    **Playwright Browser Installation:** The `BrowserUseAgentToolkit` requires Playwright browsers to be installed. After installing requirements, run:
    ```bash
    playwright install chromium
    ```

## Configuration

Ensure the required environment variables listed in **Prerequisites** are set. You can place them in a `.env` file in the project root directory. Example `.env` file:

```dotenv
OPENAI_API_KEY="sk-..."
TAVILY_API_KEY="tvly-..."
PINECONE_API_KEY="..."
PINECONE_ENVIRONMENT="..."
BRAVE_API_KEY="YOUR_BRAVE_SEARCH_API_KEY" # Required for BraveSearchToolkit
SMITHERY_API_KEY="YOUR_SMITHERY_API_KEY" # Required for MCP connections
PERPLEXITY_API_KEY="YOUR_PERPLEXITY_API_KEY" # Required for PerplexitySearchToolkit
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY" # Required for BrowserUseAgentToolkit (Gemini)
PERPLEXITY_API_KEY="YOUR_PERPLEXITY_API_KEY"
```

The application uses `python-dotenv` to load these variables automatically if the library is installed and included in the loading mechanism (e.g., in `config/settings.py` if created later, or directly loaded in `main`).

## Running the System (CLI)

You can interact with the Nova SHIFT system using the command-line interface (`cli.py`). Provide a task description in natural language as an argument.

**Prerequisites:**
*   Ensure Redis is running.
*   Ensure all required environment variables (`.env` file or system variables) are set (see **Prerequisites** section above).
*   Activate your virtual environment (`source venv/bin/activate` or `.\venv\Scripts\activate`).

**Usage:**

Navigate to the project root directory (`Desktop/NOVA-SHIFT-2.0`) and run:

```bash
python cli.py "Your task description here"
```

**Example:**

```bash
python cli.py "Research the latest advancements in quantum computing and provide a summary."
```

The script will:
1.  Initialize the Nova SHIFT core components (Dispatcher, Shared Memory, LTM, Tool Registry).
2.  Initialize the Planner Agent and Specialist Agents.
3.  Pass your task description to the Planner Agent for decomposition.
4.  Dispatch the generated subtasks to the Specialist Agents.
5.  Monitor the execution progress by checking Shared Memory (Redis).
6.  Print the final status and results of the task execution.

*(Note: The Planner Agent's LLM call for decomposition might still be mocked in the current version. Check `agents/planner_agent.py` if results seem unexpected.)*

## Running Tests

Tests are implemented using `pytest`. Ensure you have installed the development dependencies (including `pytest`).

To run all tests from the project root directory:

```bash
pytest
```

To run tests for a specific module:

```bash
pytest tests/core/test_shared_memory_interface.py
```

*(Note: Some integration tests might require a running Redis instance on the configured test database (DB 1 by default in `test_phase2_flow.py`).)*