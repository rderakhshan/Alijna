# 🧱 Deep Agents from Scratch

A comprehensive Python framework demonstrating advanced context engineering patterns in modern autonomous agents using [LangGraph](https://github.com/langchain-ai/langgraph).

Rather than relying on simple, single-prompt loops, this project implements the three core architectural patterns found in state-of-the-art agent systems (like Manus and Claude Code) to handle long-horizon tasks:
1. **Task Planning (TODO Lists)**: Recitation and tracking to prevent task-drift.
2. **Virtual File Systems (VFS)**: Context offloading to state-persisted files to prevent token overflow.
3. **Sub-agent Delegation (Context Isolation)**: Running specialized sub-agents with clean, isolated context windows to prevent prompt and history contamination.

---

## 🛠️ Key Architectural Patterns

### 1. Task Planning via TODO Lists
Long-horizon execution is highly susceptible to "agent drift," where the LLM forgets its original goal after multiple tool calls. 
* **Implementation**: Managed via `DeepAgentState.todos`.
* **Workflow**: The agent builds a structured TODO list at the beginning of a query using the `write_todos` tool, checks its progress using `read_todos`, and updates task statuses (`pending` ➡️ `in_progress` ➡️ `completed`) dynamically.
* **Reference**: [src/deep_agents_from_scratch/todo_tools.py](file:///d:/AI%20Engineering%20LAB/Deepagent/src/deep_agents_from_scratch/todo_tools.py)

### 2. Virtual File System (VFS)
Storing large amounts of web-search outputs directly in the LLM conversation history quickly pollutes the context window and drives up token costs.
* **Implementation**: The agent state maintains a virtual file system (`DeepAgentState.files`) mapped to ephemeral key-value contents.
* **Workflow**:
  - `ls()`: List existing files in state.
  - `read_file()`: Read file contents with support for line offset and limit pagination (to prevent context overflow).
  - `write_file()`: Create or overwrite files.
* **Reference**: [src/deep_agents_from_scratch/file_tools.py](file:///d:/AI%20Engineering%20LAB/Deepagent/src/deep_agents_from_scratch/file_tools.py)

### 3. Sub-agent Delegation with Context Isolation
When solving complex multi-faceted queries, letting a single agent manage everything leads to context clash.
* **Implementation**: The framework spawns specialized sub-agents (e.g., `research-agent`) with limited tools.
* **Workflow**: The parent agent delegates a specific subtask using the `task` tool. The sub-agent runs in an isolated LangGraph sequence, with its message history completely quarantined. Once completed, only the final answer and any modified virtual files are merged back into the parent state.
* **Reference**: [src/deep_agents_from_scratch/task_tool.py](file:///d:/AI%20Engineering%20LAB/Deepagent/src/deep_agents_from_scratch/task_tool.py)

---

## 📂 Project Structure

```
.
├── main.py                     # Entry point wrapper for CLI execution
├── pyproject.toml              # Project dependencies, build setup, and linter rules
├── uv.lock                     # Lockfile guaranteeing reproducible environment states
├── src/
│   └── deep_agents_from_scratch/
│       ├── __init__.py         # Package entry point
│       ├── cli.py              # CLI runner, prompt composition, and interactive chat loop
│       ├── state.py            # LangGraph schemas (DeepAgentState, Todo) and state reducers
│       ├── file_tools.py       # Virtual file system tools (ls, read_file, write_file)
│       ├── todo_tools.py       # Task planning tools (write_todos, read_todos)
│       ├── task_tool.py        # Sub-agent creation, registry, and isolation (task tool)
│       ├── research_tools.py   # Web search wrapper and AI page summarization
│       ├── prompts.py          # Detailed system prompts and guidelines
│       └── utils.py            # CLI output formatters and agent streaming logic
└── experiment/                 # Research notebooks exploring agent progression
```

---

## 🚀 Getting Started

### Prerequisites
* **Python**: `^3.11` (Requires Python >= 3.11, < 3.14)
* **Package Manager**: [uv](https://docs.astral.sh/uv/) (recommended for speed and reliability)

### Installation
Clone the repository and install the package dependencies:
```bash
# Clone the repository
git clone https://github.com/rderakhshan/Alijna.git
cd Alijna

# Install dependencies and sync virtual environment
uv sync
```

### Environment Configuration
Create a `.env` file in the root of the project to define the API keys for the language model provider and external search APIs:
```ini
# Required for research tools
TAVILY_API_KEY=your_tavily_api_key_here

# LLM Providers (Provide at least one)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: LangSmith Tracing & Observability
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=deep-agents-from-scratch
```

---

## 💻 Running the Application

To run the interactive CLI interface and chat with the Deep Agent, execute:
```bash
uv run python main.py
```

### Model Selection Order
The runner dynamically binds your LLM client on startup according to the keys present in `.env`:
1. **DeepSeek** (`deepseek-chat` via OpenAI-compatible endpoints) - *First priority if key is set*
2. **OpenAI** (`gpt-4o`) - *Second priority*
3. **Anthropic** (`claude-3-5-sonnet-latest`) - *Third priority*
4. **Fallback** (`gpt-4o-mini` using default system client)

---

## ⚙️ Development & Code Quality

Maintain codebase standards using the preconfigured linting and formatting tooling:

### Formatting & Linting (Ruff)
Verify code quality and automatically fix format issues:
```bash
# Sync development tools
uv sync --extra dev

# Run ruff checks
uv run ruff check

# Auto-fix linting issues
uv run ruff check --fix

# Format code layout
uv run ruff format
```

### Type Checking (Mypy)
Check type hints across source code:
```bash
uv run mypy src/
```
