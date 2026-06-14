# Spec: MemEvolve Integration into Deepagent

## ASSUMPTIONS I'M MAKING

1. MemEvolve source is Apache 2.0 licensed — we can vendor it directly
2. We copy the full MemEvolve repo (`EvolveLab/`, `MemEvolve/`, plus supporting files) into Deepagent's source tree, keeping internal structure intact
3. **Two tiers:** memory providers (always available, stdlib only) + evolution pipeline (heavier deps, separate CLI for power users)
4. Memory provider is optional — no change to Deepagent's default behavior
5. The provider persists data to disk — survives across sessions
6. The provider lives outside LangGraph state, injected into the agent at CLI startup
7. The evolution pipeline's heavy deps (crawl4ai, playwright, Flash-Searcher) are optional install extras — only needed when running `evolve_cli.py`

## Objective

Vendor MemEvolve's full ICML-accepted codebase into Deepagent, giving end users two capabilities:
- **Use** evolved memory providers in the agent loop (lightweight, no new deps)
- **Evolve** new memory systems from their own trajectory data (evolution pipeline, optional heavy deps)

**Success criteria:**
- User runs `python main.py` — default behavior unchanged
- User runs `MEMORY_PROVIDER=lightweight_memory python main.py` — agent uses evolved memory with recall across turns
- User runs `python -m deep_agents_from_scratch.evolve auto-evolve gaia` — full evolution pipeline works, generates new providers
- Memory providers persist learnings to disk and survive session restarts

## Tech Stack

- **Python 3.11-3.13** (matches Deepagent)
- **LangGraph 0.6+** (existing agent framework)
- **MemEvolve** vendored as-is from `bingreeky/MemEvolve`
- **Optional evolution deps:** openai, sentence-transformers, numpy, scikit-learn, crawl4ai, playwright

## Commands

```
# Use evolved memory (no new deps required)
python main.py                                                # default, no memory
MEMORY_PROVIDER=lightweight_memory python main.py             # with evolved memory
MEMORY_PROVIDER=lightweight_memory MEMORY_DB_PATH=./data python main.py  # custom storage

# Run evolution pipeline (requires optional deps + API keys)
python -m deep_agents_from_scratch.evolve auto-evolve gaia \
    --provider lightweight_memory \
    --num-rounds 3 \
    --creativity 0.5

# Manual evolution steps
python -m deep_agents_from_scratch.evolve analyze ./trajectories/ --provider lightweight_memory
python -m deep_agents_from_scratch.evolve generate --creativity 0.5
python -m deep_agents_from_scratch.evolve create
python -m deep_agents_from_scratch.evolve validate
```

## Project Structure

```
src/deep_agents_from_scratch/
├── __init__.py
├── cli.py                    # Entry point — parse --memory-provider, inject adapter
├── memory/                   # ← NEW: vendored EvolveLab/ package
│   ├── __init__.py
│   ├── base_memory.py         # BaseMemoryProvider ABC (as-is from MemEvolve)
│   ├── memory_types.py        # MemoryRequest, MemoryResponse, TrajectoryData
│   ├── config.py              # Provider config
│   └── providers/             # ICML-accepted evolved providers
│       ├── __init__.py
│       ├── lightweight_memory_provider.py   # 1597 lines, stdlib only
│       ├── cerebra_fusion_memory_provider.py # 64KB, has sklearn/sentence-tx deps
│       └── agent_kb_provider.py             # baseline for evolution
├── evolve/                   # ← NEW: vendored MemEvolve/ package + evolve_cli.py
│   ├── __init__.py
│   ├── core/                  # MemoryEvolver, AutoEvolver
│   ├── phases/                # Analyzer, Generator, Creator, Validator
│   ├── validators/
│   ├── utils/
│   └── evolve_cli.py          # Full evolution CLI
├── adapter/                  # ← NEW: thin integration layer
│   ├── __init__.py
│   └── memory_adapter.py      # Wraps provider for agent loop (our code)
├── file_tools.py
├── state.py
├── prompts.py
├── task_tool.py
└── utils.py
```

## Code Style

Follow existing Deepagent conventions (Google-style docstrings, type hinted, PEP 8). New MemEvolve vendored code keeps its original structure — only the integration adapter (`memory_adapter.py`) follows Deepagent's style.

```python
# Example: memory adapter
from deep_agents_from_scratch.memory import BaseMemoryProvider, MemoryRequest, TrajectoryData
from deep_agents_from_scratch.memory.providers import LightweightMemoryProvider

class MemoryAdapter:
    def __init__(self, provider_name: str, storage_path: str | None = None):
        self.provider: BaseMemoryProvider = _load_provider(provider_name, storage_path)
    def initialize(self) -> bool: ...
    def provide_context(self, query: str) -> str: ...
    def absorb_trajectory(self, query: str, messages: list, result: str) -> None: ...
```

## Testing Strategy

```
pytest tests/ -v --cov=src/deep_agents_from_scratch/memory
```

- Unit test the adapter with each provider (mock file I/O)
- Integration test: run CLI with `MEMORY_PROVIDER=lightweight_memory`, send 2 queries, verify second recalls first
- No need to test MemEvolve's own code — it's ICML-accepted

## Boundaries

- **Always:** Keep providers optional; default behavior unchanged. Vendor code as-is (no refactoring upstream). Run `ruff check` after edits.
- **Ask first:** Adding a new provider beyond `lightweight_memory`. Adding non-stdlib dependencies. Changing the vendored MemEvolve code.
- **Never:** Modify MemEvolve's `BaseMemoryProvider` interface. Introduce regressions to the existing agent loop. Embed API keys in the code.

## Open Questions

1. Just `lightweight_memory` for v1, or also `cerebra_fusion_memory` and baselines?
2. Exact hook points: `provide_memory()` before each user query and `take_in_memory()` after — or also mid-step?
3. Persistence path: default to `./memories/` in CWD?

## Task Plan

- [ ] 1. Clone MemEvolve repo temporarily, extract `EvolveLab/` → `src/deep_agents_from_scratch/memory/`
- [ ] 2. Extract `MemEvolve/` → `src/deep_agents_from_scratch/evolve/`
- [ ] 3. Create `adapter/memory_adapter.py` — wraps any BaseMemoryProvider for the agent loop
- [ ] 4. Update `cli.py` — parse `MEMORY_PROVIDER` env var, inject adapter
- [ ] 5. Update agent loop — call `provide_memory()` before query, `take_in_memory()` after
- [ ] 6. Add optional deps to `pyproject.toml` (`[project.optional-dependencies] evolve = [...]`)
- [ ] 7. Test: run 2+ queries with memory, verify recall across turns
- [ ] 8. Test: run evolution CLI with `--help` to confirm it loads
