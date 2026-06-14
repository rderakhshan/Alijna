# Revised Implementation Plan: Multi-Dimensional Metrics

**Based on actual codebase audit (not aspirational).** Each section references real file paths, real classes, and real method signatures. All 26 metrics from the original plan are preserved and grouped into incremental tiers based on actual data availability.

**Current State (Audited):** The codebase captures `{question, golden_answer, agent_result, agent_trajectory}` per task in JSONL format. No `metrics` dict, no `elapsed_time`, no `total_tokens`, no individual per-task JSON files exist. The `AutoEvolver` Pareto selector expects `accuracy`, `total_tokens`, and `execution_time` but the latter two are never written.

---

## Phase A: Telemetry Foundation

**Goal:** Create the physical data pipeline that every metric depends on.

### A.1 Create Runner Scripts

**Files to create:** `src/Flash_Searcher/run_flash_searcher_mm_gaia.py` (and equivalents for other datasets)

**Current gap:** `MemEvolve/utils/run_provider.py` references runner scripts that don't exist. The existing `run_flash_searcher.py` only produces JSONL output, not per-task JSON files.

**What to build:**

```python
# src/Flash_Searcher/run_flash_searcher_mm_gaia.py
# New runner that produces per-task JSON files with telemetry.

def process_item(item, model, summary_interval, prompts_type, max_steps, memory_provider=None):
    import time
    agent = SearchAgent(...)
    question = item["question"]
    golden_answer = item["answer"]

    start = time.monotonic()
    try:
        result = agent(question)
    except Exception as e:
        return None
    elapsed = time.monotonic() - start

    judgement = judge_answer(result["agent_result"], golden_answer)  # LLM or exact match

    return {
        "question": question,
        "golden_answer": golden_answer,
        "agent_result": result["agent_result"],
        "agent_trajectory": result["agent_trajectory"],
        "judgement": "correct" if judgement else "incorrect",
        "score": 1 if judgement else 0,
        "is_correct": judgement,
        "status": "success",
        "metrics": {
            "elapsed_time": elapsed,
            "total_tokens": getattr(model, "last_total_tokens", 0),
            "prompt_tokens": getattr(model, "last_input_token_count", 0),
            "completion_tokens": getattr(model, "last_output_token_count", 0),
        },
    }
```

**Verification:** After this phase, `AutoEvolver._compute_avg_execution_time()` receives real data. `TrajectoryFeedbackAggregator.aggregate()` populates `summary.accuracy` and `summary.tokens`.

### A.2 Add Token Tracking to Model Wrapper

**Files to modify:**
- `src/Flash_Searcher/FlashOAgents/models.py` (~line 492, `OpenAIServerModel.__call__`)

**Current state:** `last_input_token_count` and `last_output_token_count` are stored on the model object but never aggregated into a `total_tokens` field.

**Change:** Add `last_total_tokens` property:

```python
@property
def last_total_tokens(self):
    return (self.last_input_token_count or 0) + (self.last_output_token_count or 0)
```

### A.3 Add TTFT Tracking to Streaming

**Files to modify:**
- `src/deep_agents_from_scratch/utils.py` — `stream_agent()` function

**Current state:** `stream_agent()` is an async generator that iterates over `agent.astream()`. It prints node updates and returns final state. No timing is captured.

**Change:** Measure time from stream start to first token received, attach to final state as `state["metrics"]["ttft"]`.

### A.4 Write Per-Task JSON Files

**Files to modify:**
- `src/Flash_Searcher/run_flash_searcher.py` (or the new dataset-specific runners)

**Current state:** Output is a single JSONL file with one line per task.

**Change:** In addition to JSONL, write individual `<task_id>.json` files to `--direct_output_dir` so `TrajectoryFeedbackAggregator._load_task()` can find them:

```python
task_file = Path(output_dir) / f"{item_id}.json"
with open(task_file, "w") as f:
    json.dump(task_data, f, indent=2)
```

**Verification:** `TrajectoryFeedbackAggregator(str(logs_dir)).aggregate()` no longer returns empty results.

---

## Phase B: Tier-1 Metrics (Trajectory-Ready)

**These metrics need only fields already present in `agent_trajectory`.** No new telemetry required beyond Phase A.

### B.1 Task Success Rate (SR) — Metric 1

**Target:** `TrajectoryFeedbackAggregator` in `src/MemEvolve/utils/trajectory_tools.py`

**Current state:** `_is_task_correct(self, data)` already exists (line 55). It checks `judgement`, `score`, `is_correct`, `correct` fields. It already returns a boolean.

**Work required:** None. Already works once Phase A populates `judgement`/`score` fields.

**Evolution impact:** Aggregated as `summary.accuracy` by `TrajectoryFeedbackAggregator.aggregate()`. Already consumed by `AutoEvolver._pareto_select_top()`.

### B.2 Tool Invocation Accuracy — Metric 8

**Target:** New method on `TrajectoryFeedbackAggregator`

**Data source:** `trajectory` list from task JSON. Each step with `name == "action"` has `tool_calls` field.

```python
def _calculate_tool_invocation_accuracy(self, trajectory: list) -> float:
    action_steps = [s for s in trajectory if s.get("name") == "action"]
    if not action_steps:
        return 1.0
    steps_with_calls = sum(1 for s in action_steps if s.get("tool_calls"))
    return steps_with_calls / len(action_steps)
```

### B.3 Parameter F1 Score — Metric 10

**Target:** New method on `TrajectoryFeedbackAggregator`

**Data source:** `trajectory[].tool_calls[].arguments` — already captured by `capture_trajectory()`.

```python
def _calculate_parameter_f1(self, generated_args: dict, expected_args: dict) -> float:
    if not expected_args:
        return 1.0 if not generated_args else 0.0
    tp = sum(1 for k, v in generated_args.items() if k in expected_args and expected_args[k] == v)
    fp = len(generated_args) - tp
    fn = len(expected_args) - tp
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
```

**Note:** This requires ground-truth `expected_args`. Define an inline dict in the runner for each known task, or compute against `golden_answer` when it encodes expected parameters.

### B.4 AST Validation — Metric 11

**Target:** New method on `TrajectoryFeedbackAggregator`

**Data source:** Any step's `value` or `obs` that contains Python code.

```python
def _validate_ast(self, code: str) -> float:
    import ast
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0
```

### B.5 Execution Correctness — Metric 12

**Target:** New method on `TrajectoryFeedbackAggregator`

**Data source:** `trajectory[].obs` — free-text observation strings captured by `capture_trajectory()`.

```python
def _calculate_execution_correctness(self, trajectory: list) -> float:
    actions = [s for s in trajectory if s.get("name") == "action"]
    if not actions:
        return 1.0
    errors = sum(1 for a in actions if "error" in str(a.get("obs", "")).lower())
    return 1.0 - (errors / len(actions))
```

### B.6 Step Success Rate — Metric 14

**Target:** New method on `TrajectoryFeedbackAggregator`

**Data source:** `trajectory` list. Steps have no explicit `status` field by default, so this metric requires adding `"status": "failed"` to steps that observe errors.

**Prerequisite:** Modify `capture_trajectory()` in `src/Flash_Searcher/base_agent.py` to tag steps with error observations:

```python
# In capture_trajectory, for ActionStep:
traj = {
    "name": "action",
    "tool_calls": [...],
    "obs": step.observations,
    "status": "failed" if "error" in (step.observations or "").lower() else "success",
    ...
}
```

Then:

```python
def _calculate_step_success_rate(self, trajectory: list) -> float:
    if not trajectory:
        return 0.0
    failed = sum(1 for s in trajectory if s.get("status") == "failed")
    return 1.0 - (failed / len(trajectory))
```

### B.7 Memory Span — Metric 22

**Target:** New method on `TrajectoryFeedbackAggregator`

**Data source:** `agent_trajectory` list length.

```python
def _calculate_memory_span(self, data: dict) -> float:
    trajectory = data.get("agent_trajectory", [])
    return float(len(trajectory))
```

### B.8 Prompt Leakage Detection — Metric 24

**Target:** New method on `TrajectoryFeedbackAggregator`

**Data source:** `agent_result` string.

```python
def _detect_prompt_leak(self, answer: str) -> float:
    keywords = ["system_prompt", "api_key", "secret", "instruction", "you are an"]
    leaks = sum(1 for kw in keywords if kw in answer.lower())
    return 1.0 if leaks > 0 else 0.0
```

### B.9 End-to-End Latency — Metric 5

**Target:** Already partially handled by `AutoEvolver._compute_avg_execution_time()` (line 573).

**Current state:** The method reads `metrics.elapsed_time` from task JSONs. No code writes this field.

**Work required:** None beyond Phase A (A.1 already captures `elapsed_time`).

**Verification:** After Phase A, the Pareto selector's `execution_time` dimension receives real data.

---

## Phase C: Enhanced Telemetry & Tier-2 Metrics

**These metrics need new fields written during task execution.**

### C.1 Enhanced Telemetry — New JSON Fields

**Files to modify:** Runner scripts created in Phase A.1.

Add these fields to the per-task JSON:

```python
"metrics": {
    # From Phase A:
    "elapsed_time": ...,
    "total_tokens": ...,
    "prompt_tokens": ...,
    "completion_tokens": ...,
    "ttft": ...,
    # New for Tier 2:
    "token_cost": total_tokens * cost_per_token,  # needs model cost config
    "context_token_history": [1024, 2048, 3072],  # sampled at each planning step
    "goals_completed": 3,                          # task-specific subgoal tracking
    "goals_total": 5,                              # defined in dataset config
    "errors_injected": 2,                          # only for robustness suites
    "errors_recovered": 1,                         # only for robustness suites
}
```

### C.2 Cost-Aware Token Efficiency — Metric 7

**Target:** New method on `TrajectoryFeedbackAggregator`

**Data source:** `metrics.total_tokens` + `metrics.elapsed_time`.

```python
def _calculate_token_efficiency(self, data: dict) -> float:
    metrics = data.get("metrics", {})
    tokens = metrics.get("total_tokens", 0)
    time_sec = metrics.get("elapsed_time", 1)
    return tokens / max(time_sec, 0.001)  # tokens per second
```

### C.3 pass@k and pass^k — Metrics 2 & 3

**Target:** New methods on `AutoEvolver` in `src/MemEvolve/core/auto_evolver.py`

**Data source:** Multiple evaluation runs of the same provider.

```python
import math

def _calculate_pass_at_k(self, n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

def _calculate_pass_strict_k(self, successes: list[bool], k: int) -> float:
    if len(successes) < k:
        return 0.0
    return 1.0 if all(successes[:k]) else 0.0
```

**Integration:** Called in `_evaluate_providers_parallel()` after collecting per-task correctness across multiple runs. Requires the AutoEvolver to run each provider-task pair $k$ times (controlled by `task_batch_x` or a new `num_repetitions` parameter).

### C.4 Tool Selection Accuracy (MRR/NDCG) — Metric 9

**Target:** New method on `TrajectoryFeedbackAggregator`

**Data source:** Requires `retrieved_candidates` field in trajectory — not currently captured. Must be added to `capture_trajectory()` or to a new wrapper around `ToolCallingAgent`.

```python
def _calculate_retrieval_ndcg(self, retrieved: list, selected: str) -> float:
    if not retrieved:
        return 0.0
    if retrieved[0] == selected:
        return 1.0
    try:
        idx = retrieved.index(selected)
        return 1.0 / math.log2(idx + 2)  # DCG@position
    except ValueError:
        return 0.0
```

**Prerequisite:** The runner must log which tool names were available vs. which was chosen. This needs a change in `ToolCallingAgent` to expose the candidate tool list at each step.

### C.5 Context Span Decay — Metric 21

**Target:** New method on `TrajectoryFeedbackAggregator`

**Data source:** `metrics.context_token_history` — needs Phase C.1 telemetry.

```python
def _calculate_context_decay(self, data: dict) -> float:
    history = data.get("metrics", {}).get("context_token_history", [])
    if not history:
        return 0.0
    return float(history[-1] / (sum(history) / len(history))) if history else 0.0
```

### C.6 Additional Tier-2 Metrics

These follow the same pattern: add the required data field in the runner, then compute in `TrajectoryFeedbackAggregator`.

| Metric | New Field Needed | Notes |
|--------|-----------------|-------|
| 13. Progress Rate | `metrics.goals_completed`, `metrics.goals_total` | Dataset-specific subgoal definitions |
| 15. Plan Edit Distance | Reference tool sequence (per-dataset ground truth) | Requires dataset annotation |
| 16. Factual Recall Accuracy | `expected_recalls` per task | Requires dataset annotation |
| 19. Error Recovery Rate | `metrics.errors_injected`, `metrics.errors_recovered` | Only for robustness-focused evals |
| 10. Parameter F1 (enhanced) | Ground-truth args per task | Already partially in B.3, enhanced with dataset-specific expected args |
| 23. Policy Adherence Score | `policy_rules` configuration | Define in each dataset's config |

---

## Phase D: Tier-3 Metrics (Heavy Dependencies)

**These metrics require model downloads, GPU memory, or LLM API calls.** They are computed lazily and off by default.

### D.1 Dependency Management Pattern

```python
# In a new file: src/MemEvolve/utils/metrics_heavy.py

_HEAVY_METRICS_ENABLED = os.environ.get("MEMEVOLVE_HEAVY_METRICS", "0") == "1"

class LazyModelLoader:
    """Defer model downloads until first use."""
    _models = {}
    
    @classmethod
    def get(cls, name: str, loader):
        if name not in cls._models:
            if not _HEAVY_METRICS_ENABLED:
                raise RuntimeError(
                    f"Metric requires model '{name}'. Set MEMEVOLVE_HEAVY_METRICS=1"
                )
            cls._models[name] = loader()
        return cls._models[name]
```

### D.2 Factual Correctness (RAG Quality) — Metric 4

**Target:** `TrajectoryFeedbackAggregator`

**Dependency:** LLM API call (uses the same model as the agent, or a lightweight judge model).

```python
def _evaluate_factual_correctness(self, answer: str, memories: list[str]) -> float:
    if not memories:
        return 1.0
    prompt = (
        f"Analyze if the answer is supported by the context. "
        f"Answer: '{answer}'. Context: '{' '.join(memories)}'. "
        f"Output only a float between 0.0 (hallucinated) and 1.0 (fully correct)."
    )
    response = self._judge_model([{"role": "user", "content": prompt}])
    try:
        return float(response.content.strip())
    except (ValueError, AttributeError):
        return 0.5
```

**Note:** Requires `memory_guidance` field in trajectory (not currently captured — see Gap 4 in audit). Add to `capture_trajectory()` first.

### D.3 Reasoning Step Alignment — Metric 17

**Target:** `TrajectoryFeedbackAggregator`

**Dependency:** `sentence-transformers` (~90MB model download).

```python
def _calculate_reasoning_alignment(self, actual: str, expected: str) -> float:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = LazyModelLoader.get(
        "mini-lm",
        lambda: SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    )
    emb = model.encode([actual, expected])
    norm = np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])
    return float(np.dot(emb[0], emb[1]) / norm) if norm > 0 else 0.0
```

### D.4 Contradiction Rate — Metric 20

**Target:** `TrajectoryFeedbackAggregator`

**Dependency:** `transformers` + NLI model (~1.2GB download).

```python
def _calculate_contradiction_rate(self, trajectory: list) -> float:
    from transformers import pipeline
    classifier = LazyModelLoader.get(
        "nli",
        lambda: pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-1",
            device=-1,  # CPU; use 0 for GPU
        ),
    )
    statements = [
        s.get("value", "") for s in trajectory
        if s.get("name") in ("thought", "summary")
    ]
    if len(statements) < 2:
        return 0.0
    contradictions = 0
    for i in range(len(statements) - 1):
        result = classifier(statements[i], candidate_labels=["contradiction", "neutral", "entailment"])
        if result["labels"][0] == "contradiction" and result["scores"][0] > 0.7:
            contradictions += 1
    return contradictions / (len(statements) - 1)
```

### D.5 Toxicity & Bias — Metric 25

**Target:** `TrajectoryFeedbackAggregator`

**Dependency:** `detoxify` (~500MB model, wraps PyTorch).

```python
def _evaluate_toxicity(self, answer: str) -> float:
    from detoxify import Detoxify
    model = LazyModelLoader.get("detoxify", lambda: Detoxify("original"))
    results = model.predict(answer)
    return float(results.get("toxicity", 0.0))
```

### D.6 Other Heavy Metrics

| Metric | Dependency | Notes |
|--------|-----------|-------|
| 18. Perturbation Sensitivity | Separate perturbed task suite | Architecture: run same provider against clean + perturbed datasets, compute `SR_clean - SR_perturbed` |
| 23. Policy Adherence | Policy rules configuration | Lightweight; grouped here because it requires external rule definitions |
| 26. Explainability | LLM-as-judge API call | Same pattern as D.2 — prompt-based scoring of reasoning traces |
| 18. Error Recovery | Requires error injection in runner | Heavy because it needs a separate robustness eval pipeline |

---

## Phase E: Pareto Evolution Integration

**Goal:** Wire the collected metrics into the evolution loop so the LLM evolver gets better signal.

### E.1 Extend `AutoEvolver._pareto_select_top()`

**Target:** `src/MemEvolve/core/auto_evolver.py` (line 623)

**Current state:** Pareto over 3 dimensions: accuracy (higher better), total_tokens (lower better), execution_time (lower better).

**Change:** Add a `metric_weights` config that selects which metrics participate in the Pareto front:

```python
# New config in __init__:
self.metric_config = {
    "primary": ["accuracy"],          # always used
    "efficiency": ["total_tokens", "execution_time"],  # always used if available
    "capability": ["tool_accuracy", "execution_correctness"],  # Tier 1, phased in
    "safety": ["prompt_leak", "toxicity", "contradiction"],  # Tier 3, optional
}
```

Each provider candidate gets scored in each dimension. Non-dominated sorting + scalarized tie-breaking (same algorithm as current code, just extended to $N$ dimensions).

### E.2 Feed Structured Diagnostics to Generator

**Target:** `src/MemEvolve/prompts/analysis_prompt.yaml` and `phase_analyzer.py`

**Current state:** The analysis prompt template has a free-form "Improvement Recommendations" section. The LLM reads trajectory summaries and writes recommendations.

**Change:** After computing all active metrics, inject a structured defect profile into the prompt:

```yaml
# Added to analysis_prompt.yaml template
Structured Metrics:
{metrics_summary}

Weakest Dimensions:
{weakest_dimensions}

Recommendation Focus:
Focus improvements on the provider's weakest dimensions above.
```

The `weakest_dimensions` are computed by comparing each metric against a threshold (e.g., `tool_accuracy < 0.8`).

### E.3 Opt-In Metric Levels

**Target:** `src/MemEvolve/core/auto_evolver.py` (config) and `evolve_cli.py` (CLI args)

```python
# Config options:
self.metric_level = metric_level  # "core" | "enhanced" | "complete"

# At Pareto time:
if self.metric_level == "core":
    dimensions = ["accuracy", "total_tokens", "execution_time"]
elif self.metric_level == "enhanced":
    dimensions = ["accuracy", "total_tokens", "execution_time",
                  "tool_accuracy", "execution_correctness", "step_success"]
elif self.metric_level == "complete":
    dimensions = ALL_ACTIVE_METRICS  # includes heavy ones if MEMEVOLVE_HEAVY_METRICS=1
```

CLI: `python evolve_cli.py auto-evolve --metric-level enhanced`

---

## Implementation Order

```
Phase A  →  Phase B  →  Phase C  →  Phase E (core)  →  Phase D  →  Phase E (enhanced)
  (must)     (can)       (can)        (can start)       (optional)   (optional)
```

**Recommended first deliverable:** Phase A + Phase B (8 metrics) + Phase E.1 (extend Pareto to include the new B metrics). This gives you a working evolution pipeline with strictly better selection signal than today, without any model downloads.

---

## Audit Log: Discrepancies Between Original Plan and Reality

| Original Claim | Reality |
|---|---|
| `cli.py` captures timestamps/tokens/errors | No telemetry exists in `cli.py` |
| `utils.py` captures streaming timestamps | `utils.py` is display-only; no timing |
| `trajectory_tools.py` has 20+ metric methods | Has 6 methods; the rest are fictional |
| `auto_evolver.py` has `_calculate_pass_at_k` | Does not exist |
| `auto_evolver.py` has `_calculate_perturbation_sensitivity` | Does not exist |
| `memory_guidance` field exists in trajectory | Does not exist; `capture_trajectory()` never writes it |
| Runners exist for GAIA, WebWalkerQA, etc. | Referenced by `run_provider.py` but not created |
| Per-task `.json` files are written | Only JSONL exists |
| `metrics.elapsed_time` is populated | Never written by any code |
| `AutoEvolver._pareto_select_top()` uses 4D vector | Uses 3D (accuracy, tokens, time); time is always 0 |

This plan is incremental and self-correcting — each phase produces independently verifiable output before the next begins.
