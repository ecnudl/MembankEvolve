# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flash-Searcher is a DAG-based parallel execution agent framework for multi-step reasoning tasks. It combines a tool-calling agent system (FlashOAgents), a pluggable memory evolution system (EvolveLab with 15+ providers), and an automated memory system generator (MemEvolve). Benchmarked on GAIA, WebWalkerQA, xbench, BrowseComp.

## Environment & Running

**Conda environment:** `dl` (Python 3.10)

```bash
# Activate environment
conda activate dl

# Run GAIA benchmark (core entry point)
cd /home/MemEvolve/Flash-Searcher-main
python run_flash_searcher_mm_gaia.py \
  --infile ./data/gaia/validation/metadata.jsonl \
  --task_indices "1-10" \
  --memory_provider prompt_based \
  --concurrency 1 \
  --max_steps 40 \
  --direct_output_dir ./experiments/my_run \
  --outfile ./experiments/my_run/results.jsonl

# Run with specific prompt combination (env var override)
ENABLED_PROMPTS="tip,trajectory" PROMPT_STORAGE_DIR="./storage/my_run" \
  python run_flash_searcher_mm_gaia.py --memory_provider prompt_based --task_indices "1-10" ...

# One-click 100-task evaluation script
./run_gaia100_tip_trajectory.sh

# Memory evolution CLI
python evolve_cli.py analyze --task-logs ./experiments/my_run --dataset gaia
python evolve_cli.py generate --analysis-report ./evolve_workspace/analysis_report.json
python evolve_cli.py validate --provider-name my_provider

# Run memory extraction test
python test_memory_extraction.py
```

**Task index syntax:** `"1-10"`, `"1,3,5-8"`, `"[level1]1-53"`, `"[level2]"` (all level 2), `"[level1] [ignore]3,5"`.

## Architecture

### Three-Layer System

```
FlashOAgents (Agent Execution)
    ↕ memory_guidance injection
EvolveLab (Memory Providers)
    ↕ 4-phase evolution
MemEvolve (Memory System Generator)
```

### FlashOAgents — Agent Framework (`FlashOAgents/`)
- `agents.py`: `ToolCallingAgent` — DAG-based planning with adaptive parallel tool execution. Steps: `PlanningStep → ActionStep → SummaryStep`. Memory guidance injected as special user messages between steps.
- `memory.py`: `AgentMemory` stores step history. Each `PlanningStep`/`ActionStep` has a `memory_guidance` field populated by the memory provider at `BEGIN`/`IN` phases.
- `search_tools.py`: `WebSearchTool`, `CrawlPageTool`, `GitHubSearchTool`, `GitHubIssueTool`
- `mm_tools.py`: `VisualInspectorTool`, `TextInspectorTool`, `AudioInspectorTool`
- `prompts/`: YAML prompt templates per `prompts_type`

### EvolveLab — Memory System (`EvolveLab/`)
- `memory_types.py`: `MemoryType` enum + `PROVIDER_MAPPING` dict (class name → module). Add new providers here.
- `config.py`: `DEFAULT_CONFIG["providers"]` — per-provider config keyed by `MemoryType`.
- `memory_schema.py`: `MemoryUnit` (atomic memory with type, content, embedding, relations). `split_extraction_output()` parses LLM JSON into units.
- `base_memory.py`: `BaseMemoryProvider` interface — `initialize()`, `take_in_memory(TrajectoryData)`, `provide_memory(MemoryRequest) → MemoryResponse`.
- `providers/`: 15+ implementations. Key one: `prompt_based_memory_provider.py` — configurable multi-prompt extraction (tip/trajectory/workflow/shortcut/insight) with sentence-transformers cosine similarity retrieval.

### MemEvolve — Evolution Engine (`MemEvolve/`)
4-phase pipeline: Analyze trajectories → Generate provider code → Create files & register → Validate (AST + import + simulation). `auto_evolver.py` runs multi-round tournaments.

### Agent Classes (`base_agent.py`)
- `MMSearchAgent`: Full multimodal agent (web + crawl + visual + text + audio + GitHub tools). Used by GAIA runner.
- `SearchAgent`: Text-only (web + crawl).
- `AnalysisAgent`: Trajectory analysis tools for MemEvolve.
- All inherit `BaseAgent` which provides `forward()` and `capture_trajectory()`.

### Evaluation Pipeline
- `run_flash_searcher_mm_gaia.py`: Main GAIA runner. Loads JSONL, filters by level/indices, runs `process_item()` per task with `ThreadPoolExecutor`, judges via `lasj.py`, saves per-task JSON + report.
- `lasj.py`: `judge_equivalence(question, golden, predicted, model)` — LLM-based answer comparison returning `{"judgement": "correct"|"incorrect"}`.
- `eval_utils.py`: `TaskTimer`, `TokenCounter`, `generate_unified_report()`.

## Configuration

All LLM calls use `DEFAULT_MODEL` from `.env`. API keys, base URLs, and tool tokens (SERPER_API_KEY, GITHUB_TOKEN) also in `.env`. Models are accessed through OpenAI-compatible API (`OPENAI_API_BASE`).

Current model: `qwen3-max` via DashScope (`https://dashscope.aliyuncs.com/compatible-mode/v1`).

## Data

- **GAIA dataset**: `data/gaia/validation/metadata.jsonl` (165 tasks, levels 1-3) + attachment files (xlsx, pdf, mp3, png, etc.)
- **Prompt templates**: `tips_prompt.txt`, `trajectory_prompt.txt`, `workflow_prompt.txt`, `shortcut_prompt.txt`, `insights_prompt.txt` in project root
- **Memory storage**: `storage/prompt_based_tip_trajectory/memory_db.json` (persisted MemoryUnits with embeddings)
- **Experiment results**: `experiments/<run_name>/` — per-task JSON files + `report.txt`

## Key Patterns

- Memory providers are selected via `--memory_provider prompt_based` CLI arg; prompt subset controlled by `ENABLED_PROMPTS` env var (comma-separated: `tip,trajectory,workflow,shortcut,insight`).
- `PROMPT_STORAGE_DIR` env var isolates memory storage for parallel experiments.
- Memory is injected only at `MemoryStatus.BEGIN` phase (planning). Retrieval uses cosine similarity of task query against stored MemoryUnit embeddings (top-k=5).
- The `take_in_memory()` path: task completes → trajectory + correctness passed to provider → LLM extracts structured JSON via prompt template → `split_extraction_output()` creates `MemoryUnit`s → dedup by signature → embed & persist.
- `tips_prompt.txt` extracts 3 categories: `planning_and_decision_tips`, `tool_and_search_tips`, `answer_format_tips`.
