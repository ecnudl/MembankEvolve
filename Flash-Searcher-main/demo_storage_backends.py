#!/usr/bin/env python
# coding=utf-8
"""
End-to-end demo: Insight extraction from GAIA trajectories + all 4 storage backends.

Steps:
  1. Load 5 existing GAIA trajectories from memory_trajectories/
  2. Run "insight" + "tip" extraction via PromptBasedMemoryProvider (calls LLM)
  3. Collect all extracted MemoryUnits
  4. Store them into all 4 backends: JsonStorage, VectorStorage, HybridStorage, GraphStore
  5. Verify each backend: count, dedup, retrieval, persistence round-trip
  6. Print comparison report

Usage:
  conda activate memevolve
  python demo_storage_backends.py
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from FlashOAgents import OpenAIServerModel
from EvolveLab.memory_types import MemoryType, TrajectoryData
from EvolveLab.memory_schema import (
    MemoryUnit,
    MemoryUnitType,
    MemoryRelation,
    RelationType,
    split_extraction_output,
)
from EvolveLab.providers.prompt_based_memory_provider import (
    _build_template_context,
    _render_prompt,
    _parse_json_from_response,
    _load_embedding_model,
    PROMPT_FILE_NAMES,
    PROMPT_TO_UNIT_TYPE,
)
from EvolveLab.storage import JsonStorage, VectorStorage, HybridStorage, GraphStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")

TRAJECTORIES_DIR = "./memory_trajectories"
ENABLED_PROMPTS = ["tip", "insight"]
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE_DIR = "./storage/models"


# ======================================================================
# Phase 1: Load trajectories
# ======================================================================

def load_trajectories(n: int = 5):
    """Load the first n trajectory files."""
    trajectories = []
    for i in range(1, n + 1):
        fpath = os.path.join(TRAJECTORIES_DIR, f"{i}.json")
        if not os.path.exists(fpath):
            logger.warning(f"Trajectory file not found: {fpath}")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        trajectories.append(data)
    return trajectories


def trajectory_to_data(traj: dict) -> TrajectoryData:
    """Convert raw trajectory JSON to TrajectoryData for extraction."""
    is_correct = traj.get("judgement") == "correct"
    return TrajectoryData(
        query=traj.get("question", traj.get("full_query", "")),
        trajectory=traj.get("agent_trajectory", []),
        result=traj.get("agent_result"),
        metadata={
            "task_id": traj.get("task_id", ""),
            "is_correct": is_correct,
            "golden_answer": traj.get("golden_answer", ""),
            "task_order": traj.get("task_order", 0),
        },
    )


# ======================================================================
# Phase 2: Extract memories via LLM
# ======================================================================

def extract_memories(
    trajectories: list,
    model,
    embedding_model,
) -> list:
    """Run extraction prompts on trajectories, return list of MemoryUnits."""
    # Load prompt templates
    templates = {}
    for pname in ENABLED_PROMPTS:
        fname = PROMPT_FILE_NAMES.get(pname)
        if fname and os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                templates[pname] = f.read()
            logger.info(f"Loaded prompt template: {pname} ({fname})")

    all_units = []
    existing_sigs = set()

    for idx, traj in enumerate(trajectories):
        td = trajectory_to_data(traj)
        is_correct = td.metadata.get("is_correct", False)
        task_outcome = "success" if is_correct else "failure"
        task_id = td.metadata.get("task_id", f"task_{idx}")
        context = _build_template_context(td, is_correct)

        logger.info(
            f"\n{'='*60}\n"
            f"Task {idx+1}/{len(trajectories)}: {task_id}\n"
            f"  Question: {td.query[:80]}...\n"
            f"  Outcome: {task_outcome}\n"
            f"{'='*60}"
        )

        for pname, template_str in templates.items():
            unit_type = PROMPT_TO_UNIT_TYPE.get(pname)
            if unit_type is None:
                continue

            # Skip conditions
            if pname == "insight" and is_correct:
                logger.info(f"  [{pname}] Skipped (insight is failure-only, task succeeded)")
                continue
            if pname == "workflow" and not is_correct:
                logger.info(f"  [{pname}] Skipped (workflow is success-only, task failed)")
                continue

            logger.info(f"  [{pname}] Calling LLM for extraction...")
            t0 = time.time()

            try:
                filled_prompt = _render_prompt(template_str, context)
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": filled_prompt}]}
                ]
                response = model(messages)
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )
            except Exception as e:
                logger.error(f"  [{pname}] LLM call failed: {e}")
                continue

            elapsed = time.time() - t0
            logger.info(f"  [{pname}] LLM responded in {elapsed:.1f}s")

            parsed = _parse_json_from_response(response_text)
            if parsed is None:
                logger.warning(f"  [{pname}] Failed to parse JSON from response")
                continue

            if isinstance(parsed, dict) and parsed.get("skipped"):
                logger.info(f"  [{pname}] Extraction skipped by LLM")
                continue

            try:
                units = split_extraction_output(
                    extraction_result=parsed,
                    unit_type=unit_type,
                    source_task_id=task_id,
                    source_task_query=td.query,
                    task_outcome=task_outcome,
                    extraction_model=getattr(model, "model_id", "unknown"),
                )
            except Exception as e:
                logger.error(f"  [{pname}] split_extraction_output failed: {e}")
                continue

            # Dedup + embed
            added = 0
            for unit in units:
                if unit.signature in existing_sigs:
                    continue
                text = unit.content_text()
                if text and embedding_model is not None:
                    unit.embedding = embedding_model.encode(text, convert_to_numpy=True)
                all_units.append(unit)
                existing_sigs.add(unit.signature)
                added += 1

            logger.info(
                f"  [{pname}] Extracted {added} units "
                f"(total so far: {len(all_units)})"
            )

    return all_units


# ======================================================================
# Phase 3: Add COOCCURS relations between units from same task
# ======================================================================

def add_cooccurs_relations(units: list) -> None:
    """Add COOCCURS relations between MemoryUnits from the same task."""
    by_task = {}
    for u in units:
        by_task.setdefault(u.source_task_id, []).append(u)

    for task_id, task_units in by_task.items():
        if len(task_units) < 2:
            continue
        for i, u in enumerate(task_units):
            for j, other in enumerate(task_units):
                if i != j:
                    u.relations.append(
                        MemoryRelation(
                            target_id=other.id,
                            relation_type=RelationType.COOCCURS,
                            weight=1.0,
                        )
                    )


# ======================================================================
# Phase 4: Test all storage backends
# ======================================================================

def test_storage_backend(name: str, store, units: list) -> dict:
    """Test a single storage backend and return metrics."""
    logger.info(f"\n{'─'*50}\nTesting {name}\n{'─'*50}")
    results = {"name": name, "errors": []}

    # Initialize
    t0 = time.time()
    ok = store.initialize()
    results["init_time"] = time.time() - t0
    if not ok:
        results["errors"].append("initialize() failed")
        return results
    logger.info(f"  Initialized in {results['init_time']:.3f}s")

    # Add units
    t0 = time.time()
    added = store.add(units)
    results["add_time"] = time.time() - t0
    results["added"] = added
    results["count"] = store.count()
    logger.info(f"  Added {added} units in {results['add_time']:.3f}s (count={store.count()})")

    # Dedup test
    added2 = store.add(units)
    results["dedup_added"] = added2
    logger.info(f"  Dedup re-add: {added2} (should be 0)")
    if added2 != 0:
        results["errors"].append(f"Dedup failed: re-add returned {added2}")

    # Get by ID
    first = units[0]
    got = store.get(first.id)
    results["get_ok"] = got is not None
    logger.info(f"  Get by ID: {'OK' if got else 'FAIL'}")

    # Filter
    tips = store.get_all(unit_type=MemoryUnitType.TIP)
    insights = store.get_all(unit_type=MemoryUnitType.INSIGHT)
    results["tips"] = len(tips)
    results["insights"] = len(insights)
    logger.info(f"  Filter: {len(tips)} tips, {len(insights)} insights")

    # Embedding index
    mat, emb_units = store.get_embedding_index(active_only=True)
    results["emb_count"] = len(emb_units)
    results["emb_shape"] = mat.shape if mat is not None else None
    logger.info(
        f"  Embedding index: {mat.shape if mat is not None else 'None'}, "
        f"{len(emb_units)} units"
    )

    # Similarity search (using embedding index)
    if mat is not None and len(emb_units) > 0:
        from sklearn.metrics.pairwise import cosine_similarity

        query_emb = emb_units[0].embedding.reshape(1, -1)
        sims = cosine_similarity(query_emb, mat)[0]
        top3_idx = sims.argsort()[-3:][::-1]
        top3 = [(emb_units[i].type.value, round(float(sims[i]), 3)) for i in top3_idx]
        results["top3_search"] = top3
        logger.info(f"  Similarity search top-3: {top3}")

    # Save
    t0 = time.time()
    store.save()
    results["save_time"] = time.time() - t0
    logger.info(f"  Saved in {results['save_time']:.3f}s")

    # Reload test
    store2 = store.__class__(store.config)
    store2.initialize()
    results["reload_count"] = store2.count()
    logger.info(f"  Reload verification: {store2.count()} units (expected {store.count()})")
    if store2.count() != store.count():
        results["errors"].append(
            f"Reload mismatch: {store2.count()} vs {store.count()}"
        )

    # Embedding persistence
    mat2, eu2 = store2.get_embedding_index(active_only=True)
    results["reload_emb_count"] = len(eu2)
    if mat2 is not None and mat is not None:
        results["emb_match"] = np.allclose(mat, mat2, atol=1e-5)
        logger.info(f"  Embedding persistence: {'MATCH' if results['emb_match'] else 'MISMATCH'}")
    else:
        results["emb_match"] = mat2 is None and mat is None

    # Graph-specific tests
    if name == "GraphStore":
        _test_graph_specific(store2, units, results)

    return results


def _test_graph_specific(store, units, results):
    """Additional tests for GraphStore."""
    stats = store.stats()
    results["graph_stats"] = stats
    logger.info(f"  Graph stats: {json.dumps(stats, indent=2)}")

    # Query layer
    task_ids = set(u.source_task_id for u in units)
    for tid in list(task_ids)[:2]:
        found = store.get_units_by_query(tid)
        logger.info(f"  Units by query '{tid[:12]}...': {len(found)}")

    # Entity layer
    if stats["nodes_by_layer"].get("entity", 0) > 0:
        # Find an entity node
        for nid, data in store._graph.nodes(data=True):
            if data.get("layer") == "entity":
                etype = data.get("entity_type", "")
                ename = data.get("normalized_name", "")
                found = store.get_units_by_entity(etype, ename)
                logger.info(
                    f"  Units by entity ({etype}:{ename}): {len(found)}"
                )
                break

    # Neighbors
    first_nid = f"m:{units[0].id}"
    nbrs = store.neighbors(first_nid, direction="both")
    logger.info(f"  Neighbors of {first_nid[:20]}...: {len(nbrs)}")


# ======================================================================
# Main
# ======================================================================

def main():
    logger.info("=" * 60)
    logger.info("Storage Backend E2E Demo")
    logger.info("=" * 60)

    # Load trajectories
    trajectories = load_trajectories(5)
    logger.info(f"Loaded {len(trajectories)} trajectories")

    if not trajectories:
        logger.error("No trajectories found. Run GAIA tasks first.")
        sys.exit(1)

    # Initialize LLM
    model_id = os.environ.get("DEFAULT_MODEL", "qwen3-max")
    api_base = os.environ.get(
        "OPENAI_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in .env")
        sys.exit(1)

    logger.info(f"LLM: {model_id} @ {api_base}")
    model = OpenAIServerModel(
        model_id=model_id, api_base=api_base, api_key=api_key
    )

    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = _load_embedding_model(EMBEDDING_MODEL_NAME, EMBEDDING_CACHE_DIR)
    logger.info("Embedding model loaded")

    # ---- Phase 2: Extract memories ----
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Extracting memories from trajectories")
    logger.info("=" * 60)
    units = extract_memories(trajectories, model, embedding_model)
    logger.info(f"\nTotal extracted: {len(units)} MemoryUnits")

    if not units:
        logger.error("No units extracted. Check LLM connectivity.")
        sys.exit(1)

    # Show summary
    by_type = {}
    for u in units:
        by_type[u.type.value] = by_type.get(u.type.value, 0) + 1
    logger.info(f"By type: {by_type}")
    logger.info(
        f"By outcome: "
        f"success={sum(1 for u in units if u.task_outcome=='success')}, "
        f"failure={sum(1 for u in units if u.task_outcome=='failure')}"
    )

    # ---- Phase 3: Add relations ----
    add_cooccurs_relations(units)
    total_rels = sum(len(u.relations) for u in units)
    logger.info(f"Added COOCCURS relations: {total_rels} total edges")

    # ---- Phase 4: Test all storage backends ----
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4: Testing all storage backends")
    logger.info("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="storage_demo_")
    logger.info(f"Temp directory: {tmpdir}")

    backends = [
        (
            "JsonStorage",
            JsonStorage({"db_path": os.path.join(tmpdir, "json", "db.json")}),
        ),
        (
            "VectorStorage",
            VectorStorage(
                {"storage_dir": os.path.join(tmpdir, "vector"), "embedding_dim": 384}
            ),
        ),
        (
            "HybridStorage",
            HybridStorage(
                {"storage_dir": os.path.join(tmpdir, "hybrid"), "embedding_dim": 384}
            ),
        ),
        (
            "GraphStore",
            GraphStore({"storage_dir": os.path.join(tmpdir, "graph")}),
        ),
    ]

    all_results = []
    for name, store in backends:
        try:
            res = test_storage_backend(name, store, units)
            all_results.append(res)
        except Exception as e:
            logger.error(f"{name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"name": name, "errors": [str(e)]})

    # ---- Final Report ----
    logger.info("\n" + "=" * 60)
    logger.info("FINAL REPORT")
    logger.info("=" * 60)

    header = f"{'Backend':<16} {'Added':>6} {'Count':>6} {'Dedup':>6} {'Tips':>5} {'Insights':>8} {'Embs':>5} {'Save(s)':>8} {'Reload':>7} {'Status'}"
    logger.info(header)
    logger.info("─" * len(header))

    for r in all_results:
        status = "OK" if not r.get("errors") else f"ERR({len(r['errors'])})"
        line = (
            f"{r['name']:<16} "
            f"{r.get('added', '?'):>6} "
            f"{r.get('count', '?'):>6} "
            f"{r.get('dedup_added', '?'):>6} "
            f"{r.get('tips', '?'):>5} "
            f"{r.get('insights', '?'):>8} "
            f"{r.get('emb_count', '?'):>5} "
            f"{r.get('save_time', 0):>7.3f}s "
            f"{r.get('reload_count', '?'):>7} "
            f"{status}"
        )
        logger.info(line)

    errors = [r for r in all_results if r.get("errors")]
    if errors:
        logger.warning(f"\nErrors found in {len(errors)} backend(s):")
        for r in errors:
            for e in r["errors"]:
                logger.warning(f"  [{r['name']}] {e}")
    else:
        logger.info("\nAll 4 storage backends passed all checks!")

    # Cleanup
    shutil.rmtree(tmpdir)
    logger.info(f"Cleaned up {tmpdir}")


if __name__ == "__main__":
    main()
