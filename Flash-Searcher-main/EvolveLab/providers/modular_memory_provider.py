"""
ModularMemoryProvider — Adapter that bridges decoupled Storage + Retrieval
layers into the existing BaseMemoryProvider interface.

Extraction reuses existing prompt-based logic from prompt_based_memory_provider.
Storage and Retrieval are pluggable via config.

Config keys:
    enabled_prompts: List[str]       — ["tip", "insight", ...]
    storage_type: str                — "json" | "graph"
    retriever_type: str              — "semantic" | "keyword" | "hybrid" | "contrastive" | "graph"
    retriever_config: Dict           — strategy-specific params (weights, top_k, etc.)
    storage_dir: str                 — base dir for persistence
    top_k: int                       — max memories to retrieve (default 5)
    embedding_model_name: str        — sentence-transformers model id
    embedding_cache_dir: str         — local model cache
    prompt_dir: str                  — directory containing prompt .txt files

Usage:
    provider = ModularMemoryProvider(config={
        "enabled_prompts": ["tip", "insight"],
        "storage_type": "json",
        "retriever_type": "hybrid",
        "retriever_config": {"weights": {"SemanticRetriever": 0.7, "KeywordRetriever": 0.3}},
        "storage_dir": "./storage/modular_experiment_1",
    })
    provider.config["model"] = llm_model
    provider.initialize()
"""

import logging
import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryItem,
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    TrajectoryData,
)
from ..memory_schema import MemoryUnit, MemoryUnitType, split_extraction_output

from .prompt_based_memory_provider import (
    PROMPT_FILE_NAMES,
    PROMPT_TO_UNIT_TYPE,
    _build_template_context,
    _load_embedding_model,
    _parse_json_from_response,
    _render_prompt,
)

logger = logging.getLogger(__name__)


# Storage type → (class, module path, config builder)
_STORAGE_FACTORIES = {
    "json": lambda cfg: _make_json_storage(cfg),
    "graph": lambda cfg: _make_graph_storage(cfg),
}


def _make_json_storage(cfg):
    from ..storage import JsonStorage
    db_path = os.path.join(cfg["storage_dir"], "memory_db.json")
    return JsonStorage({"db_path": db_path})


def _make_graph_storage(cfg):
    from ..storage import GraphStore
    return GraphStore({"storage_dir": cfg["storage_dir"]})


def _make_retriever(retriever_type, store, embedding_model, retriever_config):
    """Create a retriever instance by type string."""
    from ..retrieval import (
        SemanticRetriever,
        KeywordRetriever,
        HybridRetriever,
        GraphRetriever,
        ContrastiveRetriever,
    )

    if retriever_type == "semantic":
        return SemanticRetriever(store, embedding_model, retriever_config)

    elif retriever_type == "keyword":
        return KeywordRetriever(store, retriever_config)

    elif retriever_type == "hybrid":
        semantic = SemanticRetriever(store, embedding_model, retriever_config)
        keyword = KeywordRetriever(store, retriever_config)
        return HybridRetriever(store, [semantic, keyword], retriever_config)

    elif retriever_type == "graph":
        return GraphRetriever(store, embedding_model, retriever_config)

    elif retriever_type == "contrastive":
        return ContrastiveRetriever(store, embedding_model, retriever_config)

    else:
        logger.warning(f"Unknown retriever_type '{retriever_type}', falling back to semantic")
        return SemanticRetriever(store, embedding_model, retriever_config)


class ModularMemoryProvider(BaseMemoryProvider):
    """
    Adapter provider that composes:
      - Extraction: reuses prompt_based_memory_provider logic
      - Storage: pluggable backend (JsonStorage / GraphStore)
      - Retrieval: pluggable strategy (5 options)

    Implements BaseMemoryProvider interface so it works with existing
    GAIA runner and agent framework unchanged.
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(MemoryType.MODULAR, config)

        self.model = self.config.get("model")

        # Environment variable overrides (same pattern as prompt_based)
        # MODULAR_STORAGE_DIR, MODULAR_STORAGE_TYPE, MODULAR_RETRIEVER_TYPE,
        # MODULAR_RETRIEVER_CONFIG (JSON), MODULAR_ENABLED_PROMPTS (comma-sep)
        self.storage_dir = (
            os.environ.get("MODULAR_STORAGE_DIR")
            or self.config.get("storage_dir", "./storage/modular")
        )
        self.storage_type = (
            os.environ.get("MODULAR_STORAGE_TYPE")
            or self.config.get("storage_type", "json")
        )
        self.retriever_type = (
            os.environ.get("MODULAR_RETRIEVER_TYPE")
            or self.config.get("retriever_type", "semantic")
        )

        # Retriever config: env var as JSON, or from config dict
        env_retriever_config = os.environ.get("MODULAR_RETRIEVER_CONFIG", "").strip()
        if env_retriever_config:
            import json as _json
            try:
                self.retriever_config = _json.loads(env_retriever_config)
            except Exception:
                logger.warning(f"Invalid MODULAR_RETRIEVER_CONFIG JSON, using config default")
                self.retriever_config = self.config.get("retriever_config", {})
        else:
            self.retriever_config = self.config.get("retriever_config", {})

        self.top_k = self.config.get("top_k", 5)

        # Injection threshold: memories below this relevance score are discarded
        env_min_rel = os.environ.get("MODULAR_MIN_RELEVANCE", "").strip()
        if env_min_rel:
            self.min_relevance = float(env_min_rel)
        else:
            self.min_relevance = float(self.config.get("min_relevance", 0.0))

        # Extraction config
        env_prompts = os.environ.get("MODULAR_ENABLED_PROMPTS", "").strip()
        if env_prompts:
            self.enabled_prompts = [p.strip() for p in env_prompts.split(",") if p.strip()]
        else:
            self.enabled_prompts: List[str] = self.config.get(
                "enabled_prompts", ["tip", "insight"]
            )
        self.prompt_dir = self.config.get("prompt_dir", ".")

        # Embedding model config
        self.embedding_model_name = self.config.get(
            "embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.embedding_cache_dir = self.config.get(
            "embedding_cache_dir", "./storage/models"
        )

        # Internal state (initialized during initialize())
        self.store = None
        self.retriever = None
        self.embedding_model = None
        self._prompt_templates: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            os.makedirs(self.storage_dir, exist_ok=True)

            # 1. Load embedding model
            self.embedding_model = _load_embedding_model(
                self.embedding_model_name, self.embedding_cache_dir
            )

            # 2. Create and initialize storage backend
            factory = _STORAGE_FACTORIES.get(self.storage_type)
            if factory is None:
                logger.error(f"Unknown storage_type: {self.storage_type}")
                return False

            self.store = factory({"storage_dir": self.storage_dir})
            if not self.store.initialize():
                logger.error("Storage backend initialization failed")
                return False

            # 3. Create retriever
            self.retriever = _make_retriever(
                self.retriever_type,
                self.store,
                self.embedding_model,
                self.retriever_config,
            )

            # 4. Load prompt templates
            for prompt_name in self.enabled_prompts:
                fname = PROMPT_FILE_NAMES.get(prompt_name)
                if not fname:
                    logger.warning(f"Unknown prompt name: {prompt_name}, skipping")
                    continue
                fpath = os.path.join(self.prompt_dir, fname)
                if not os.path.exists(fpath):
                    logger.warning(f"Prompt file not found: {fpath}, skipping")
                    continue
                with open(fpath, "r", encoding="utf-8") as f:
                    self._prompt_templates[prompt_name] = f.read()

            logger.info(
                f"ModularMemoryProvider initialized: "
                f"storage={self.storage_type}, retriever={self.retriever_type}, "
                f"prompts={list(self._prompt_templates.keys())}, "
                f"existing_units={self.store.count()}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ModularMemoryProvider: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Memory ingestion (reuses prompt_based extraction logic)
    # ------------------------------------------------------------------

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple:
        if not self.model:
            return False, "No model provided for memory extraction"

        if not self._prompt_templates:
            return False, "No prompt templates loaded"

        metadata = trajectory_data.metadata or {}
        is_correct = metadata.get("is_correct", False)
        task_outcome = "success" if is_correct else "failure"
        task_id = metadata.get("task_id", str(uuid.uuid4())[:8])

        context = _build_template_context(trajectory_data, is_correct)

        new_units: List[MemoryUnit] = []
        prompts_used = []

        for prompt_name, template_str in self._prompt_templates.items():
            unit_type = PROMPT_TO_UNIT_TYPE.get(prompt_name)
            if unit_type is None:
                continue

            # Skip conditions
            if prompt_name == "insight" and is_correct:
                continue
            if prompt_name == "workflow" and not is_correct:
                continue

            try:
                filled_prompt = _render_prompt(template_str, context)
            except Exception as e:
                logger.error(f"Template rendering failed for {prompt_name}: {e}")
                continue

            try:
                messages = [
                    {"role": "user", "content": [{"type": "text", "text": filled_prompt}]}
                ]
                response = self.model(messages)
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )
            except Exception as e:
                logger.error(f"LLM call failed for {prompt_name}: {e}")
                continue

            parsed = _parse_json_from_response(response_text)
            if parsed is None:
                logger.warning(f"Failed to parse extraction result for {prompt_name}")
                continue

            if isinstance(parsed, dict) and parsed.get("skipped"):
                continue

            try:
                units = split_extraction_output(
                    extraction_result=parsed,
                    unit_type=unit_type,
                    source_task_id=task_id,
                    source_task_query=trajectory_data.query,
                    task_outcome=task_outcome,
                    extraction_model=str(getattr(self.model, "model_id", "unknown")),
                )
            except Exception as e:
                logger.error(f"split_extraction_output failed for {prompt_name}: {e}")
                continue

            # Dedup + embed + add to storage
            for unit in units:
                if self.store.exists_signature(unit.signature):
                    continue

                text = unit.content_text()
                if text and self.embedding_model is not None:
                    unit.embedding = self.embedding_model.encode(
                        text, convert_to_numpy=True
                    )

                new_units.append(unit)

            prompts_used.append(prompt_name)

        # Batch add to storage
        if new_units:
            if self.storage_type == "graph" and hasattr(self.store, 'upsert_memory_unit'):
                # For GraphStore: extract entities and use upsert_memory_unit
                from ..storage.graph_storage import extract_entities_from_unit
                added = 0
                for unit in new_units:
                    entities = extract_entities_from_unit(unit)
                    self.store.upsert_memory_unit(unit, entities=entities)
                    added += 1
            else:
                added = self.store.add(new_units)
            self.store.save()
            logger.info(
                f"ModularMemoryProvider: added {added} units from "
                f"{', '.join(prompts_used)} (total: {self.store.count()})"
            )

        msg = (
            f"Extracted {len(new_units)} units from "
            f"{len(prompts_used)} prompts ({', '.join(prompts_used)})"
        )
        return True, msg

    # ------------------------------------------------------------------
    # Memory retrieval (delegates to Retriever → MemoryPack)
    # ------------------------------------------------------------------

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        empty_response = MemoryResponse(
            memories=[],
            memory_type=self.memory_type,
            total_count=0,
            request_id=str(uuid.uuid4()),
        )

        # Only provide at BEGIN phase
        if request.status != MemoryStatus.BEGIN:
            return empty_response

        if self.retriever is None or self.store is None:
            return empty_response

        if self.store.count() == 0:
            return empty_response

        # Build query context
        from ..retrieval import QueryContext

        query_emb = None
        if self.embedding_model is not None:
            query_emb = self.embedding_model.encode(
                request.query, convert_to_numpy=True
            )

        ctx = QueryContext(
            query=request.query,
            embedding=query_emb,
        )

        # Retrieve
        pack = self.retriever.retrieve(ctx, top_k=self.top_k)

        if pack.is_empty():
            return empty_response

        # Apply min_relevance threshold: discard low-relevance memories
        if self.min_relevance > 0:
            original_count = len(pack.scored_units)
            pack.scored_units = [
                su for su in pack.scored_units if su.score >= self.min_relevance
            ]
            filtered_count = original_count - len(pack.scored_units)
            if filtered_count > 0:
                logger.info(
                    f"min_relevance={self.min_relevance}: filtered {filtered_count}/{original_count} "
                    f"low-relevance memories, {len(pack.scored_units)} remaining"
                )
            if pack.is_empty():
                logger.info("All memories below min_relevance threshold, returning empty")
                return empty_response

        # Convert MemoryPack → guidance text → MemoryResponse
        guidance_text = pack.to_guidance_text()

        memory_item = MemoryItem(
            id=f"modular_{uuid.uuid4()}",
            content=guidance_text,
            metadata={
                "retriever": pack.retriever_name,
                "num_units": len(pack.scored_units),
                "by_type": {k: len(v) for k, v in pack.by_type.items()},
                "total_memory_units": self.store.count(),
            },
            score=float(np.mean([su.score for su in pack.scored_units])) if pack.scored_units else 0.0,
        )

        logger.info(
            f"provide_memory: {pack.retriever_name} returned {len(pack.scored_units)} units "
            f"(query='{request.query[:60]}...', total={self.store.count()})"
        )

        return MemoryResponse(
            memories=[memory_item],
            memory_type=self.memory_type,
            total_count=1,
            request_id=str(uuid.uuid4()),
        )
