"""
HybridStorage — JSON metadata + FAISS vector index combined storage backend.

Combines the full-fidelity JSON storage of MemoryUnit metadata with a FAISS
index for efficient vector similarity search. Supports structured filtering
(by type, active status, domains, etc.) combined with vector search in a
single query path.

Config:
    storage_dir: str          — Directory for all storage files
    embedding_dim: int        — Embedding dimension (default: 384)
    index_type: str           — FAISS index type: "flat", "ivfflat" (default: "flat")
    nlist: int                — IVFFlat Voronoi cells (default: 100)
    nprobe: int               — IVFFlat probe count (default: 10)

Files created:
    <storage_dir>/memory_db.json   — Full MemoryUnit data (with embeddings as lists)
    <storage_dir>/faiss.index      — FAISS binary index (synced with JSON)
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..memory_schema import MemoryUnit, MemoryUnitType
from .base_storage import BaseMemoryStorage

logger = logging.getLogger(__name__)


def _get_faiss():
    """Lazy import faiss."""
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "HybridStorage requires the 'faiss-cpu' or 'faiss-gpu' package. "
            "Install with: pip install faiss-cpu"
        )


class HybridStorage(BaseMemoryStorage):
    """
    Hybrid JSON + FAISS storage.

    Maintains two synchronized data stores:
      1. JSON file: Full MemoryUnit serialization (human-readable, supports
         structured queries and metadata filtering)
      2. FAISS index: Normalized embeddings for fast similarity search

    The JSON store is the source of truth. The FAISS index is always
    rebuildable from JSON data and is treated as a derived acceleration
    structure.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.storage_dir: str = self.config.get(
            "storage_dir", "./storage/hybrid"
        )
        self.embedding_dim: int = self.config.get("embedding_dim", 384)
        self.index_type: str = self.config.get("index_type", "flat")
        self.nlist: int = self.config.get("nlist", 100)
        self.nprobe: int = self.config.get("nprobe", 10)

        self._json_path = os.path.join(self.storage_dir, "memory_db.json")
        self._index_path = os.path.join(self.storage_dir, "faiss.index")

        # In-memory state
        self._units: List[MemoryUnit] = []
        self._id_index: Dict[str, int] = {}     # unit.id -> list position
        self._sig_set: set = set()
        self._faiss_index = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            faiss = _get_faiss()
            os.makedirs(self.storage_dir, exist_ok=True)

            # Always load from JSON (source of truth)
            if os.path.exists(self._json_path):
                self._load_json()
            else:
                self._units = []

            self._rebuild_indices()

            # Build or rebuild FAISS index
            self._rebuild_faiss_index(faiss)

            logger.info(
                f"HybridStorage initialized: {len(self._units)} units, "
                f"FAISS entries={self._faiss_index.ntotal if self._faiss_index else 0}"
            )
            return True
        except Exception as e:
            logger.error(f"HybridStorage initialization failed: {e}")
            return False

    def save(self) -> None:
        faiss = _get_faiss()
        os.makedirs(self.storage_dir, exist_ok=True)

        # Save JSON (source of truth, includes embeddings)
        data = [u.to_dict() for u in self._units]
        with open(self._json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save FAISS index
        if self._faiss_index is not None:
            faiss.write_index(self._faiss_index, self._index_path)

    def _load_json(self) -> None:
        with open(self._json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._units = [MemoryUnit.from_dict(d) for d in data]

    def _rebuild_indices(self) -> None:
        self._id_index = {u.id: i for i, u in enumerate(self._units)}
        self._sig_set = {u.signature for u in self._units if u.signature}

    def _create_faiss_index(self, faiss):
        if self.index_type == "ivfflat":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.nprobe = self.nprobe
            return index
        else:
            return faiss.IndexFlatIP(self.embedding_dim)

    def _rebuild_faiss_index(self, faiss=None) -> None:
        if faiss is None:
            faiss = _get_faiss()

        self._faiss_index = self._create_faiss_index(faiss)

        embs = []
        for unit in self._units:
            if unit.embedding is not None:
                emb = unit.embedding.astype(np.float32).copy()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb /= norm
                embs.append(emb)
            else:
                embs.append(np.zeros(self.embedding_dim, dtype=np.float32))

        if embs:
            emb_matrix = np.vstack(embs).astype(np.float32)
            if self.index_type == "ivfflat" and not self._faiss_index.is_trained:
                self._faiss_index.train(emb_matrix)
            self._faiss_index.add(emb_matrix)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, units: List[MemoryUnit]) -> int:
        faiss = _get_faiss()
        added = 0
        new_embs = []

        for unit in units:
            if unit.signature and unit.signature in self._sig_set:
                logger.debug(f"Duplicate signature {unit.signature}, skipping")
                continue

            pos = len(self._units)
            self._units.append(unit)
            self._id_index[unit.id] = pos
            if unit.signature:
                self._sig_set.add(unit.signature)

            # Prepare embedding for FAISS
            if unit.embedding is not None:
                emb = unit.embedding.astype(np.float32).copy()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb /= norm
                new_embs.append(emb)
            else:
                new_embs.append(np.zeros(self.embedding_dim, dtype=np.float32))

            added += 1

        # Batch-add to FAISS
        if new_embs:
            emb_matrix = np.vstack(new_embs).astype(np.float32)
            if self.index_type == "ivfflat" and not self._faiss_index.is_trained:
                self._faiss_index.train(emb_matrix)
            self._faiss_index.add(emb_matrix)

        if added > 0:
            self.save()
        return added

    def update(self, unit: MemoryUnit) -> bool:
        pos = self._id_index.get(unit.id)
        if pos is None:
            return False

        old_sig = self._units[pos].signature
        if old_sig:
            self._sig_set.discard(old_sig)

        self._units[pos] = unit
        if unit.signature:
            self._sig_set.add(unit.signature)

        # Rebuild FAISS if embedding changed
        self._rebuild_faiss_index()
        self.save()
        return True

    def delete(self, unit_id: str) -> bool:
        pos = self._id_index.get(unit_id)
        if pos is None:
            return False

        unit = self._units[pos]
        if unit.signature:
            self._sig_set.discard(unit.signature)

        self._units.pop(pos)
        self._rebuild_indices()
        self._rebuild_faiss_index()
        self.save()
        return True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, unit_id: str) -> Optional[MemoryUnit]:
        pos = self._id_index.get(unit_id)
        if pos is None:
            return None
        return self._units[pos]

    def get_all(
        self,
        active_only: bool = False,
        unit_type: Optional[MemoryUnitType] = None,
    ) -> List[MemoryUnit]:
        result = self._units
        if active_only:
            result = [u for u in result if u.is_active]
        if unit_type is not None:
            result = [u for u in result if u.type == unit_type]
        return result

    def count(self) -> int:
        return len(self._units)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def exists_signature(self, signature: str) -> bool:
        return signature in self._sig_set

    # ------------------------------------------------------------------
    # Embedding access
    # ------------------------------------------------------------------

    def get_embedding_index(
        self,
        active_only: bool = True,
    ) -> Tuple[Optional[np.ndarray], List[MemoryUnit]]:
        if active_only:
            units = [u for u in self._units if u.is_active and u.embedding is not None]
        else:
            units = [u for u in self._units if u.embedding is not None]

        if not units:
            return None, []

        emb_matrix = np.vstack([u.embedding for u in units])
        return emb_matrix, units

    # ------------------------------------------------------------------
    # Hybrid search (structured filter + vector similarity)
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        active_only: bool = True,
        unit_type: Optional[MemoryUnitType] = None,
        min_confidence: float = 0.0,
        domains: Optional[List[str]] = None,
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        Hybrid search: FAISS vector similarity with post-hoc structured filtering.

        The search retrieves a larger candidate set from FAISS, then applies
        metadata filters (type, active status, confidence, domains) to produce
        the final top-k results.

        Args:
            query_embedding: Query vector, shape (dim,).
            top_k: Number of results to return after filtering.
            active_only: Filter out inactive units.
            unit_type: Filter by MemoryUnitType.
            min_confidence: Minimum confidence threshold.
            domains: If set, unit must have at least one matching domain.

        Returns:
            List of (MemoryUnit, score) tuples sorted by descending similarity.
        """
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []

        # Normalize query
        qe = query_embedding.astype(np.float32).copy().reshape(1, -1)
        norm = np.linalg.norm(qe)
        if norm > 0:
            qe /= norm

        # Retrieve more candidates to allow filtering
        search_k = min(top_k * 5, self._faiss_index.ntotal)
        scores, indices = self._faiss_index.search(qe, search_k)

        results = []
        domain_set = set(domains) if domains else None

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._units):
                continue

            unit = self._units[idx]

            # Apply filters
            if active_only and not unit.is_active:
                continue
            if unit_type is not None and unit.type != unit_type:
                continue
            if unit.confidence < min_confidence:
                continue
            if domain_set and not domain_set.intersection(unit.applicable_domains):
                continue

            results.append((unit, float(score)))
            if len(results) >= top_k:
                break

        return results

    def filtered_get(
        self,
        active_only: bool = True,
        unit_type: Optional[MemoryUnitType] = None,
        min_confidence: float = 0.0,
        task_outcome: Optional[str] = None,
        domains: Optional[List[str]] = None,
    ) -> List[MemoryUnit]:
        """
        Structured metadata-only query without vector similarity.

        Useful for management operations (pruning, statistics, etc.)
        that need to filter by metadata fields.
        """
        result = self._units
        if active_only:
            result = [u for u in result if u.is_active]
        if unit_type is not None:
            result = [u for u in result if u.type == unit_type]
        if min_confidence > 0:
            result = [u for u in result if u.confidence >= min_confidence]
        if task_outcome is not None:
            result = [u for u in result if u.task_outcome == task_outcome]
        if domains:
            domain_set = set(domains)
            result = [
                u for u in result
                if domain_set.intersection(u.applicable_domains)
            ]
        return result
