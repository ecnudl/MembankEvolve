"""
VectorStorage — FAISS-indexed vector storage backend for MemoryUnit.

Stores embeddings in a FAISS index for efficient similarity search,
with a JSON sidecar file for MemoryUnit metadata. This backend is optimized
for large-scale embedding-based retrieval.

Config:
    storage_dir: str          — Directory for index + metadata files
    embedding_dim: int        — Embedding dimension (default: 384 for all-MiniLM-L6-v2)
    index_type: str           — FAISS index type: "flat", "ivfflat" (default: "flat")
    nlist: int                — Number of Voronoi cells for IVFFlat (default: 100)
    nprobe: int               — Number of cells to probe at search time (default: 10)

Files created:
    <storage_dir>/faiss.index      — FAISS binary index
    <storage_dir>/metadata.json    — MemoryUnit metadata (everything except embedding)
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
    """Lazy import faiss to avoid hard dependency at module level."""
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "VectorStorage requires the 'faiss-cpu' or 'faiss-gpu' package. "
            "Install with: pip install faiss-cpu"
        )


class VectorStorage(BaseMemoryStorage):
    """
    FAISS-indexed vector storage.

    Embeddings are stored in a FAISS index for O(1) or O(log N) similarity
    search. Metadata (all MemoryUnit fields except embedding) is stored in
    a JSON sidecar file. A mapping from FAISS internal IDs to MemoryUnit IDs
    is maintained for consistent lookup.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.storage_dir: str = self.config.get(
            "storage_dir", "./storage/vector"
        )
        self.embedding_dim: int = self.config.get("embedding_dim", 384)
        self.index_type: str = self.config.get("index_type", "flat")
        self.nlist: int = self.config.get("nlist", 100)
        self.nprobe: int = self.config.get("nprobe", 10)

        self._index_path = os.path.join(self.storage_dir, "faiss.index")
        self._meta_path = os.path.join(self.storage_dir, "metadata.json")

        # In-memory state
        self._index = None                          # faiss.Index
        self._units: List[MemoryUnit] = []          # ordered by FAISS internal position
        self._id_to_pos: Dict[str, int] = {}        # unit.id -> position in _units
        self._sig_set: set = set()                   # signatures for dedup

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            faiss = _get_faiss()
            os.makedirs(self.storage_dir, exist_ok=True)

            if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
                self._load(faiss)
            else:
                self._index = self._create_index(faiss)
                self._units = []
                self._id_to_pos = {}
                self._sig_set = set()

            logger.info(
                f"VectorStorage initialized: {len(self._units)} units, "
                f"index_type={self.index_type}, dim={self.embedding_dim}"
            )
            return True
        except Exception as e:
            logger.error(f"VectorStorage initialization failed: {e}")
            return False

    def _create_index(self, faiss):
        """Create a new FAISS index based on config."""
        if self.index_type == "ivfflat":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            index.nprobe = self.nprobe
            return index
        else:
            # Default: flat index with inner product (cosine sim on normalized vectors)
            return faiss.IndexFlatIP(self.embedding_dim)

    def save(self) -> None:
        faiss = _get_faiss()
        os.makedirs(self.storage_dir, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, self._index_path)

        # Save metadata (MemoryUnits without embeddings to avoid duplication)
        meta_list = []
        for unit in self._units:
            d = unit.to_dict()
            d.pop("embedding", None)  # stored in FAISS, not in JSON
            meta_list.append(d)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_list, f, indent=2, ensure_ascii=False)

    def _load(self, faiss) -> None:
        """Load existing index and metadata from disk."""
        self._index = faiss.read_index(self._index_path)

        with open(self._meta_path, "r", encoding="utf-8") as f:
            meta_list = json.load(f)

        self._units = []
        self._id_to_pos = {}
        self._sig_set = set()

        for i, d in enumerate(meta_list):
            # Reconstruct embedding from FAISS index
            emb = np.zeros(self.embedding_dim, dtype=np.float32)
            try:
                self._index.reconstruct(i, emb)
                d["embedding"] = emb.tolist()
            except RuntimeError:
                d["embedding"] = None

            unit = MemoryUnit.from_dict(d)
            self._units.append(unit)
            self._id_to_pos[unit.id] = i
            if unit.signature:
                self._sig_set.add(unit.signature)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, units: List[MemoryUnit]) -> int:
        faiss = _get_faiss()
        added = 0

        vectors_to_add = []
        units_to_add = []

        for unit in units:
            if unit.signature and unit.signature in self._sig_set:
                logger.debug(f"Duplicate signature {unit.signature}, skipping")
                continue
            if unit.embedding is None:
                logger.warning(f"Unit {unit.id} has no embedding, skipping for VectorStorage")
                continue

            units_to_add.append(unit)
            # Normalize for cosine similarity via inner product
            emb = unit.embedding.astype(np.float32).copy()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm
            vectors_to_add.append(emb)

        if not vectors_to_add:
            return 0

        emb_matrix = np.vstack(vectors_to_add).astype(np.float32)

        # Train IVF index if needed and not yet trained
        if self.index_type == "ivfflat" and not self._index.is_trained:
            self._index.train(emb_matrix)

        self._index.add(emb_matrix)

        for unit in units_to_add:
            pos = len(self._units)
            self._units.append(unit)
            self._id_to_pos[unit.id] = pos
            if unit.signature:
                self._sig_set.add(unit.signature)
            added += 1

        if added > 0:
            self.save()
        return added

    def update(self, unit: MemoryUnit) -> bool:
        pos = self._id_to_pos.get(unit.id)
        if pos is None:
            return False

        old_sig = self._units[pos].signature
        if old_sig:
            self._sig_set.discard(old_sig)

        self._units[pos] = unit
        if unit.signature:
            self._sig_set.add(unit.signature)

        # FAISS flat index does not support in-place update.
        # Rebuild the index to reflect the updated embedding.
        if unit.embedding is not None:
            self._rebuild_faiss_index()

        self.save()
        return True

    def delete(self, unit_id: str) -> bool:
        pos = self._id_to_pos.get(unit_id)
        if pos is None:
            return False

        unit = self._units[pos]
        if unit.signature:
            self._sig_set.discard(unit.signature)

        self._units.pop(pos)

        # Rebuild position mapping and FAISS index
        self._id_to_pos = {u.id: i for i, u in enumerate(self._units)}
        self._rebuild_faiss_index()
        self.save()
        return True

    def _rebuild_faiss_index(self) -> None:
        """Rebuild the entire FAISS index from current units."""
        faiss = _get_faiss()
        self._index = self._create_index(faiss)

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
            if self.index_type == "ivfflat" and not self._index.is_trained:
                self._index.train(emb_matrix)
            self._index.add(emb_matrix)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, unit_id: str) -> Optional[MemoryUnit]:
        pos = self._id_to_pos.get(unit_id)
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

        # Reconstruct from unit embeddings (already in memory)
        emb_matrix = np.vstack([u.embedding for u in units])
        return emb_matrix, units

    # ------------------------------------------------------------------
    # FAISS-native search (bonus: direct similarity search)
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        active_only: bool = True,
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        Perform FAISS-native similarity search.

        This is a convenience method that bypasses the retrieval layer for
        direct vector search. The retrieval layer can also use
        get_embedding_index() for its own search logic.

        Args:
            query_embedding: Query vector, shape (dim,).
            top_k: Number of results to return.
            active_only: If True, filter out inactive units from results.

        Returns:
            List of (MemoryUnit, score) tuples sorted by descending similarity.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Normalize query
        qe = query_embedding.astype(np.float32).copy().reshape(1, -1)
        norm = np.linalg.norm(qe)
        if norm > 0:
            qe /= norm

        # Search more than top_k to allow filtering
        search_k = min(top_k * 3, self._index.ntotal)
        scores, indices = self._index.search(qe, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._units):
                continue
            unit = self._units[idx]
            if active_only and not unit.is_active:
                continue
            results.append((unit, float(score)))
            if len(results) >= top_k:
                break

        return results
