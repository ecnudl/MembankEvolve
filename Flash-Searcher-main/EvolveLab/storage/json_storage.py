"""
JsonStorage — Pure JSON file storage backend for MemoryUnit.

Stores all MemoryUnit data (including embeddings as float lists) in a single
JSON file. Simple, human-readable, and zero-dependency beyond stdlib + numpy.

Config:
    db_path: str  — Path to the JSON file (default: ./storage/json/memory_db.json)

This is the simplest storage backend and mirrors the original storage approach
used in PromptBasedMemoryProvider.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..memory_schema import MemoryUnit, MemoryUnitType
from .base_storage import BaseMemoryStorage

logger = logging.getLogger(__name__)


class JsonStorage(BaseMemoryStorage):
    """
    Pure JSON file storage.

    All MemoryUnits are kept in-memory as a list and serialized to a single
    JSON file on save(). Embeddings are stored as float arrays within each
    unit's JSON representation.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.db_path: str = self.config.get("db_path", "./storage/json/memory_db.json")
        self._units: List[MemoryUnit] = []
        self._id_index: Dict[str, int] = {}       # id -> list index
        self._sig_index: Dict[str, str] = {}       # signature -> id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._load()
            logger.info(
                f"JsonStorage initialized: {len(self._units)} units from {self.db_path}"
            )
            return True
        except Exception as e:
            logger.error(f"JsonStorage initialization failed: {e}")
            return False

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        data = [u.to_dict() for u in self._units]
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load(self) -> None:
        if not os.path.exists(self.db_path):
            self._units = []
            self._rebuild_indices()
            return
        with open(self.db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._units = [MemoryUnit.from_dict(d) for d in data]
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        self._id_index = {u.id: i for i, u in enumerate(self._units)}
        self._sig_index = {u.signature: u.id for u in self._units if u.signature}

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, units: List[MemoryUnit]) -> int:
        added = 0
        for unit in units:
            if unit.signature and unit.signature in self._sig_index:
                logger.debug(f"Duplicate signature {unit.signature}, skipping")
                continue
            idx = len(self._units)
            self._units.append(unit)
            self._id_index[unit.id] = idx
            if unit.signature:
                self._sig_index[unit.signature] = unit.id
            added += 1
        if added > 0:
            self.save()
        return added

    def update(self, unit: MemoryUnit) -> bool:
        idx = self._id_index.get(unit.id)
        if idx is None:
            return False
        old_sig = self._units[idx].signature
        if old_sig and old_sig in self._sig_index:
            del self._sig_index[old_sig]
        self._units[idx] = unit
        if unit.signature:
            self._sig_index[unit.signature] = unit.id
        self.save()
        return True

    def delete(self, unit_id: str) -> bool:
        idx = self._id_index.get(unit_id)
        if idx is None:
            return False
        unit = self._units[idx]
        if unit.signature and unit.signature in self._sig_index:
            del self._sig_index[unit.signature]
        self._units.pop(idx)
        self._rebuild_indices()
        self.save()
        return True

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, unit_id: str) -> Optional[MemoryUnit]:
        idx = self._id_index.get(unit_id)
        if idx is None:
            return None
        return self._units[idx]

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
        return signature in self._sig_index

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
