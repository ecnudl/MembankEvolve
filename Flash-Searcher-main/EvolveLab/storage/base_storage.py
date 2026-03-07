"""
Abstract base class for memory storage backends.

The storage layer is responsible ONLY for persistence and basic CRUD operations
on MemoryUnit objects. It is orthogonal to:
  - Extraction: how MemoryUnits are created from trajectories
  - Retrieval:  how MemoryUnits are searched/ranked at query time
  - Management: how MemoryUnits are pruned, merged, or decayed over time
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..memory_schema import MemoryUnit, MemoryUnitType


class BaseMemoryStorage(ABC):
    """
    Abstract interface for memory storage backends.

    All implementations must support:
      1. Persist and load MemoryUnits
      2. Basic CRUD by ID
      3. Signature-based deduplication check
      4. Provide embedding matrix for the retrieval layer
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> bool:
        """
        Set up storage backend (create dirs, load existing data, build indices).

        Returns:
            True if initialization succeeded.
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """Flush all in-memory state to persistent storage."""
        pass

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    def add(self, units: List[MemoryUnit]) -> int:
        """
        Add new MemoryUnits to storage. Skips units whose signature already exists.

        Args:
            units: List of MemoryUnit objects to store.

        Returns:
            Number of units actually added (after dedup).
        """
        pass

    @abstractmethod
    def update(self, unit: MemoryUnit) -> bool:
        """
        Update an existing MemoryUnit (matched by id).

        Returns:
            True if the unit was found and updated.
        """
        pass

    @abstractmethod
    def delete(self, unit_id: str) -> bool:
        """
        Remove a MemoryUnit by id.

        Returns:
            True if the unit was found and removed.
        """
        pass

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    @abstractmethod
    def get(self, unit_id: str) -> Optional[MemoryUnit]:
        """Retrieve a single MemoryUnit by id, or None if not found."""
        pass

    @abstractmethod
    def get_all(
        self,
        active_only: bool = False,
        unit_type: Optional[MemoryUnitType] = None,
    ) -> List[MemoryUnit]:
        """
        Retrieve all stored MemoryUnits, optionally filtered.

        Args:
            active_only: If True, return only units with is_active=True.
            unit_type: If set, return only units of that type.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored MemoryUnits."""
        pass

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @abstractmethod
    def exists_signature(self, signature: str) -> bool:
        """Check whether a MemoryUnit with the given signature already exists."""
        pass

    # ------------------------------------------------------------------
    # Embedding access (for retrieval layer)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_embedding_index(
        self,
        active_only: bool = True,
    ) -> Tuple[Optional[np.ndarray], List[MemoryUnit]]:
        """
        Return the embedding matrix and corresponding MemoryUnit list.

        This is the primary interface for the retrieval layer to perform
        similarity search without knowing storage internals.

        Args:
            active_only: If True, only include active units with embeddings.

        Returns:
            (embedding_matrix, units) where embedding_matrix is shape (N, dim)
            and units[i] corresponds to embedding_matrix[i].
            Returns (None, []) if no embeddings are available.
        """
        pass
