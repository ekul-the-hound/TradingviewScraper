from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class AnnIndex:
    def __init__(self, dimension: int, storage_path: Path) -> None:
        self.dimension = dimension
        self.storage_path = storage_path
        self._vectors: list[np.ndarray] = []
        self._ids: list[str] = []

    def add(self, indicator_id: str, vector: Iterable[float]) -> None:
        arr = np.asarray(list(vector), dtype=float)
        if arr.shape[0] != self.dimension:
            raise ValueError(f"Expected vector dimension {self.dimension}, got {arr.shape[0]}")
        self._ids.append(indicator_id)
        self._vectors.append(arr)

    def query(self, vector: Iterable[float], top_k: int = 10) -> list[tuple[str, float]]:
        if not self._vectors:
            return []
        target = np.asarray(list(vector), dtype=float)
        all_vecs = np.vstack(self._vectors)
        norms = np.linalg.norm(all_vecs, axis=1) * np.linalg.norm(target)
        norms = np.where(norms == 0, 1e-12, norms)
        sims = (all_vecs @ target) / norms
        idx = np.argsort(-sims)[:top_k]
        return [(self._ids[i], float(sims[i])) for i in idx]

    def save(self) -> None:
        if not self._vectors:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.storage_path, ids=np.array(self._ids), vectors=np.vstack(self._vectors))

    @classmethod
    def load(cls, dimension: int, storage_path: Path) -> "AnnIndex":
        instance = cls(dimension=dimension, storage_path=storage_path)
        if not storage_path.exists():
            return instance
        data = np.load(storage_path, allow_pickle=True)
        instance._ids = list(data["ids"])
        instance._vectors = [row for row in data["vectors"]]
        return instance
