from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from annoy import AnnoyIndex


class ANNIndex:
    def __init__(self, vector_size: int, metric: str = "angular") -> None:
        self.vector_size = vector_size
        self.metric = metric
        self._index = AnnoyIndex(vector_size, metric)
        self._id_to_name: Dict[int, str] = {}
        self._name_to_id: Dict[str, int] = {}
        self._built = False

    def add(self, name: str, vector: Sequence[float]) -> None:
        idx = len(self._id_to_name)
        self._index.add_item(idx, list(vector))
        self._id_to_name[idx] = name
        self._name_to_id[name] = idx
        self._built = False

    def build(self, trees: int = 20) -> None:
        if self._id_to_name:
            self._index.build(trees)
            self._built = True

    def query_by_vector(self, vector: Sequence[float], top_k: int = 10) -> List[Tuple[str, float]]:
        if not self._built:
            self.build()
        if not self._id_to_name:
            return []
        ids, distances = self._index.get_nns_by_vector(list(vector), top_k, include_distances=True)
        return [(self._id_to_name[idx], 1 / (1 + dist)) for idx, dist in zip(ids, distances)]

    def save(self, index_path: Path, mapping_path: Path) -> None:
        if not self._built:
            self.build()
        self._index.save(str(index_path))
        mapping_path.write_text("\n".join(f"{idx},{name}" for idx, name in self._id_to_name.items()))

    def load(self, index_path: Path, mapping_path: Path) -> None:
        self._index.load(str(index_path))
        self._id_to_name = {}
        self._name_to_id = {}
        for line in mapping_path.read_text().splitlines():
            idx_text, name = line.split(",", 1)
            idx = int(idx_text)
            self._id_to_name[idx] = name
            self._name_to_id[name] = idx
        self._built = True
