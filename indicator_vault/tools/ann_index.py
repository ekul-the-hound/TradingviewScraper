from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnnCandidate:
    name: str
    score: float


class AnnIndex:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._vectors: list[tuple[str, list[float]]] = []

    def add(self, name: str, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError("invalid_vector_dimension")
        self._vectors.append((name, vector))

    def query(self, vector: list[float], top_k: int = 10) -> list[AnnCandidate]:
        if len(vector) != self.dimension:
            raise ValueError("invalid_vector_dimension")

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(y * y for y in b) ** 0.5
            if norm_a == 0.0 or norm_b == 0.0:
                return 0.0
            return dot / (norm_a * norm_b)

        ranked = sorted(
            (AnnCandidate(name=n, score=cosine(vector, v)) for n, v in self._vectors),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:top_k]
