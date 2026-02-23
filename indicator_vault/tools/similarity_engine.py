from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ast_normalizer import cosine_similarity, jaccard_similarity, shape_similarity


@dataclass(frozen=True)
class SimilarityResult:
    similarity_score: float
    relation: str


class SimilarityEngine:
    def __init__(self) -> None:
        self.weights = {
            "ast_cosine": 0.4,
            "primitive_jaccard": 0.3,
            "shape": 0.2,
            "parameter_count": 0.1,
        }

    def parameter_similarity(self, left_count: int, right_count: int) -> float:
        denom = max(left_count, right_count, 1)
        return 1.0 - min(abs(left_count - right_count) / denom, 1.0)

    def score(self, left: dict[str, Any], right: dict[str, Any]) -> SimilarityResult:
        if left.get("fingerprint") == right.get("fingerprint"):
            return SimilarityResult(similarity_score=1.0, relation="near_duplicate")

        ast_cosine = cosine_similarity(left.get("counts", {}), right.get("counts", {}))
        primitive_jaccard = jaccard_similarity(left.get("primitives", {}), right.get("primitives", {}))
        shape = shape_similarity(left.get("shape", {}), right.get("shape", {}))
        parameter = self.parameter_similarity(int(left.get("parameter_count", 0)), int(right.get("parameter_count", 0)))

        score = (
            self.weights["ast_cosine"] * ast_cosine
            + self.weights["primitive_jaccard"] * primitive_jaccard
            + self.weights["shape"] * shape
            + self.weights["parameter_count"] * parameter
        )

        relation = self.tag_relation(score)
        return SimilarityResult(similarity_score=score, relation=relation)

    @staticmethod
    def tag_relation(score: float) -> str:
        if score >= 0.95:
            return "near_duplicate"
        if score >= 0.85:
            return "variant"
        if score >= 0.70:
            return "same_family"
        return "structurally_distinct"
