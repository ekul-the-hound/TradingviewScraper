from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SimilarityBreakdown:
    ast_cosine: float
    primitive_jaccard: float
    tree_shape: float
    parameter_count: float
    final_score: float
    classification: str


def cosine_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    keys = sorted(set(a) | set(b))
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    mag_a = math.sqrt(sum(a.get(k, 0) ** 2 for k in keys))
    mag_b = math.sqrt(sum(b.get(k, 0) ** 2 for k in keys))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def jaccard_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    keys = sorted(set(a) | set(b))
    intersection = 0
    union = 0
    for key in keys:
        va = 1 if a.get(key, 0) else 0
        vb = 1 if b.get(key, 0) else 0
        intersection += 1 if va and vb else 0
        union += 1 if va or vb else 0
    return (intersection / union) if union else 1.0


def tree_shape_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    fields = ["depth", "branching_factor", "rolling_window_count", "cross_logic_count"]
    if not fields:
        return 1.0
    diffs = []
    for field in fields:
        va = a.get(field, 0.0)
        vb = b.get(field, 0.0)
        denom = max(abs(va), abs(vb), 1.0)
        diffs.append(1.0 - min(abs(va - vb) / denom, 1.0))
    return sum(diffs) / len(diffs)


def parameter_count_similarity(a: int, b: int) -> float:
    denom = max(a, b, 1)
    return 1.0 - (abs(a - b) / denom)


def classify_similarity(score: float) -> str:
    if score >= 0.95:
        return "near_duplicate"
    if 0.85 <= score < 0.95:
        return "variant"
    if 0.70 <= score < 0.85:
        return "same_family"
    return "structurally_distinct"


def compute_similarity(
    fingerprint_a: str,
    fingerprint_b: str,
    ast_vec_a: dict[str, int],
    ast_vec_b: dict[str, int],
    primitive_a: dict[str, int],
    primitive_b: dict[str, int],
    tree_shape_a: dict[str, float],
    tree_shape_b: dict[str, float],
    params_a: int,
    params_b: int,
) -> SimilarityBreakdown:
    if fingerprint_a == fingerprint_b:
        return SimilarityBreakdown(
            ast_cosine=1.0,
            primitive_jaccard=1.0,
            tree_shape=1.0,
            parameter_count=1.0,
            final_score=1.0,
            classification="near_duplicate",
        )

    ast_cosine = cosine_similarity(ast_vec_a, ast_vec_b)
    primitive_jaccard = jaccard_similarity(primitive_a, primitive_b)
    shape_score = tree_shape_similarity(tree_shape_a, tree_shape_b)
    param_score = parameter_count_similarity(params_a, params_b)

    final = 0.4 * ast_cosine + 0.3 * primitive_jaccard + 0.2 * shape_score + 0.1 * param_score
    return SimilarityBreakdown(
        ast_cosine=ast_cosine,
        primitive_jaccard=primitive_jaccard,
        tree_shape=shape_score,
        parameter_count=param_score,
        final_score=final,
        classification=classify_similarity(final),
    )
