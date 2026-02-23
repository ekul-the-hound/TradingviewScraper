from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Iterable

from .ast_normalizer import normalize_code

PRIMITIVES = [
    "ema",
    "sma",
    "rsi",
    "atr",
    "std",
    "highest",
    "lowest",
    "crossover",
    "volatility",
    "momentum",
]


@dataclass(frozen=True)
class FeatureSet:
    frequency_vector: dict[str, int]
    primitive_signature: dict[str, int]
    tree_shape: dict[str, float]
    parameter_count: int


@dataclass(frozen=True)
class SimilarityResult:
    similarity_score: float
    tag: str


class FeatureExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.freq = {
            "add": 0,
            "sub": 0,
            "mul": 0,
            "div": 0,
            "rolling": 0,
            "ewm": 0,
            "comparisons": 0,
            "logical": 0,
            "shift": 0,
            "rolling_mean": 0,
            "rolling_std": 0,
            "gt": 0,
            "lt": 0,
        }
        self.primitives = {name: 0 for name in PRIMITIVES}
        self.max_depth = 0
        self.node_count = 0
        self.branching_total = 0
        self.cross_logic_count = 0
        self.parameter_count = 0

    def extract(self, tree: ast.AST) -> FeatureSet:
        self.visit(tree)
        branching_factor = self.branching_total / max(self.node_count, 1)
        return FeatureSet(
            frequency_vector=self.freq,
            primitive_signature=self.primitives,
            tree_shape={
                "depth": float(self.max_depth),
                "branching_factor": float(branching_factor),
                "rolling_count": float(self.freq["rolling"]),
                "cross_logic_count": float(self.cross_logic_count),
            },
            parameter_count=self.parameter_count,
        )

    def generic_visit(self, node: ast.AST, depth: int = 0) -> None:  # type: ignore[override]
        self.node_count += 1
        self.max_depth = max(self.max_depth, depth)
        children = list(ast.iter_child_nodes(node))
        self.branching_total += len(children)
        for child in children:
            self.generic_visit(child, depth + 1)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.parameter_count = len(node.args.args) + len(node.args.kwonlyargs)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Add):
            self.freq["add"] += 1
        elif isinstance(node.op, ast.Sub):
            self.freq["sub"] += 1
        elif isinstance(node.op, ast.Mult):
            self.freq["mul"] += 1
        elif isinstance(node.op, ast.Div):
            self.freq["div"] += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.freq["logical"] += 1
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.freq["comparisons"] += 1
        for op in node.ops:
            if isinstance(op, (ast.Gt, ast.GtE)):
                self.freq["gt"] += 1
            if isinstance(op, (ast.Lt, ast.LtE)):
                self.freq["lt"] += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = ""
        if isinstance(node.func, ast.Attribute):
            target = node.func.attr.lower()
        elif isinstance(node.func, ast.Name):
            target = node.func.id.lower()

        if target == "rolling":
            self.freq["rolling"] += 1
        if target == "ewm":
            self.freq["ewm"] += 1
        if target == "shift":
            self.freq["shift"] += 1
        if target == "mean" and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Attribute):
                if node.func.value.func.attr.lower() == "rolling":
                    self.freq["rolling_mean"] += 1
        if target == "std" and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Attribute):
                if node.func.value.func.attr.lower() == "rolling":
                    self.freq["rolling_std"] += 1

        for primitive in self.primitives:
            if primitive in target:
                self.primitives[primitive] = 1

        if target in {"crossover", "crossunder"}:
            self.cross_logic_count += 1

        self.generic_visit(node)


def cosine_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    keys = sorted(set(a) | set(b))
    dot = sum(float(a.get(k, 0) * b.get(k, 0)) for k in keys)
    norm_a = math.sqrt(sum(float(a.get(k, 0) ** 2) for k in keys))
    norm_b = math.sqrt(sum(float(b.get(k, 0) ** 2) for k in keys))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    a_set = {k for k, v in a.items() if v}
    b_set = {k for k, v in b.items() if v}
    union = a_set | b_set
    if not union:
        return 1.0
    return len(a_set & b_set) / len(union)


def tree_shape_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    scores = []
    for key in ["depth", "branching_factor", "rolling_count", "cross_logic_count"]:
        av = a.get(key, 0.0)
        bv = b.get(key, 0.0)
        denom = max(abs(av), abs(bv), 1.0)
        scores.append(1.0 - (abs(av - bv) / denom))
    return max(0.0, min(1.0, sum(scores) / len(scores)))


def parameter_count_similarity(a: int, b: int) -> float:
    denom = max(a, b, 1)
    return 1.0 - (abs(a - b) / denom)


def classify_similarity(score: float) -> str:
    if score >= 0.95:
        return "near_duplicate"
    if score >= 0.85:
        return "variant"
    if score >= 0.70:
        return "same_family"
    return "structurally_distinct"


def extract_features(code: str) -> tuple[str, FeatureSet]:
    normalized = normalize_code(code)
    extractor = FeatureExtractor()
    features = extractor.extract(normalized.normalized_tree)
    return normalized.fingerprint, features


def similarity_between(code_a: str, code_b: str) -> SimilarityResult:
    fp_a, features_a = extract_features(code_a)
    fp_b, features_b = extract_features(code_b)
    if fp_a == fp_b:
        return SimilarityResult(similarity_score=1.0, tag="near_duplicate")

    cosine = cosine_similarity(features_a.frequency_vector, features_b.frequency_vector)
    jaccard = jaccard_similarity(features_a.primitive_signature, features_b.primitive_signature)
    shape = tree_shape_similarity(features_a.tree_shape, features_b.tree_shape)
    param = parameter_count_similarity(features_a.parameter_count, features_b.parameter_count)

    final_score = 0.4 * cosine + 0.3 * jaccard + 0.2 * shape + 0.1 * param
    return SimilarityResult(similarity_score=final_score, tag=classify_similarity(final_score))


def vectorize_features(features: FeatureSet) -> list[float]:
    ordered_freq_keys = sorted(features.frequency_vector)
    ordered_primitive_keys = sorted(features.primitive_signature)
    vector: list[float] = [float(features.frequency_vector[k]) for k in ordered_freq_keys]
    vector.extend(float(features.primitive_signature[k]) for k in ordered_primitive_keys)
    vector.extend(
        [
            features.tree_shape["depth"],
            features.tree_shape["branching_factor"],
            features.tree_shape["rolling_count"],
            features.tree_shape["cross_logic_count"],
            float(features.parameter_count),
        ]
    )
    return vector


def top_k_neighbors(query_vector: list[float], vectors: Iterable[tuple[str, list[float]]], k: int = 10) -> list[tuple[str, float]]:
    def cosine_dense(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    scored = [(name, cosine_dense(query_vector, vec)) for name, vec in vectors]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:k]
