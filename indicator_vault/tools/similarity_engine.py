from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

from .ast_normalizer import ASTNormalizer, NormalizationResult

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
class ShapeMetrics:
    depth: int
    avg_branching_factor: float
    rolling_window_count: int
    cross_logic_count: int


@dataclass(frozen=True)
class IndicatorFeatures:
    fingerprint: str
    ast_frequency: Dict[str, float]
    primitive_signature: Dict[str, int]
    shape: ShapeMetrics
    parameter_count: int


@dataclass(frozen=True)
class SimilarityResult:
    score: float
    label: str
    breakdown: Dict[str, float]


class _FeatureExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.freq: Dict[str, float] = {
            "add": 0,
            "sub": 0,
            "mul": 0,
            "div": 0,
            "rolling": 0,
            "ewm": 0,
            "comparisons": 0,
            "logical": 0,
            "shift": 0,
        }
        self.primitives: Set[str] = set()
        self._depth = 0
        self.max_depth = 0
        self._branch_sum = 0
        self._branch_nodes = 0
        self.rolling_window_count = 0
        self.cross_logic_count = 0
        self.parameter_count = 0

    def generic_visit(self, node: ast.AST) -> None:
        self._depth += 1
        self.max_depth = max(self.max_depth, self._depth)
        children = list(ast.iter_child_nodes(node))
        if children:
            self._branch_sum += len(children)
            self._branch_nodes += 1
        super().generic_visit(node)
        self._depth -= 1

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
        self.freq["comparisons"] += len(node.ops)
        if any(isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)) for op in node.ops):
            self.cross_logic_count += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = _call_name(node.func)
        lower_name = func_name.lower()
        if "rolling" in lower_name:
            self.freq["rolling"] += 1
            self.rolling_window_count += 1
        if "ewm" in lower_name:
            self.freq["ewm"] += 1
        if "shift" in lower_name:
            self.freq["shift"] += 1
        for primitive in PRIMITIVES:
            if primitive in lower_name:
                self.primitives.add(primitive)
        if "cross" in lower_name:
            self.cross_logic_count += 1
        self.generic_visit(node)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}"
    return ""


def _cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    left_values = [left.get(key, 0.0) for key in keys]
    right_values = [right.get(key, 0.0) for key in keys]
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _jaccard_binary(left: Dict[str, int], right: Dict[str, int]) -> float:
    keys = sorted(set(left) | set(right))
    intersection = sum(1 for key in keys if left.get(key, 0) and right.get(key, 0))
    union = sum(1 for key in keys if left.get(key, 0) or right.get(key, 0))
    if union == 0:
        return 0.0
    return intersection / union


def _shape_similarity(left: ShapeMetrics, right: ShapeMetrics) -> float:
    components = [
        _ratio_similarity(left.depth, right.depth),
        _ratio_similarity(left.avg_branching_factor, right.avg_branching_factor),
        _ratio_similarity(left.rolling_window_count, right.rolling_window_count),
        _ratio_similarity(left.cross_logic_count, right.cross_logic_count),
    ]
    return sum(components) / len(components)


def _ratio_similarity(left: float, right: float) -> float:
    high = max(left, right)
    if high == 0:
        return 1.0
    return 1.0 - abs(left - right) / high


def _parameter_similarity(left: int, right: int) -> float:
    return _ratio_similarity(float(left), float(right))


def _label(score: float) -> str:
    if score >= 0.95:
        return "near_duplicate"
    if score >= 0.85:
        return "variant"
    if score >= 0.70:
        return "same_family"
    return "structurally_distinct"


class SimilarityEngine:
    def __init__(self) -> None:
        self._normalizer = ASTNormalizer()

    def extract_features(self, code: str) -> IndicatorFeatures:
        normalized = self._normalizer.normalize(code)
        extractor = _FeatureExtractor()
        extractor.visit(normalized.normalized_tree)
        primitive_signature = {name: int(name in extractor.primitives) for name in PRIMITIVES}
        avg_branch = extractor._branch_sum / extractor._branch_nodes if extractor._branch_nodes else 0.0
        shape = ShapeMetrics(
            depth=extractor.max_depth,
            avg_branching_factor=avg_branch,
            rolling_window_count=extractor.rolling_window_count,
            cross_logic_count=extractor.cross_logic_count,
        )
        return IndicatorFeatures(
            fingerprint=normalized.fingerprint,
            ast_frequency=extractor.freq,
            primitive_signature=primitive_signature,
            shape=shape,
            parameter_count=extractor.parameter_count,
        )

    def similarity(self, left: IndicatorFeatures, right: IndicatorFeatures) -> SimilarityResult:
        if left.fingerprint == right.fingerprint:
            return SimilarityResult(score=1.0, label="near_duplicate", breakdown={"exact_fingerprint": 1.0})
        cosine = _cosine_similarity(left.ast_frequency, right.ast_frequency)
        jaccard = _jaccard_binary(left.primitive_signature, right.primitive_signature)
        shape_sim = _shape_similarity(left.shape, right.shape)
        param_sim = _parameter_similarity(left.parameter_count, right.parameter_count)
        score = 0.4 * cosine + 0.3 * jaccard + 0.2 * shape_sim + 0.1 * param_sim
        return SimilarityResult(
            score=score,
            label=_label(score),
            breakdown={
                "cosine_ast_freq": cosine,
                "jaccard_primitive": jaccard,
                "shape_similarity": shape_sim,
                "parameter_similarity": param_sim,
            },
        )

    def nearest_family_cluster(self, target: IndicatorFeatures, peers: Iterable[Tuple[str, IndicatorFeatures]]) -> Tuple[str, float]:
        best_name = ""
        best_score = -1.0
        for name, peer in peers:
            score = self.similarity(target, peer).score
            if score > best_score:
                best_name = name
                best_score = score
        return best_name, max(best_score, 0.0)
