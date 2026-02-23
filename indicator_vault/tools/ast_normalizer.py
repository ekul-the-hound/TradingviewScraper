from __future__ import annotations

import ast
import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

COMMUTATIVE_OPS: tuple[type[ast.AST], ...] = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.Eq)
PRIMITIVES = (
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
)


@dataclass(frozen=True)
class NormalizedAstResult:
    normalized_tree: ast.AST
    normalized_dump: str
    fingerprint: str
    parameter_count: int


class AstNormalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._name_map: dict[str, str] = {}
        self._name_counter = 0
        self.parameter_count = 0

    def _canonical_name(self, original: str) -> str:
        if original not in self._name_map:
            self._name_counter += 1
            self._name_map[original] = f"VAR_{self._name_counter}"
        return self._name_map[original]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        node.name = "indicator"
        self.parameter_count = len(node.args.args)
        for arg in node.args.args:
            arg.arg = self._canonical_name(arg.arg)
        node.args.defaults = []
        node.args.kw_defaults = []
        node.args.kwonlyargs = []
        node.args.vararg = None
        node.args.kwarg = None
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_Name(self, node: ast.Name) -> Any:
        node.id = self._canonical_name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> Any:
        node.arg = self._canonical_name(node.arg)
        return node

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, complex)):
            return ast.copy_location(ast.Name(id="CONST", ctx=ast.Load()), node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        node = self.generic_visit(node)
        if isinstance(node.op, COMMUTATIVE_OPS):
            left_dump = ast.dump(node.left, annotate_fields=False)
            right_dump = ast.dump(node.right, annotate_fields=False)
            if right_dump < left_dump:
                node.left, node.right = node.right, node.left
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        node = self.generic_visit(node)
        values = sorted(node.values, key=lambda v: ast.dump(v, annotate_fields=False))
        node.values = values
        return node


class FeatureExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.flags: dict[str, bool] = {
            "has_rolling_std": False,
            "has_ema": False,
            "has_crossover": False,
            "uses_volume": False,
            "uses_zscore": False,
            "uses_normalization": False,
            "uses_difference": False,
            "has_rolling_mean": False,
        }
        self.primitives = {key: 0 for key in PRIMITIVES}
        self._max_depth = 0
        self._node_count = 0
        self._children_count = 0

    def _track_depth(self, node: ast.AST, depth: int = 1) -> None:
        self._max_depth = max(self._max_depth, depth)
        children = list(ast.iter_child_nodes(node))
        self._node_count += 1
        self._children_count += len(children)
        for child in children:
            self._track_depth(child, depth + 1)

    def extract(self, tree: ast.AST) -> dict[str, Any]:
        self._track_depth(tree)
        self.visit(tree)
        avg_branching = self._children_count / self._node_count if self._node_count else 0.0
        return {
            "counts": dict(self.counts),
            "flags": self.flags,
            "primitives": self.primitives,
            "shape": {
                "depth": self._max_depth,
                "branching_factor": avg_branching,
                "rolling_window_count": self.counts.get("rolling", 0),
                "cross_logic_count": self.counts.get("cross", 0),
            },
        }

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_map = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}
        for op_type, key in op_map.items():
            if isinstance(node.op, op_type):
                self.counts[key] += 1
                break
        if isinstance(node.op, ast.Sub):
            self.flags["uses_difference"] = True
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> Any:
        self.counts["comparisons"] += len(node.ops)
        for op in node.ops:
            self.counts[type(op).__name__.lower()] += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        self.counts["logical_ops"] += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr.lower()
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id.lower()

        if func_name == "rolling":
            self.counts["rolling"] += 1
        if func_name == "ewm":
            self.counts["ewm"] += 1
            self.flags["has_ema"] = True
        if func_name == "shift":
            self.counts["shift"] += 1
        if func_name in {"std", "rolling_std"}:
            self.counts["rolling_std"] += 1
            self.flags["has_rolling_std"] = True
        if func_name in {"mean", "rolling_mean"}:
            self.counts["rolling_mean"] += 1
            self.flags["has_rolling_mean"] = True
        if func_name in {"crossover", "crossunder"}:
            self.counts["cross"] += 1
            self.flags["has_crossover"] = True

        for primitive in PRIMITIVES:
            if primitive in func_name:
                self.primitives[primitive] = 1

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        token = node.id.lower()
        if token == "volume":
            self.flags["uses_volume"] = True
        if "zscore" in token or token == "z":
            self.flags["uses_zscore"] = True
        if any(keyword in token for keyword in ("norm", "normalize", "scaled", "scale")):
            self.flags["uses_normalization"] = True


def normalize_code(code_string: str) -> NormalizedAstResult:
    tree = ast.parse(code_string)
    normalizer = AstNormalizer()
    normalized_tree = normalizer.visit(tree)
    ast.fix_missing_locations(normalized_tree)
    normalized_dump = ast.dump(normalized_tree, annotate_fields=False, include_attributes=False)
    fingerprint = hashlib.sha256(normalized_dump.encode()).hexdigest()
    return NormalizedAstResult(
        normalized_tree=normalized_tree,
        normalized_dump=normalized_dump,
        fingerprint=fingerprint,
        parameter_count=normalizer.parameter_count,
    )


def extract_structural_features(code_string: str) -> dict[str, Any]:
    normalized = normalize_code(code_string)
    extractor = FeatureExtractor()
    features = extractor.extract(normalized.normalized_tree)
    features["fingerprint"] = normalized.fingerprint
    features["parameter_count"] = normalized.parameter_count
    return features


def category_from_features(features: dict[str, Any]) -> str:
    flags = features["flags"]
    counts = features["counts"]
    trend = bool((counts.get("ewm", 0) or counts.get("rolling_mean", 0)) and counts.get("comparisons", 0))
    momentum = bool(flags["uses_difference"] or features["primitives"].get("momentum", 0) or features["primitives"].get("rsi", 0))
    volatility = bool(flags["has_rolling_std"] or features["primitives"].get("atr", 0) or features["primitives"].get("volatility", 0))
    volume = bool(flags["uses_volume"])
    mean_reversion = bool(flags["uses_zscore"] or (flags["has_rolling_mean"] and counts.get("cross", 0) > 0))

    active = [trend, momentum, volatility, volume, mean_reversion]
    if sum(active) >= 2:
        return "composite"
    if trend:
        return "trend"
    if momentum:
        return "momentum"
    if volatility:
        return "volatility"
    if volume:
        return "volume"
    if mean_reversion:
        return "mean_reversion"
    return "trend"


def shape_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    keys = ("depth", "branching_factor", "rolling_window_count", "cross_logic_count")
    scores = []
    for key in keys:
        l_val = float(left.get(key, 0.0))
        r_val = float(right.get(key, 0.0))
        denom = max(abs(l_val), abs(r_val), 1.0)
        scores.append(1.0 - min(abs(l_val - r_val) / denom, 1.0))
    return sum(scores) / len(scores)


def cosine_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 1.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    mag_a = math.sqrt(sum(a.get(k, 0) ** 2 for k in keys))
    mag_b = math.sqrt(sum(b.get(k, 0) ** 2 for k in keys))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def jaccard_similarity(a: dict[str, int], b: dict[str, int]) -> float:
    keys = sorted(set(a) | set(b))
    if not keys:
        return 1.0
    inter = 0
    union = 0
    for key in keys:
        av = int(bool(a.get(key, 0)))
        bv = int(bool(b.get(key, 0)))
        inter += av & bv
        union += av | bv
    return inter / union if union else 1.0
