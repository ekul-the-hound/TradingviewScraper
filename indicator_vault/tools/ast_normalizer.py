from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from typing import Any

COMMUTATIVE_OPS = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr)
COMMUTATIVE_CMP_OPS = (ast.Eq,)


@dataclass(frozen=True)
class NormalizationResult:
    normalized_tree: ast.AST
    normalized_dump: str
    fingerprint: str


class _Normalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._name_map: dict[str, str] = {}
        self._name_counter = 0

    def _canonical_name(self, value: str) -> str:
        if value not in self._name_map:
            self._name_counter += 1
            self._name_map[value] = f"VAR_{self._name_counter}"
        return self._name_map[value]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        node.name = "FUNCTION"
        node.body = [stmt for stmt in node.body if not self._is_docstring(stmt)]
        for arg in node.args.args:
            arg.arg = self._canonical_name(arg.arg)
            arg.annotation = None
        node.args.defaults = [ast.Constant(value="CONST") for _ in node.args.defaults]
        node.decorator_list = []
        node.returns = None
        self.generic_visit(node)
        return node

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, complex, str, bytes, bool)) or node.value is None:
            return ast.copy_location(ast.Constant(value="CONST"), node)
        return node

    def visit_Name(self, node: ast.Name) -> Any:
        return ast.copy_location(ast.Name(id=self._canonical_name(node.id), ctx=node.ctx), node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        node.attr = "ATTR"
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        self.generic_visit(node)
        if isinstance(node.op, COMMUTATIVE_OPS):
            operands = sorted([node.left, node.right], key=lambda n: ast.dump(n, include_attributes=False))
            node.left, node.right = operands
        return node

    def visit_Compare(self, node: ast.Compare) -> Any:
        self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], COMMUTATIVE_CMP_OPS):
            operands = sorted([node.left, node.comparators[0]], key=lambda n: ast.dump(n, include_attributes=False))
            node.left, node.comparators[0] = operands
        return node

    @staticmethod
    def _is_docstring(stmt: ast.stmt) -> bool:
        return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)


def normalize_code(code_string: str) -> NormalizationResult:
    tree = ast.parse(code_string)
    normalized = _Normalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    normalized_dump = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    fingerprint = hashlib.sha256(normalized_dump.encode("utf-8")).hexdigest()
    return NormalizationResult(normalized_tree=normalized, normalized_dump=normalized_dump, fingerprint=fingerprint)


def extract_subtree_frequency(tree: ast.AST) -> dict[str, int]:
    counts = {
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
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Add):
                counts["add"] += 1
            elif isinstance(node.op, ast.Sub):
                counts["sub"] += 1
            elif isinstance(node.op, ast.Mult):
                counts["mul"] += 1
            elif isinstance(node.op, ast.Div):
                counts["div"] += 1
        elif isinstance(node, ast.BoolOp):
            counts["logical"] += 1
        elif isinstance(node, ast.Compare):
            counts["comparisons"] += len(node.ops)
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if "rolling" in name:
                counts["rolling"] += 1
            if "ewm" in name:
                counts["ewm"] += 1
            if "shift" in name:
                counts["shift"] += 1
    return counts


def extract_primitive_signature(tree: ast.AST) -> dict[str, int]:
    primitives = {
        "ema": 0,
        "sma": 0,
        "rsi": 0,
        "atr": 0,
        "std": 0,
        "highest": 0,
        "lowest": 0,
        "crossover": 0,
        "volatility": 0,
        "momentum": 0,
    }
    raw = ast.dump(tree, annotate_fields=False, include_attributes=False).lower()
    for key in primitives:
        if key in raw:
            primitives[key] = 1
    if "rolling" in raw and "std" in raw:
        primitives["volatility"] = 1
    if "diff" in raw or "momentum" in raw:
        primitives["momentum"] = 1
    return primitives


def extract_tree_shape(tree: ast.AST) -> dict[str, float]:
    max_depth = _tree_depth(tree)
    child_counts = [len(list(ast.iter_child_nodes(node))) for node in ast.walk(tree)]
    branch_factor = (sum(child_counts) / len(child_counts)) if child_counts else 0.0
    rolling_windows = 0
    cross_logic = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and "rolling" in _call_name(node):
            rolling_windows += 1
        if isinstance(node, ast.BoolOp) or isinstance(node, ast.Compare):
            cross_logic += 1
    return {
        "depth": float(max_depth),
        "branching_factor": branch_factor,
        "rolling_window_count": float(rolling_windows),
        "cross_logic_count": float(cross_logic),
    }


def detect_category_flags(tree: ast.AST) -> dict[str, bool]:
    flags = {
        "has_rolling_std": False,
        "has_ema": False,
        "has_crossover": False,
        "uses_volume": False,
        "uses_zscore": False,
        "uses_normalization": False,
        "uses_difference": False,
        "has_atr": False,
        "has_rolling_mean": False,
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.lower() == "volume":
            flags["uses_volume"] = True
        if isinstance(node, ast.Call):
            call_name = _call_name(node)
            low_name = call_name.lower()
            if "rolling" in low_name and "std" in low_name:
                flags["has_rolling_std"] = True
            if "rolling" in low_name and "mean" in low_name:
                flags["has_rolling_mean"] = True
            if "ewm" in low_name or "ema" in low_name:
                flags["has_ema"] = True
            if "atr" in low_name:
                flags["has_atr"] = True
            if "zscore" in low_name:
                flags["uses_zscore"] = True
            if "diff" in low_name:
                flags["uses_difference"] = True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            flags["uses_difference"] = True
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            flags["has_crossover"] = True
        if isinstance(node, ast.Compare):
            if any(isinstance(op, (ast.Gt, ast.Lt, ast.GtE, ast.LtE)) for op in node.ops):
                flags["uses_normalization"] = True
    return flags


def assign_category(flags: dict[str, bool]) -> str:
    matches: set[str] = set()
    if (flags["has_ema"] or flags["has_rolling_mean"]) and not flags["uses_zscore"]:
        matches.add("trend")
    if flags["uses_difference"] or flags["uses_normalization"]:
        matches.add("momentum")
    if flags["has_rolling_std"] or flags["has_atr"]:
        matches.add("volatility")
    if flags["uses_volume"]:
        matches.add("volume")
    if flags["uses_zscore"] and flags["has_rolling_mean"]:
        matches.add("mean_reversion")
    if len(matches) >= 2:
        return "composite"
    return next(iter(matches), "composite")


def serialize_metadata(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _tree_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    if not children:
        return 1
    return 1 + max(_tree_depth(child) for child in children)
