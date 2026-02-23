from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, Set


@dataclass(frozen=True)
class CategoryResult:
    category: str
    flags: Dict[str, bool]


class _CategoryVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.flags: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        fn = self._name(node.func).lower()
        if "rolling" in fn and "std" in fn:
            self.flags.add("has_rolling_std")
        if "ewm" in fn or "ema" in fn:
            self.flags.add("has_ema")
        if "cross" in fn:
            self.flags.add("has_crossover")
        if "zscore" in fn or "z_score" in fn:
            self.flags.add("uses_zscore")
        if "normalize" in fn or "normaliz" in fn:
            self.flags.add("uses_normalization")
        if "diff" in fn or "momentum" in fn:
            self.flags.add("uses_difference")
        if "atr" in fn or "true_range" in fn:
            self.flags.add("uses_atr")
        if "vwap" in fn or "obv" in fn:
            self.flags.add("uses_volume")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.lower() == "volume":
            self.flags.add("uses_volume")
        self.generic_visit(node)

    @staticmethod
    def _name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{_CategoryVisitor._name(node.value)}.{node.attr}"
        return ""


class CategoryDetector:
    def detect(self, code: str) -> CategoryResult:
        tree = ast.parse(code)
        visitor = _CategoryVisitor()
        visitor.visit(tree)
        flags = {
            "has_rolling_std": "has_rolling_std" in visitor.flags,
            "has_ema": "has_ema" in visitor.flags,
            "has_crossover": "has_crossover" in visitor.flags,
            "uses_volume": "uses_volume" in visitor.flags,
            "uses_zscore": "uses_zscore" in visitor.flags,
            "uses_normalization": "uses_normalization" in visitor.flags,
            "uses_difference": "uses_difference" in visitor.flags,
            "uses_atr": "uses_atr" in visitor.flags,
        }
        matches = []
        if flags["has_ema"] and not flags["uses_zscore"]:
            matches.append("trend")
        if flags["uses_difference"] or flags["uses_normalization"]:
            matches.append("momentum")
        if flags["has_rolling_std"] or flags["uses_atr"]:
            matches.append("volatility")
        if flags["uses_volume"]:
            matches.append("volume")
        if flags["uses_zscore"]:
            matches.append("mean_reversion")
        category = "composite" if len(set(matches)) >= 2 else (matches[0] if matches else "trend")
        return CategoryResult(category=category, flags=flags)
