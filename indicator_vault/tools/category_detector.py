from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryFlags:
    has_rolling_std: bool = False
    has_ema: bool = False
    has_crossover: bool = False
    uses_volume: bool = False
    uses_zscore: bool = False
    uses_normalization: bool = False
    uses_difference: bool = False
    has_rolling_mean: bool = False
    has_atr: bool = False
    has_true_range: bool = False
    bounded_oscillator: bool = False


class FlagExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.flags = {
            "has_rolling_std": False,
            "has_ema": False,
            "has_crossover": False,
            "uses_volume": False,
            "uses_zscore": False,
            "uses_normalization": False,
            "uses_difference": False,
            "has_rolling_mean": False,
            "has_atr": False,
            "has_true_range": False,
            "bounded_oscillator": False,
        }

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.lower() == "volume":
            self.flags["uses_volume"] = True

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Sub):
            self.flags["uses_difference"] = True
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(isinstance(op, (ast.Gt, ast.GtE, ast.Lt, ast.LtE)) for op in node.ops):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value in {0, 1, 30, 70, -1}:
                    self.flags["bounded_oscillator"] = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = ""
        if isinstance(node.func, ast.Attribute):
            target = node.func.attr.lower()
        elif isinstance(node.func, ast.Name):
            target = node.func.id.lower()

        if target == "std":
            self.flags["has_rolling_std"] = True
        if target == "mean":
            self.flags["has_rolling_mean"] = True
        if target == "ewm" or target == "ema":
            self.flags["has_ema"] = True
        if target in {"crossover", "crossunder"}:
            self.flags["has_crossover"] = True
        if target == "atr":
            self.flags["has_atr"] = True
        if target in {"maximum", "minimum", "abs"}:
            self.flags["has_true_range"] = True
        if target in {"clip", "divide", "div"}:
            self.flags["uses_normalization"] = True
        if target == "zscore":
            self.flags["uses_zscore"] = True

        self.generic_visit(node)


def detect_category(code: str) -> tuple[str, CategoryFlags]:
    tree = ast.parse(code)
    extractor = FlagExtractor()
    extractor.visit(tree)
    flags = CategoryFlags(**extractor.flags)

    signatures: list[str] = []

    trend = (flags.has_ema or flags.has_rolling_mean) and not flags.uses_zscore
    momentum = flags.uses_difference and (flags.uses_normalization or flags.bounded_oscillator)
    volatility = flags.has_rolling_std or flags.has_atr or flags.has_true_range
    volume = flags.uses_volume
    mean_reversion = flags.uses_zscore or (flags.has_rolling_mean and flags.has_crossover)

    if trend:
        signatures.append("trend")
    if momentum:
        signatures.append("momentum")
    if volatility:
        signatures.append("volatility")
    if volume:
        signatures.append("volume")
    if mean_reversion:
        signatures.append("mean_reversion")

    if len(signatures) >= 2:
        return "composite", flags
    if len(signatures) == 1:
        return signatures[0], flags
    return "trend", flags
