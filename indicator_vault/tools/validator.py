from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    error: str = ""


FORBIDDEN_CALLS = {"print"}
FORBIDDEN_IMPORTS = {"matplotlib", "seaborn", "plotly"}


def validate_indicator_code(code: str) -> ValidationResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ValidationResult(valid=False, error=f"Syntax error: {exc}")

    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(funcs) != 1:
        return ValidationResult(valid=False, error="File must contain exactly one function")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    return ValidationResult(valid=False, error="Plotting imports are not allowed")
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in FORBIDDEN_IMPORTS:
                return ValidationResult(valid=False, error="Plotting imports are not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                return ValidationResult(valid=False, error="print statements are not allowed")

    return ValidationResult(valid=True)
