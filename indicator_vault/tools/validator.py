from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    is_valid: bool
    reason: str


def validate_indicator_code(code: str) -> ValidationResult:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return ValidationResult(False, f"syntax_error: {exc.msg}")

    function_defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(function_defs) != 1:
        return ValidationResult(False, "indicator_must_contain_exactly_one_function")

    function = function_defs[0]
    if not function.args.args:
        return ValidationResult(False, "function_must_accept_df")

    first_arg = function.args.args[0].arg
    if first_arg != "df":
        return ValidationResult(False, "first_parameter_must_be_df")

    if not any(isinstance(n, ast.Return) for n in ast.walk(function)):
        return ValidationResult(False, "function_must_return_series_or_dataframe")

    return ValidationResult(True, "ok")
