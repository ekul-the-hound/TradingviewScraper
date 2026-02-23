from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    message: str


class IndicatorValidator:
    def validate_python_indicator(self, code: str) -> ValidationResult:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return ValidationResult(False, f"syntax_error:{exc}")
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if len(functions) != 1:
            return ValidationResult(False, "must_contain_exactly_one_function")
        fn = functions[0]
        if not fn.args.args:
            return ValidationResult(False, "first_parameter_must_be_dataframe")
        if fn.args.args[0].arg != "df":
            return ValidationResult(False, "first_parameter_must_be_df")
        if not any(isinstance(node, ast.Return) for node in ast.walk(fn)):
            return ValidationResult(False, "function_must_return_value")
        return ValidationResult(True, "ok")
