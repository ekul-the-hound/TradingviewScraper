from __future__ import annotations

import ast


REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def validate_python_indicator(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax_error:{exc.msg}"

    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(funcs) != 1:
        return False, "must_contain_exactly_one_function"

    func = funcs[0]
    if not func.args.args:
        return False, "first_argument_must_be_df"
    if func.args.args[0].arg != "df":
        return False, "first_argument_must_be_df"

    returns = [node for node in ast.walk(func) if isinstance(node, ast.Return)]
    if not returns:
        return False, "missing_return"

    if any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(func)):
        return False, "row_loops_forbidden"

    return True, "ok"


def validate_dataframe_columns(df_columns: set[str]) -> tuple[bool, str]:
    missing = REQUIRED_COLUMNS - df_columns
    if missing:
        return False, f"missing_columns:{','.join(sorted(missing))}"
    return True, "ok"
