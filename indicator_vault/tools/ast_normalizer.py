from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Dict

_COMMUTATIVE_OPS = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.Eq)


class ASTNormalizationError(ValueError):
    pass


@dataclass(frozen=True)
class NormalizationResult:
    normalized_tree: ast.AST
    normalized_dump: str
    fingerprint: str


class _Normalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self._name_map: Dict[str, str] = {}
        self._next_name_idx = 1

    def _canonical_name(self, source: str) -> str:
        if source not in self._name_map:
            self._name_map[source] = f"VAR_{self._next_name_idx}"
            self._next_name_idx += 1
        return self._name_map[source]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = "indicator_fn"
        node.decorator_list = []
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        node.args.defaults = []
        node.args.kw_defaults = [None for _ in node.args.kwonlyargs]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = "indicator_fn"
        node.decorator_list = []
        node.args.defaults = []
        node.args.kw_defaults = [None for _ in node.args.kwonlyargs]
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in {"True", "False", "None"}:
            return node
        return ast.copy_location(ast.Name(id=self._canonical_name(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node = self.generic_visit(node)
        node.arg = self._canonical_name(node.arg)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float, complex)):
            return ast.copy_location(ast.Constant(value="CONST"), node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.op, _COMMUTATIVE_OPS):
            left_dump = ast.dump(node.left, annotate_fields=True, include_attributes=False)
            right_dump = ast.dump(node.right, annotate_fields=True, include_attributes=False)
            if right_dump < left_dump:
                node.left, node.right = node.right, node.left
        return node


class ASTNormalizer:
    def normalize(self, code_string: str) -> NormalizationResult:
        try:
            tree = ast.parse(code_string)
        except SyntaxError as exc:
            raise ASTNormalizationError(str(exc)) from exc
        normalized_tree = _Normalizer().visit(tree)
        ast.fix_missing_locations(normalized_tree)
        normalized_dump = ast.dump(normalized_tree, annotate_fields=True, include_attributes=False)
        fingerprint = hashlib.sha256(normalized_dump.encode("utf-8")).hexdigest()
        return NormalizationResult(
            normalized_tree=normalized_tree,
            normalized_dump=normalized_dump,
            fingerprint=fingerprint,
        )
