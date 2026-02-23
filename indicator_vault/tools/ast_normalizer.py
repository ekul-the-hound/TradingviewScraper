from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from typing import Any

COMMUTATIVE_BINOPS = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr)
COMMUTATIVE_CMPOPS = (ast.Eq,)


@dataclass(frozen=True)
class NormalizationResult:
    normalized_tree: ast.AST
    normalized_dump: str
    fingerprint: str


class DeterministicNormalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.name_map: dict[str, str] = {}
        self.arg_map: dict[str, str] = {}
        self._name_counter = 0
        self._arg_counter = 0

    def _canonical_name(self, old: str) -> str:
        if old not in self.name_map:
            self._name_counter += 1
            self.name_map[old] = f"VAR_{self._name_counter}"
        return self.name_map[old]

    def _canonical_arg(self, old: str) -> str:
        if old not in self.arg_map:
            self._arg_counter += 1
            self.arg_map[old] = f"ARG_{self._arg_counter}"
        return self.arg_map[old]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        node.name = "FUNC"
        node.decorator_list = []
        node.returns = None
        node.type_comment = None
        node.args.defaults = []
        node.args.kw_defaults = []
        self.visit(node.args)
        node.body = [self.visit(stmt) for stmt in node.body if not self._is_docstring(stmt)]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self.visit_FunctionDef(node)

    def visit_arguments(self, node: ast.arguments) -> Any:
        for arg in node.posonlyargs:
            self.visit(arg)
        for arg in node.args:
            self.visit(arg)
        if node.vararg:
            self.visit(node.vararg)
        for arg in node.kwonlyargs:
            self.visit(arg)
        if node.kwarg:
            self.visit(node.kwarg)
        return node

    def visit_arg(self, node: ast.arg) -> Any:
        node.arg = self._canonical_arg(node.arg)
        node.annotation = None
        node.type_comment = None
        return node

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in {"True", "False", "None"}:
            return node
        node.id = self._canonical_name(node.id)
        return node

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float, complex)):
            return ast.copy_location(ast.Name(id="CONST", ctx=ast.Load()), node)
        if isinstance(node.value, str):
            return ast.copy_location(ast.Name(id="STR", ctx=ast.Load()), node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.op, COMMUTATIVE_BINOPS):
            left_key = ast.dump(node.left, annotate_fields=False, include_attributes=False)
            right_key = ast.dump(node.right, annotate_fields=False, include_attributes=False)
            if right_key < left_key:
                node.left, node.right = node.right, node.left
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        node.values = [self.visit(v) for v in node.values]
        node.values = sorted(
            node.values,
            key=lambda v: ast.dump(v, annotate_fields=False, include_attributes=False),
        )
        return node

    def visit_Compare(self, node: ast.Compare) -> Any:
        node.left = self.visit(node.left)
        node.comparators = [self.visit(c) for c in node.comparators]
        if len(node.ops) == 1 and isinstance(node.ops[0], COMMUTATIVE_CMPOPS):
            first = node.left
            second = node.comparators[0]
            first_key = ast.dump(first, annotate_fields=False, include_attributes=False)
            second_key = ast.dump(second, annotate_fields=False, include_attributes=False)
            if second_key < first_key:
                node.left = second
                node.comparators[0] = first
        return node

    @staticmethod
    def _is_docstring(stmt: ast.stmt) -> bool:
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )


def normalize_code(code_string: str) -> NormalizationResult:
    tree = ast.parse(code_string)
    normalized_tree = DeterministicNormalizer().visit(tree)
    ast.fix_missing_locations(normalized_tree)
    normalized_dump = ast.dump(normalized_tree, annotate_fields=True, include_attributes=False)
    fingerprint = hashlib.sha256(normalized_dump.encode("utf-8")).hexdigest()
    return NormalizationResult(
        normalized_tree=normalized_tree,
        normalized_dump=normalized_dump,
        fingerprint=fingerprint,
    )
