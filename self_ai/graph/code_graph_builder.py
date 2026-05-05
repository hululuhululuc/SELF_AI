# coding=utf-8
"""Python code graph builders for Neo4j structural memory."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .graph_models import CodeSymbol, GraphEdge, GraphNode


def _module_name_from_path(path: str) -> str:
    p = Path(path)
    stem = p.stem
    return stem if stem else "module"


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = _decorator_name(node.value) if isinstance(node.value, ast.expr) else ""
        return f"{root}.{node.attr}".strip(".")
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return "unknown"


def _call_target_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            # method call base name (e.g. obj.method -> obj)
            return func.value.id
        if isinstance(func.value, ast.Attribute):
            return func.value.attr
        return func.attr
    return None


class _CodeVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, module_name: str) -> None:
        self.file_path = file_path
        self.module_name = module_name
        self.class_stack: list[str] = []
        self.symbols: list[CodeSymbol] = []

    def _build_symbol_id(self, symbol_type: str, name: str, lineno: int) -> str:
        return f"{self.file_path}::{symbol_type}::{name}::{lineno}"

    def _collect_calls(self, node: ast.AST) -> list[str]:
        calls: list[str] = []
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call):
                target = _call_target_name(inner)
                if target:
                    calls.append(target)
        return calls

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        symbol = CodeSymbol(
            symbol_id=self._build_symbol_id("class", node.name, int(getattr(node, "lineno", 0) or 0)),
            name=node.name,
            symbol_type="class",
            file_path=self.file_path,
            module_name=self.module_name,
            class_name=None,
            decorators=[_decorator_name(item) for item in node.decorator_list],
            calls=[],
            lineno=int(getattr(node, "lineno", 0) or 0),
            end_lineno=int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0),
        )
        self.symbols.append(symbol)

        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function_like(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function_like(node, is_async=True)

    def _visit_function_like(self, node: ast.AST, *, is_async: bool) -> None:
        fn_name = getattr(node, "name", "unknown")
        lineno = int(getattr(node, "lineno", 0) or 0)
        end_lineno = int(getattr(node, "end_lineno", lineno) or lineno)
        decorators = [
            _decorator_name(item)
            for item in getattr(node, "decorator_list", [])
            if isinstance(item, ast.expr)
        ]
        class_name = self.class_stack[-1] if self.class_stack else None
        symbol_type = "method" if class_name else "function"
        if is_async and symbol_type == "function":
            symbol_type = "async_function"
        if is_async and symbol_type == "method":
            symbol_type = "async_method"

        calls = self._collect_calls(node)
        symbol = CodeSymbol(
            symbol_id=self._build_symbol_id(symbol_type, fn_name, lineno),
            name=fn_name,
            symbol_type=symbol_type,
            file_path=self.file_path,
            module_name=self.module_name,
            class_name=class_name,
            decorators=decorators,
            calls=calls,
            lineno=lineno,
            end_lineno=end_lineno,
        )
        self.symbols.append(symbol)
        self.generic_visit(node)


def parse_python_file(path: str) -> list[CodeSymbol]:
    """Parse one Python file and return discovered code symbols.

    Syntax errors return an empty list instead of raising.
    """
    file_path = str(Path(path).resolve())
    try:
        source = Path(file_path).read_text(encoding="utf-8")
    except Exception:
        return []

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []
    except Exception:
        return []

    visitor = _CodeVisitor(file_path=file_path, module_name=_module_name_from_path(file_path))
    visitor.visit(tree)
    return visitor.symbols


def build_code_graph_for_file(path: str) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Build graph nodes/edges for one Python file."""
    symbols = parse_python_file(path)
    if not symbols:
        return [], []

    file_path = str(Path(path).resolve())
    file_node_id = f"file:{file_path}"
    nodes: list[GraphNode] = [
        GraphNode(
            id=file_node_id,
            label="File",
            properties={"path": file_path, "module": _module_name_from_path(file_path)},
        )
    ]
    edges: list[GraphEdge] = []

    known_symbol_ids: set[str] = set()
    call_target_ids: dict[str, str] = {}

    for symbol in symbols:
        label = "Class" if symbol.symbol_type == "class" else "Function"
        node = GraphNode(
            id=symbol.symbol_id,
            label=label,
            properties={
                "name": symbol.name,
                "symbol_type": symbol.symbol_type,
                "file_path": symbol.file_path,
                "module_name": symbol.module_name,
                "class_name": symbol.class_name,
                "decorators": list(symbol.decorators),
                "lineno": symbol.lineno,
                "end_lineno": symbol.end_lineno,
            },
        )
        nodes.append(node)
        known_symbol_ids.add(symbol.symbol_id)
        edges.append(
            GraphEdge(
                source_id=file_node_id,
                target_id=symbol.symbol_id,
                relation="CONTAINS",
                properties={},
            )
        )

    for symbol in symbols:
        if symbol.symbol_type == "class":
            continue
        for callee in symbol.calls:
            if not callee:
                continue
            target_id = call_target_ids.get(callee)
            if target_id is None:
                target_id = f"call_target:{file_path}:{callee}"
                call_target_ids[callee] = target_id
                nodes.append(
                    GraphNode(
                        id=target_id,
                        label="Function",
                        properties={
                            "name": callee,
                            "symbol_type": "call_target",
                            "file_path": file_path,
                            "external_ref": True,
                        },
                    )
                )
            edges.append(
                GraphEdge(
                    source_id=symbol.symbol_id,
                    target_id=target_id,
                    relation="CALLS",
                    properties={},
                )
            )

    return nodes, edges


def build_code_graph_for_paths(paths: list[str]) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Build merged code graph for multiple files."""
    all_nodes: dict[str, GraphNode] = {}
    all_edges: dict[tuple[str, str, str], GraphEdge] = {}

    for path in paths:
        nodes, edges = build_code_graph_for_file(path)
        for node in nodes:
            all_nodes[node.id] = node
        for edge in edges:
            key = (edge.source_id, edge.target_id, edge.relation)
            all_edges[key] = edge

    return list(all_nodes.values()), list(all_edges.values())
