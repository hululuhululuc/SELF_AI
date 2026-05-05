# coding=utf-8
"""Tests for Python AST code graph builder."""

from pathlib import Path

from self_ai.graph.code_graph_builder import (
    build_code_graph_for_file,
    build_code_graph_for_paths,
    parse_python_file,
)


def test_parse_python_file_identifies_function_class_method_calls_decorators(tmp_path: Path) -> None:
    code = '''
import functools

def deco(fn):
    return fn

def helper():
    return 1

@deco
@functools.lru_cache(maxsize=128)
def top():
    helper()

class A:
    @staticmethod
    def m(self):
        top()
        self.run()
'''
    file_path = tmp_path / "sample.py"
    file_path.write_text(code, encoding="utf-8")

    symbols = parse_python_file(str(file_path))

    names = {s.name for s in symbols}
    assert "helper" in names
    assert "top" in names
    assert "A" in names
    assert "m" in names

    top_symbol = next(s for s in symbols if s.name == "top")
    assert "deco" in top_symbol.decorators
    assert any("functools.lru_cache" in d for d in top_symbol.decorators)
    assert "helper" in top_symbol.calls

    method_symbol = next(s for s in symbols if s.name == "m")
    assert method_symbol.class_name == "A"
    assert "top" in method_symbol.calls
    assert "self" in method_symbol.calls
    assert method_symbol.lineno > 0
    assert method_symbol.end_lineno >= method_symbol.lineno


def test_build_code_graph_for_file_generates_nodes_and_edges(tmp_path: Path) -> None:
    code = '''

def f1():
    f2()

def f2():
    return 2

class B:
    def m1(self):
        f1()
'''
    file_path = tmp_path / "graph_file.py"
    file_path.write_text(code, encoding="utf-8")

    nodes, edges = build_code_graph_for_file(str(file_path))

    labels = {n.label for n in nodes}
    relations = {e.relation for e in edges}

    assert "File" in labels
    assert "Function" in labels
    assert "Class" in labels
    assert "CONTAINS" in relations
    assert "CALLS" in relations


def test_build_code_graph_for_paths_merges_results(tmp_path: Path) -> None:
    p1 = tmp_path / "a.py"
    p2 = tmp_path / "b.py"
    p1.write_text("def a():\n    return 1\n", encoding="utf-8")
    p2.write_text("def b():\n    a()\n", encoding="utf-8")

    nodes, edges = build_code_graph_for_paths([str(p1), str(p2)])

    assert nodes
    assert edges


def test_parse_python_file_syntax_error_safe_return(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.py"
    bad_file.write_text("def broken(:\n    pass\n", encoding="utf-8")

    symbols = parse_python_file(str(bad_file))
    nodes, edges = build_code_graph_for_file(str(bad_file))

    assert symbols == []
    assert nodes == []
    assert edges == []
