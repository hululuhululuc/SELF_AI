# coding=utf-8
"""Tests for Neo4j structural memory store with fake driver/session."""

from self_ai.graph.graph_models import CodeSymbol, GraphEdge, GraphNode, TaskGraphRecord
from self_ai.memory.neo4j_store import Neo4jGraphMemory, NullNeo4jGraphMemory


class FakeResult:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def data(self):
        return self._rows

    def __iter__(self):
        return iter(self._rows)

    def consume(self) -> None:
        return None


class FakeSession:
    def __init__(self, driver: "FakeDriver") -> None:
        self.driver = driver

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query: str, params: dict):
        self.driver.calls.append((query, params))
        if self.driver.raise_error:
            raise RuntimeError("neo4j boom")
        if self.driver.read_queue:
            rows = self.driver.read_queue.pop(0)
        else:
            rows = []
        return FakeResult(rows)


class FakeDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.read_queue: list[list[dict]] = []
        self.raise_error = False
        self.session_kwargs: list[dict] = []

    def session(self, **kwargs):
        self.session_kwargs.append(kwargs)
        return FakeSession(self)


def test_neo4j_memory_supports_dependency_injection() -> None:
    driver = FakeDriver()
    store = Neo4jGraphMemory(driver=driver)
    assert store.driver is driver


def test_upsert_node_uses_label_whitelist() -> None:
    driver = FakeDriver()
    store = Neo4jGraphMemory(driver=driver)

    ok = store.upsert_node(GraphNode(id="task:1", label="Task", properties={"run_id": "r1"}))

    assert ok
    assert driver.calls
    assert "MERGE (n:Task" in driver.calls[0][0]


def test_upsert_node_rejects_invalid_label() -> None:
    driver = FakeDriver()
    store = Neo4jGraphMemory(driver=driver)

    ok = store.upsert_node(GraphNode(id="x", label="Hacker", properties={}))

    assert ok is False
    assert driver.calls == []


def test_upsert_edge_uses_relation_whitelist() -> None:
    driver = FakeDriver()
    store = Neo4jGraphMemory(driver=driver)

    ok = store.upsert_edge(
        GraphEdge(source_id="task:1", target_id="profile:1", relation="HAS_PROFILE")
    )

    assert ok
    assert "MERGE (a)-[r:HAS_PROFILE]->(b)" in driver.calls[0][0]


def test_upsert_edge_rejects_invalid_relation() -> None:
    driver = FakeDriver()
    store = Neo4jGraphMemory(driver=driver)

    ok = store.upsert_edge(GraphEdge(source_id="a", target_id="b", relation="PWNED"))

    assert ok is False
    assert driver.calls == []


def test_upsert_task_graph_writes_task_profile_evidence_agent_review() -> None:
    driver = FakeDriver()
    store = Neo4jGraphMemory(driver=driver)
    record = TaskGraphRecord(
        run_id="run-1",
        session_id="session-1",
        task_id="task:run-1",
        task_profile_id="task_profile:run-1",
        task_preview="preview",
        intent="architecture_design",
        domain="ai_agent",
        evidence_refs=[{"id": "evidence:1", "source_type": "doc", "preview": "p"}],
        agent_outputs=[{"id": "agent_output:1", "agent_name": "architect", "preview": "p"}],
        review_reports=[{"id": "review:1", "pass_review": False, "preview": "p"}],
    )

    ok = store.upsert_task_graph(record)

    assert ok
    queries = "\n".join(q for q, _ in driver.calls)
    assert "MERGE (n:Task" in queries
    assert "MERGE (n:TaskProfile" in queries
    assert "MERGE (n:Evidence" in queries
    assert "MERGE (n:AgentRun" in queries
    assert "MERGE (n:ReviewReport" in queries
    assert "MERGE (a)-[r:HAS_PROFILE]->(b)" in queries
    assert "MERGE (a)-[r:USED_EVIDENCE]->(b)" in queries
    assert "MERGE (a)-[r:HAS_OUTPUT]->(b)" in queries
    assert "MERGE (a)-[r:HAS_REVIEW]->(b)" in queries


def test_upsert_code_symbols_writes_file_function_class_and_calls() -> None:
    driver = FakeDriver()
    store = Neo4jGraphMemory(driver=driver)
    symbols = [
        CodeSymbol(
            symbol_id="s:class:A",
            name="A",
            symbol_type="class",
            file_path="x.py",
            module_name="x",
        ),
        CodeSymbol(
            symbol_id="s:function:f",
            name="f",
            symbol_type="function",
            file_path="x.py",
            module_name="x",
            calls=["helper"],
        ),
    ]

    ok = store.upsert_code_symbols(symbols)

    assert ok
    queries = "\n".join(q for q, _ in driver.calls)
    assert "MERGE (n:File" in queries
    assert "MERGE (n:Function" in queries
    assert "MERGE (n:Class" in queries
    assert "MERGE (a)-[r:CONTAINS]->(b)" in queries
    assert "MERGE (a)-[r:CALLS]->(b)" in queries


def test_record_issue_fix_writes_issue_fix_test() -> None:
    driver = FakeDriver()
    store = Neo4jGraphMemory(driver=driver)

    ok = store.record_issue_fix(
        run_id="run-2",
        issue={"title": "null pointer", "preview": "stack trace"},
        fix={"preview": "added guard"},
        test={"name": "test_null_pointer", "status": "pass"},
    )

    assert ok
    queries = "\n".join(q for q, _ in driver.calls)
    assert "MERGE (n:Issue" in queries
    assert "MERGE (n:Fix" in queries
    assert "MERGE (n:Test" in queries
    assert "MERGE (a)-[r:FLAGS]->(b)" in queries
    assert "MERGE (a)-[r:FIXED_BY]->(b)" in queries
    assert "MERGE (a)-[r:TESTS]->(b)" in queries


def test_find_task_lineage_returns_graph_path() -> None:
    driver = FakeDriver()
    driver.read_queue.append(
        [
            {
                "source": {"id": "task:run-1", "label": "Task", "properties": {"run_id": "run-1"}},
                "rel": {"relation": "HAS_PROFILE", "properties": {}},
                "target": {
                    "id": "task_profile:run-1",
                    "label": "TaskProfile",
                    "properties": {"intent": "debugging"},
                },
            }
        ]
    )
    store = Neo4jGraphMemory(driver=driver)

    path = store.find_task_lineage("run-1")

    assert len(path.nodes) == 2
    assert len(path.edges) == 1
    assert path.edges[0].relation == "HAS_PROFILE"


def test_find_related_symbols_returns_paths() -> None:
    driver = FakeDriver()
    driver.read_queue.append(
        [
            {
                "source": {"id": "f1", "label": "Function", "properties": {"name": "foo"}},
                "rel": {"relation": "CALLS", "properties": {}},
                "target": {"id": "f2", "label": "Function", "properties": {"name": "bar"}},
            }
        ]
    )
    store = Neo4jGraphMemory(driver=driver)

    paths = store.find_related_symbols("foo")

    assert len(paths) == 1
    assert paths[0].edges[0].relation == "CALLS"


def test_find_prior_issues_returns_paths() -> None:
    driver = FakeDriver()
    driver.read_queue.append(
        [
            {
                "issue": {"id": "issue:1", "label": "Issue", "properties": {"preview": "null pointer"}},
                "rel": {"relation": "FIXED_BY", "properties": {}},
                "related": {"id": "fix:1", "label": "Fix", "properties": {}},
            }
        ]
    )
    store = Neo4jGraphMemory(driver=driver)

    paths = store.find_prior_issues("null")

    assert len(paths) == 1
    assert paths[0].nodes[0].label == "Issue"


def test_neo4j_exceptions_do_not_raise() -> None:
    driver = FakeDriver()
    driver.raise_error = True
    store = Neo4jGraphMemory(driver=driver)

    assert store.upsert_node(GraphNode(id="x", label="Task")) is False
    assert store.upsert_edge(GraphEdge(source_id="a", target_id="b", relation="RELATED_TO")) is False
    assert store.find_task_lineage("run")
    assert store.find_related_symbols("f") == []
    assert store.find_prior_issues("issue") == []


def test_null_neo4j_memory_interface_compatible() -> None:
    null_store = NullNeo4jGraphMemory()
    assert null_store.upsert_node(GraphNode(id="x", label="Task")) is False
    assert null_store.upsert_edge(GraphEdge(source_id="a", target_id="b", relation="RELATED_TO")) is False
    assert null_store.upsert_task_graph(
        TaskGraphRecord(run_id="r1", task_id="task:r1", task_profile_id="profile:r1")
    ) is False
    assert null_store.upsert_code_symbols([]) is False
    assert (
        null_store.record_issue_fix(run_id="r1", issue={"title": "x"}, fix=None, test=None)
        is False
    )
    assert null_store.find_related_symbols("x") == []
    assert null_store.find_prior_issues("x") == []
