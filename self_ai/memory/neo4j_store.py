# coding=utf-8
"""Neo4j structural graph memory store for Phase 4."""

from __future__ import annotations

import hashlib
from typing import Any

from ..graph.graph_models import CodeSymbol, GraphEdge, GraphNode, GraphPath, TaskGraphRecord
from ..graph.task_graph_builder import task_graph_to_nodes_edges
from ..observability import trace


class Neo4jGraphMemory:
    """Neo4j-backed structural graph memory with safety guards.

    Driver/session must be injected for testability.
    """

    LABEL_WHITELIST: set[str] = {
        "Task",
        "TaskProfile",
        "Plan",
        "PlanStep",
        "AgentRun",
        "ReviewReport",
        "Issue",
        "Fix",
        "File",
        "Module",
        "Class",
        "Function",
        "Test",
        "Decision",
        "Evidence",
    }

    RELATION_WHITELIST: set[str] = {
        "HAS_PROFILE",
        "HAS_PLAN",
        "HAS_STEP",
        "EXECUTED_BY",
        "PRODUCED",
        "FLAGS",
        "FIXED_BY",
        "CONTAINS",
        "CALLS",
        "TESTS",
        "USED_EVIDENCE",
        "AFFECTS",
        "HAS_OUTPUT",
        "HAS_REVIEW",
        "RELATED_TO",
    }

    def __init__(
        self,
        driver: Any | None = None,
        *,
        session_factory: Any | None = None,
        database: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.driver = driver
        self.session_factory = session_factory
        self.database = database
        self.enabled = enabled

    def _is_valid_label(self, label: str) -> bool:
        return label in self.LABEL_WHITELIST

    def _is_valid_relation(self, relation: str) -> bool:
        return relation in self.RELATION_WHITELIST

    def _get_session(self) -> Any:
        if self.session_factory is not None:
            return self.session_factory()
        if self.driver is None:
            raise RuntimeError("Neo4j driver is not configured")
        kwargs: dict[str, Any] = {}
        if self.database:
            kwargs["database"] = self.database
        return self.driver.session(**kwargs)

    @staticmethod
    def _record_to_dict(record: Any) -> dict[str, Any]:
        if isinstance(record, dict):
            return record
        if hasattr(record, "data") and callable(record.data):
            data = record.data()
            if isinstance(data, dict):
                return data
        if hasattr(record, "items"):
            try:
                return dict(record.items())
            except Exception:
                pass
        if hasattr(record, "keys") and hasattr(record, "__getitem__"):
            try:
                return {k: record[k] for k in record.keys()}
            except Exception:
                pass
        return {}

    def _run_write(self, cypher: str, params: dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        try:
            with self._get_session() as session:
                result = session.run(cypher, params)
                if hasattr(result, "consume"):
                    result.consume()
            return True
        except Exception as exc:
            trace(
                "neo4j_store.write.error",
                error=type(exc).__name__,
                message=str(exc)[:200],
            )
            return False

    def _run_read(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            with self._get_session() as session:
                result = session.run(cypher, params)
                if isinstance(result, list):
                    raw_records = result
                elif hasattr(result, "data") and callable(result.data):
                    raw_records = result.data()
                else:
                    raw_records = list(result)
                return [self._record_to_dict(item) for item in raw_records]
        except Exception as exc:
            trace(
                "neo4j_store.read.error",
                error=type(exc).__name__,
                message=str(exc)[:200],
            )
            return []

    @staticmethod
    def _safe_id(prefix: str, value: str) -> str:
        digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()
        return f"{prefix}:{digest}"

    @staticmethod
    def _extract_node(raw: Any, fallback_label: str = "Task") -> GraphNode | None:
        if raw is None:
            return None
        if isinstance(raw, GraphNode):
            return raw
        if isinstance(raw, dict):
            props = raw.get("properties")
            if not isinstance(props, dict):
                props = {
                    k: v
                    for k, v in raw.items()
                    if k not in {"id", "label", "labels", "properties"}
                }
            labels = raw.get("labels")
            label = str(raw.get("label") or (labels[0] if isinstance(labels, list) and labels else fallback_label))
            node_id = str(raw.get("id") or props.get("id") or "")
            if not node_id:
                return None
            return GraphNode(id=node_id, label=label, properties=props)

        props: dict[str, Any] = {}
        try:
            props = dict(raw)
        except Exception:
            props = {}

        label = fallback_label
        labels = getattr(raw, "labels", None)
        if labels:
            try:
                label = next(iter(labels))
            except Exception:
                pass

        node_id = str(props.get("id") or "")
        if not node_id and hasattr(raw, "get"):
            try:
                node_id = str(raw.get("id") or "")
            except Exception:
                node_id = ""
        if not node_id:
            return None
        return GraphNode(id=node_id, label=label, properties=props)

    @staticmethod
    def _extract_edge(
        raw: Any,
        *,
        source_id: str,
        target_id: str,
        fallback_relation: str = "RELATED_TO",
    ) -> GraphEdge:
        if isinstance(raw, GraphEdge):
            return raw
        if isinstance(raw, dict):
            rel = str(raw.get("relation") or raw.get("type") or fallback_relation)
            props = raw.get("properties")
            if not isinstance(props, dict):
                props = {
                    k: v
                    for k, v in raw.items()
                    if k
                    not in {
                        "relation",
                        "type",
                        "properties",
                        "source_id",
                        "target_id",
                    }
                }
            return GraphEdge(
                source_id=str(raw.get("source_id") or source_id),
                target_id=str(raw.get("target_id") or target_id),
                relation=rel,
                properties=props,
            )

        rel = getattr(raw, "type", None)
        relation = str(rel or fallback_relation)
        props: dict[str, Any] = {}
        try:
            props = dict(raw)
        except Exception:
            props = {}
        return GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            properties=props,
        )

    def upsert_node(self, node: GraphNode) -> bool:
        if not self._is_valid_label(node.label):
            return False
        cypher = f"MERGE (n:{node.label} {{id: $id}}) SET n += $props"
        params = {"id": node.id, "props": dict(node.properties)}
        return self._run_write(cypher, params)

    def upsert_edge(self, edge: GraphEdge) -> bool:
        if not self._is_valid_relation(edge.relation):
            return False
        cypher = (
            f"MATCH (a {{id: $source_id}}) "
            f"MATCH (b {{id: $target_id}}) "
            f"MERGE (a)-[r:{edge.relation}]->(b) "
            "SET r += $props"
        )
        params = {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "props": dict(edge.properties),
        }
        return self._run_write(cypher, params)

    def upsert_task_graph(self, record: TaskGraphRecord) -> bool:
        nodes, edges = task_graph_to_nodes_edges(record)
        ok = True
        for node in nodes:
            ok = self.upsert_node(node) and ok
        for edge in edges:
            ok = self.upsert_edge(edge) and ok
        return ok

    def upsert_code_symbols(self, symbols: list[CodeSymbol]) -> bool:
        ok = True
        file_nodes: dict[str, GraphNode] = {}

        for symbol in symbols:
            file_id = f"file:{symbol.file_path}"
            if file_id not in file_nodes:
                file_nodes[file_id] = GraphNode(
                    id=file_id,
                    label="File",
                    properties={"path": symbol.file_path, "module": symbol.module_name},
                )

        for node in file_nodes.values():
            ok = self.upsert_node(node) and ok

        for symbol in symbols:
            label = "Class" if symbol.symbol_type == "class" else "Function"
            symbol_node = GraphNode(
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
            ok = self.upsert_node(symbol_node) and ok

            file_id = f"file:{symbol.file_path}"
            ok = (
                self.upsert_edge(
                    GraphEdge(
                        source_id=file_id,
                        target_id=symbol.symbol_id,
                        relation="CONTAINS",
                        properties={},
                    )
                )
                and ok
            )

            if symbol.symbol_type == "class":
                continue

            for callee in symbol.calls:
                if not callee:
                    continue
                call_target_id = f"call_target:{symbol.file_path}:{callee}"
                call_target_node = GraphNode(
                    id=call_target_id,
                    label="Function",
                    properties={
                        "name": callee,
                        "symbol_type": "call_target",
                        "file_path": symbol.file_path,
                        "external_ref": True,
                    },
                )
                ok = self.upsert_node(call_target_node) and ok
                ok = (
                    self.upsert_edge(
                        GraphEdge(
                            source_id=symbol.symbol_id,
                            target_id=call_target_id,
                            relation="CALLS",
                            properties={},
                        )
                    )
                    and ok
                )

        return ok

    def record_issue_fix(
        self,
        *,
        run_id: str,
        issue: dict[str, Any],
        fix: dict[str, Any] | None = None,
        test: dict[str, Any] | None = None,
    ) -> bool:
        ok = True
        task_id = f"task:{run_id}"
        issue_preview = str(issue.get("preview") or issue.get("title") or issue.get("description") or "")
        issue_id = str(issue.get("id") or self._safe_id("issue", f"{run_id}:{issue_preview}"))

        ok = self.upsert_node(
            GraphNode(id=task_id, label="Task", properties={"run_id": run_id})
        ) and ok
        ok = self.upsert_node(
            GraphNode(
                id=issue_id,
                label="Issue",
                properties={
                    "run_id": run_id,
                    "title": issue.get("title"),
                    "preview": issue_preview[:200],
                    "severity": issue.get("severity"),
                },
            )
        ) and ok
        ok = (
            self.upsert_edge(
                GraphEdge(
                    source_id=task_id,
                    target_id=issue_id,
                    relation="FLAGS",
                    properties={"run_id": run_id},
                )
            )
            and ok
        )

        fix_id: str | None = None
        if fix:
            fix_preview = str(fix.get("preview") or fix.get("summary") or "")
            fix_id = str(fix.get("id") or self._safe_id("fix", f"{run_id}:{fix_preview}"))
            ok = self.upsert_node(
                GraphNode(
                    id=fix_id,
                    label="Fix",
                    properties={
                        "run_id": run_id,
                        "preview": fix_preview[:200],
                        "patch_ref": fix.get("patch_ref"),
                    },
                )
            ) and ok
            ok = (
                self.upsert_edge(
                    GraphEdge(
                        source_id=issue_id,
                        target_id=fix_id,
                        relation="FIXED_BY",
                        properties={"run_id": run_id},
                    )
                )
                and ok
            )

        if test:
            test_name = str(test.get("name") or "test")
            test_id = str(test.get("id") or self._safe_id("test", f"{run_id}:{test_name}"))
            ok = self.upsert_node(
                GraphNode(
                    id=test_id,
                    label="Test",
                    properties={
                        "run_id": run_id,
                        "name": test_name,
                        "status": test.get("status"),
                    },
                )
            ) and ok
            target_id = fix_id or issue_id
            ok = (
                self.upsert_edge(
                    GraphEdge(
                        source_id=test_id,
                        target_id=target_id,
                        relation="TESTS",
                        properties={"run_id": run_id},
                    )
                )
                and ok
            )

        return ok

    def find_task_lineage(self, run_id: str, limit: int = 20) -> GraphPath:
        cypher = (
            "MATCH (t:Task {run_id: $run_id}) "
            "OPTIONAL MATCH (t)-[r]->(n) "
            "RETURN t AS source, labels(t) AS source_labels, "
            "r AS rel, type(r) AS rel_type, n AS target, labels(n) AS target_labels "
            "LIMIT $limit"
        )
        rows = self._run_read(cypher, {"run_id": run_id, "limit": int(limit)})
        if not rows:
            return GraphPath(nodes=[], edges=[], metadata={"run_id": run_id, "result_count": 0})

        nodes_by_id: dict[str, GraphNode] = {}
        edges: list[GraphEdge] = []

        for row in rows:
            source_labels = row.get("source_labels", [])
            target_labels = row.get("target_labels", [])
            source_label = (
                str(source_labels[0])
                if isinstance(source_labels, list) and source_labels
                else "Task"
            )
            target_label = (
                str(target_labels[0])
                if isinstance(target_labels, list) and target_labels
                else "Task"
            )
            src = self._extract_node(row.get("source"), fallback_label=source_label)
            tgt = self._extract_node(row.get("target"), fallback_label=target_label)
            if src:
                nodes_by_id[src.id] = src
            if tgt:
                nodes_by_id[tgt.id] = tgt
            if src and tgt and row.get("rel") is not None:
                edges.append(
                    self._extract_edge(
                        row.get("rel"),
                        source_id=src.id,
                        target_id=tgt.id,
                        fallback_relation=str(row.get("rel_type") or "RELATED_TO"),
                    )
                )

        return GraphPath(
            nodes=list(nodes_by_id.values()),
            edges=edges,
            metadata={"run_id": run_id, "result_count": len(edges)},
        )

    def find_related_symbols(self, symbol_name: str, limit: int = 10) -> list[GraphPath]:
        cypher = (
            "MATCH (s)-[r]-(n) "
            "WHERE (s:Function OR s:Class) AND (n:Function OR n:Class OR n:File) "
            "AND (s.name = $symbol_name OR n.name = $symbol_name) "
            "RETURN s AS source, r AS rel, n AS target LIMIT $limit"
        )
        rows = self._run_read(cypher, {"symbol_name": symbol_name, "limit": int(limit)})
        paths: list[GraphPath] = []
        for row in rows:
            src = self._extract_node(row.get("source"), fallback_label="Function")
            tgt = self._extract_node(row.get("target"), fallback_label="Function")
            if not src or not tgt:
                continue
            edge = self._extract_edge(
                row.get("rel"),
                source_id=src.id,
                target_id=tgt.id,
                fallback_relation="RELATED_TO",
            )
            paths.append(GraphPath(nodes=[src, tgt], edges=[edge], metadata={"symbol_name": symbol_name}))
        return paths

    def find_prior_issues(self, query: str, limit: int = 10) -> list[GraphPath]:
        cypher = (
            "MATCH (i:Issue) "
            "WHERE toLower(coalesce(i.preview, '')) CONTAINS toLower($query) "
            "OPTIONAL MATCH (i)-[r]-(n) "
            "RETURN i AS issue, r AS rel, n AS related LIMIT $limit"
        )
        rows = self._run_read(cypher, {"query": query, "limit": int(limit)})
        paths: list[GraphPath] = []
        for row in rows:
            issue_node = self._extract_node(row.get("issue"), fallback_label="Issue")
            related_node = self._extract_node(row.get("related"), fallback_label="Task")
            if issue_node is None:
                continue
            if related_node is None:
                paths.append(GraphPath(nodes=[issue_node], edges=[], metadata={"query": query}))
                continue
            edge = self._extract_edge(
                row.get("rel"),
                source_id=issue_node.id,
                target_id=related_node.id,
                fallback_relation="RELATED_TO",
            )
            paths.append(
                GraphPath(nodes=[issue_node, related_node], edges=[edge], metadata={"query": query})
            )
        return paths


class NullNeo4jGraphMemory:
    """No-op Neo4j memory for disabled runtime or failure isolation."""

    def upsert_node(self, node: GraphNode) -> bool:
        return False

    def upsert_edge(self, edge: GraphEdge) -> bool:
        return False

    def upsert_task_graph(self, record: TaskGraphRecord) -> bool:
        return False

    def upsert_code_symbols(self, symbols: list[CodeSymbol]) -> bool:
        return False

    def record_issue_fix(
        self,
        *,
        run_id: str,
        issue: dict[str, Any],
        fix: dict[str, Any] | None = None,
        test: dict[str, Any] | None = None,
    ) -> bool:
        return False

    def find_task_lineage(self, run_id: str, limit: int = 20) -> GraphPath:
        return GraphPath(nodes=[], edges=[], metadata={"run_id": run_id, "result_count": 0})

    def find_related_symbols(self, symbol_name: str, limit: int = 10) -> list[GraphPath]:
        return []

    def find_prior_issues(self, query: str, limit: int = 10) -> list[GraphPath]:
        return []
