# coding=utf-8
"""Core schemas for storage-aware task understanding (Phase 1)."""

from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskIntent(StrEnum):
    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    DEBUGGING = "debugging"
    CODE_REVIEW = "code_review"
    ARCHITECTURE_DESIGN = "architecture_design"
    RESEARCH_SUMMARY = "research_summary"
    RAG_ANSWERING = "rag_answering"
    DOCUMENT_WRITING = "document_writing"
    PLANNING = "planning"
    GENERAL_QA = "general_qa"


class TaskDomain(StrEnum):
    SOFTWARE = "software"
    AI_AGENT = "ai_agent"
    RAG = "rag"
    DATA = "data"
    ACADEMIC = "academic"
    GENERAL = "general"


class ExpectedArtifact(StrEnum):
    ANSWER = "answer"
    CODE_PATCH = "code_patch"
    CODE_REVIEW_REPORT = "code_review_report"
    DESIGN_DOC = "design_doc"
    IMPLEMENTATION_PLAN = "implementation_plan"
    RESEARCH_REPORT = "research_report"
    TEST_PLAN = "test_plan"


class ComplexityLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StorageTarget(StrEnum):
    REDIS = "redis"
    QDRANT = "qdrant"
    NEO4J = "neo4j"


class RunIdentity(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = "default"
    user_id: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class SurfaceSignals(BaseModel):
    has_code_block: bool = False
    has_file_path: bool = False
    has_error_trace: bool = False
    has_url: bool = False
    has_chinese: bool = False
    has_markdown_table: bool = False
    length: int = 0


class MemoryWritePolicy(BaseModel):
    write_redis: bool = True
    write_qdrant: bool = False
    write_neo4j: bool = False
    redis_ttl_seconds: int | None = 86400
    qdrant_collection: str | None = None
    neo4j_label: str | None = None
    reason: str = ""


class RetrievalPolicy(BaseModel):
    use_vector: bool = True
    use_graph: bool = True
    use_web: bool = False
    use_code_index: bool = False

    top_k_vector: int = 8
    top_k_graph: int = 5
    rerank: bool = True
    freshness_required: bool = False

    qdrant_collections: list[str] = Field(default_factory=list)
    neo4j_node_labels: list[str] = Field(default_factory=list)
    allowed_sources: list[str] = Field(default_factory=list)
    min_score: float = 0.0
    max_context_chars: int = 6000
    collection_weights: dict[str, float] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    content: str
    source_type: str = "unknown"
    source: str | None = None
    path: str | None = None
    symbol: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    title: str | None = None
    url: str | None = None
    token_count: int = 0
    created_at: str | None = None
    embedding_model: str | None = None
    metadata: dict = Field(default_factory=dict)


class TaskProfile(BaseModel):
    identity: RunIdentity = Field(default_factory=RunIdentity)

    intent: TaskIntent
    domain: TaskDomain = TaskDomain.GENERAL

    requires_code: bool = False
    requires_repo_context: bool = False
    requires_research: bool = False
    requires_planning: bool = False
    requires_review: bool = False

    expected_artifact: ExpectedArtifact = ExpectedArtifact.ANSWER
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    risk_level: RiskLevel = RiskLevel.MEDIUM

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""

    surface_signals: SurfaceSignals = Field(default_factory=SurfaceSignals)
    memory_write_policy: MemoryWritePolicy = Field(default_factory=MemoryWritePolicy)
    retrieval_policy: RetrievalPolicy = Field(default_factory=RetrievalPolicy)


class AgentOutput(BaseModel):
    run_id: str
    agent_name: str
    output_type: str
    content: str
    metadata: dict = Field(default_factory=dict)


class ReviewReport(BaseModel):
    run_id: str
    pass_review: bool
    major_issues: list[str] = Field(default_factory=list)
    minor_issues: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    revision_instruction: str = ""
    risk_level: RiskLevel = RiskLevel.MEDIUM
    storage_evidence: list[EvidenceItem] = Field(default_factory=list)
