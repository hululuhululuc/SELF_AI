# coding=utf-8
"""Storage policy builders for Phase 1 storage-aware task understanding."""

from .schemas import MemoryWritePolicy, RetrievalPolicy, TaskIntent, TaskProfile


def build_memory_write_policy(profile: TaskProfile) -> MemoryWritePolicy:
    """Build memory write policy from task intent.

    Phase 1 only generates strategy and does not perform real writes.
    """
    intent = profile.intent
    policy = MemoryWritePolicy(
        write_redis=True,
        write_qdrant=False,
        write_neo4j=False,
        reason=f"default policy for intent={intent.value}",
    )

    if intent in {TaskIntent.ARCHITECTURE_DESIGN, TaskIntent.PLANNING}:
        policy.write_qdrant = True
        policy.write_neo4j = True
        policy.qdrant_collection = "cf_task_memory"
        policy.neo4j_label = "Task"
        policy.reason = "architecture/planning needs task memory and graph trace"
        return policy

    if intent in {TaskIntent.CODE_REVIEW, TaskIntent.DEBUGGING}:
        policy.write_qdrant = True
        policy.write_neo4j = True
        policy.qdrant_collection = "cf_review_memory"
        policy.neo4j_label = "Issue"
        policy.reason = "debug/review tasks benefit from issue and fix memory"
        return policy

    if intent in {TaskIntent.RESEARCH_SUMMARY, TaskIntent.RAG_ANSWERING}:
        policy.write_qdrant = True
        policy.qdrant_collection = "cf_doc_chunks"
        policy.reason = "research tasks retain reusable document memory"
        return policy

    if intent == TaskIntent.GENERAL_QA:
        policy.reason = "general QA keeps only short-term runtime state"
        return policy

    return policy


def build_retrieval_policy(profile: TaskProfile) -> RetrievalPolicy:
    """Build retrieval policy from task intent."""
    intent = profile.intent
    policy = RetrievalPolicy(
        use_vector=True,
        use_graph=True,
        use_web=False,
        use_code_index=False,
        qdrant_collections=[],
        neo4j_node_labels=[],
        allowed_sources=[],
    )

    if intent == TaskIntent.CODE_GENERATION:
        policy.use_graph = False
        policy.use_code_index = True
        policy.qdrant_collections = ["cf_code_chunks", "cf_doc_chunks"]
        policy.neo4j_node_labels = []
        return policy

    if intent in {
        TaskIntent.CODE_MODIFICATION,
        TaskIntent.DEBUGGING,
        TaskIntent.CODE_REVIEW,
    }:
        policy.use_graph = True
        policy.use_code_index = True
        policy.qdrant_collections = ["cf_code_chunks", "cf_review_memory"]
        policy.neo4j_node_labels = ["File", "Function", "Issue", "Fix"]
        return policy

    if intent in {TaskIntent.ARCHITECTURE_DESIGN, TaskIntent.PLANNING}:
        policy.use_graph = True
        policy.use_code_index = False
        policy.qdrant_collections = [
            "cf_task_memory",
            "cf_doc_chunks",
            "cf_review_memory",
        ]
        policy.neo4j_node_labels = ["Task", "Plan", "Decision", "Module"]
        return policy

    if intent in {TaskIntent.RESEARCH_SUMMARY, TaskIntent.RAG_ANSWERING}:
        policy.use_graph = False
        policy.use_web = True
        policy.use_code_index = False
        policy.qdrant_collections = ["cf_doc_chunks", "cf_web_chunks"]
        policy.neo4j_node_labels = []
        return policy

    if intent == TaskIntent.DOCUMENT_WRITING:
        policy.use_graph = False
        policy.use_code_index = False
        policy.qdrant_collections = ["cf_doc_chunks", "cf_task_memory"]
        policy.neo4j_node_labels = []
        return policy

    if intent == TaskIntent.GENERAL_QA:
        policy.use_graph = False
        policy.use_code_index = False
        policy.qdrant_collections = ["cf_doc_chunks"]
        policy.neo4j_node_labels = []
        return policy

    return policy


def enrich_profile_with_storage_policies(profile: TaskProfile) -> TaskProfile:
    """Attach memory and retrieval policies to profile."""
    profile.memory_write_policy = build_memory_write_policy(profile)
    profile.retrieval_policy = build_retrieval_policy(profile)
    return profile
