# coding=utf-8
"""Configuration management for Self AI using Pydantic Settings."""

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
_ENV_FILE = PROJECT_ROOT / ".env"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    resolved = (PROJECT_ROOT / path).resolve()
    if resolved.exists():
        return resolved
    if str(value).replace("\\", "/").strip("./") == "model/bge-m3":
        packaged_model = (PACKAGE_ROOT / "model" / "bge-m3").resolve()
        if packaged_model.exists():
            return packaged_model
    return resolved


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        nested_model_default_partial_update=True,
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",),
    )

    use_async: bool = Field(default=False, validation_alias=AliasChoices("SELF_AI_USE_ASYNC", "CF_USE_ASYNC"), description="Toggle async DB operations.")
    use_sparse: bool = Field(default=False, validation_alias=AliasChoices("SELF_AI_USE_SPARSE", "CF_USE_SPARSE"), description="Toggle sparse embeddings.")
    use_gpu: bool = Field(default=False, validation_alias=AliasChoices("SELF_AI_USE_GPU", "CF_USE_GPU"), description="Toggle GPU usage.")
    use_structured: bool = Field(
        default=False,
        validation_alias=AliasChoices("SELF_AI_USE_STRUCTURED", "CF_USE_STRUCTURED"),
        description="Toggle structured outputs.",
    )
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_timeout_s: float = Field(default=2.0, alias="QDRANT_TIMEOUT_S")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_enabled: bool = Field(default=True, alias="NEO4J_ENABLED")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    graph_read_enabled: bool = Field(default=False, alias="GRAPH_READ_ENABLED")
    graph_max_paths: int = Field(default=5, alias="GRAPH_MAX_PATHS")
    graph_timeout_ms: int = Field(default=1000, alias="GRAPH_TIMEOUT_MS")
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_key_prefix: str = Field(default="self_ai", alias="REDIS_KEY_PREFIX")
    redis_default_ttl_seconds: int = Field(
        default=86400,
        alias="REDIS_DEFAULT_TTL_SECONDS",
    )
    redis_enabled: bool = Field(default=True, alias="REDIS_ENABLED")
    embedder_path: str = Field(default="./self_ai/model/bge-m3", validation_alias=AliasChoices("SELF_AI_EMBEDDER_PATH", "CF_EMBEDDER_PATH"))
    embedder_fp16: bool = Field(default=False, validation_alias=AliasChoices("SELF_AI_EMBEDDER_FP16", "CF_EMBEDDER_FP16"))
    embedder_prewarm: bool = Field(default=False, validation_alias=AliasChoices("SELF_AI_EMBEDDER_PREWARM", "CF_EMBEDDER_PREWARM"))
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")
    qdrant_strict_schema: bool = Field(default=True, validation_alias=AliasChoices("SELF_AI_QDRANT_STRICT_SCHEMA", "CF_QDRANT_STRICT_SCHEMA"))
    disable_legacy_retrieval: bool = Field(default=True, validation_alias=AliasChoices("SELF_AI_DISABLE_LEGACY_RETRIEVAL", "CF_DISABLE_LEGACY_RETRIEVAL"))
    trace_sanitize_control: bool = Field(
        default=True,
        validation_alias=AliasChoices("SELF_AI_TRACE_SANITIZE_CONTROL", "CF_TRACE_SANITIZE_CONTROL"),
    )
    trace_repair_mojibake: bool = Field(
        default=True,
        validation_alias=AliasChoices("SELF_AI_TRACE_REPAIR_MOJIBAKE", "CF_TRACE_REPAIR_MOJIBAKE"),
    )

    # LLM gateway: DashScope (Alibaba Bailian) OpenAI-compatible mode.
    dashscope_api_key: Optional[str] = Field(default=None, alias="DASHSCOPE_API_KEY")
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="DASHSCOPE_BASE_URL",
    )
    dashscope_enable_thinking: bool = Field(
        default=True,
        alias="DASHSCOPE_ENABLE_THINKING",
    )
    dashscope_timeout_s: int = Field(default=80, alias="DASHSCOPE_TIMEOUT_S")
    router_selector_model: str = Field(
        default="deepseek-v4-flash",
        validation_alias=AliasChoices("SELF_AI_ROUTER_SELECTOR_MODEL", "CF_ROUTER_SELECTOR_MODEL"),
    )
    router_use_selector: bool = Field(default=False, validation_alias=AliasChoices("SELF_AI_ROUTER_USE_SELECTOR", "CF_ROUTER_USE_SELECTOR"))
    router_default_tier: str = Field(default="balanced", validation_alias=AliasChoices("SELF_AI_ROUTER_DEFAULT_TIER", "CF_ROUTER_DEFAULT_TIER"))
    router_default_enable_thinking: bool = Field(
        default=False,
        validation_alias=AliasChoices("SELF_AI_ROUTER_DEFAULT_ENABLE_THINKING", "CF_ROUTER_DEFAULT_ENABLE_THINKING"),
    )
    router_max_retries: int = Field(default=2, validation_alias=AliasChoices("SELF_AI_ROUTER_MAX_RETRIES", "CF_ROUTER_MAX_RETRIES"))
    permission_mode: str = Field(default="dev_write", validation_alias=AliasChoices("SELF_AI_PERMISSION_MODE", "CF_PERMISSION_MODE"))
    shell_enabled: bool = Field(default=False, validation_alias=AliasChoices("SELF_AI_SHELL_ENABLED", "CF_SHELL_ENABLED"))
    shell_allowlist: str = Field(
        default="git status;git diff;python -m pytest;pytest;ruff check;mypy;streamlit run",
        validation_alias=AliasChoices("SELF_AI_SHELL_ALLOWLIST", "CF_SHELL_ALLOWLIST"),
    )
    shell_denylist: str = Field(
        default="rm,del,sudo,shutdown,reboot,curl,wget,powershell,cmd,bash,sh,pip install",
        validation_alias=AliasChoices("SELF_AI_SHELL_DENYLIST", "CF_SHELL_DENYLIST"),
    )
    prompt_cache_enabled: bool = Field(default=True, validation_alias=AliasChoices("SELF_AI_PROMPT_CACHE_ENABLED", "CF_PROMPT_CACHE_ENABLED"))
    prompt_cache_max_entries: int = Field(default=64, validation_alias=AliasChoices("SELF_AI_PROMPT_CACHE_MAX_ENTRIES", "CF_PROMPT_CACHE_MAX_ENTRIES"))
    prompt_max_chars_mainloop: int = Field(default=180000, validation_alias=AliasChoices("SELF_AI_PROMPT_MAX_CHARS_MAINLOOP", "CF_PROMPT_MAX_CHARS_MAINLOOP"))
    prompt_dynamic_budget_ratio: float = Field(default=0.65, validation_alias=AliasChoices("SELF_AI_PROMPT_DYNAMIC_BUDGET_RATIO", "CF_PROMPT_DYNAMIC_BUDGET_RATIO"))
    prompt_recent_errors: int = Field(default=12, validation_alias=AliasChoices("SELF_AI_PROMPT_RECENT_ERRORS", "CF_PROMPT_RECENT_ERRORS"))
    prompt_recent_tool_results: int = Field(default=24, validation_alias=AliasChoices("SELF_AI_PROMPT_RECENT_TOOL_RESULTS", "CF_PROMPT_RECENT_TOOL_RESULTS"))
    prompt_selected_agents_max: int = Field(default=16, validation_alias=AliasChoices("SELF_AI_PROMPT_SELECTED_AGENTS_MAX", "CF_PROMPT_SELECTED_AGENTS_MAX"))
    prompt_tool_catalog_max: int = Field(default=160, validation_alias=AliasChoices("SELF_AI_PROMPT_TOOL_CATALOG_MAX", "CF_PROMPT_TOOL_CATALOG_MAX"))
    mainloop_goal_contract_model_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("SELF_AI_MAINLOOP_GOAL_CONTRACT_MODEL_ENABLED", "CF_MAINLOOP_GOAL_CONTRACT_MODEL_ENABLED"),
    )
    mainloop_goal_contract_verify_vote_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("SELF_AI_MAINLOOP_GOAL_CONTRACT_VERIFY_VOTE_ENABLED", "CF_MAINLOOP_GOAL_CONTRACT_VERIFY_VOTE_ENABLED"),
    )
    mainloop_semantic_completion_gate_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("SELF_AI_MAINLOOP_SEMANTIC_COMPLETION_GATE_ENABLED", "CF_MAINLOOP_SEMANTIC_COMPLETION_GATE_ENABLED"),
    )
    mainloop_semantic_gate_side_effect_only: bool = Field(
        default=True,
        validation_alias=AliasChoices("SELF_AI_MAINLOOP_SEMANTIC_GATE_SIDE_EFFECT_ONLY", "CF_MAINLOOP_SEMANTIC_GATE_SIDE_EFFECT_ONLY"),
    )
    chat_memory_enabled: bool = Field(default=True, validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_ENABLED", "CF_CHAT_MEMORY_ENABLED"))
    chat_memory_root: str = Field(default="./memory/chats", validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_ROOT", "CF_CHAT_MEMORY_ROOT"))
    chat_memory_max_shard_bytes: int = Field(
        default=1_048_576,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_MAX_SHARD_BYTES", "CF_CHAT_MEMORY_MAX_SHARD_BYTES"),
    )
    chat_memory_max_turns_per_shard: int = Field(
        default=200,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_MAX_TURNS_PER_SHARD", "CF_CHAT_MEMORY_MAX_TURNS_PER_SHARD"),
    )
    chat_memory_recent_turns: int = Field(
        default=16,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_RECENT_TURNS", "CF_CHAT_MEMORY_RECENT_TURNS"),
    )
    chat_memory_injection_max_chars: int = Field(
        default=96000,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_INJECTION_MAX_CHARS", "CF_CHAT_MEMORY_INJECTION_MAX_CHARS"),
    )
    chat_memory_injection_item_max_chars: int = Field(
        default=6000,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_INJECTION_ITEM_MAX_CHARS", "CF_CHAT_MEMORY_INJECTION_ITEM_MAX_CHARS"),
    )
    chat_memory_sidecar_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_SIDECAR_ENABLED", "CF_CHAT_MEMORY_SIDECAR_ENABLED"),
    )
    chat_memory_sidecar_timeout_s: int = Field(
        default=14,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_SIDECAR_TIMEOUT_S", "CF_CHAT_MEMORY_SIDECAR_TIMEOUT_S"),
    )
    chat_memory_sidecar_retry_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_SIDECAR_RETRY_ENABLED", "CF_CHAT_MEMORY_SIDECAR_RETRY_ENABLED"),
    )
    chat_memory_sidecar_fast_retry_timeout_s: int = Field(
        default=6,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_SIDECAR_FAST_RETRY_TIMEOUT_S", "CF_CHAT_MEMORY_SIDECAR_FAST_RETRY_TIMEOUT_S"),
    )
    chat_memory_fusion_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_FUSION_ENABLED", "CF_CHAT_MEMORY_FUSION_ENABLED"),
    )
    chat_memory_l2_collection_default: str = Field(
        default="cf_chat_memory",
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_L2_COLLECTION_DEFAULT", "CF_CHAT_MEMORY_L2_COLLECTION_DEFAULT"),
    )
    chat_memory_compact_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_COMPACT_ENABLED", "CF_CHAT_MEMORY_COMPACT_ENABLED"),
    )
    chat_memory_compact_min_shards: int = Field(
        default=3,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_COMPACT_MIN_SHARDS", "CF_CHAT_MEMORY_COMPACT_MIN_SHARDS"),
    )
    chat_memory_cold_after_days: int = Field(
        default=7,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_COLD_AFTER_DAYS", "CF_CHAT_MEMORY_COLD_AFTER_DAYS"),
    )
    chat_memory_summary_max_chars: int = Field(
        default=4000,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_SUMMARY_MAX_CHARS", "CF_CHAT_MEMORY_SUMMARY_MAX_CHARS"),
    )
    chat_memory_compact_max_docs: int = Field(
        default=400,
        validation_alias=AliasChoices("SELF_AI_CHAT_MEMORY_COMPACT_MAX_DOCS", "CF_CHAT_MEMORY_COMPACT_MAX_DOCS"),
    )

    # Model tiers (Claude-Code style): fast / balanced / high / code
    model_fast: str = Field(default="qwen3.6-flash", validation_alias=AliasChoices("SELF_AI_MODEL_FAST", "CF_MODEL_FAST"))
    model_balanced: str = Field(
        default="qwen3.5-plus-2026-02-15",
        validation_alias=AliasChoices("SELF_AI_MODEL_BALANCED", "CF_MODEL_BALANCED"),
    )
    model_high: str = Field(default="qwen3.6-plus", validation_alias=AliasChoices("SELF_AI_MODEL_HIGH", "CF_MODEL_HIGH"))
    model_code: str = Field(default="kimi-k2.6", validation_alias=AliasChoices("SELF_AI_MODEL_CODE", "CF_MODEL_CODE"))

    @field_validator(
        "use_async",
        "use_sparse",
        "use_gpu",
        "use_structured",
        "redis_enabled",
        "neo4j_enabled",
        "graph_read_enabled",
        "embedder_fp16",
        "embedder_prewarm",
        "qdrant_strict_schema",
        "disable_legacy_retrieval",
        "trace_sanitize_control",
        "trace_repair_mojibake",
        "dashscope_enable_thinking",
        "router_use_selector",
        "router_default_enable_thinking",
        "shell_enabled",
        "prompt_cache_enabled",
        "mainloop_goal_contract_model_enabled",
        "mainloop_goal_contract_verify_vote_enabled",
        "mainloop_semantic_completion_gate_enabled",
        "mainloop_semantic_gate_side_effect_only",
        "chat_memory_enabled",
        "chat_memory_sidecar_enabled",
        "chat_memory_sidecar_retry_enabled",
        "chat_memory_fusion_enabled",
        "chat_memory_compact_enabled",
        mode="before",
    )
    @classmethod
    def parse_bool(cls, v: str) -> bool:
        """Parse boolean from string per Pydantic best practices."""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    @model_validator(mode="after")
    def check_api_keys(self) -> "Settings":
        """Validate critical API keys.

        API key validation is deferred to runtime call sites so local startup and
        non-model workflows can still run without immediate hard failure.
        """
        if self.router_max_retries < 1:
            raise ValueError("SELF_AI_ROUTER_MAX_RETRIES must be >= 1")
        if self.router_default_tier not in {"fast", "balanced", "high", "code"}:
            raise ValueError("SELF_AI_ROUTER_DEFAULT_TIER must be one of fast|balanced|high|code")
        if self.dashscope_timeout_s < 5:
            raise ValueError("DASHSCOPE_TIMEOUT_S must be >= 5")
        if self.qdrant_timeout_s <= 0:
            raise ValueError("QDRANT_TIMEOUT_S must be > 0")
        if self.permission_mode not in {"readonly", "dev_write", "dev_full"}:
            raise ValueError("SELF_AI_PERMISSION_MODE must be one of readonly|dev_write|dev_full")
        if self.prompt_cache_max_entries < 1:
            raise ValueError("SELF_AI_PROMPT_CACHE_MAX_ENTRIES must be >= 1")
        if self.prompt_max_chars_mainloop < 1200:
            raise ValueError("SELF_AI_PROMPT_MAX_CHARS_MAINLOOP must be >= 1200")
        if not (0.1 <= self.prompt_dynamic_budget_ratio <= 0.8):
            raise ValueError("SELF_AI_PROMPT_DYNAMIC_BUDGET_RATIO must be in [0.1, 0.8]")
        if self.prompt_recent_errors < 1:
            raise ValueError("SELF_AI_PROMPT_RECENT_ERRORS must be >= 1")
        if self.prompt_recent_tool_results < 1:
            raise ValueError("SELF_AI_PROMPT_RECENT_TOOL_RESULTS must be >= 1")
        if self.prompt_selected_agents_max < 1:
            raise ValueError("SELF_AI_PROMPT_SELECTED_AGENTS_MAX must be >= 1")
        if self.prompt_tool_catalog_max < 1:
            raise ValueError("SELF_AI_PROMPT_TOOL_CATALOG_MAX must be >= 1")
        if self.chat_memory_max_shard_bytes < 4096:
            raise ValueError("SELF_AI_CHAT_MEMORY_MAX_SHARD_BYTES must be >= 4096")
        if self.chat_memory_max_turns_per_shard < 10:
            raise ValueError("SELF_AI_CHAT_MEMORY_MAX_TURNS_PER_SHARD must be >= 10")
        if self.chat_memory_recent_turns < 1:
            raise ValueError("SELF_AI_CHAT_MEMORY_RECENT_TURNS must be >= 1")
        if self.chat_memory_injection_max_chars < 400:
            raise ValueError("SELF_AI_CHAT_MEMORY_INJECTION_MAX_CHARS must be >= 400")
        if self.chat_memory_injection_item_max_chars < 80:
            raise ValueError("SELF_AI_CHAT_MEMORY_INJECTION_ITEM_MAX_CHARS must be >= 80")
        if self.chat_memory_injection_item_max_chars > self.chat_memory_injection_max_chars:
            raise ValueError(
                "SELF_AI_CHAT_MEMORY_INJECTION_ITEM_MAX_CHARS must be <= SELF_AI_CHAT_MEMORY_INJECTION_MAX_CHARS"
            )
        dynamic_budget = int(self.prompt_max_chars_mainloop * self.prompt_dynamic_budget_ratio)
        # Reserve most dynamic budget for chat recent turns + execution state + tool summaries.
        # Keep a 10% headroom to avoid accidental truncation pressure.
        if self.chat_memory_injection_max_chars > int(dynamic_budget * 0.9):
            raise ValueError(
                "SELF_AI_CHAT_MEMORY_INJECTION_MAX_CHARS is too large for prompt dynamic budget; "
                "reduce it or increase prompt budget."
            )
        if self.chat_memory_sidecar_timeout_s < 5:
            raise ValueError("SELF_AI_CHAT_MEMORY_SIDECAR_TIMEOUT_S must be >= 5")
        if self.chat_memory_sidecar_fast_retry_timeout_s < 1:
            raise ValueError("SELF_AI_CHAT_MEMORY_SIDECAR_FAST_RETRY_TIMEOUT_S must be >= 1")
        if self.chat_memory_sidecar_fast_retry_timeout_s >= self.chat_memory_sidecar_timeout_s:
            raise ValueError(
                "SELF_AI_CHAT_MEMORY_SIDECAR_FAST_RETRY_TIMEOUT_S must be < SELF_AI_CHAT_MEMORY_SIDECAR_TIMEOUT_S"
            )
        if not str(self.chat_memory_l2_collection_default or "").strip():
            raise ValueError("SELF_AI_CHAT_MEMORY_L2_COLLECTION_DEFAULT must be non-empty")
        if self.chat_memory_compact_min_shards < 2:
            raise ValueError("SELF_AI_CHAT_MEMORY_COMPACT_MIN_SHARDS must be >= 2")
        if self.chat_memory_cold_after_days < 1:
            raise ValueError("SELF_AI_CHAT_MEMORY_COLD_AFTER_DAYS must be >= 1")
        if self.chat_memory_summary_max_chars < 200:
            raise ValueError("SELF_AI_CHAT_MEMORY_SUMMARY_MAX_CHARS must be >= 200")
        if self.chat_memory_compact_max_docs < 1:
            raise ValueError("SELF_AI_CHAT_MEMORY_COMPACT_MAX_DOCS must be >= 1")
        return self


settings = Settings()  # Global instance for app-wide use
