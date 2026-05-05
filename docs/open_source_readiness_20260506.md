# Open-Source Readiness Summary (2026-05-06)

## 1) Frontend/Backend Compatibility Check

Execution environment:
- Python interpreter: `.venv\\Scripts\\python.exe`
- Date: 2026-05-06

Results:
- `tests/test_frontend_app.py` + `tests/test_frontend_formatting.py` + `tests/test_frontend_smoke.py` + `tests/test_main.py`: **34 passed**
- `tests/test_engine_loop_goal_contract.py`: **8 passed**
- `tests/test_engine_state.py` + `tests/test_config_memory_params.py`: **7 passed**
- Runtime smoke: `python -m self_ai.frontend_smoke` completed without error.

Note:
- System Python in this machine is missing `redis` package; use project `.venv` for release validation.

## 2) Capability-Oriented Metrics (README Recommended View)

Instead of only showing final pass/fail, present agent ability in four dimensions: execution reliability, structured tool competence, code-task effectiveness, and memory-grounded reasoning.

### A. Execution Reliability

| Metric | Value | Source |
|---|---:|---|
| Runtime smoke workflow success | **5/5 (100%)** | `benchmarks/runs/runtime_smoke_batch01_offset0_size5_20260505_1044/metrics.json` |
| Runtime smoke avg latency | 20.628s | same as above |
| Runtime smoke p95 latency | 28.687s | same as above |
| Public real eval completion | **25/25 executed (100%)** | `benchmarks/runs/public_real_eval_25_humaneval_bfcl_simpleqa_20260505_1136/summary.json` |

Interpretation: end-to-end orchestration and workflow stability are strong under release-like smoke/protocol workloads.

### B. Structured Tool Competence (Function-Calling Quality)

| Metric | Value | Source |
|---|---:|---|
| BFCL schema_valid_rate | **1.0** | `public_real_eval_25.../summary.json` (2 BFCL batches combined) |
| BFCL function_name_accuracy | **1.0** | same |
| BFCL argument_exact_match | **0.9** (9/10) | same |
| BFCL task success rate | **0.9** (9/10) | same |

Interpretation: tool call formatting and function targeting are highly stable; residual errors are mainly argument-level mismatch.

### C. Code Task Effectiveness

| Metric | Value | Source |
|---|---:|---|
| HumanEval+ pass@1 | **1.0** (5/5) | `public_real_eval_25.../summary.json` |
| HumanEval+ syntax_pass_rate | **1.0** | same |
| SWE-Lite replace-edit patch_apply_rate | **0.7** (7/10) | `benchmarks/runs/swebench_lite_l1_replace_mini10_summary_20260505_1538/summary.json` |

Interpretation: for coding slices, the agent can both generate runnable code solutions (HumanEval+) and produce executable patch artifacts (SWE replace-edit mode).

### D. Long-Context Memory Grounding

LongMemEval-S mini10 improvement (baseline -> rerun with default3, valid completed runs only):

| Metric | Baseline | Rerun | Delta | Source |
|---|---:|---:|---:|---|
| Answer accuracy | 0.2 | **0.4** | +0.2 | `longmemeval_s_mini10_summary_20260505_1644/summary.json` + batch01/02 default3 metrics |
| Evidence hit rate | 0.3 | **1.0** | +0.7 | same |
| Memory recall rate | 0.2 | **0.4** | +0.2 | same |
| Quota exhausted cases | - | **0** | - | batch01/02 default3 metrics |

Self-defined ABCDE memory regression (Docker + all memory modules ON):

| Metric | Value | Source |
|---|---:|---|
| All-layer hit rate (L1/L2/L3 all > 0) | **5/5 (100%)** | `docs/memory_eval_v2/summary/metrics.json` |
| Total L1/L2/L3 hits | 52 / 103 / 33 | same |
| Functional pass rate | 2/5 | same |

Interpretation: memory pipeline engagement is complete and evidence grounding improved significantly; remaining failures are quality constraints (`final_content_mismatch`, `missing_redis`, `compaction_not_observed`) rather than workflow deadlock.

### E. Suggested README Headline Metrics

Use this short block as primary showcase:

- Public real eval micro-suite: **72% (18/25)** overall success.
- Runtime smoke stability: **100% (5/5)** workflow success.
- Tool-calling robustness (BFCL): **schema/function accuracy 100%**, argument exact match **90%**.
- Code generation slice (HumanEval+): **pass@1 = 100% (5/5)**.
- Long memory grounding: evidence hit rate improved **0.3 -> 1.0**, answer accuracy **0.2 -> 0.4** on mini10 rerun.
- SWE patch executability slice: replace-edit patch apply rate **70% (7/10)**.

### F. Transparency Notes (keep in README footnote)

- All results are fixed public subsets / mini slices, not official leaderboard submissions.
- Some historical runs were affected by external quota limits (`quota_exhausted`) and are excluded from capability comparison.
- SWE replace-edit result is a patch-executability signal, not a full issue-resolution claim.

## 3) Legacy Compatibility Cleanup Decision

### Removed in this pass (low risk, confirmed unused)
- `MODELSCOPE_API_KEY` legacy config field removed from `self_ai/config.py`.
- `EngineState.from_legacy_state` removed from `self_ai/kernel/engine_state.py`.
- Related legacy-only tests removed/updated in `tests/test_engine_state.py`.
- Outdated frontend guide rewritten to current stack and env vars: `docs/guides/frontend-usage.md`.

### Kept intentionally (still part of real runtime behavior)
- `SELF_AI_DISABLE_LEGACY_RETRIEVAL`: despite name, this is an active retrieval behavior switch used in runtime/tests.
- Runtime safety fallbacks (planner/reviewer/workflow policy): kept because they are reliability guards, not version compatibility shims.

## 4) Files Changed for This Readiness Pass

- `self_ai/config.py`
- `self_ai/kernel/engine_state.py`
- `tests/test_engine_state.py`
- `docs/guides/frontend-usage.md`
