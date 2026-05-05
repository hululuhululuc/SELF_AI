# GitHub Release Upload Checklist

This checklist reflects the current release policy:

- Do not upload large local model files.
- Do not upload full local debug/run logs.
- Upload the curated raw benchmark evidence referenced by README metrics.

## A) Must Upload

### Core Project Files

- `README.md`
- `README.frontpage.md`
- `LICENSE`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `pyproject.toml`
- `docker-compose.yml`
- `.env.example`
- `.gitignore`
- `self_ai/`
- `scripts/`
- `tests/`

### Curated Benchmark Evidence (README-linked)

- `benchmarks/runs/public_real_eval_25_humaneval_bfcl_simpleqa_20260505_1136/**`
- `benchmarks/runs/runtime_smoke_batch01_offset0_size5_20260505_1044/**`
- `benchmarks/runs/swebench_lite_l1_replace_mini10_summary_20260505_1538/**`
- `benchmarks/runs/longmemeval_s_mini10_summary_20260505_1644/**`
- `benchmarks/runs/longmemeval_s_batch01_offset0_size5_20260506_default3/**`
- `benchmarks/runs/longmemeval_s_batch02_offset5_size5_20260506_default3/**`
- `docs/memory_eval_v2/summary/metrics.json`
- `docs/memory_eval_v2/summary/failures.json`
- `docs/memory_eval_v2/summary/report.md`

### Dataset Subset Specs (for reproducibility)

- `benchmarks/public_subsets/**`

## B) Must Not Upload

- `.env`
- `.venv/`
- `self_ai/model/**`
- `memory/chats/**`
- `artifacts/**`
- `tmp_dbg/**`
- `tmp_real_samples/**`
- Full raw debug traces outside curated evidence (for example large `docs/memory_eval_v2/raw/**` dumps)

## C) Pre-push Verification

```powershell
git status
git check-ignore -v self_ai/model
git check-ignore -v benchmarks/runs/some_non_curated_dir/metrics.json
git check-ignore -v benchmarks/runs/public_real_eval_25_humaneval_bfcl_simpleqa_20260505_1136/summary.json
```

Expected:

- `self_ai/model` should be ignored.
- Non-curated run dirs under `benchmarks/runs` should be ignored.
- Curated run dirs listed above should NOT be ignored.

## D) Suggested Commit Order

1. Commit docs + README + policies.
2. Commit curated benchmark evidence subset.
3. Final check with `git status`, then push release branch.
