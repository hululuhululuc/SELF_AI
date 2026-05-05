# Self AI SWE-bench Lite L1 Replace Edit Mini 10 Summary

Generated: `2026-05-05T07:50:39+00:00`

First 10 SWE-bench Lite test cases, evaluated only for L1 patch applicability with replace_edit mode. Not official SWE-bench resolved results.

## Totals

| Metric | Value |
|---|---:|
| cases | 10 |
| ok_cases | 0 |
| failed_cases | 10 |
| patch_applied_cases | 7 |
| validation_blocked_cases | 7 |
| ok_rate | 0.0 |
| patch_apply_rate | 0.7 |
| validation_blocked_rate | 0.7 |

## Runs

| Run | Cases | Patch Apply Rate | Failure Reasons |
|---|---:|---:|---|
| SWE-bench Lite L1 replace_edit batch01 offset0 size5 | 5 | 0.8 | `{"validation_blocked": 4, "edit_find_not_unique": 1}` |
| SWE-bench Lite L1 replace_edit batch02 offset5 size5 | 5 | 0.6 | `{"edit_find_not_unique": 2, "validation_blocked": 3}` |

## Interpretation

- L1 only: measures whether Self AI can produce executable replace edits that become real git patches.
- `validation_blocked` is expected in minimal_tree mode and means full repository tests were intentionally not run.
- This does not claim official SWE-bench issue resolution.
- Compared with direct diff mode patch_apply_rate=0.0, replace_edit reached patch_apply_rate=0.7 on the same 10-case slice.

## Artifacts

- SWE-bench Lite L1 replace_edit batch01 offset0 size5: `benchmarks/runs/swebench_lite_l1_replace_batch01_offset0_size5_20260505_1515`
- SWE-bench Lite L1 replace_edit batch02 offset5 size5: `benchmarks/runs/swebench_lite_l1_replace_batch02_offset5_size5_20260505_1518`
- cases: `benchmarks/public_subsets/swebench_lite_mini10_20260505/swebench_lite_cases.jsonl`