# Self AI Real Public Eval Summary

Generated: `2026-05-05T03:35:30+00:00`

Lightweight fixed-subset results; not official leaderboard submissions.

## Totals

| Metric | Value |
|---|---:|
| cases | 25 |
| ok_cases | 18 |
| failed_cases | 7 |
| ok_rate | 0.72 |

## Runs

| Run | Stage | Cases | OK | Rate | Key Metrics |
|---|---|---:|---:|---:|---|
| HumanEval+ batch02 HumanEval/5-HumanEval/9 | humaneval_plus | 5 | 5 | 1.0 | pass_at_1=1.0 |
| BFCL batch01 simple_0-simple_4 | bfcl | 5 | 5 | 1.0 | schema_valid_rate=1.0; function_name_accuracy=1.0; argument_exact_match=1.0 |
| BFCL batch02 simple_5-simple_9 | bfcl | 5 | 4 | 0.8 | schema_valid_rate=1.0; function_name_accuracy=1.0; argument_exact_match=0.8 |
| SimpleQA batch01 simpleqa_0-simpleqa_4 | simpleqa | 5 | 3 | 0.6 | correct_rate=0.6; incorrect_rate=0.4 |
| SimpleQA batch02 simpleqa_5-simpleqa_9 | simpleqa | 5 | 1 | 0.2 | correct_rate=0.2; incorrect_rate=0.8 |

## Artifacts

- HumanEval+ batch02 HumanEval/5-HumanEval/9: `benchmarks/runs/humaneval_plus_real_batch02_offset5_size5_20260505_1125`
- cases: `benchmarks/public_subsets/humaneval_plus_mini10_20260505/humaneval_plus_cases.jsonl`
- predictions: `benchmarks/predictions/humaneval_plus_mini10_20260505_batch02_predictions.jsonl`
- BFCL batch01 simple_0-simple_4: `benchmarks/runs/bfcl_real_batch01_offset0_size5_20260505_1132`
- cases: `benchmarks/public_subsets/bfcl_real_mini10_20260505/bfcl_cases.jsonl`
- predictions: `benchmarks/predictions/bfcl_real_mini10_20260505_direct_predictions.jsonl`
- BFCL batch02 simple_5-simple_9: `benchmarks/runs/bfcl_real_batch02_offset5_size5_20260505_1133`
- cases: `benchmarks/public_subsets/bfcl_real_mini10_20260505/bfcl_cases.jsonl`
- predictions: `benchmarks/predictions/bfcl_real_mini10_20260505_direct_predictions.jsonl`
- SimpleQA batch01 simpleqa_0-simpleqa_4: `benchmarks/runs/simpleqa_real_batch01_offset0_size5_20260505_1134`
- cases: `benchmarks/public_subsets/simpleqa_real_mini10_20260505/simpleqa_cases.jsonl`
- predictions: `benchmarks/predictions/simpleqa_real_mini10_20260505_predictions.jsonl`
- SimpleQA batch02 simpleqa_5-simpleqa_9: `benchmarks/runs/simpleqa_real_batch02_offset5_size5_20260505_1135`
- cases: `benchmarks/public_subsets/simpleqa_real_mini10_20260505/simpleqa_cases.jsonl`
- predictions: `benchmarks/predictions/simpleqa_real_mini10_20260505_predictions.jsonl`