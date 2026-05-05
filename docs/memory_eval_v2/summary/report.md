# Memory Eval V2 Report

- generated_at: `2026-05-05T18:25:43Z`
- stop_reason: `completed`

## Case Summary

| Case | Mode | Valid | Reason | Turns | Avg ms | Avg tools | Errors | L1/L2/L3 | Quota |
|---|---|---|---|---:|---:|---:|---:|---|---|
| A preference_persistence | ON | True |  | 4 | 25663.08 | 0.0 | 3 | 6/11/5 | False |
| B file_state_consistency | ON | False | final_content_mismatch | 4 | 21253.63 | 0.0 | 0 | 6/12/5 | False |
| C issue_fix_reuse | ON | True |  | 4 | 29509.27 | 0.0 | 3 | 6/15/5 | False |
| D research_summary_continuity | ON | False | missing_redis | 4 | 19838.11 | 0.0 | 0 | 6/12/5 | False |
| E compaction_cold_recall | ON | False | compaction_not_observed | 8 | 17178.28 | 0.0 | 0 | 28/53/13 | False |

## ON vs OFF

| Case | ON valid | OFF valid | ON ms | OFF ms | ON tools | OFF tools | ON L2 | OFF L2 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A | True | False | 25663.08 | None | 0.0 | None | 11 | None |
| B | False | False | 21253.63 | None | 0.0 | None | 12 | None |
| C | True | False | 29509.27 | None | 0.0 | None | 15 | None |
| D | False | False | 19838.11 | None | 0.0 | None | 12 | None |
| E | False | False | 17178.28 | None | 0.0 | None | 53 | None |