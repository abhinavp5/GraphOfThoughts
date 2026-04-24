# BFS Final Evaluation Protocol

This document defines the frozen, BFS-only protocol for final project results.

## 1) Scope

- Task: **BFS only**
- Objective: compare baseline vs tuned models on next-operation prediction and failure behavior
- Exclusions: DFS and Dijkstra are out of scope for final claims (can be listed as future work)

## 2) Models

Evaluate the following models under identical settings:

- `M0` Baseline: `Qwen/Qwen2.5-0.5B-Instruct` (no adapter)
- `M1` Best BFS SFT adapter
- `M2` Optional Stage-2 adapter (e.g., DPO/GRPO or DAgger-style update)

## 3) Frozen Settings

Use the same settings across all model comparisons:

- `algorithm`: `bfs`
- `seed`: fixed (recommend `100`)
- inference mode: teacher-forced for primary results
- `few-shot`: `0` (unless changed for all models and explicitly documented)
- same `limit` and `bench-limit` for all compared models
- same benchmark input files for all compared models

## 4) Required Runs Per Model

For each model (`M0`, `M1`, `M2`):

1. Main BFS pipeline run
2. NLGraph BFS evaluation
3. GLBench BFS evaluation
4. Save operation and failure metrics JSON outputs

## 5) Canonical Command Template

Run from repo root:

```bash
bash scripts/run_pipeline.sh \
  --model <BASE_MODEL_ID> \
  --adapter <OPTIONAL_ADAPTER_PATH> \
  --algorithm bfs \
  --family erdos_renyi \
  --n 10 \
  --count 100 \
  --seed 100 \
  --limit 100 \
  --nlgraph-input data/benchmarks/nlgraph_eval.json \
  --glbench-input data/benchmarks/glbench_eval.json \
  --bench-limit 100
```

Notes:

- Baseline (`M0`) omits `--adapter`.
- For short smoke checks, reduce `--count`, `--limit`, and `--bench-limit` consistently across models.

## 6) Metrics to Report

At minimum, report:

- step accuracy
- trace accuracy
- mean first-error step
- overall failure rate
- failure rate by operation type
- top operation confusions

Use outputs from:

- main run:
  - `out/metrics_<tag>.json`
  - `out/failures_<tag>.json`
- NLGraph:
  - `out/nlgraph_<tag>_operation_accuracy.json`
  - `out/nlgraph_<tag>_failure_analysis.json`
- GLBench:
  - `out/glbench_<tag>_operation_accuracy.json`
  - `out/glbench_<tag>_failure_analysis.json`

## 7) Final Comparison Table

Rows:

- `M0` baseline
- `M1` tuned SFT
- `M2` optional stage-2

Columns:

- Main step accuracy
- Main mean first-error step
- Main overall failure rate
- NLGraph step accuracy
- GLBench step accuracy

## 8) Required Figures

Produce these three final figures:

1. Step accuracy comparison (`M0` vs `M1` vs `M2`)
2. First-error-step distribution
3. Failure rate by operation type (before vs after tuning)

## 9) Claim Boundaries

- Final claims are **BFS-only**.
- Do not generalize results to DFS/Dijkstra.
- Any alternate setting (free-running, few-shot, different seeds) must be clearly labeled as supplemental.

## 10) Reproducibility Checklist

For every final run:

- record git commit hash
- record exact command used
- record model ID and adapter path
- keep seed/settings fixed across comparisons
- do not mix outputs from different settings in one comparison

