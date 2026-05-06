#!/usr/bin/env bash
# GLBench-only eval: two BFS + two DFS rows (same tasks duplicated) in
# data/benchmarks/glbench_eval_quick2x.json — ~2x the inference of a single row per algo.
# Override with INPUT=data/benchmarks/glbench_eval.json for the minimal 1+1 shard.
# Writes small JSONs (predictions + metrics) under OUT_DIR.
#
# Rivanna / shared clusters: use one conda env end-to-end and Conda's libstdc++.
#   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate got
#   export PYTHONNOUSERSITE=1
#   export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
# Otherwise you may see: ImportError: libstdc++.so.6: GLIBCXX_3.4.29 not found
#
# Example (SFT only):
#   MODEL="Qwen/Qwen2.5-7B-Instruct" \
#   ADAPTER="$HOME/out/.../pre_sft_adapter" \
#   OUT_DIR="$HOME/out/glbench_sft" \
#   bash scripts/run_glbench_quick.sh
#
# Example (SFT + DAgger — path may be post_retrain_model, post_retrain_model_r3, or checkpoint-*):
#   DAGGER_ADAPTER="$HOME/out/.../post_retrain_model_r3" \
#   bash scripts/run_glbench_quick.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER="${ADAPTER:-}"
DAGGER_ADAPTER="${DAGGER_ADAPTER:-}"
OUT_DIR="${OUT_DIR:-$ROOT/out/glbench_quick}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-float16}"
INPUT="${INPUT:-data/benchmarks/glbench_eval_quick2x.json}"

mkdir -p "$OUT_DIR"

run_algo () {
  local algo="$1"
  local prefix="$OUT_DIR/glbench_${algo}"
  echo "=== GLBench --algorithm $algo → ${prefix}_*.json"
  local args=(
    -m evaluation.benchmarks.glbench
    --input "$INPUT"
    --model "$MODEL"
    --out-prefix "$prefix"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --algorithm "$algo"
  )
  [[ -n "$ADAPTER" ]] && args+=(--adapter "$ADAPTER")
  [[ -n "$DAGGER_ADAPTER" ]] && args+=(--dagger-adapter "$DAGGER_ADAPTER")
  python "${args[@]}"
}

run_algo bfs
run_algo dfs

echo "=== step_accuracy and n_samples (no jq needed)"
export OUT_DIR
python -c '
import json, os
from pathlib import Path
out = Path(os.environ["OUT_DIR"])
for algo in ("bfs", "dfs"):
    p = out / f"glbench_{algo}_operation_accuracy.json"
    if p.is_file():
        d = json.loads(p.read_text())
        print(algo, "step_accuracy=", d["step_accuracy"], " n_samples=", d.get("n_samples"))
    else:
        print(algo, "MISSING", p)
'
