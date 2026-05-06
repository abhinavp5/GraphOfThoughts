#!/usr/bin/env bash
# GLBench-only eval: one graph each for BFS and DFS in data/benchmarks/glbench_eval.json.
# Fast on a spare GPU; writes small JSONs (predictions + metrics) under OUT_DIR.
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
INPUT="${INPUT:-data/benchmarks/glbench_eval.json}"

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

echo "Done. Step accuracy: grep step_accuracy ${OUT_DIR}/glbench_*_operation_accuracy.json"
