#!/usr/bin/env bash
# -----------------------------------------------------------------------
# run_pipeline.sh — Full Graph of Thoughts pipeline in one command.
#
#   data generation → inference → evaluation
#
# Defaults to an auth-free base model (Qwen/Qwen2.5-0.5B-Instruct) on a
# tiny BFS / Erdős–Rényi test set, so `bash scripts/run_pipeline.sh` works
# out of the box. Override anything via flags or env vars.
#
# Unified console log: default $OUT_DIR/pipeline_${TAG}_<timestamp>.log
# Override with --log-file PATH or env PIPELINE_LOG.
# W&B: set WANDB_PROJECT; aggregate metrics logged at end via scripts/log_pipeline_run.py
#
# Usage:
#   bash scripts/run_pipeline.sh [options]
#   bash scripts/run_pipeline.sh --help
# -----------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")/.."

# -------------------------------------------------------------------
# Defaults (override via env var or CLI flag)
# -------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER="${ADAPTER:-}"
ALGORITHM="${ALGORITHM:-bfs}"
FAMILY="${FAMILY:-erdos_renyi}"
N="${N:-10}"
COUNT="${COUNT:-5}"
LIMIT="${LIMIT:-5}"
SEED="${SEED:-100}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUT_DIR="${OUT_DIR:-out}"
MAX_STEPS="${MAX_STEPS:-}"
FREE_RUNNING="${FREE_RUNNING:-0}"
FEW_SHOT="${FEW_SHOT:-0}"
WEIGHTED_FLAG=""
SMOKE_ONLY="${SMOKE_ONLY:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_DATA="${SKIP_DATA:-0}"
NLGRAPH_INPUT="${NLGRAPH_INPUT:-}"
GLBENCH_INPUT="${GLBENCH_INPUT:-}"
BENCH_LIMIT="${BENCH_LIMIT:-}"
CLI_LOG_FILE="${CLI_LOG_FILE:-}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the full pipeline: data → inference → evaluation.

Options (all also settable as env vars):
  -m, --model MODEL        HF model id or local path (default: $MODEL)
  -a, --adapter DIR        LoRA adapter directory (optional)
  -A, --algorithm ALG      bfs | dfs | dijkstra (default: $ALGORITHM)
  -f, --family FAM         Graph family (default: $FAMILY)
                           Regular: erdos_renyi, barabasi_albert, random_tree, grid
                           Hard:    bridge, bottleneck, high_girth
  -n, --n INT              Nodes per graph (default: $N)
  -c, --count INT          Samples to generate (default: $COUNT)
  -l, --limit INT          Inference sample limit (default: $LIMIT)
      --max-steps INT      Override per-sample step budget (default: match gold)
  -s, --seed INT           RNG seed (default: $SEED)
  -d, --device DEV         auto | mps | cuda | cpu (default: $DEVICE)
  -t, --dtype DT           auto | float16 | bfloat16 | float32 (default: $DTYPE)
  -o, --out-dir DIR        Output dir (default: $OUT_DIR)
      --log-file PATH      Tee full stdout/stderr to this file (default:
                           \$OUT_DIR/pipeline_<TAG>_<timestamp>.log)
                           Env PIPELINE_LOG overrides the default path if set
                           (CLI --log-file takes precedence over PIPELINE_LOG).
      --free-running       Model's op drives state (default: teacher-forced)
      --few-shot N         Prepend N demo traces as in-context examples so
                           untrained base models learn the output grammar
                           (default: 0, i.e. zero-shot)
      --weighted           Attach random edge weights (required for Dijkstra)
      --smoke-only         Only run the mock-model scaffolding test, then exit
      --skip-smoke         Skip the smoke test
      --skip-data          Reuse existing trace file if present
  -h, --help               Show this message
      --nlgraph-input F    Optional NLGraph JSON to evaluate via benchmark adapter
      --glbench-input F    Optional GLBench JSON to evaluate via benchmark adapter
      --bench-limit N      Optional sample cap for benchmark adapters

  W&B: export WANDB_PROJECT=<name> before running; aggregate metrics from
  pipe / NLGraph / GLBench are logged at end (requires wandb install + auth).

Examples:
  # Zero-config baseline — base Qwen 0.5B on 5 BFS samples, teacher-forced
  bash scripts/run_pipeline.sh

  # With a trained LoRA adapter
  bash scripts/run_pipeline.sh \\
      --model meta-llama/Llama-3.2-1B-Instruct \\
      --adapter checkpoints/llama-bfs/final \\
      --limit 20

  # Drift experiment — free-running, longer
  bash scripts/run_pipeline.sh --free-running --limit 10

  # Dijkstra on weighted ER
  bash scripts/run_pipeline.sh --algorithm dijkstra --weighted

  # Larger graphs (structural generalization spot check)
  bash scripts/run_pipeline.sh --n 50 --count 20 --limit 20
EOF
}

# -------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)      MODEL="$2"; shift 2 ;;
    -a|--adapter)    ADAPTER="$2"; shift 2 ;;
    -A|--algorithm)  ALGORITHM="$2"; shift 2 ;;
    -f|--family)     FAMILY="$2"; shift 2 ;;
    -n|--n)          N="$2"; shift 2 ;;
    -c|--count)      COUNT="$2"; shift 2 ;;
    -l|--limit)      LIMIT="$2"; shift 2 ;;
    --max-steps)     MAX_STEPS="$2"; shift 2 ;;
    -s|--seed)       SEED="$2"; shift 2 ;;
    -d|--device)     DEVICE="$2"; shift 2 ;;
    -t|--dtype)      DTYPE="$2"; shift 2 ;;
    -o|--out-dir)    OUT_DIR="$2"; shift 2 ;;
    --log-file)      CLI_LOG_FILE="$2"; shift 2 ;;
    --free-running)  FREE_RUNNING=1; shift ;;
    --few-shot)      FEW_SHOT="$2"; shift 2 ;;
    --weighted)      WEIGHTED_FLAG="--weighted"; shift ;;
    --smoke-only)    SMOKE_ONLY=1; shift ;;
    --skip-smoke)    SKIP_SMOKE=1; shift ;;
    --skip-data)     SKIP_DATA=1; shift ;;
    --nlgraph-input) NLGRAPH_INPUT="$2"; shift 2 ;;
    --glbench-input) GLBENCH_INPUT="$2"; shift 2 ;;
    --bench-limit)   BENCH_LIMIT="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "Unknown flag: $1" >&2; usage; exit 2 ;;
  esac
done

# Dijkstra implies weighted
if [[ "$ALGORITHM" == "dijkstra" ]] && [[ -z "$WEIGHTED_FLAG" ]]; then
  echo "[note] --algorithm dijkstra implies --weighted; enabling."
  WEIGHTED_FLAG="--weighted"
fi

# Few-shot needs at least (few_shot + 1) samples in the file so demos and
# target never overlap. Bump COUNT if user asked for more demos than samples.
if [[ "$FEW_SHOT" != "0" ]] && (( COUNT <= FEW_SHOT )); then
  NEW_COUNT=$((FEW_SHOT + 1))
  echo "[note] --few-shot $FEW_SHOT needs at least $NEW_COUNT samples; bumping --count from $COUNT to $NEW_COUNT"
  COUNT="$NEW_COUNT"
fi

TAG="${ALGORITHM}_${FAMILY}_n${N}_c${COUNT}_s${SEED}"
if [[ "$FEW_SHOT" != "0" ]]; then
  TAG="${TAG}_fs${FEW_SHOT}"
fi

RUN_ID="${TAG}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUT_DIR"

# Resolve unified log path: CLI > env PIPELINE_LOG > default
if [[ -n "$CLI_LOG_FILE" ]]; then
  PIPELINE_LOG_FILE="$CLI_LOG_FILE"
elif [[ -n "${PIPELINE_LOG:-}" ]]; then
  PIPELINE_LOG_FILE="$PIPELINE_LOG"
else
  PIPELINE_LOG_FILE="${OUT_DIR}/pipeline_${TAG}_$(date +%Y%m%d_%H%M%S).log"
fi

mkdir -p "$(dirname "$PIPELINE_LOG_FILE")"

RUN_START="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# Tee all subsequent output to console and log file
exec > >(tee -a "$PIPELINE_LOG_FILE") 2>&1

_log_abs() {
  if command -v realpath &>/dev/null; then
    realpath "$PIPELINE_LOG_FILE" 2>/dev/null || echo "$PIPELINE_LOG_FILE"
  elif readlink -f "$PIPELINE_LOG_FILE" &>/dev/null; then
    readlink -f "$PIPELINE_LOG_FILE"
  else
    echo "$PIPELINE_LOG_FILE"
  fi
}
echo "[pipeline] Unified log: $(_log_abs)"

# -------------------------------------------------------------------
# Auto-detect device + dtype
# -------------------------------------------------------------------
if [[ "$DEVICE" == "auto" ]] || [[ "$DTYPE" == "auto" ]]; then
  detected=$(python - <<'PY' 2>/dev/null || true
try:
    import torch
    if torch.cuda.is_available():
        print("cuda float16")
    elif torch.backends.mps.is_available():
        print("mps float32")
    else:
        print("cpu float32")
except Exception:
    print("cpu float32")
PY
)
  auto_device="${detected% *}"
  auto_dtype="${detected##* }"
  [[ "$DEVICE" == "auto" ]] && DEVICE="${auto_device:-cpu}"
  [[ "$DTYPE" == "auto" ]]  && DTYPE="${auto_dtype:-float32}"
fi

# -------------------------------------------------------------------
# Banner
# -------------------------------------------------------------------
banner() {
  printf "\n\033[1;34m==>\033[0m \033[1m%s\033[0m\n" "$1"
}

banner "Graph of Thoughts — pipeline run"
echo "  model:     $MODEL"
[[ -n "$ADAPTER" ]] && echo "  adapter:   $ADAPTER" || echo "  adapter:   (none — base model)"
echo "  algorithm: $ALGORITHM    family: $FAMILY    n=$N    count=$COUNT"
echo "  limit:     $LIMIT        seed=$SEED"
echo "  device:    $DEVICE       dtype=$DTYPE"
if [[ "$FREE_RUNNING" == "1" ]]; then
  echo "  mode:      free-running (model drives state)"
else
  echo "  mode:      teacher-forced (gold drives state — default)"
fi
if [[ "$FEW_SHOT" != "0" ]]; then
  echo "  few-shot:  $FEW_SHOT demo trace(s) in context"
else
  echo "  few-shot:  disabled (zero-shot)"
fi
echo "  out_dir:   $OUT_DIR"
echo "  log_file:  $PIPELINE_LOG_FILE"
[[ -n "$NLGRAPH_INPUT" ]] && echo "  nlgraph:   $NLGRAPH_INPUT"
[[ -n "$GLBENCH_INPUT" ]] && echo "  glbench:   $GLBENCH_INPUT"

# -------------------------------------------------------------------
# 0. Scaffolding smoke test (no ML deps required)
# -------------------------------------------------------------------
if [[ "$SKIP_SMOKE" != "1" ]]; then
  banner "Scaffolding smoke test (mock model, no ML deps)"
  SMOKE_TRACE="data/traces/test_${ALGORITHM}.json"
  if [[ ! -f "$SMOKE_TRACE" ]]; then
    echo "  [info] $SMOKE_TRACE missing; generating a small stand-in"
    python -m data.generators.generate_dataset \
      --algorithm "$ALGORITHM" --family "$FAMILY" \
      --n 10 --count 3 --seed 42 \
      $WEIGHTED_FLAG \
      --out "$SMOKE_TRACE"
  fi
  python scripts/smoke_test_inference.py "$SMOKE_TRACE"
fi

if [[ "$SMOKE_ONLY" == "1" ]]; then
  banner "Smoke-only mode — stopping here."
  exit 0
fi

# -------------------------------------------------------------------
# 1. Dataset generation
# -------------------------------------------------------------------
TRACE_FILE="data/traces/pipe_${TAG}.json"

if [[ "$SKIP_DATA" == "1" ]] && [[ -f "$TRACE_FILE" ]]; then
  banner "Dataset (reusing $TRACE_FILE)"
else
  banner "Dataset generation → $TRACE_FILE"
  python -m data.generators.generate_dataset \
    --algorithm "$ALGORITHM" \
    --family "$FAMILY" \
    --n "$N" --count "$COUNT" --seed "$SEED" \
    $WEIGHTED_FLAG \
    --out "$TRACE_FILE"
fi

# -------------------------------------------------------------------
# 2. Inference
# -------------------------------------------------------------------
PRED_FILE="$OUT_DIR/pred_${TAG}.json"

banner "Inference → $PRED_FILE"
INF_ARGS=(
  --model "$MODEL"
  --trace "$TRACE_FILE"
  --out "$PRED_FILE"
  --limit "$LIMIT"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --verbose
)
[[ -n "$ADAPTER" ]]     && INF_ARGS+=(--adapter "$ADAPTER")
[[ -n "$MAX_STEPS" ]]   && INF_ARGS+=(--max-steps "$MAX_STEPS")
[[ "$FREE_RUNNING" == "1" ]] && INF_ARGS+=(--free-running)
[[ "$FEW_SHOT" != "0" ]] && INF_ARGS+=(--few-shot "$FEW_SHOT")

python -m inference.run_inference "${INF_ARGS[@]}"

# -------------------------------------------------------------------
# 3. Evaluation
# -------------------------------------------------------------------
METRICS_FILE="$OUT_DIR/metrics_${TAG}.json"
banner "Scoring → $METRICS_FILE"
python -m evaluation.metrics.operation_accuracy "$PRED_FILE" --out "$METRICS_FILE"
FAILURES_FILE="$OUT_DIR/failures_${TAG}.json"
banner "Failure analysis → $FAILURES_FILE"
python -m evaluation.metrics.failure_analysis "$PRED_FILE" --out "$FAILURES_FILE"

# -------------------------------------------------------------------
# 4. Optional benchmark integrations (NLGraph / GLBench)
# -------------------------------------------------------------------
NL_PREFIX=""
NL_PRED=""
NL_OA=""
NL_FA=""

if [[ -n "$NLGRAPH_INPUT" ]]; then
  NL_PREFIX="$OUT_DIR/nlgraph_${TAG}"
  banner "NLGraph benchmark → ${NL_PREFIX}_*.json"
  NL_ARGS=(
    --input "$NLGRAPH_INPUT"
    --model "$MODEL"
    --out-prefix "$NL_PREFIX"
    --device "$DEVICE"
    --dtype "$DTYPE"
  )
  [[ -n "$ADAPTER" ]] && NL_ARGS+=(--adapter "$ADAPTER")
  [[ -n "$MAX_STEPS" ]] && NL_ARGS+=(--max-steps "$MAX_STEPS")
  [[ -n "$BENCH_LIMIT" ]] && NL_ARGS+=(--limit "$BENCH_LIMIT")
  [[ "$FREE_RUNNING" == "1" ]] && NL_ARGS+=(--free-running)
  python -m evaluation.benchmarks.nlgraph "${NL_ARGS[@]}"
  NL_PRED="${NL_PREFIX}_predictions.json"
  NL_OA="${NL_PREFIX}_operation_accuracy.json"
  NL_FA="${NL_PREFIX}_failure_analysis.json"
fi

GL_PREFIX=""
GL_PRED=""
GL_OA=""
GL_FA=""

if [[ -n "$GLBENCH_INPUT" ]]; then
  GL_PREFIX="$OUT_DIR/glbench_${TAG}"
  banner "GLBench benchmark → ${GL_PREFIX}_*.json"
  GL_ARGS=(
    --input "$GLBENCH_INPUT"
    --model "$MODEL"
    --out-prefix "$GL_PREFIX"
    --device "$DEVICE"
    --dtype "$DTYPE"
  )
  [[ -n "$ADAPTER" ]] && GL_ARGS+=(--adapter "$ADAPTER")
  [[ -n "$MAX_STEPS" ]] && GL_ARGS+=(--max-steps "$MAX_STEPS")
  [[ -n "$BENCH_LIMIT" ]] && GL_ARGS+=(--limit "$BENCH_LIMIT")
  [[ "$FREE_RUNNING" == "1" ]] && GL_ARGS+=(--free-running)
  python -m evaluation.benchmarks.glbench "${GL_ARGS[@]}"
  GL_PRED="${GL_PREFIX}_predictions.json"
  GL_OA="${GL_PREFIX}_operation_accuracy.json"
  GL_FA="${GL_PREFIX}_failure_analysis.json"
fi

# -------------------------------------------------------------------
# 5. Run manifest (JSON) + optional W&B
# -------------------------------------------------------------------
RUN_END="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || true)"

MANIFEST_FILE="$OUT_DIR/run_manifest_${TAG}.json"

if [[ "$FREE_RUNNING" == "1" ]]; then
  JSON_FREE=true
else
  JSON_FREE=false
fi

# Build paths JSON (nlgraph / glbench blocks optional)
export MANIFEST_FILE RUN_ID TAG RUN_START RUN_END PIPELINE_LOG_FILE
export MODEL ADAPTER ALGORITHM FAMILY N COUNT LIMIT SEED DEVICE DTYPE
export BENCH_LIMIT MAX_STEPS SMOKE_ONLY TRACE_FILE PRED_FILE METRICS_FILE FAILURES_FILE
export NL_PREFIX NL_PRED NL_OA NL_FA GL_PREFIX GL_PRED GL_OA GL_FA
export JSON_FREE
export GIT_COMMIT="${GIT_COMMIT:-}"
export NLGRAPH_INPUT="${NLGRAPH_INPUT:-}"
export GLBENCH_INPUT="${GLBENCH_INPUT:-}"

python <<'PY'
import json, os

def path_or_null(p: str) -> str | None:
    return p if p else None

git = os.environ.get("GIT_COMMIT", "").strip() or None

pipe = {
    "predictions": os.environ["PRED_FILE"],
    "operation_accuracy": os.environ["METRICS_FILE"],
    "failure_analysis": os.environ["FAILURES_FILE"],
}

nl = None
if os.environ.get("NL_PREFIX", "").strip():
    nl = {
        "prefix": os.environ.get("NL_PREFIX", ""),
        "predictions": os.environ.get("NL_PRED") or None,
        "operation_accuracy": os.environ.get("NL_OA") or None,
        "failure_analysis": os.environ.get("NL_FA") or None,
    }

gl = None
if os.environ.get("GL_PREFIX", "").strip():
    gl = {
        "prefix": os.environ.get("GL_PREFIX", ""),
        "predictions": os.environ.get("GL_PRED") or None,
        "operation_accuracy": os.environ.get("GL_OA") or None,
        "failure_analysis": os.environ.get("GL_FA") or None,
    }

bench = os.environ.get("BENCH_LIMIT", "").strip()
cfg = {
    "model": os.environ["MODEL"],
    "adapter": os.environ.get("ADAPTER", "") or None,
    "algorithm": os.environ["ALGORITHM"],
    "family": os.environ["FAMILY"],
    "n": int(os.environ["N"]),
    "count": int(os.environ["COUNT"]),
    "limit": int(os.environ["LIMIT"]),
    "bench_limit": int(bench) if bench else None,
    "max_steps": os.environ.get("MAX_STEPS") or None,
    "device": os.environ["DEVICE"],
    "dtype": os.environ["DTYPE"],
    "free_running": os.environ.get("JSON_FREE", "false") == "true",
    "smoke_only": bool(int(os.environ.get("SMOKE_ONLY", "0"))),
    "out_dir": os.path.dirname(os.environ["PRED_FILE"]) or ".",
    "trace_file": os.environ["TRACE_FILE"],
    "nlgraph_input": path_or_null(os.environ.get("NLGRAPH_INPUT", "")),
    "glbench_input": path_or_null(os.environ.get("GLBENCH_INPUT", "")),
}

manifest = {
    "run_id": os.environ["RUN_ID"],
    "tag": os.environ["TAG"],
    "started_at": os.environ["RUN_START"],
    "finished_at": os.environ["RUN_END"],
    "git_commit": git,
    "config": cfg,
    "paths": {
        "log_file": os.environ.get("PIPELINE_LOG_FILE", ""),
        "trace_file": os.environ["TRACE_FILE"],
        "pipe": pipe,
        "nlgraph": nl,
        "glbench": gl,
    },
}
out = os.environ["MANIFEST_FILE"]
with open(out, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
print(f"[pipeline] Wrote run manifest: {out}")
PY

banner "Done"
echo "  trace file:  $TRACE_FILE"
echo "  predictions: $PRED_FILE"
echo "  metrics:     $METRICS_FILE"
echo "  failures:    $FAILURES_FILE"
echo "  manifest:    $MANIFEST_FILE"
echo "  log file:    $PIPELINE_LOG_FILE"

if [[ -n "${WANDB_PROJECT:-}" ]]; then
  banner "Weights & Biases (WANDB_PROJECT=$WANDB_PROJECT)"
  if python scripts/log_pipeline_run.py --manifest "$MANIFEST_FILE"; then
    echo "[pipeline] W&B run completed."
  else
    echo "[pipeline][warn] W&B logging failed (see messages above). Continuing." >&2
  fi
fi
