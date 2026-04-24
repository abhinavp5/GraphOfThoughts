# Graph of Thoughts (GoT)

**Decoupling Logic from Memory in LLMs for Graph Algorithm Reasoning**

A research framework that separates the LLM's reasoning role (CPU / Logic)
from algorithmic state management (RAM / Memory). A deterministic,
non-neural **State Executor** maintains the ground-truth graph state while
the LLM is supervised to predict only the next operation at each step.
This prevents the "state drift" that causes Chain-of-Thought (CoT) to fail
on long deterministic graph traces.

**Team:** Jonas Lee, Neil Morgan, Abhinav Pappu, Paris Phan, Andrew Thepvongs
(University of Virginia).

---

```
   CoT prompt             GoT loop                 Ground truth
   ---------             --------                 ------------
   model generates       model predicts           State Executor
   full trace in         one op at a time         maintains state
   free form             given ground-truth       s_t deterministically
                         state s_t
           ↓                  ↓                         ↓
    drifts after ~5-10    stable at any length     never wrong
    steps, hallucinates
    edges
```

The project's goal is to measure and publish this gap across graph
families, algorithms (BFS, DFS, Dijkstra), and trace lengths.




## Development & Testing

Requires Python 3.11. The ML stack (torch + transformers + peft + accelerate)
is cross-platform; `bitsandbytes` is only needed for 4-bit quantization on
Linux/CUDA.

```bash
conda create -n got python=3.11 -y
conda activate got
pip install -r requirements.txt
```

### smoke-test for inference scaffolding (for testing functionality)

test to validate graph reconstruction, prompt-building, and the
State-Executor-in-the-loop by replaying gold operations through the
pipeline:

```bash
python scripts/smoke_test_inference.py data/traces/<test_bfs.json>
```


## Functions


### Generate a dataset

```bash
python -m data.generators.generate_dataset \
  --algorithm bfs \
  --family erdos_renyi \
  --n 20 --count 1000 \
  --out data/traces/train_bfs_er.json \
  --seed 42
```

Families: `erdos_renyi`, `barabasi_albert`, `random_tree`, `grid`,
`bridge`, `bottleneck`, `high_girth`. For Dijkstra traces, add `--weighted`.


### Running the full pipeline

The wrapper script does data generation → inference → evaluation in one
command. Defaults to an auth-free base model so `bash scripts/run_pipeline.sh`
works out of the box.

```bash
# Zero-config baseline: Qwen 0.5B (no adapter), BFS, 5 samples, teacher-forced
bash scripts/run_pipeline.sh

# With a trained LoRA adapter
bash scripts/run_pipeline.sh \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --adapter checkpoints/llama-bfs/final \
  --limit 20

# Drift experiment (model drives state instead of gold)
bash scripts/run_pipeline.sh --free-running --limit 10

# Dijkstra on weighted ER graphs
bash scripts/run_pipeline.sh --algorithm dijkstra --weighted

# See all flags
bash scripts/run_pipeline.sh --help
```

The untrained Qwen-0.5B baseline will show near-0% step accuracy — that is
the **floor**. A GoT-fine-tuned adapter should clear 90% on the same test
set.

Each run writes three artefacts to `out/`:
- the generated trace JSON
- the model's predictions JSON
- the metrics JSON

With current pipeline updates, runs also emit:
- failure analysis JSON (`out/failures_<tag>.json`)
- run manifest JSON (`out/run_manifest_<tag>.json`)
- optional NLGraph / GLBench outputs when `--nlgraph-input` / `--glbench-input` are set

Benchmark usage:

```bash
bash scripts/run_pipeline.sh \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --nlgraph-input data/benchmarks/nlgraph_eval.json \
  --glbench-input data/benchmarks/glbench_eval.json \
  --bench-limit 100
```



### Evaluation: Operation accuracy

```bash
python -m evaluation.metrics.operation_accuracy out/preds_llama.json \
  --out out/metrics_llama.json
```

Prints and optionally writes:
- `step_accuracy` — fraction of steps where `pred_op == gold_op`
- `trace_accuracy` — fraction of traces with 100% matching ops
- `mean_first_error_step` — average depth at which the first mismatch occurs
- `mean_coverage` — fraction of gold steps the model attempted

### Evaluation: Failure analysis

```bash
python -m evaluation.metrics.failure_analysis out/preds_llama.json \
  --out out/failures_llama.json
```

### Evaluation: State consistency

```bash
python -m evaluation.metrics.state_consistency out/preds_llama.json \
  --out out/state_consistency_llama.json
```

### Evaluation: Structural generalization

```bash
python -m evaluation.metrics.structural_generalization out/preds_llama.json \
  --train-max-n 20 \
  --out out/structural_generalization_llama.json
```

### Stage-2 recovery training (minimal DAgger loop)

Collect recovery examples from free-running rollouts:

```bash
python -m training.dagger collect \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --trace-dir data/traces \
  --trace-pattern "train_*.json" \
  --out data/traces/dagger_recovery_bfs.json
```

Fine-tune on collected recovery examples:

```bash
python -m training.dagger finetune \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --recovery-json data/traces/dagger_recovery_bfs.json \
  --output-dir checkpoints/dagger-bfs-stage2
```

### Final reproduction (BFS-only)

See `BFS_FINAL_PROTOCOL.md` and `FINAL_EXECUTION_CHECKLIST.md` for the frozen BFS-only run protocol, final figure list, and done criteria.

```









this page left intentionally blank type

















```



# Some Useful Information Courtesy of Claude

## How the pieces fit

```
┌──────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Graph        │───▶│ Deterministic│───▶│ JSON trace      │
│ generator    │    │ solver +     │    │ {graph, source, │
│ (family, n)  │    │ StateExecutor│    │  steps: [...]}  │
└──────────────┘    └──────────────┘    └────────┬────────┘
                                                 │
                        ┌────────────────────────┴──────────┐
                        │                                   │
                        ▼                                   ▼
              ┌──────────────────┐               ┌──────────────────┐
              │ SFT training     │               │ Inference        │
              │ (teacher_forcing │               │ (StateExecutor   │
              │  + neg sampling) │               │  in the loop)    │
              └────────┬─────────┘               └────────┬─────────┘
                       │                                  │
                       ▼                                  ▼
                 LoRA adapter ────────────────▶  Predicted trace JSON
                                                         │
                                                         ▼
                                              ┌─────────────────────┐
                                              │ Evaluation metrics  │
                                              │ + plots (vs CoT)    │
                                              └─────────────────────┘
```


## Training

Configs in [`training/configs/`](training/configs/). Single-GPU:

```bash
python -m training.sft --config training/configs/llama_3_1b.yaml
```

Multi-GPU (e.g. 4 H200 on Rivanna) with DDP:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m training.sft --config training/configs/llama_3_1b.yaml
```

Llama requires HF auth; set `HF_TOKEN` or run `huggingface-cli login` once.
On HPC with tight home-dir quota, put the hub cache on scratch:

```bash
export HF_HUB_CACHE=/sfs/weka/scratch/$USER/hf_hub
```

---

## Inference

### Teacher-forced mode (default, for eval)

The State Executor is advanced with **gold** ops so every prompt contains
the correct state. The model's prediction is logged and scored but does
not drive state. This isolates next-operation prediction accuracy from
error compounding.

```bash
python -m inference.run_inference \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --adapter path/to/lora/checkpoint \
  --trace data/traces/test_bfs.json \
  --limit 20 \
  --out out/preds_llama.json \
  --device mps --dtype float16
```

### Free-running mode (for drift experiments)

The model's op drives the executor; an invalid op terminates the run.
This is the setup that makes CoT fail and GoT (with negative-sampling
recovery) shine.

```bash
python -m inference.run_inference \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --adapter path/to/lora/checkpoint \
  --trace data/traces/test_bfs.json \
  --free-running \
  --out out/preds_llama_drift.json \
  --device mps --dtype float16
```

Output JSON is a per-sample list with `gold_steps` and `predicted_steps`
(each predicted step records `operation_predicted`, `operation_gold`,
`match`, `applied`, and any `error`).
