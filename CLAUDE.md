# Graph of Thoughts (GoT): Decoupling Logic from Memory in LLMs

## Project Overview

This project implements the **Graph of Thoughts (GoT)** framework — a method for improving LLM performance on deterministic graph algorithms (BFS, DFS, Dijkstra's) by separating the model's reasoning role (CPU/Logic) from state management (RAM/Memory). A deterministic, non-neural state executor maintains ground-truth graph state, while the LLM is supervised to predict the next operation at each step.

**Team:** Jonas Lee, Neil Morgan, Abhinav Pappu, Paris Phan, Andrew Thepvongs  
**Affiliation:** University of Virginia  
**Deadline:** May 4, 2026

---

## Repo Structure

```
got/
├── CLAUDE.md                  ← You are here
├── README.md
├── requirements.txt
│
├── data/
│   ├── generators/
│   │   ├── graph_families.py      # Erdős–Rényi, Barabási–Albert, Tree, Grid generators
│   │   ├── hard_cases.py          # Bridge graphs, bottleneck graphs, high-girth graphs
│   │   └── generate_dataset.py    # CLI entry point: generates and saves JSON trace datasets
│   ├── traces/                    # Generated JSON trace files (gitignored if large)
│   └── linearization.py          # Graph-Language linearization: (u)-[w]->(v) encoding
│
├── solvers/
│   ├── bfs.py                     # Deterministic BFS with step-by-step state logging
│   ├── dfs.py                     # Deterministic DFS with step-by-step state logging
│   ├── dijkstra.py                # Deterministic Dijkstra's with step-by-step state logging
│   └── state_executor.py          # Core state executor: applies operations, tracks ground truth
│
├── training/
│   ├── dataset.py                 # PyTorch Dataset class wrapping JSON traces
│   ├── sft.py                     # Supervised fine-tuning loop (token-level cross-entropy)
│   ├── teacher_forcing.py         # Teacher forcing strategy implementation
│   ├── negative_sampling.py       # Broken trace injection + CORRECTION token training
│   └── configs/
│       ├── qwen_2_5_7b.yaml       # Training config for Qwen 2.5 7B Instruct
│       └── llama_3_1b.yaml        # Training config for Llama 3.1B
│
├── inference/
│   ├── prompt_forcing.py          # Approach 1: Prompt LLM to generate subgraph at each step
│   ├── decoding_forcing.py        # Approach 3: Constrain decoding to valid subgraph formats
│   └── run_inference.py           # End-to-end inference pipeline with state executor in loop
│
├── evaluation/
│   ├── metrics/
│   │   ├── operation_accuracy.py  # % of correctly predicted next operations (target: ≥90%)
│   │   ├── state_consistency.py   # Alignment between LLM reasoning and state executor
│   │   └── structural_generalization.py  # Accuracy on graphs larger than training set
│   ├── benchmarks/
│   │   ├── nlgraph.py             # NLGraph benchmark integration
│   │   └── glbench.py             # GLBench baseline evaluation
│   ├── robustness/
│   │   ├── zero_shot_transfer.py  # Erdős–Rényi → Barabási–Albert transfer test
│   │   ├── faithfulness_audit.py  # Detect contradictions between LLM text and state executor
│   │   └── negative_recovery.py   # Evaluate recovery from broken traces
│   └── run_eval.py                # Master eval runner: all metrics, CoT vs GoT comparison
│
├── plots/
│   ├── length_vs_accuracy.py      # Path length vs. accuracy: GoT vs CoT degradation curve
│   └── trace_comparison.py        # Side-by-side reasoning trace visualizer
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_training_run.ipynb
    └── 03_results_analysis.ipynb
```

---

## Key Concepts & Terminology

These definitions are used throughout the codebase:

- **`s_t`** — Algorithmic state at step `t` (visited set, queue/frontier, distances, parent pointers)
- **`H_t`** — Induced subgraph at step `t` (predecessor tree / explored edges) derived from `s_t`
- **`o_t`** — Operation taken at step `t` that maps `s_{t-1} → s_t`
- **State Executor** — Deterministic, non-neural module that maintains ground-truth graph state
- **Graph-Language Linearization** — Compact token encoding: `(u)-[w]->(v)` for edge `u→v` with weight `w`
- **CORRECTION token** — Special token the model must emit when recovering from a broken trace

---

## Subgraph Generation Approaches

Three approaches are implemented and compared (see `inference/` and `training/`):

1. **Prompt Forcing** (`inference/prompt_forcing.py`) — Prompt the LLM to generate a subgraph at each step
2. **Supervised Fine-Tuning (SFT)** (`training/sft.py`) — Fine-tune on deterministic solver traces
3. **Decoding-Level Forcing** (`inference/decoding_forcing.py`) — Constrain decoding to valid subgraph format

---

## Graph Families

Generated in `data/generators/graph_families.py`:

| Family | Notes |
|---|---|
| Erdős–Rényi | Random graphs, baseline distribution |
| Barabási–Albert | Scale-free graphs, zero-shot transfer target |
| Tree | Acyclic, simple path structure |
| Grid | Regular structure, good for BFS/DFS |

Hard cases in `data/generators/hard_cases.py`:
- **Bridge graphs** — All paths traverse a single edge
- **Bottleneck graphs** — High-traffic single-node choke points
- **High-girth graphs** — Long cycles requiring long-term memory

---

## Evaluation Metrics

| Metric | Description | Success Threshold |
|---|---|---|
| Operation Accuracy | Correct next-step prediction | ≥ 90% |
| State Consistency | LLM reasoning aligns with state executor | Maximize |
| Structural Generalization | Accuracy on graphs >> training size | Evaluate at 10x, 100x scale |

Comparisons are run against **Chain of Thought (CoT)** baseline on identical graphs.

---

## Training Loss

Token-level cross-entropy (see `training/sft.py`):

```
L_SFT = - sum_{t=1}^{T} log p_theta(y_t | x, y_{<t})
```

Where:
- `x` = prompt (graph + query + instructions)
- `y_t` = target token (operation) at position `t`

---

## Models

- **Primary:** Qwen 2.5 7B Instruct
- **Alternative:** Llama 3.1B

Configs in `training/configs/`.

---

## Project Timeline

| Phase | Weeks | Goal |
|---|---|---|
| Synthetic Foundation | 1–2 | Data pipeline, JSON trace generation, linearization |
| Supervised Fine-Tuning | 3–5 | SFT training, teacher forcing, negative sampling |
| Evaluation | 6–8 | Full benchmark suite, CoT vs GoT comparison |
| Paper & Publication | 9–10 | Final paper, figures, reasoning trace comparisons |

---

## Development Notes for Claude Code

- **Start with `solvers/` and `data/generators/`** — all downstream work depends on correct trace generation
- **JSON trace format** must record: `{ "step": t, "operation": o_t, "state": s_t, "induced_subgraph": H_t }`
- **State executor is the source of truth** — LLM outputs are always validated against it, never trusted blindly
- When implementing SFT, use **teacher forcing on states** to prevent error compounding during training
- **Negative sampling**: explicitly corrupt ~20% of traces and require the model to emit `CORRECTION` before continuing
- For evaluation, always run CoT baseline and GoT on **identical graphs** to ensure fair comparison
- Plot `path_length vs accuracy` for both methods — this is the key figure showing CoT state drift vs GoT stability
