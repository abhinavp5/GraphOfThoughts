"""
Supervised Fine-Tuning (SFT) — main training script for Graph of Thoughts.

Fine-tunes a causal LM (default: Llama 3.2-1B-Instruct) with LoRA on
graph algorithm traces.  Uses HuggingFace Trainer for the training loop.

Usage
-----
# From repo root:
python -m training.sft --config training/configs/llama_3_1b.yaml

# 4 GPUs (DDP) on one node:
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m training.sft \\
    --config training/configs/llama_3_1b.yaml

# Or with CLI overrides:
python -m training.sft \\
    --config training/configs/llama_3_1b.yaml \\
    --epochs 5 \\
    --lr 1e-4

Gated models (Llama on HF)
-------------------------
`meta-llama/*` requires (1) accepting the license on the model page and
(2) authentication. On HPC, set a token in the job environment, e.g.:

    export HF_TOKEN="$(cat ~/.hf_token)"   # or your site's secret injection

Alternatively run `huggingface-cli login` once so
`~/.cache/huggingface/token` exists on shared / home storage, or set
`HF_HOME` to that cache. Without this, downloads fail with 401 / GatedRepoError.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dataset import GoTDataset, GoTDataCollator, load_traces
from training.negative_sampling import CORRECTION_TOKEN


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("got.sft")


def _is_local_main_process() -> bool:
    """True if this is the only process or LOCAL_RANK 0 (under torchrun)."""
    lr = int(os.environ.get("LOCAL_RANK", "-1"))
    return lr in (-1, 0)


def _hf_token_configured() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


def _maybe_warn_hf_gated_model(model_id: str) -> None:
    """Llama (and similar) need HF login; fail early with a clear message."""
    mid = model_id.lower()
    if "meta-llama" not in mid:
        return
    if _hf_token_configured():
        return
    logger.error(
        "Hugging Face auth missing for gated model %r. Accept the license at "
        "https://huggingface.co/%s then set HF_TOKEN or run huggingface-cli login "
        "(see docstring at top of training/sft.py).",
        model_id,
        model_id,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config, returning a nested dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Override config values with CLI arguments when provided."""
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.output_dir is not None:
        config["training"]["output_dir"] = args.output_dir
    return config


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def setup_tokenizer(model_name: str, special_tokens: list[str] | None = None):
    """Load tokenizer and add any special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=True,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Add special tokens (e.g., CORRECTION)
    if special_tokens:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        if num_added > 0:
            logger.info(f"Added {num_added} special token(s): {special_tokens}")

    return tokenizer


def setup_model(model_name: str, tokenizer, lora_config: dict):
    """Load model, resize embeddings, apply LoRA."""
    logger.info(f"Loading model: {model_name}")

    # DDP (torchrun): one full replica per rank — do not use device_map="auto".
    # Single process: device_map="auto" is fine for multi-GPU single-node sharding.
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    load_kw: dict = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if local_rank == -1:
        load_kw["device_map"] = "auto"

    load_kw["token"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)

    # Resize embeddings if we added special tokens
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized embeddings: {model.config.vocab_size} → {len(tokenizer)}")

    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def setup_datasets(config: dict, tokenizer) -> tuple:
    """Load training and validation datasets."""
    data_cfg = config["data"]
    trace_dir = data_cfg.get("train_dir", "data/traces")
    max_length = data_cfg.get("max_seq_length", 2048)
    neg_ratio = data_cfg.get("negative_sample_ratio", 0.2)

    logger.info(f"Loading training traces from {trace_dir}")
    train_traces = load_traces(trace_dir, data_cfg.get("train_pattern", "train_*.json"))
    logger.info(f"  Loaded {len(train_traces)} training traces")

    logger.info(f"Loading validation traces from {trace_dir}")
    val_traces = load_traces(trace_dir, data_cfg.get("val_pattern", "val_*.json"))
    logger.info(f"  Loaded {len(val_traces)} validation traces")

    train_dataset = GoTDataset(
        traces=train_traces,
        tokenizer=tokenizer,
        max_length=max_length,
        negative_ratio=neg_ratio,
        seed=42,
    )

    val_dataset = GoTDataset(
        traces=val_traces,
        tokenizer=tokenizer,
        max_length=max_length,
        negative_ratio=0.0,  # no corruption for validation
        seed=123,
    )

    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict):
    """Run the full SFT training pipeline."""
    model_cfg = config["model"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    data_cfg = config["data"]

    # Seed
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    _maybe_warn_hf_gated_model(model_cfg["name"])

    # Tokenizer
    special_tokens = config.get("special_tokens", [CORRECTION_TOKEN])
    tokenizer = setup_tokenizer(model_cfg["name"], special_tokens)

    # Model
    model = setup_model(model_cfg["name"], tokenizer, lora_cfg)

    # Datasets
    train_dataset, val_dataset = setup_datasets(config, tokenizer)

    # Data collator
    collator = GoTDataCollator(
        tokenizer=tokenizer,
        max_length=data_cfg.get("max_seq_length", 2048),
    )

    # Output directory
    output_dir = train_cfg.get("output_dir", "checkpoints/got-lora")
    os.makedirs(output_dir, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=train_cfg.get("batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        bf16=train_cfg.get("bf16", True),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",  # set to "wandb" if you have wandb configured
        seed=seed,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Log final metrics
    metrics = train_result.metrics
    logger.info(f"Training complete. Final metrics: {metrics}")

    # Save
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Final eval
    logger.info("Running final evaluation...")
    eval_metrics = trainer.evaluate()
    logger.info(f"Eval metrics: {eval_metrics}")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("Done!")
    return metrics, eval_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM on Graph of Thoughts traces."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config = merge_cli_overrides(config, args)

    if _is_local_main_process():
        logger.info(f"Config: {config}")
    train(config)


if __name__ == "__main__":
    main()
