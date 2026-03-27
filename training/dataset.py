"""
GoT Dataset — PyTorch Dataset wrapping JSON traces for causal LM SFT.

Loads trace files, formats them with teacher forcing (ground-truth states),
optionally corrupts a fraction via negative sampling, and tokenizes
everything into input_ids + labels tensors ready for HuggingFace Trainer.

Public API
----------
GoTDataset(trace_files, tokenizer, max_length, negative_ratio, seed)
    → torch.utils.data.Dataset yielding {input_ids, attention_mask, labels}

load_traces(directory, pattern) → list[dict]
    Load all JSON trace files matching a glob pattern.
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.teacher_forcing import format_full_chat
from training.negative_sampling import maybe_corrupt, CORRECTION_TOKEN


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_traces(directory: str, pattern: str = "train_*.json") -> list[dict]:
    """
    Load all JSON trace files matching `pattern` under `directory`.

    Each file is expected to contain a JSON array of trace dicts.

    Returns
    -------
    list[dict]
        Flat list of all trace dicts across all matching files.
    """
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in '{directory}'"
        )

    traces = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        if isinstance(data, list):
            traces.extend(data)
        else:
            traces.append(data)

    return traces


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GoTDataset(Dataset):
    """
    PyTorch Dataset for Graph of Thoughts SFT training.

    Each sample is a tokenized chat conversation (system + user + assistant)
    where:
      - system/user tokens have labels=-100 (ignored in loss)
      - assistant tokens have labels=input_ids (supervised)

    Parameters
    ----------
    trace_files : list[dict]
        List of raw trace dicts (from load_traces).
    tokenizer : PreTrainedTokenizer
        Tokenizer with chat template support.
    max_length : int
        Maximum sequence length (truncated if exceeded).
    negative_ratio : float
        Fraction of traces to corrupt with negative sampling.
    seed : int
        Random seed for negative sampling reproducibility.
    """

    def __init__(
        self,
        traces: list[dict],
        tokenizer,
        max_length: int = 2048,
        negative_ratio: float = 0.2,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        rng = random.Random(seed)

        for trace in traces:
            # Optionally corrupt the trace
            trace = maybe_corrupt(trace, ratio=negative_ratio, rng=rng)

            # Format as chat messages
            messages = format_full_chat(trace)

            # Tokenize the full conversation
            sample = self._tokenize_chat(messages)
            if sample is not None:
                self.samples.append(sample)

    def _tokenize_chat(self, messages: list[dict]) -> Optional[dict]:
        """
        Tokenize a chat conversation and create labels with prompt masking.

        The prompt (system + user messages) gets labels=-100 so the loss
        is only computed on the assistant's completion tokens.
        """
        # Tokenize the full conversation (all messages)
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # Tokenize just the prompt (system + user) to find where to start labels
        prompt_messages = messages[:-1]  # everything except assistant
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,  # adds the assistant header
        )
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]

        # Create labels: -100 for prompt tokens, input_ids for completion tokens
        prompt_len = len(prompt_encoding["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        # Ensure labels length matches input_ids
        labels = labels[:len(input_ids)]

        if len(input_ids) < 4:
            return None  # skip degenerate samples

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class GoTDataCollator:
    """
    Pads a batch of GoT samples to the same length.

    Uses left-padding for input_ids/attention_mask and -100 padding for labels.
    """

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict:
        # Find max length in this batch
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length,
        )

        input_ids = []
        attention_mask = []
        labels = []

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            if pad_len > 0:
                # Right-pad
                input_ids.append(
                    torch.cat([f["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
                )
                attention_mask.append(
                    torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
                )
                labels.append(
                    torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
                )
            else:
                # Truncate
                input_ids.append(f["input_ids"][:max_len])
                attention_mask.append(f["attention_mask"][:max_len])
                labels.append(f["labels"][:max_len])

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }
