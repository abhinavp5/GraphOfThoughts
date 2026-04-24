"""
Quick validation for training config files.

Checks required keys and (optionally) tokenizer accessibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REQ_TOP = ("model", "lora", "training", "data")
REQ_MODEL = ("name",)
REQ_DATA = ("train_dir", "train_pattern")


def validate_config(path: Path) -> list[str]:
    errs = []
    with path.open() as f:
        cfg = yaml.safe_load(f)
    for k in REQ_TOP:
        if k not in cfg:
            errs.append(f"missing top-level key: {k}")
    for k in REQ_MODEL:
        if k not in cfg.get("model", {}):
            errs.append(f"missing model.{k}")
    for k in REQ_DATA:
        if k not in cfg.get("data", {}):
            errs.append(f"missing data.{k}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs-dir", default="training/configs")
    ap.add_argument("--check-tokenizer", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.configs_dir).glob("*.yaml"))
    if not files:
        raise SystemExit(f"No yaml files in {args.configs_dir}")

    bad = 0
    for p in files:
        errs = validate_config(p)
        if errs:
            bad += 1
            print(f"[FAIL] {p}")
            for e in errs:
                print(f"  - {e}")
            continue
        print(f"[OK]   {p}")

        if args.check_tokenizer:
            try:
                from transformers import AutoTokenizer

                AutoTokenizer.from_pretrained(
                    yaml.safe_load(p.read_text())["model"]["name"],
                    trust_remote_code=True,
                )
                print("      tokenizer load OK")
            except Exception as e:
                bad += 1
                print(f"      tokenizer load FAIL: {e}")

    if bad:
        raise SystemExit(f"Config smoke test failed on {bad} file(s)")
    print("All configs passed.")


if __name__ == "__main__":
    main()
