"""
Canonicalize and compare Graph-of-Thoughts operation strings.

Benchmarks (NLGraph / GLBench) and training traces may use slightly different
spellings for the same transition (e.g. ``visit(x)`` vs ``mark_visited(x)``).
Base models often emit natural language or wrong casing on the first line;
exact string equality then yields 0% accuracy even when partially correct.

This module provides:
  - robust extraction of an ``op(args)`` substring from messy generation text
  - normalization to a canonical form for fair scoring against gold ops
"""

from __future__ import annotations

import re
from typing import Any

# Mirror solvers.state_executor: visit and mark_visited are equivalent.
_VISIT_ALIASES = frozenset({"visit", "mark_visited"})


def strip_step_prefix(line: str) -> str:
    """Remove leading ``Step t:`` chat-style prefix if present."""
    s = line.strip()
    m = re.match(r"^Step\s+\d+\s*:\s*", s, flags=re.IGNORECASE)
    if m:
        return s[m.end() :].strip()
    return s


def extract_operation_call(text: str) -> str:
    """
    Best-effort extract a single operation call ``name(args)`` from model text.

    Order:
    1. First non-empty line, with ``Step t:`` stripped.
    2. If that does not contain ``(``, scan full text for a valid-looking call.
    3. Normalize common typos (``Mark visited`` -> ``mark_visited``).
    """
    if not text or not text.strip():
        return ""

    # Collapse "Mark visited(5)" style into mark_visited(5)
    collapsed = re.sub(r"(?i)mark\s+visited\s*\(", "mark_visited(", text)

    def first_call_in(s: str) -> str:
        m = re.search(
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)",
            s,
        )
        if m:
            return f"{m.group(1)}({m.group(2)})".strip()
        return ""

    line = extract_first_nonblank_line(collapsed)
    line = strip_step_prefix(line)
    if "(" in line:
        # Often the whole line is exactly ``enqueue(1)`` or ``Step 0: enqueue(1)``
        inner = first_call_in(line)
        if inner:
            return inner

    return first_call_in(collapsed)


def extract_first_nonblank_line(text: str) -> str:
    for line in text.split("\n"):
        if line.strip():
            return line
    return ""


def _parse_op(operation: str) -> tuple[str, list[Any]]:
    """Same convention as ``solvers.state_executor._parse_op`` (local copy to avoid cycles)."""
    operation = operation.strip()
    if "(" not in operation:
        return operation, []
    op = operation[: operation.index("(")]
    raw_args = operation[operation.index("(") + 1 : operation.rindex(")")]
    if not raw_args.strip():
        return op, []
    parsed: list[Any] = []
    for a in raw_args.split(","):
        a = a.strip()
        if a == "None":
            parsed.append(None)
            continue
        try:
            parsed.append(int(a))
        except ValueError:
            try:
                parsed.append(float(a))
            except ValueError:
                parsed.append(a)
    return op, parsed


def _fmt_arg(a: Any) -> str:
    if a is None:
        return "None"
    return str(a)


def canonical_operation_string(op: str | None) -> str | None:
    """
    Return a canonical string representation for scoring equality.

    Rules:
    - Lowercase operation name.
    - ``visit`` and ``mark_visited`` -> ``mark_visited`` (executor-equivalent).
    - Normalize numeric args: 5.0 -> 5 when integral.
    """
    if op is None:
        return None
    raw = op.strip()
    if not raw:
        return None
    # So "_parse_op" sees a single token name (same as extract_operation_call).
    raw = re.sub(r"(?i)mark\s+visited\s*\(", "mark_visited(", raw)

    name, args = _parse_op(raw)
    if not name:
        return None

    name_l = name.lower()
    if name_l in _VISIT_ALIASES:
        name_l = "mark_visited"

    norm_args: list[Any] = []
    for a in args:
        if isinstance(a, float) and a.is_integer():
            norm_args.append(int(a))
        else:
            norm_args.append(a)

    if not norm_args:
        return f"{name_l}()"
    return f"{name_l}({', '.join(_fmt_arg(x) for x in norm_args)})"


def operations_match(
    predicted: str | None,
    gold: str | None,
    *,
    algorithm: str | None = None,
) -> bool:
    """True iff predicted and gold denote the same operation under normalization."""
    cp = canonical_operation_string(predicted or "")
    cg = canonical_operation_string(gold or "")
    if cp is None or cg is None:
        return False
    return cp == cg
