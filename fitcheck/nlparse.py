"""Natural language spec parser for fitcheck plan commands.

Parses shorthand like:
    "qlora meta-llama/Llama-3.1-8B on 3090"
    "lora llama-3.1-8b on a100 with data.jsonl"
    "full phi-3-mini on h100 at 2048"

into structured fields for api.plan().
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ParsedSpec:
    """Result of parsing a natural language plan spec."""

    method: str
    model_id: str
    gpu: str
    dataset_path: str | None = None
    seq_len: int | None = None


# Pattern: METHOD MODEL on GPU [with DATASET] [at SEQ_LEN]
# METHOD must be one of: full, lora, qlora (case insensitive)
# MODEL can contain letters, numbers, hyphens, dots, slashes, underscores
# GPU can contain letters, numbers, hyphens
# DATASET is a file path (non-whitespace)
# SEQ_LEN is an integer
_SPEC_PATTERN = re.compile(
    r"^(full|lora|qlora)"  # method
    r"\s+"
    r"(\S+)"  # model_id
    r"\s+on\s+"
    r"(\S+)"  # gpu
    r"(?:\s+with\s+(\S+))?"  # optional dataset
    r"(?:\s+at\s+(\d+))?"  # optional seq_len
    r"$",
    re.IGNORECASE,
)


def parse_spec(spec: str) -> ParsedSpec | None:
    """Parse a natural language plan spec string.

    Returns ParsedSpec if the string matches the expected pattern,
    or None if it doesn't match (caller should fall through to
    flag-based parsing).
    """
    match = _SPEC_PATTERN.match(spec.strip())
    if match is None:
        return None

    method, model_id, gpu, dataset_path, seq_len_str = match.groups()

    return ParsedSpec(
        method=method.lower(),
        model_id=model_id,
        gpu=gpu,
        dataset_path=dataset_path,
        seq_len=int(seq_len_str) if seq_len_str else None,
    )
