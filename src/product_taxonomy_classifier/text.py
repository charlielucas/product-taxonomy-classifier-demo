"""Text helpers for product records."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize simple product text."""
    return TOKEN_RE.findall(text.lower())


def record_text(record: dict[str, str]) -> str:
    """Combine useful product fields into one classification string."""
    parts = [
        record.get("vendor", ""),
        record.get("part_number", ""),
        record.get("description", ""),
    ]
    return " ".join(part for part in parts if part)


def count_tokens(records: Iterable[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(tokenize(record_text(record)))
    return counts
