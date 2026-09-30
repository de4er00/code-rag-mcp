"""Redaction of secret-looking strings before anything is written to the index.

Applied to every chunk's text right before it is stored. Order matters: the
specific token/key patterns run first so the generic "credential assignment"
pattern does not re-match (and mis-classify) something already handled.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Tuple

PLACEHOLDER = "[REDACTED:{kind}]"

# (kind, compiled regex, group index to replace or None to replace whole match)
_PATTERNS: list[tuple[str, re.Pattern, "int | None"]] = [
    ("telegram_token", re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{35}\b"), None),
    ("openai_key", re.compile(r"\bsk-(?:ant-|or-)?[A-Za-z0-9_-]{16,}\b"), None),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), None),
    ("github_token", re.compile(r"\b(?:ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{20,})\b"), None),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{10,}=*"), None),
    # NAME (containing key/token/secret/password/api) = "value" / : value
    (
        "credential",
        re.compile(
            r"(?im)\b(\w*(?:key|token|secret|password|passwd|api)\w*\s*[:=]\s*)"
            r"(['\"]?)([A-Za-z0-9+/_\-]{12,})\2"
        ),
        3,
    ),
    ("phone", re.compile(r"(?<![\w.])(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\w)"), None),
    ("ip_address", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), None),
]


def redact_text(text: str) -> Tuple[str, Counter]:
    """Return (redacted_text, counts_by_kind)."""
    counts: Counter = Counter()

    def make_sub(kind: str, group: "int | None"):
        def _sub(m: re.Match) -> str:
            counts[kind] += 1
            placeholder = PLACEHOLDER.format(kind=kind)
            if group is None:
                return placeholder
            # replace only the captured group, keep the rest of the match
            start, end = m.span(group)
            whole_start = m.start()
            return m.group(0)[: start - whole_start] + placeholder + m.group(0)[end - whole_start :]
        return _sub

    for kind, pattern, group in _PATTERNS:
        text = pattern.sub(make_sub(kind, group), text)

    return text, counts
