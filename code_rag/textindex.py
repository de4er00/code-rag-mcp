"""Text preparation for the keyword index: contextual header, identifier splitting (path separators
and camelCase), stopword removal and RU/EN stemming. The same functions run at index and query time,
so a stemmed query matches stemmed text.
"""
from __future__ import annotations

import re
from pathlib import Path

import snowballstemmer

_ru_stemmer = snowballstemmer.stemmer("russian")
_en_stemmer = snowballstemmer.stemmer("english")

_CYRILLIC_RE = re.compile(r"[а-яА-ЯёЁ]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_IDENT_SPLIT_RE = re.compile(r"[\\/_\-.\s]+")
_CAMEL_BOUNDARY_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_BOUNDARY_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")

# Small, hand-picked stopword lists: common RU/EN question/function words
# that carry no retrieval signal on their own and would otherwise dominate
# an OR-of-all-tokens BM25 query.
STOPWORDS_RU = {
    "как", "что", "где", "когда", "почему", "зачем", "какой", "какая",
    "какие", "каким", "какого", "который", "которая", "которые", "это",
    "этот", "эта", "эти", "для", "при", "если", "или", "либо", "чтобы",
    "чтоб", "на", "по", "в", "во", "с", "со", "о", "об", "от", "до", "из",
    "у", "к", "не", "ни", "и", "а", "но", "же", "ли", "бы", "то", "так",
    "тот", "та", "те", "его", "её", "их", "он", "она", "оно", "они",
}
STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "does", "do", "did", "how", "what", "why", "where", "when", "which",
    "who", "whom", "this", "that", "these", "those", "in", "on", "for",
    "of", "to", "and", "or", "but", "with", "without", "into", "onto",
    "from", "by", "as", "at", "it", "its", "there", "here", "if", "so",
}
STOPWORDS = STOPWORDS_RU | STOPWORDS_EN


def relpath_str(path: str) -> str:
    """Chunks already carry a logical 'repo/relative/path'; normalise separators only."""
    return path.replace("\\", "/")


def make_header(path: str, symbol: str, kind: str) -> str:
    """Human-readable header, e.g. 'httpx/httpx/_client.py | Client.send | method'."""
    return f"{relpath_str(path)} | {symbol} | {kind}"


def split_identifier(s: str) -> list[str]:
    """Split on path separators / _ . - and camelCase boundaries; lowercase."""
    out: list[str] = []
    for piece in _IDENT_SPLIT_RE.split(s):
        if not piece:
            continue
        sub = _CAMEL_BOUNDARY_1.sub(" ", piece)
        sub = _CAMEL_BOUNDARY_2.sub(" ", sub)
        for tok in sub.split():
            if tok:
                out.append(tok.lower())
    return out


def header_tokens_text(path: str, repo: str, symbol: str, kind: str) -> str:
    """Space-joined tokens for the FTS `header_tokens` column."""
    tokens = split_identifier(relpath_str(path)) + split_identifier(symbol) + [kind.lower(), repo.lower()]
    return " ".join(tokens)


def is_stopword(token: str) -> bool:
    return token.lower() in STOPWORDS


def stem_token(token: str) -> str:
    low = token.lower()
    if _CYRILLIC_RE.search(low):
        return _ru_stemmer.stemWord(low)
    return _en_stemmer.stemWord(low)


def stem_text(text: str) -> str:
    """Lowercase, drop stopwords, stem — used for the stored `body_stemmed`
    FTS column, and for query tokens at search time (same function, same
    behavior, so index-time and query-time stemming can never drift apart)."""
    out = []
    for tok in _WORD_RE.findall(text):
        low = tok.lower()
        if is_stopword(low):
            continue
        out.append(stem_token(low))
    return " ".join(out)


def query_tokens(query: str) -> list[str]:
    """Raw lowercase tokens from a query, stopwords removed, de-duplicated,
    order preserved."""
    seen = set()
    out = []
    for tok in _WORD_RE.findall(query):
        low = tok.lower()
        if low in seen or is_stopword(low):
            continue
        seen.add(low)
        out.append(low)
    return out
