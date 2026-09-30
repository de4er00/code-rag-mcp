"""Walk a repository and decide which files to index, counting the reason for every skip."""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass
class WalkResult:
    included: list[Path]
    skipped_by_reason: Counter
    skipped_dirs_by_reason: Counter


def _dir_skip_reason(dirname: str, rel_path: str) -> str | None:
    if dirname.lower() in config.SKIP_DIR_NAMES:
        return "skip_dir_name"
    rel = rel_path.lower().replace("\\", "/")
    if any(sub in rel for sub in config.SKIP_DIR_PATH_SUBSTRINGS):
        return "vendored_or_generated"
    return None


def _file_skip_reason(name: str, size: int) -> str | None:
    lower = name.lower()
    if any(sub in lower for sub in config.SKIP_FILE_SUBSTRINGS) or lower.endswith(tuple(config.SKIP_FILE_SUFFIXES)):
        return "secret_like_name"
    if Path(lower).suffix not in config.INCLUDE_EXTENSIONS:
        return "extension_not_included"
    if size > config.MAX_FILE_SIZE:
        return "too_large"
    return None


def walk_source(root: Path) -> WalkResult:
    included: list[Path] = []
    skipped: Counter = Counter()
    skipped_dirs: Counter = Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        keep = []
        for d in dirnames:
            reason = _dir_skip_reason(d, d if rel_dir == "." else os.path.join(rel_dir, d))
            if reason:
                skipped_dirs[reason] += 1
            else:
                keep.append(d)
        dirnames[:] = keep
        for fname in filenames:
            fpath = Path(dirpath) / fname
            try:
                size = fpath.stat().st_size
            except OSError:
                skipped["stat_error"] += 1
                continue
            reason = _file_skip_reason(fname, size)
            if reason:
                skipped[reason] += 1
            else:
                included.append(fpath)
    return WalkResult(included, skipped, skipped_dirs)


def print_summary(repo: str, result: WalkResult) -> None:
    print(f"[{repo}] included files: {len(result.included)}")
    for reason, count in result.skipped_by_reason.most_common():
        print(f"  skipped {reason}: {count}")
    for reason, count in result.skipped_dirs_by_reason.most_common():
        print(f"  skipped dirs {reason}: {count}")
