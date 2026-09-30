"""Clone the benchmark corpus: four well-known Python projects at pinned tags.

Pinned tags keep the evaluation numbers reproducible: the expected paths in eval/*.yaml refer to
these exact versions.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOS = {
    "requests": ("https://github.com/psf/requests.git", "v2.32.3"),
    "httpx": ("https://github.com/encode/httpx.git", "0.28.1"),
    "click": ("https://github.com/pallets/click.git", "8.1.7"),
    "rich": ("https://github.com/Textualize/rich.git", "v13.9.4"),
}
CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


def main() -> int:
    CORPUS_DIR.mkdir(exist_ok=True)
    for name, (url, tag) in REPOS.items():
        target = CORPUS_DIR / name
        if target.exists():
            print(f"{name}: already present")
            continue
        print(f"{name}: cloning {tag}")
        subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--depth", "1", "--branch", tag, url, str(target)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
