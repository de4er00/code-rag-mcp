"""Command line: index, search, eval."""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

from . import config, store
from .indexer import index_corpus, print_report

for _stream in (sys.stdout, sys.stderr):  # legacy Windows code pages cannot print every character in a corpus
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

MODES = ("bm25", "dense", "hybrid")


def cmd_index(args: argparse.Namespace) -> None:
    models = args.models.split(",") if args.models else [config.DEFAULT_MODEL]
    print_report(index_corpus(config.load_corpus(Path(args.corpus)), models=models))


def cmd_search(args: argparse.Namespace) -> None:
    from . import retrieval

    conn = store.connect()
    results = retrieval.search(conn, args.query, k=args.k, mode=args.mode, repo=args.repo, model=args.model)
    conn.close()
    if not results:
        print("No results.")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.path}:{r.start_line}-{r.end_line}  {r.symbol} ({r.kind})  score={r.score:.4f}  id={r.chunk_id}")
        print("   " + r.snippet.replace("\n", " | "))


def cmd_eval(args: argparse.Namespace) -> None:
    from . import evaluation

    model = args.model or config.DEFAULT_MODEL
    report = index_corpus(config.load_corpus(Path(args.corpus)), models=[model])
    print_report(report)
    conn = store.connect()
    sets = {name: evaluation.load_questions(evaluation.EVAL_DIR / f"{name}.yaml") for name in ("questions", "holdout")}
    results = {
        name: {mode: evaluation.run_mode(conn, qs, mode, model=None if mode == "bm25" else model) for mode in MODES}
        for name, qs in sets.items()
    }
    conn.close()
    for name, by_mode in results.items():
        print(f"\n{name} ({len(sets[name])} questions)")
        for mode, m in by_mode.items():
            print(f"  {mode:7s} recall@5={m['recall@5']:.2f} recall@10={m['recall@10']:.2f} mrr={m['mrr']:.2f}"
                  f"  misses@5={','.join(m['misses']) or '-'}")
    out = evaluation.EVAL_DIR / f"results-{model.split('/')[-1].replace('@', '-')}.md"
    out.write_text(_results_md(model, report, sets, results), encoding="utf-8")
    print(f"\nWrote {out}")


def _results_md(model: str, report, sets: dict, results: dict) -> str:
    lines = [f"# Retrieval results: {model}", "",
             f"Run on {datetime.date.today().isoformat()}; {report.chunks_total} chunks in the index.", ""]
    for name, by_mode in results.items():
        lines += [f"## {name}.yaml ({len(sets[name])} questions)", "",
                  "| mode | recall@5 | recall@10 | MRR | missed in top 5 |", "|---|---|---|---|---|"]
        for mode, m in by_mode.items():
            lines.append(f"| {mode} | {m['recall@5']:.2f} | {m['recall@10']:.2f} | {m['mrr']:.2f} | "
                         f"{', '.join(m['misses']) or '—'} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="code-rag")
    parser.add_argument("--corpus", default=str(config.DEFAULT_CORPUS_FILE), help="YAML list of repositories")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="walk, chunk, redact, store and embed the corpus")
    p.add_argument("--models", help="comma-separated embedding model ids (default: %(default)s)")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="search the index")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--repo")
    p.add_argument("--model")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("eval", help="index if needed, then score every mode on eval/questions.yaml and eval/holdout.yaml")
    p.add_argument("--model")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
