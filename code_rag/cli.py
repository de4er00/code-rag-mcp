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

MODES = config.SEARCH_MODES


def cmd_index(args: argparse.Namespace) -> None:
    models = args.models.split(",") if args.models else [config.DEFAULT_MODEL]
    print_report(index_corpus(config.load_corpus(Path(args.corpus)), models=models))


def cmd_search(args: argparse.Namespace) -> None:
    from . import retrieval

    conn = store.connect()
    results = retrieval.search(conn, args.query, k=args.k, mode=args.mode, repo=args.repo, model=args.model,
                               reranker=args.reranker)
    conn.close()
    if not results:
        print("No results.")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.path}:{r.start_line}-{r.end_line}  {r.symbol} ({r.kind})  score={r.score:.4f}  id={r.chunk_id}")
        print("   " + r.snippet.replace("\n", " | "))


def cmd_eval(args: argparse.Namespace) -> None:
    from . import evaluation, rerank

    model = args.model or config.DEFAULT_MODEL
    reranker = args.reranker or config.RERANK_MODEL
    modes = [m for m in MODES if not (args.no_rerank and m == "hybrid+rerank")]
    report = index_corpus(config.load_corpus(Path(args.corpus)), models=[model])
    print_report(report)
    conn = store.connect()
    sets = {name: evaluation.load_questions(evaluation.EVAL_DIR / f"{name}.yaml") for name in evaluation.SETS}
    results = {
        name: {
            mode: evaluation.run_mode(conn, qs, mode, model=None if mode == "bm25" else model, reranker=reranker)
            for mode in modes
        }
        for name, qs in sets.items()
    }
    conn.close()
    rerank.unload()
    for name, by_mode in results.items():
        spans = _has_spans(sets[name])
        print(f"\n{name} ({len(sets[name])} questions)")
        for mode, m in by_mode.items():
            fn = f" | function level recall@5={m['fn_recall@5']:.2f} mrr={m['fn_mrr']:.2f}" if spans else ""
            print(f"  {mode:13s} recall@5={m['recall@5']:.2f} recall@10={m['recall@10']:.2f} mrr={m['mrr']:.2f}"
                  f"{fn}  {m['latency_ms']:.0f} ms  misses@5={','.join(m['misses']) or '-'}")
    name = "-".join(m.split("/")[-1].replace("@", "-") for m in [model] + ([args.reranker] if args.reranker else []))
    out = evaluation.EVAL_DIR / f"results-{name}.md"
    out.write_text(_results_md(model, reranker, report, sets, results), encoding="utf-8")
    print(f"\nWrote {out}")


def _has_spans(questions: list[dict]) -> bool:
    return any("expected_spans" in q or "expected_all" in q for q in questions)


def _results_md(model: str, reranker: str, report, sets: dict, results: dict) -> str:
    lines = [f"# Retrieval results: {model}", "",
             f"Run on {datetime.date.today().isoformat()}; {report.chunks_total} chunks in the index. "
             f"hybrid+rerank reorders the top {config.RERANK_CANDIDATES} fused candidates with {reranker}. "
             "Latency is the mean wall time of one search, after the models are loaded.", ""]
    for name, by_mode in results.items():
        spans = _has_spans(sets[name])
        lines += [f"## {name}.yaml ({len(sets[name])} questions)", ""]
        if spans:
            lines += ["| mode | recall@5 | recall@10 | MRR | function recall@5 | function MRR | ms/query | missed in top 5 |",
                      "|---|---|---|---|---|---|---|---|"]
        else:
            lines += ["| mode | recall@5 | recall@10 | MRR | ms/query | missed in top 5 |", "|---|---|---|---|---|---|"]
        for mode, m in by_mode.items():
            fn = f"{m['fn_recall@5']:.2f} | {m['fn_mrr']:.2f} | " if spans else ""
            lines.append(f"| {mode} | {m['recall@5']:.2f} | {m['recall@10']:.2f} | {m['mrr']:.2f} | {fn}"
                         f"{m['latency_ms']:.0f} | {', '.join(m['misses']) or '—'} |")
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
    p.add_argument("--reranker", help=f"cross-encoder for hybrid+rerank (default: {config.RERANK_MODEL})")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("eval", help="index if needed, then score every mode on every question set in eval/")
    p.add_argument("--model")
    p.add_argument("--reranker", help=f"cross-encoder for hybrid+rerank (default: {config.RERANK_MODEL})")
    p.add_argument("--no-rerank", action="store_true", help="skip hybrid+rerank (no reranker download)")
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
