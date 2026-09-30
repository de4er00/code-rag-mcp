# code-rag-mcp

Local code search for coding agents. It indexes your repositories into a single SQLite file, searches them with a mix of keyword and embedding retrieval, and serves the results to Claude Code or any other MCP client. It uses no API keys and no cloud services, and a laptop CPU is enough.

I built the first version to let an agent find its way around a private platform of about 580 files: seven services, some docs in Russian, some code comments in English. grep does not understand "why are the Google clients built per thread", and pasting everything into the context does not fit. This repository is the same tool, generalised and measured on public code, so the numbers can be reproduced.

## How it works

- **Chunking follows the code.** Python files are split with the `ast` module: the module docstring and imports, each function, and each class. Large classes become a header chunk plus one chunk per method. A gap-filling pass turns any line not covered by a definition (constants, `if __name__ == "__main__"` blocks) into its own chunk, so nothing escapes indexing or redaction. Markdown is split by headings, everything else by line windows.
- **Secrets are redacted before anything is stored:** tokens, API keys, bearer headers, credential assignments, phone numbers, IP addresses.
- **Two retrievers.**
  - Keyword search uses SQLite FTS5 with BM25 over two columns: a header (path, symbol, kind; split on separators and camelCase, weighted 3×) and the stemmed, stopword-filtered body.
  - Dense search is cosine similarity over embeddings kept as blobs in the same database. The default model is `bge-small-en-v1.5` through fastembed on CPU; `multilingual-e5-large` on a CUDA GPU is optional.
- **Fusion and ranking.** The two lists are merged with weighted Reciprocal Rank Fusion. Tests, docs and changelogs get half weight, and one file contributes at most two chunks to a result list.
- **Incremental indexing.** Files are tracked by sha256. Unchanged files are skipped, deleted files are removed from all three tables, and embedding is committed in small batches, so an interrupted run resumes where it stopped. The index stores logical paths (`repo/relative/path`), never local ones, so it can be copied between machines.
- **MCP server** (stdio) with three tools: `search_code`, `read_chunk`, `list_repos`.

## Benchmark

The benchmark indexes four well-known projects at pinned tags (requests 2.32.3, httpx 0.28.1, click 8.1.7, rich 13.9.4): 602 files, 6,718 chunks. There are 40 questions, phrased the way a developer asks an agent, without copying identifiers. Each lists the file where the answer lives. 30 questions are for development; 10 were held out and looked at once, after the settings below were fixed.

In the first run hybrid search was worse than dense search alone:

| dev set, 30 questions | recall@5 | recall@10 | MRR |
|---|---|---|---|
| BM25 | 0.63 | 0.80 | 0.31 |
| dense | 0.77 | 0.90 | 0.47 |
| hybrid | 0.67 | 0.90 | 0.40 |

Looking at the misses explained it. For "where does the client stop following redirects after too many hops", BM25's first answer was `tests/client/test_redirects.py` and dense search's first answer was `docs/user/quickstart.rst`. Tests and docs describe behaviour in exactly the words people use in questions, but an agent asking "where" wants the implementation.

Two changes:
- tests, docs and changelogs get half the score, and are still returned;
- the keyword list gets half the weight of the dense list in fusion.

A sweep on the dev set picked the values: for the source weight {1.0, 0.7, 0.5, 0.3}, for the fusion weight {1.0, 0.5}.

| after the change | dev recall@5 | dev MRR | held-out recall@5 | held-out MRR |
|---|---|---|---|---|
| BM25 | 0.83 | 0.70 | 1.00 | 0.87 |
| dense | 0.93 | 0.74 | 1.00 | 0.83 |
| hybrid | **1.00** | **0.74** | **1.00** | **0.93** |

Before the change, the held-out set gave hybrid 0.90 recall@5 and 0.65 MRR. Ten questions is a small sample, so read the held-out column as "did not overfit", not as a precise number. Full tables: `eval/results-*.md`.

## Usage

```bash
python -m pip install -e ".[dev]"            # add ",gpu" for sentence-transformers on CUDA
python scripts/fetch_corpus.py               # the benchmark repos; or list your own in corpus.yaml
python -m code_rag index                     # walk, chunk, redact, embed (about 12 min for 6.7k chunks on a laptop CPU)
python -m code_rag search "where are gzip and brotli responses decompressed"
python -m code_rag eval                      # reproduces the tables above
python -m pytest -q
```

Connect it to Claude Code:

```bash
claude mcp add code-rag -- python -m code_rag.mcp_server
```

To search your own code, replace the list in `corpus.yaml`:

```yaml
repos:
  - name: backend
    path: ../my-backend
  - name: infra
    path: ../infra
```

## Layout

```
code_rag/
  walker.py      which files to index and why others are skipped
  chunker.py     AST, markdown and window chunking
  redact.py      secret redaction
  textindex.py   headers, identifier splitting, stemming (EN and RU)
  store.py       SQLite schema: files, chunks, FTS5, embeddings
  embeddings.py  fastembed on CPU, sentence-transformers on CUDA
  retrieval.py   BM25, dense, weighted RRF, source prior, per-file cap
  indexer.py     incremental indexing
  evaluation.py  recall@k and MRR at file level
  mcp_server.py  MCP tools
eval/            question sets and results
scripts/         benchmark corpus fetcher, MCP smoke test
```

## Limitations

- Dense search loads all vectors into memory and does exact cosine. That is fine up to a few hundred thousand chunks; beyond that it needs an approximate index.
- The source prior helps "where is X implemented" questions and hurts "how do I use X" questions, which docs answer best. `NON_SOURCE_WEIGHT = 1.0` in `config.py` turns it off; a per-query switch would be better.
- Evaluation is at file level: finding the right file counts, finding the right function inside it is not checked.
- 40 questions over four repositories is a small benchmark.

## What I'd do next

- A per-query intent switch (implementation vs usage) in place of the fixed source prior.
- A function-level benchmark and a cross-encoder reranker over the top 20.
- An approximate nearest-neighbour index for large monorepos.
