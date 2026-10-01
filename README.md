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
- **Optional reranking.** In `hybrid+rerank` mode a cross-encoder reads the question together with each of the top 20 candidates, and its order is fused with the retrieval order. The default reranker (`ms-marco-MiniLM-L-6-v2`) runs on CPU through fastembed; `bge-reranker-v2-m3` runs on a CUDA GPU. The model is loaded on the first reranked query only.
- **Incremental indexing.** Files are tracked by sha256. Unchanged files are skipped, deleted files are removed from all three tables, and embedding is committed in small batches, so an interrupted run resumes where it stopped. The index stores logical paths (`repo/relative/path`), never local ones, so it can be copied between machines.
- **MCP server** (stdio) with three tools: `search_code`, `read_chunk`, `list_repos`.

## Benchmark

The benchmark indexes four well-known projects at pinned tags (requests 2.32.3, httpx 0.28.1, click 8.1.7, rich 13.9.4): 602 files, 6,718 chunks. There are two question sets, both phrased the way a developer asks an agent, without copying identifiers. Each set has a development part, used to choose settings, and a held-out part, committed before any tuning on that set.

### First set: 40 questions, file level

Each question lists the file where the answer lives. 30 questions are for development; 10 were held out and looked at once, after the settings below were fixed.

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

Before the change, the held-out set gave hybrid 0.90 recall@5 and 0.65 MRR. Ten questions is a small sample, so read the held-out column as "did not overfit", not as a precise number.

After this, every method scored 1.00 recall@5 on the held-out questions, so the set could no longer tell the methods apart.

### Second set: 30 harder questions, function level

The second set (`eval/hard.yaml`, 18 for development; `eval/hard_holdout.yaml`, 12 held out) has three kinds of question:
- 18 ask where something is implemented inside a large file. Each lists the line spans of the right functions, and a result counts at function level only if its lines overlap one of them.
- 6 need two places at once (for example, where an exception is translated and where the exception classes are defined). Every listed file must appear; the rank is where the last one first appears.
- 6 are "how do I" questions, where the docs page and the implementation both count.

Results with the settings above, recall@5 / recall@10 / MRR at file level, then recall@5 / MRR at function level:

| | dev, 18 questions | held-out, 12 questions |
|---|---|---|
| BM25 | 0.83 / 0.89 / 0.61, function 0.56 / 0.36 | 0.75 / 0.92 / 0.61, function 0.58 / 0.41 |
| dense | 1.00 / 1.00 / 0.77, function 0.94 / 0.69 | 1.00 / 1.00 / 0.73, function 0.92 / 0.64 |
| hybrid | 1.00 / 1.00 / 0.71, function 0.89 / 0.65 | 1.00 / 1.00 / 0.70, function 0.83 / 0.62 |

What this shows:
- At file level the new questions are still easy for dense and hybrid search. The function-level columns and MRR are where the methods differ: BM25 finds the right function in the top 5 for about half of the questions.
- On these questions dense search alone ranks the right function higher than hybrid. A new sweep over the source and fusion weights on both development sets kept the current values (0.5 and 0.5); no combination closed that gap.
- The source prior has a cost on "how do I" questions, as expected. On the development set, function-level recall@5 for those questions is 0.75 without the prior and 0.50 with it.

### Reranking

`hybrid+rerank` reorders the top 20 fused candidates with a cross-encoder. Letting the cross-encoder's order replace the retrieval order was worse than plain hybrid with every model tried on the development questions (both sets): MiniLM-L-6, MiniLM-L-12, jina-reranker-v1-turbo-en and bge-reranker-base on CPU, bge-reranker-v2-m3 on GPU. They put docs, tests, and look-alike code from the other HTTP library at the top. Fusing the two orders with RRF did better, and is what the mode does.

| hybrid+rerank | first set, dev (30) | first set, held-out (10) | second set, dev (18) | second set, held-out (12) |
|---|---|---|---|---|
| plain hybrid, for reference | 1.00 / 1.00 / 0.74 | 1.00 / 1.00 / 0.93 | 1.00 / 1.00 / 0.71, function 0.89 / 0.65 | 1.00 / 1.00 / 0.70, function 0.83 / 0.62 |
| MiniLM-L-6 on CPU (default), ~1 s per query | 0.93 / 1.00 / 0.75 | 1.00 / 1.00 / 0.95 | 0.83 / 1.00 / 0.77, function 0.83 / 0.70 | 0.92 / 1.00 / 0.78, function 0.83 / 0.67 |
| bge-reranker-v2-m3 on GPU, ~0.3 s per query | 0.90 / 1.00 / 0.80 | 1.00 / 1.00 / 0.93 | 0.89 / 0.94 / 0.75, function 0.83 / 0.69 | 1.00 / 1.00 / 0.81, function 0.92 / 0.67 |

Reranking raises MRR or leaves it equal on every set, so the right answer moves up on average, but it lowers file-level recall@5 on several sets and takes 7 to 25 times longer than hybrid search (about 40 ms). That is why it is a mode and the default stays hybrid. On a private, partly Russian codebase the same GPU reranker used on its own, replacing the retrieval order, did help: held-out recall@5 went from 0.60 to 0.70. On these four English libraries it did not.

A note on process: the first held-out run of the reranked mode used a version that broke RRF ties in favour of the cross-encoder, while the development comparison had broken them in favour of the retrieval order. The code was changed to match the development comparison and the held-out sets were run again. In the first run the first set's held-out MRR was 0.90 instead of 0.95; the second set's held-out numbers were the same.

Full tables: `eval/results-*.md`.

## Usage

```bash
python -m pip install -e ".[dev]"            # add ",gpu" for sentence-transformers on CUDA
python scripts/fetch_corpus.py               # the benchmark repos; or list your own in corpus.yaml
python -m code_rag index                     # walk, chunk, redact, embed (about 12 min for 6.7k chunks on a laptop CPU)
python -m code_rag search "where are gzip and brotli responses decompressed"
python -m code_rag search "where are gzip and brotli responses decompressed" --mode hybrid+rerank
python -m code_rag eval                      # reproduces the tables above (downloads the CPU reranker, 90 MB)
python -m code_rag eval --no-rerank          # the three retrieval modes only
python -m code_rag eval --reranker BAAI/bge-reranker-v2-m3@st-cuda   # GPU reranker, needs ".[gpu]"
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
  retrieval.py   BM25, dense, weighted RRF, source prior, per-file cap, rerank fusion
  rerank.py      optional cross-encoder, CPU (fastembed) or CUDA (sentence-transformers)
  indexer.py     incremental indexing
  evaluation.py  recall@k and MRR at file and function level
  mcp_server.py  MCP tools
eval/            question sets and results
scripts/         benchmark corpus fetcher, MCP smoke test
```

## Limitations

- Dense search loads all vectors into memory and does exact cosine. That is fine up to a few hundred thousand chunks; beyond that it needs an approximate index.
- The source prior helps "where is X implemented" questions and hurts "how do I use X" questions, which docs answer best. `NON_SOURCE_WEIGHT = 1.0` in `config.py` turns it off; a per-query switch would be better.
- Function-level scoring uses hand-labelled line spans at the pinned tags. A chunk that overlaps a span counts, even when it is a whole class.
- Some questions do not name a library, and requests and httpx both implement redirects, digest auth and multipart bodies. The labels list the files each question was written for, so a correct answer from the other library counts as a miss.
- The rerankers tried here were trained on web search data. They mostly reorder the top 20 and rarely bring in a file that hybrid search missed.
- 70 questions over four repositories is still a small benchmark: on the held-out parts one question moves recall@5 by 0.08 to 0.10.

## What I'd do next

- A per-query intent switch (implementation vs usage) in place of the fixed source prior.
- A reranker trained on code search pairs; none of the general ones tried here beats hybrid search when used on its own.
- An approximate nearest-neighbour index for large monorepos.
