# Retrieval results: BAAI/bge-small-en-v1.5

Run on 2026-10-01; 6718 chunks in the index. hybrid+rerank reorders the top 20 fused candidates with Xenova/ms-marco-MiniLM-L-6-v2. Latency is the mean wall time of one search, after the models are loaded.

## questions.yaml (30 questions)

| mode | recall@5 | recall@10 | MRR | ms/query | missed in top 5 |
|---|---|---|---|---|---|
| bm25 | 0.83 | 0.87 | 0.70 | 5 | req-redirect-limit, click-testing, click-parser, rich-cell-width, rich-markup |
| dense | 0.93 | 1.00 | 0.74 | 38 | httpx-multipart, click-help-format |
| hybrid | 1.00 | 1.00 | 0.74 | 40 | — |
| hybrid+rerank | 0.93 | 1.00 | 0.75 | 1012 | req-redirect-limit, httpx-multipart |

## holdout.yaml (10 questions)

| mode | recall@5 | recall@10 | MRR | ms/query | missed in top 5 |
|---|---|---|---|---|---|
| bm25 | 1.00 | 1.00 | 0.87 | 5 | — |
| dense | 1.00 | 1.00 | 0.83 | 36 | — |
| hybrid | 1.00 | 1.00 | 0.93 | 41 | — |
| hybrid+rerank | 1.00 | 1.00 | 0.95 | 1062 | — |

## hard.yaml (18 questions)

| mode | recall@5 | recall@10 | MRR | function recall@5 | function MRR | ms/query | missed in top 5 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.83 | 0.89 | 0.61 | 0.56 | 0.36 | 8 | req-body-length, httpx-line-split, click-strip-colors |
| dense | 1.00 | 1.00 | 0.77 | 0.94 | 0.69 | 38 | — |
| hybrid | 1.00 | 1.00 | 0.71 | 0.89 | 0.65 | 48 | — |
| hybrid+rerank | 0.83 | 1.00 | 0.77 | 0.83 | 0.70 | 999 | httpx-line-split, click-strip-colors, rich-track-loop |

## hard_holdout.yaml (12 questions)

| mode | recall@5 | recall@10 | MRR | function recall@5 | function MRR | ms/query | missed in top 5 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.75 | 0.92 | 0.61 | 0.58 | 0.41 | 7 | req-netrc, httpx-error-mapping, click-choice-case |
| dense | 1.00 | 1.00 | 0.73 | 0.92 | 0.64 | 37 | — |
| hybrid | 1.00 | 1.00 | 0.70 | 0.83 | 0.62 | 43 | — |
| hybrid+rerank | 0.92 | 1.00 | 0.78 | 0.83 | 0.67 | 1076 | httpx-error-mapping |
