# Retrieval results: BAAI/bge-small-en-v1.5

Run on 2026-10-01; 6718 chunks in the index. hybrid+rerank reorders the top 20 fused candidates with BAAI/bge-reranker-v2-m3@st-cuda. Latency is the mean wall time of one search, after the models are loaded.

## questions.yaml (30 questions)

| mode | recall@5 | recall@10 | MRR | ms/query | missed in top 5 |
|---|---|---|---|---|---|
| bm25 | 0.83 | 0.87 | 0.70 | 5 | req-redirect-limit, click-testing, click-parser, rich-cell-width, rich-markup |
| dense | 0.93 | 1.00 | 0.74 | 37 | httpx-multipart, click-help-format |
| hybrid | 1.00 | 1.00 | 0.74 | 41 | — |
| hybrid+rerank | 0.90 | 1.00 | 0.80 | 307 | req-redirect-limit, httpx-multipart, rich-markup |

## holdout.yaml (10 questions)

| mode | recall@5 | recall@10 | MRR | ms/query | missed in top 5 |
|---|---|---|---|---|---|
| bm25 | 1.00 | 1.00 | 0.87 | 5 | — |
| dense | 1.00 | 1.00 | 0.83 | 34 | — |
| hybrid | 1.00 | 1.00 | 0.93 | 40 | — |
| hybrid+rerank | 1.00 | 1.00 | 0.93 | 288 | — |

## hard.yaml (18 questions)

| mode | recall@5 | recall@10 | MRR | function recall@5 | function MRR | ms/query | missed in top 5 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.83 | 0.89 | 0.61 | 0.56 | 0.36 | 7 | req-body-length, httpx-line-split, click-strip-colors |
| dense | 1.00 | 1.00 | 0.77 | 0.94 | 0.69 | 36 | — |
| hybrid | 1.00 | 1.00 | 0.71 | 0.89 | 0.65 | 44 | — |
| hybrid+rerank | 0.89 | 0.94 | 0.75 | 0.83 | 0.69 | 317 | httpx-line-split, click-strip-colors |

## hard_holdout.yaml (12 questions)

| mode | recall@5 | recall@10 | MRR | function recall@5 | function MRR | ms/query | missed in top 5 |
|---|---|---|---|---|---|---|---|
| bm25 | 0.75 | 0.92 | 0.61 | 0.58 | 0.41 | 7 | req-netrc, httpx-error-mapping, click-choice-case |
| dense | 1.00 | 1.00 | 0.73 | 0.92 | 0.64 | 38 | — |
| hybrid | 1.00 | 1.00 | 0.70 | 0.83 | 0.62 | 44 | — |
| hybrid+rerank | 1.00 | 1.00 | 0.81 | 0.92 | 0.67 | 325 | — |
