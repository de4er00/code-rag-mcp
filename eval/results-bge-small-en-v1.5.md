# Retrieval results: BAAI/bge-small-en-v1.5

Run on 2026-09-30; 6718 chunks in the index.

## questions.yaml (30 questions)

| mode | recall@5 | recall@10 | MRR | missed in top 5 |
|---|---|---|---|---|
| bm25 | 0.83 | 0.87 | 0.70 | req-redirect-limit, click-testing, click-parser, rich-cell-width, rich-markup |
| dense | 0.93 | 1.00 | 0.74 | httpx-multipart, click-help-format |
| hybrid | 1.00 | 1.00 | 0.74 | — |

## holdout.yaml (10 questions)

| mode | recall@5 | recall@10 | MRR | missed in top 5 |
|---|---|---|---|---|
| bm25 | 1.00 | 1.00 | 0.87 | — |
| dense | 1.00 | 1.00 | 0.83 | — |
| hybrid | 1.00 | 1.00 | 0.93 | — |
