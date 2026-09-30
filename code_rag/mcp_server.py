"""MCP server (stdio) that gives coding agents three tools over the index.

Written against the `mcp` 2.x SDK, where `FastMCP` became `MCPServer`.
"""
from __future__ import annotations

from typing import Optional

from mcp.server.mcpserver import MCPServer

from . import config, store
from . import retrieval

app = MCPServer("code-rag")


@app.tool()
def search_code(query: str, k: int = 8, repo: "Optional[str]" = None) -> list[dict]:
    """Hybrid (BM25 + dense) search over the indexed repositories.

    Returns a list of {chunk_id, path, repo, symbol, kind,
    start_line, end_line, snippet, score}, best match first.
    """
    conn = store.connect()
    try:
        results = retrieval.search(conn, query, k=k, mode="hybrid", repo=repo)
        return [
            {
                "chunk_id": r.chunk_id,
                "path": r.path,
                "repo": r.repo,
                "symbol": r.symbol,
                "kind": r.kind,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "snippet": r.snippet,
                "score": r.score,
            }
            for r in results
        ]
    finally:
        conn.close()


@app.tool()
def read_chunk(chunk_id: int) -> dict:
    """Return the full text and metadata of one chunk by id."""
    conn = store.connect()
    try:
        chunk = store.get_chunk(conn, chunk_id)
        return chunk or {}
    finally:
        conn.close()


@app.tool()
def list_repos() -> list[dict]:
    """List indexed repos and how many chunks each has."""
    conn = store.connect()
    try:
        rows = store.list_repos(conn)
        return [{"repo": s, "chunks": c} for s, c in rows]
    finally:
        conn.close()


def main() -> None:
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
