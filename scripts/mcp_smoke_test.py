"""Start the MCP server over stdio with the official SDK client and call each tool once.

Run after `code-rag index`:  python scripts/mcp_smoke_test.py
"""
from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "code_rag.mcp_server"],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools:", [t.name for t in tools.tools])

            repos = await session.call_tool("list_repos", {})
            print("\nlist_repos ->")
            print(repos.content[0].text if repos.content else repos)

            results = await session.call_tool(
                "search_code", {"query": "where are redirects followed and the next request built", "k": 3}
            )
            print("\nsearch_code ->")
            print(results.content[0].text if results.content else results)


if __name__ == "__main__":
    asyncio.run(main())
