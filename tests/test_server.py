"""MCP protocol surface: the expected tools exist with correct schemas."""

import asyncio

from docqa import server


def _tools():
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


def test_tools_registered():
    assert set(_tools()) == {
        "query_documents",
        "list_documents",
        "search_chunks",
        "document_stats",
    }


def test_query_documents_schema():
    tool = _tools()["query_documents"]
    props = tool.inputSchema["properties"]
    assert "question" in props
    assert tool.inputSchema.get("required") == ["question"]
    assert "top_k" in props and "document" in props


def test_tools_have_descriptions():
    for t in _tools().values():
        assert t.description and len(t.description) > 20
