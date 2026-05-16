"""
arXiv MCP Server — Reference implementation.

MCP (Model Context Protocol) is Anthropic's open standard for LLM-tool connectivity.
This server exposes arXiv search and PDF fetch as typed tools.

In production:
  - Run as a standalone process: python -m mcp.arxiv_server
  - Register in Antigravity MCP Store or Claude Desktop config
  - Agent code calls tools through Claude's tool_use mechanism
  - Swap for PubMed/Semantic Scholar MCP server without agent changes

Interview talking point:
  "MCP standardizes tool connectivity. I write one arXiv MCP server
   and any MCP client — Claude, Antigravity IDE, Claude Desktop —
   can use it. If I want PubMed, I write a new server with the same
   interface. Agent code doesn't change."
"""
import json
import urllib.request
import urllib.parse
import pymupdf4llm
from config import settings


# MCP Tool Definitions (JSON Schema format)
TOOLS = [
    {
        "name": "search_arxiv",
        "description": "Search arXiv for academic papers by query string.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5, "description": "Number of results (1-10)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_pdf",
        "description": "Fetch and parse a PDF from URL into markdown text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full PDF URL"}
            },
            "required": ["url"]
        }
    }
]


def handle_tool_call(name: str, arguments: dict) -> str:
    if name == "search_arxiv":
        return search_arxiv(arguments["query"], arguments.get("max_results", 5))
    elif name == "fetch_pdf":
        return fetch_pdf(arguments["url"])
    return json.dumps({"error": f"Unknown tool: {name}"})


def search_arxiv(query: str, max_results: int = 5) -> str:
    base = "http://export.arxiv.org/api/query"
    params = urllib.parse.urlencode({"search_query": query, "max_results": max_results})
    try:
        with urllib.request.urlopen(f"{base}?{params}", timeout=10) as resp:
            return resp.read().decode("utf-8")[:3000]
    except Exception as e:
        return json.dumps({"error": str(e)})


def fetch_pdf(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            pdf_bytes = resp.read()
        return pymupdf4llm.to_markdown(pdf_bytes)[:settings.max_paper_chars]
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    print("arXiv MCP Server")
    print(f"Available tools: {[t['name'] for t in TOOLS]}")
    print("In production, this runs as a standalone MCP server process.")
