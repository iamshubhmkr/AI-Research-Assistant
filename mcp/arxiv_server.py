"""
arXiv MCP server — reference implementation.
Run standalone in production; agents call tools via Claude tool_use.
Swap for PubMed/Semantic Scholar by writing a new server with the same interface.
"""
import json
import urllib.request
import urllib.parse
from rag.document_router import extract_text

TOOLS = [
    {"name": "search_arxiv",
     "description": "Search arXiv for academic papers.",
     "inputSchema": {"type": "object",
                     "properties": {"query": {"type": "string"},
                                    "max_results": {"type": "integer", "default": 5}},
                     "required": ["query"]}},
    {"name": "fetch_pdf",
     "description": "Fetch and parse a document URL into markdown text.",
     "inputSchema": {"type": "object",
                     "properties": {"url": {"type": "string"}},
                     "required": ["url"]}},
]


def handle_tool_call(name, arguments):
    if name == "search_arxiv":
        return search_arxiv(arguments["query"], arguments.get("max_results", 5))
    if name == "fetch_pdf":
        return fetch_pdf(arguments["url"])
    return json.dumps({"error": f"Unknown tool: {name}"})


def search_arxiv(query, max_results=5):
    params = urllib.parse.urlencode({"search_query": query, "max_results": max_results})
    try:
        with urllib.request.urlopen(f"http://export.arxiv.org/api/query?{params}", timeout=10) as r:
            return r.read().decode("utf-8")[:3000]
    except Exception as e:
        return json.dumps({"error": str(e)})


def fetch_pdf(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return extract_text(url, r.read())
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    print("arXiv MCP Server — tools:", [t["name"] for t in TOOLS])
