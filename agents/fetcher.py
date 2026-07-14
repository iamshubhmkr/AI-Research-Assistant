"""
Fetcher — ReAct loop + arXiv tools + multi-format parallel fetching.
v3.1: paper IDs are sha256-stable (survive restarts/workers); documents that
parse to empty text are dropped instead of polluting the index.
"""
import asyncio
import logging
import aiohttp
from llm_client import call_llm
from rag.document_router import extract_text
from utils import stable_id
from .state import ResearchState
from config import settings

logger = logging.getLogger(__name__)

REACT_SYSTEM = """You are a research paper fetcher using the ReAct pattern.
Before EVERY tool call write:
Thought: [why this tool, what you expect, fallback plan]
Action: <tool>
Then after the result:
Thought: [what you learned, is it enough, next step]
Rules: never repeat a search query; stop (say FINISH) at 3+ relevant papers."""

ARXIV_TOOLS = [
    {"name": "search_arxiv",
     "description": "Search arXiv. Returns titles, abstracts, IDs, PDF URLs.",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string"},
                                     "max_results": {"type": "integer", "default": 5}},
                      "required": ["query"]}},
    {"name": "fetch_pdf",
     "description": "Fetch a document URL and return its text as markdown.",
     "input_schema": {"type": "object",
                      "properties": {"url": {"type": "string"}},
                      "required": ["url"]}},
]


def run_arxiv_tool(name: str, inputs: dict) -> str:
    if name == "search_arxiv":
        return _arxiv_search(inputs["query"], inputs.get("max_results", 5))
    if name == "fetch_pdf":
        return _fetch_sync(inputs["url"])
    return f"Unknown tool: {name}"


def _arxiv_search(query: str, max_results: int = 5) -> str:
    import urllib.request
    import urllib.parse
    params = urllib.parse.urlencode({"search_query": query, "max_results": max_results})
    try:
        with urllib.request.urlopen(f"http://export.arxiv.org/api/query?{params}", timeout=10) as r:
            return r.read().decode("utf-8")[:3000]
    except Exception as e:
        return f"Search error: {e}"


def _fetch_sync(url: str) -> str:
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "research-assistant/3.1"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return extract_text(url, r.read())
    except Exception as e:
        return f"Error fetching: {e}"


async def _fetch_async(session, url):
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as r:
            content = await r.read()
        text = extract_text(url, content)
        return {"url": url, "text": text} if text else {"url": url, "text": "", "error": "empty"}
    except Exception as e:
        return {"url": url, "text": "", "error": str(e)}


async def fetch_all_parallel(urls):
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[_fetch_async(session, u) for u in urls],
                                       return_exceptions=True)
    return [r for r in results if isinstance(r, dict) and r.get("text")]


def fetcher_node(state: ResearchState) -> dict:
    messages = [{"role": "user", "content":
        f"Research query: {state['query']}\nFind 3+ relevant papers. Begin with Thought:"}]
    papers, urls_seen = [], []

    if state.get("auto_search", True):
        for _ in range(settings.max_react_iterations):
            result = call_llm(model=settings.claude_sonnet, messages=messages,
                              system=REACT_SYSTEM, max_tokens=1500,
                              tools=ARXIV_TOOLS, agent_name="fetcher")
            if result["stop_reason"] != "tool_use":
                break
            tool_results = []
            for block in result["content"]:
                if block.type == "tool_use":
                    out = run_arxiv_tool(block.name, block.input)
                    tool_results.append({"type": "tool_result",
                                         "tool_use_id": block.id, "content": out})
                    # keep only documents that actually produced text
                    if (block.name == "fetch_pdf" and out
                            and not out.startswith(("Error", "Search error"))):
                        urls_seen.append(block.input["url"])
                        papers.append({"url": block.input["url"], "text": out})
            messages.append({"role": "assistant", "content": result["content"]})
            messages.append({"role": "user", "content": tool_results})

    extra_urls = [u for u in state.get("paper_urls", []) if u not in urls_seen]
    if extra_urls:
        papers.extend(asyncio.run(fetch_all_parallel(extra_urls)))

    logger.info(f"[fetcher] collected {len(papers)} documents")
    return {"raw_papers": papers,
            "paper_ids": [stable_id(p["url"]) for p in papers]}
