"""
Fetcher Agent — ReAct pattern + arXiv tools + parallel async fetching.

ReAct = Reason + Act.
Before every tool call: Thought → Action → Observation → Thought → ...

Parallel fetching with asyncio.gather:
  Sequential: 3 papers × 3s = 9s
  Parallel:   3 papers simultaneously = ~3s  (3x speedup)

Bottleneck analysis:
  - arXiv API rate limit: 3 req/s — mitigated by batching
  - PDF size: 5-20MB each — truncate to max_paper_chars
  - Network latency: use aiohttp connection pooling
"""
import asyncio
import logging
import aiohttp
import pymupdf4llm
from llm_client import call_llm
from .state import ResearchState
from config import settings

logger = logging.getLogger(__name__)

REACT_SYSTEM = """You are a research paper fetcher using the ReAct pattern.

Before EVERY tool call, write your reasoning using this exact format:
Thought: [Why am I calling this tool? What do I expect? What's my fallback?]
Action: <tool name>
Observation: [filled in automatically]
Thought: [What did I learn? Is this enough? What next?]
Action: <next tool or FINISH>

Rules:
- Always reason BEFORE acting — never fire a tool without a Thought
- Stop (say FINISH) when you have 3+ relevant papers with content
- Never call the same search query twice
- Maximum iterations enforced externally — be efficient"""

ARXIV_TOOLS = [
    {
        "name": "search_arxiv",
        "description": "Search arXiv for papers. Returns titles, abstracts, IDs, and PDF URLs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query — be specific"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_pdf",
        "description": "Fetch a PDF from URL and return its full text as markdown.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full PDF URL"}
            },
            "required": ["url"]
        }
    }
]


def run_arxiv_tool(name: str, inputs: dict) -> str:
    """Execute tool call — replace with MCP client in production."""
    if name == "search_arxiv":
        return _arxiv_search(inputs["query"], inputs.get("max_results", 5))
    if name == "fetch_pdf":
        return _fetch_pdf_sync(inputs["url"])
    return f"Unknown tool: {name}"


def _arxiv_search(query: str, max_results: int = 5) -> str:
    import urllib.request, urllib.parse
    base = "http://export.arxiv.org/api/query"
    params = urllib.parse.urlencode({"search_query": query, "max_results": max_results})
    try:
        with urllib.request.urlopen(f"{base}?{params}", timeout=10) as resp:
            return resp.read().decode("utf-8")[:3000]
    except Exception as e:
        return f"Search error: {e}"


def _fetch_pdf_sync(url: str) -> str:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            pdf_bytes = resp.read()
        return pymupdf4llm.to_markdown(pdf_bytes)[:settings.max_paper_chars]
    except Exception as e:
        return f"Error fetching PDF: {e}"


async def _fetch_url_async(session: aiohttp.ClientSession, url: str) -> dict:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            content = await resp.read()
        text = pymupdf4llm.to_markdown(content)[:settings.max_paper_chars]
        return {"url": url, "text": text, "char_count": len(text)}
    except Exception as e:
        logger.warning(f"Async fetch failed for {url}: {e}")
        return {"url": url, "text": "", "error": str(e)}


async def fetch_all_parallel(urls: list[str]) -> list[dict]:
    """Fetch multiple PDFs simultaneously. Total wait = slowest individual."""
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_url_async(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict) and r.get("text")]


def fetcher_node(state: ResearchState) -> dict:
    """ReAct loop: reason → call arXiv tools → collect papers."""
    messages = [{"role": "user", "content":
        f"Research query: {state['query']}\nFind 3+ relevant recent papers. Begin with Thought:"}]

    collected_papers = []
    collected_urls = []

    for iteration in range(settings.max_react_iterations):
        result = call_llm(
            model=settings.claude_sonnet,
            messages=messages,
            system=REACT_SYSTEM,
            max_tokens=1500,
            tools=ARXIV_TOOLS,
            agent_name="fetcher",
        )

        if result["stop_reason"] == "end_turn":
            break

        if result["stop_reason"] == "tool_use":
            tool_results = []
            for block in result["content"]:
                if block.type == "tool_use":
                    tool_output = run_arxiv_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_output
                    })
                    if block.name == "fetch_pdf" and "Error" not in tool_output:
                        url = block.input["url"]
                        collected_urls.append(url)
                        collected_papers.append({"url": url, "text": tool_output})

            messages.append({"role": "assistant", "content": result["content"]})
            messages.append({"role": "user", "content": tool_results})

    # Fetch user-provided URLs in parallel
    provided_urls = [u for u in state.get("paper_urls", []) if u not in collected_urls]
    if provided_urls:
        extra = asyncio.run(fetch_all_parallel(provided_urls))
        collected_papers.extend(extra)

    logger.info(f"[fetcher] collected {len(collected_papers)} papers")
    return {
        "raw_papers": collected_papers,
        "paper_ids": [str(hash(p["url"])) for p in collected_papers],
    }
