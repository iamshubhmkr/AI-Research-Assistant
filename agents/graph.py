"""
LangGraph assembly — every node feeds back to the supervisor (heartbeat).
HITL pauses: interrupt_before=["synthesizer"], interrupt_after=["critic"].

Design decisions (v3.1):
  - The graph is built LAZILY via get_graph(): v3 compiled it at import time,
    which (a) required a live Postgres just to import the API module and
    (b) called .setup() on a context manager — both broke startup.
  - AsyncPostgresSaver over a psycopg connection pool, because the API layer
    drives the graph with ainvoke/aget_state/aupdate_state.
"""
import logging
from langgraph.graph import StateGraph, END
from .state import ResearchState
from .supervisor import supervisor_node
from .fetcher import fetcher_node
from .extractor import extractor_node
from .synthesizer import synthesizer_node
from .critic import critic_node
from rag.retriever import retriever_node
from config import settings

logger = logging.getLogger(__name__)

_graph = None


def build_workflow() -> StateGraph:
    """Topology only — no checkpointer, importable without infrastructure."""
    wf = StateGraph(ResearchState)
    wf.add_node("supervisor", supervisor_node)
    wf.add_node("fetcher", fetche_node)
    wf.add_node("extractor", extractor_node)
    wf.add_node("retriever", retriever_node)
    wf.add_node("synthesizer", synthesizer_node)
    wf.add_node("critic", critic_node)
    wf.set_entry_point("supervisor")
    wf.add_conditional_edges("supervisor", lambda s: s["next"],
        {"fetcher": "fetcher", "extractor": "extractor", "retriever": "retriever",
         "synthesizer": "synthesizer", "critic": "critic", "END": END})
    for node in ["fetcher", "extractor", "retriever", "synthesizer", "critic"]:
        wf.add_edge(node, "supervisor")
    return wf


async def get_graph():
    """Compile (once) with an async Postgres checkpointer."""
    global _graph
    if _graph is None:
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        pool = AsyncConnectionPool(
            settings.postgres_url, max_size=10, open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
        await pool.open()
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        _graph = build_workflow().compile(
            checkpointer=checkpointer,
            interrupt_before=["synthesizer"],
            interrupt_after=["critic"])
        logger.info("graph compiled with AsyncPostgresSaver")
    return _graph
