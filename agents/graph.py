"""
LangGraph graph assembly.

Two HITL checkpoints:
  interrupt_before=["synthesizer"]  — human reviews retrieved chunks
  interrupt_after=["critic"]        — human reviews critique + can override

PostgresSaver vs MemorySaver:
  MemorySaver: in-process memory. Restart = lost. Multi-instance = no sharing.
  PostgresSaver: every transition in PostgreSQL. Any instance resumes any thread.

Scaling consideration:
  - max_graph_iterations prevents infinite loops (default 20)
  - PostgreSQL connection pooling via connection string
  - Graph is compiled once (singleton) — reuse across requests
"""
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from .state import ResearchState
from .supervisor import supervisor_node
from .fetcher import fetcher_node
from .extractor import extractor_node
from .synthesizer import synthesizer_node
from .critic import critic_node
from rag.retriever import retriever_node
from config import settings

logger = logging.getLogger(__name__)


def build_graph():
    wf = StateGraph(ResearchState)

    wf.add_node("supervisor",  supervisor_node)
    wf.add_node("fetcher",     fetcher_node)
    wf.add_node("extractor",   extractor_node)
    wf.add_node("retriever",   retriever_node)
    wf.add_node("synthesizer", synthesizer_node)
    wf.add_node("critic",      critic_node)

    wf.set_entry_point("supervisor")

    wf.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {
            "fetcher": "fetcher", "extractor": "extractor",
            "retriever": "retriever", "synthesizer": "synthesizer",
            "critic": "critic", "END": END,
        }
    )

    for node in ["fetcher", "extractor", "retriever", "synthesizer", "critic"]:
        wf.add_edge(node, "supervisor")

    checkpointer = PostgresSaver.from_conn_string(settings.postgres_url)
    checkpointer.setup()

    return wf.compile(
        checkpointer=checkpointer,
        interrupt_before=["synthesizer"],
        interrupt_after=["critic"],
    )


graph = build_graph()
