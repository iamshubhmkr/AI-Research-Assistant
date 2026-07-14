"""LangGraph PostgreSQL checkpointer helpers.

Design decision: the API path uses the ASYNC saver owned by
agents.graph.get_graph(). This sync helper exists for scripts/notebooks;
it is a context manager because PostgresSaver.from_conn_string() yields a
saver bound to a connection's lifetime.
"""
from contextlib import contextmanager
from config import settings


@contextmanager
def checkpointer():
    from langgraph.checkpoint.postgres import PostgresSaver
    with PostgresSaver.from_conn_string(settings.postgres_url) as ckpt:
        ckpt.setup()
        yield ckpt
