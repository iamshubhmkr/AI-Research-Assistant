"""PostgreSQL LangGraph checkpointer setup."""
from langgraph.checkpoint.postgres import PostgresSaver
from config import settings

def get_checkpointer():
    ckpt = PostgresSaver.from_conn_string(settings.postgres_url)
    ckpt.setup()
    return ckpt
