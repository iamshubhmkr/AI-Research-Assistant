"""DynamoDB — session history + long-term memory (TTL 7 days).

Design decision: Dynamo is OPTIONAL (settings.enable_dynamo) and lazy —
local/dev runs work with Docker-only infra, and a missing table degrades to
a warning instead of failing the request (sessions are convenience history;
the LangGraph checkpoint in Postgres is the durable state).
"""
import time
import logging
from config import settings

logger = logging.getLogger(__name__)


class SessionStore:
    def __init__(self):
        self._table = None

    def _get_table(self):
        if self._table is None:
            import boto3
            self._table = boto3.resource(
                "dynamodb", region_name=settings.aws_region
            ).Table(settings.dynamo_sessions_table)
        return self._table

    def create(self, session_id, user_id="anonymous"):
        if not settings.enable_dynamo:
            return None
        item = {"session_id": session_id, "user_id": user_id,
                "created_at": int(time.time()), "ttl": int(time.time()) + 604800,
                "queries": [], "long_term_memory": {}}
        try:
            self._get_table().put_item(Item=item)
            return item
        except Exception as e:
            logger.warning(f"dynamo create failed ({e}); continuing without session history")
            return None

    def add_query(self, session_id, query, answer, scores):
        if not settings.enable_dynamo:
            return
        try:
            self._get_table().update_item(
                Key={"session_id": session_id},
                UpdateExpression="SET queries = list_append(queries, :q)",
                ExpressionAttributeValues={":q": [{"query": query, "answer": answer[:400],
                                                   "ragas": scores, "ts": int(time.time())}]})
        except Exception as e:
            logger.warning(f"dynamo add_query failed ({e})")

    def get_memory(self, session_id):
        if not settings.enable_dynamo:
            return {}
        try:
            return self._get_table().get_item(
                Key={"session_id": session_id}).get("Item", {}).get("long_term_memory", {})
        except Exception as e:
            logger.warning(f"dynamo get_memory failed ({e})")
            return {}
