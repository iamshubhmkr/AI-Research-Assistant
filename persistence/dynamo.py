"""DynamoDB — Sessions + conversation history + long-term memory."""
import boto3, time
from config import settings

table = boto3.resource("dynamodb", region_name=settings.aws_region).Table(settings.dynamo_sessions_table)

class SessionStore:
    def create(self, session_id: str, user_id: str = "anonymous") -> dict:
        item = {"session_id": session_id, "user_id": user_id, "created_at": int(time.time()),
                "ttl": int(time.time()) + 86400 * 7, "queries": [], "long_term_memory": {}}
        table.put_item(Item=item)
        return item

    def add_query(self, session_id: str, query: str, answer: str, scores: dict):
        table.update_item(Key={"session_id": session_id},
            UpdateExpression="SET queries = list_append(queries, :q)",
            ExpressionAttributeValues={":q": [{"query": query, "answer": answer[:400],
                "ragas": scores, "ts": int(time.time())}]})

    def get_memory(self, session_id: str) -> dict:
        result = table.get_item(Key={"session_id": session_id})
        return result.get("Item", {}).get("long_term_memory", {})

    def update_memory(self, session_id: str, memory: dict):
        table.update_item(Key={"session_id": session_id},
            UpdateExpression="SET long_term_memory = :m",
            ExpressionAttributeValues={":m": memory})
