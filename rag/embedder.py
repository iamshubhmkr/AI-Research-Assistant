"""
Bedrock Titan v2 embedder.

Design decision: embeddings moved from a local SPECTER2/torch model to
amazon.titan-embed-text-v2:0 so (a) ALL inference spend lands on the AWS
bill alongside Claude, and (b) the deployable image needs no torch — smaller,
faster cold start. encode() returns a plain list[float] (Chroma-ready).
"""
import json
import logging
from config import settings

logger = logging.getLogger(__name__)

_embedder = None
# Titan v2 caps input around 8K tokens; truncate defensively by chars.
_MAX_CHARS = 30000


class TitanEmbedder:
    def __init__(self):
        import boto3
        kwargs = {"region_name": settings.aws_region}
        if settings.aws_profile:
            kwargs["profile_name"] = settings.aws_profile
        self.client = boto3.Session(**kwargs).client("bedrock-runtime")

    def encode(self, text: str) -> list[float]:
        body = json.dumps({"inputText": text[:_MAX_CHARS],
                           "dimensions": settings.embed_dimensions,
                           "normalize": True})
        resp = self.client.invoke_model(modelId=settings.bedrock_embed_model,
                                        contentType="application/json",
                                        accept="application/json", body=body)
        return json.loads(resp["body"].read())["embedding"]


def get_embedder() -> TitanEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = TitanEmbedder()
    return _embedder


def embed_texts(texts: list[str], cache=None) -> list[list[float]]:
    """Embed a batch, going through the L2 Redis cache when available.

    Cache errors degrade to direct embedding — a dead Redis must never
    take the pipeline down (project error-handling philosophy).
    """
    embedder = get_embedder()
    out = []
    for t in texts:
        vec = None
        if cache is not None:
            try:
                vec = cache.get(t)
            except Exception as e:
                logger.warning(f"embedding cache read failed ({e}); bypassing cache")
                cache = None
        if vec is None:
            vec = embedder.encode(t)
            if cache is not None:
                try:
                    cache.set(t, vec)
                except Exception as e:
                    logger.warning(f"embedding cache write failed ({e})")
                    cache = None
        out.append(vec)
    return out
