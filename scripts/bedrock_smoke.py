"""
Bedrock smoke test — one tiny call per model the pipeline uses.
Run after `aws configure` + enabling model access in the Bedrock console.

Usage:  .venv/bin/python scripts/bedrock_smoke.py     (costs < $0.01)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings           # noqa: E402
from llm_client import call_llm       # noqa: E402
from rag.embedder import get_embedder  # noqa: E402


def main() -> int:
    ok = True
    for label, model in (("Haiku ", settings.claude_haiku),
                         ("Sonnet", settings.claude_sonnet)):
        try:
            r = call_llm(model=model, max_tokens=20, agent_name="smoke",
                         messages=[{"role": "user", "content": "Reply with exactly: OK"}])
            print(f"  PASS  {label} -> {r['content'][0].text.strip()!r}")
        except Exception as e:
            print(f"  FAIL  {label} -> {e}")
            ok = False
    try:
        vec = get_embedder().encode("bedrock titan smoke test")
        print(f"  PASS  Titan  -> {len(vec)}-dim embedding")
    except Exception as e:
        print(f"  FAIL  Titan  -> {e}")
        ok = False
    print("BEDROCK SMOKE: " + ("ALL PASS" if ok else "FAILURES — check AWS creds, "
          "region, and Bedrock model access (console -> Bedrock -> Model access)"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
