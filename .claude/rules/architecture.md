# Architecture Rules

1. **Shared-notebook pattern**: agents NEVER call each other. They communicate
   only through ResearchState. If you need agent A's output in agent B, write
   it to state in A and read it in B.

2. **Supervisor heartbeat**: every agent node edges back to the supervisor.
   Never add a direct agent→agent edge — it breaks LLM-driven routing and
   makes HITL interrupts unpredictable.

3. **HITL invariants**: interrupt_before=["synthesizer"] and
   interrupt_after=["critic"] must not be removed. The API endpoints
   approve_chunks and resolve_critique depend on these exact pause points.

4. **Loop safety**: any new loop needs a hard cap in config.py (like
   max_revision_count). An LLM judge can always find "one more issue" —
   uncapped loops are a production outage waiting to happen.

5. **Layering**: api/ → agents/ → rag|cache|persistence → llm_client → config.
   Lower layers never import higher layers (no rag importing agents).
