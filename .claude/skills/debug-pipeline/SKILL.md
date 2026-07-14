---
name: debug-pipeline
description: Diagnose pipeline failures — wrong routing, empty retrievals, stuck HITL sessions, bad answers, cost spikes. Use when the user says "it's broken", "wrong answer", "stuck", "expensive", or "debug".
---

# Debugging the Pipeline

## 1. Identify the failing stage from symptoms
| Symptom | Stage | First check |
|---|---|---|
| No papers found | Fetcher | ReAct trace in logs; arXiv query terms |
| Empty/irrelevant chunks | Retriever | ChromaDB count; retrieval_mode setting |
| Pipeline never pauses | Graph | interrupt_before/after still in compile()? |
| Session won't resume | Persistence | PostgreSQL up? thread_id correct? |
| Hallucinated answer | Synthesizer/Critic | sc_verdicts; faithfulness_score |
| Infinite loop | Supervisor | revision_count; max_graph_iterations |
| Cost spike | Any | TokenTracker().print_report() per agent |

## 2. Inspect a live session's state
```python
from agents.graph import graph
config = {"configurable": {"thread_id": "<session_id>"}}
snap = graph.get_state(config)
print(snap.values["next"], snap.values["revision_count"])
```

## 3. Replay routing decisions
Supervisor logs every decision: grep "\[supervisor\]" in logs to see the
exact route + reason at each heartbeat.

## 4. Golden rule
If the ANSWER is bad, check RETRIEVAL first (chunks in HITL-1 payload).
Bad retrieval cannot be fixed by synthesis. 80% of "bad answer" bugs are
retrieval bugs.
