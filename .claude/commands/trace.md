---
description: Trace a session's execution path through the pipeline
---

Given a session_id in $ARGUMENTS:
1. Load the checkpoint via graph.get_state
2. List which state fields are populated (papers? chunks? synthesis? critique?)
3. Infer the current stage and what the next supervisor decision will be
4. Report revision_count, faithfulness_score, and whether a HITL pause is active
