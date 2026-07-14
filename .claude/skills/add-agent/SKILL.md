---
name: add-agent
description: Add a new agent node to the LangGraph pipeline correctly. Use when the user asks to add, create, or register a new agent (e.g. a citation-formatter, a translator, a summarizer agent).
---

# Adding a New Agent — The 6-Step Checklist

Follow these steps IN ORDER. Skipping one breaks routing or state merging.

## 1. Extend the state contract (agents/state.py)
Add the fields your agent writes. Ask: can MULTIPLE agents append to this
field? If yes → `Annotated[list, operator.add]`. If only-latest-matters →
plain type.

## 2. Write the agent module (agents/<name>.py)
Template:
```python
"""<Name> — <one-line job>. Design decision: <why this model/prompting>."""
import logging
from llm_client import call_llm
from .state import ResearchState
from config import settings

logger = logging.getLogger(__name__)

def <name>_node(state: ResearchState) -> dict:
    # read inputs from state, do work via call_llm(agent_name="<name>")
    # return ONLY the fields you write — LangGraph merges them
    return {"<field>": value}
```
Rules: model = Haiku if classification/summarize, Sonnet if reasoning.
ALL LLM calls through llm_client.call_llm with agent_name set.

## 3. Register the node (agents/graph.py)
- `wf.add_node("<name>", <name>_node)`
- Add to the conditional_edges map: `"<name>": "<name>"`
- Add the heartbeat edge: `wf.add_edge("<name>", "supervisor")`

## 4. Teach the supervisor (agents/supervisor.py)
- Add the agent to the AGENTS list in SUPERVISOR_PROMPT with WHEN to run it.
- Insert a numbered rule in the RULES priority list at the right position.
- Add any new state fields to the summary dict in supervisor_node.

## 5. Cost accounting (llm_client.py)
If the agent uses Haiku, add its agent_name to the haiku_agents set in
estimate_cost() — otherwise costs are overestimated 12×.

## 6. Tests + eval
- Unit test for any pure logic in tests/unit/test_<name>.py
- Run `make test` then `make eval` — confirm no RAGAS regression.
