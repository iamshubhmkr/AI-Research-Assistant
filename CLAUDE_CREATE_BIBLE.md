# THE .claude/ CREATION BIBLE
### How to bootstrap a state-of-the-art project with Claude Code — the exact prompts, models, and order

This is the playbook used to build this project's `.claude/` folder. Follow it
for ANY new project and you'll recreate the same quality. The core idea:
**you don't write code first — you write the project's "constitution" first**,
then let Claude Code build inside those guardrails.

---

## PART 1 — The Philosophy (read once)

A `.claude/` folder is the project's brain for Claude Code:

| File/Folder | What it is | Analogy |
|---|---|---|
| `CLAUDE.md` | Project memory — loaded into EVERY conversation | The employee handbook a new hire reads on day 1 |
| `settings.json` | Tool permissions (allow/ask/deny) | The office keycard — which doors open automatically |
| `mcp.json` | External tool servers | The Rolodex of outside vendors |
| `rules/*.md` | Detailed standards (architecture, code, tests) | Department policy binders |
| `skills/*/SKILL.md` | Step-by-step procedures Claude loads on demand | Laminated how-to cards by the machine |
| `agents/*.md` | Specialized subagents (reviewer, test-writer) | Contractors with a specific job and limited keys |
| `commands/*.md` | Slash-commands (/eval, /trace) | Speed-dial buttons |

**Model strategy** (this is the part most people get wrong):
- **PLAN with your strongest model** (Opus / "thinking" mode): architecture,
  CLAUDE.md, state design, the seed. Planning errors are 100× more expensive
  than coding errors.
- **BUILD with Sonnet**: implementing well-specified modules. The spec quality
  from the planning phase is what lets a cheaper model succeed.
- **CHORE with Haiku**: docstrings, README updates, renaming, formatting.

In Claude Code: `/model opus` ↔ `/model sonnet` ↔ `/model haiku`, and
**Shift+Tab** to toggle Plan Mode (Claude proposes, you approve, THEN it edits).

---

## PART 2 — The Seed Session (Opus + Plan Mode)

Open Claude Code in an EMPTY project folder. Switch to your strongest model
and Plan Mode. Run these prompts IN ORDER. Do not skip the order — each
prompt's output feeds the next.

### Prompt 1 — The Seed (the single most important prompt)
```
I'm starting a new project: <one-paragraph description of what it does,
who uses it, and the 3 hardest technical problems it must solve>.

Before writing ANY code, act as a principal engineer and produce:
1. A proposed architecture (components, data flow, where state lives)
2. The 5-8 "golden rules" — invariants that must NEVER be violated
   (e.g. "all LLM calls go through one client", "state is the only
   communication channel between agents")
3. The technology choices with a one-line WHY for each
4. The biggest 3 risks and how the architecture mitigates them

Challenge my description: ask me up to 5 clarifying questions FIRST if
anything is ambiguous. Do not write code.
```
Answer its questions. Iterate until the plan feels right. This conversation
IS your architecture review.

### Prompt 2 — Generate CLAUDE.md
```
Based on the architecture we just agreed, write .claude/CLAUDE.md containing:
- "What This Project Is" (3 sentences)
- "Architecture in One Paragraph" (the full data flow)
- "Golden Rules (NEVER violate)" — the numbered invariants from our plan,
  each with its WHY
- "Key Files" table (file → one-line purpose)
- "Commands" (the make targets we'll create)
- "Style" (docstring philosophy: explain WHY/design decisions, not WHAT)

Keep it under 80 lines. Every line must earn its context-window cost —
Claude reads this file in every single conversation.
```

### Prompt 3 — Generate settings.json
```
Write .claude/settings.json with a permissions policy:
- allow: read-only operations (Read, Grep, Glob), test commands,
  make targets, git status/diff/log, docker-compose
- ask: anything that mutates (Edit, Write, git commit/push, pip install)
- deny: destructive ops (rm -rf), reading .env secrets, production
  deploy commands

Principle of least privilege: when unsure, put it in "ask".
```

### Prompt 4 — Generate the rules/
```
Create .claude/rules/ with three files based on our golden rules:
- architecture.md: the structural invariants (layering, communication
  patterns, loop-safety caps) with the reasoning for each
- coding-standards.md: language version, typing, docstring philosophy,
  logging levels, where constants live, error-handling philosophy
  (degrade gracefully vs crash)
- testing.md: the test taxonomy (fast unit / expensive eval / smoke),
  what each may and may not touch, and the "prompts are code — eval
  after every prompt change" rule
```

### Prompt 5 — Generate the skills/
```
Create .claude/skills/ with one SKILL.md per recurring procedure in this
project. For each skill use YAML frontmatter (name + a description that
states WHEN to trigger it) and then a precise step-by-step checklist.

For this project the recurring procedures are:
1. <e.g. "add a new agent to the pipeline"> — the N-step checklist with
   exact files to touch in order
2. <e.g. "run and interpret evaluations"> — commands + a failure-diagnosis
   table (metric dropped → likely cause → file to inspect)
3. <e.g. "debug a broken pipeline"> — symptom → stage → first-check table

A good skill is one a brand-new engineer could follow without asking
questions.
```

### Prompt 6 — Generate the subagents/
```
Create .claude/agents/ with specialized subagents, each as a .md file with
frontmatter (name, description with "use proactively when...", tools
limited to what it needs, model sized to the job):

1. code-reviewer (model: sonnet, read-only tools + git diff) — reviews
   diffs against our golden rules, outputs PASS/FAIL per rule with
   file:line references
2. test-writer (model: sonnet, can write + run pytest) — writes unit
   tests obeying rules/testing.md, iterates until green
3. docs-writer (model: haiku, read+write) — keeps README and module
   docstrings current; docstrings must state design decisions

Each agent's prompt must reference the specific rule files it enforces.
```

### Prompt 7 — Generate the commands/
```
Create .claude/commands/ slash-commands for our most-repeated requests:
- /eval: run the evaluation, compare to targets, say if it's safe to commit
- /trace <session_id>: inspect a live session's state and report its stage
Each command is a .md file with a description frontmatter and the exact
procedure.
```

### Prompt 8 — The mcp.json
```
Write .claude/mcp.json registering our project's MCP servers (e.g. the
arXiv tool server at mcp/arxiv_server.py) so Claude Code can call the same
tools the agents use.
```

**Commit the entire .claude/ folder NOW, before any source code.**
`git add .claude CLAUDE_CREATE_BIBLE.md && git commit -m "chore: project constitution"`

---

## PART 3 — The Build Sessions (Sonnet)

Switch: `/model sonnet`. Now build in DEPENDENCY ORDER — each layer only
uses layers below it. For every module use this prompt shape:

```
Implement <file> per CLAUDE.md and rules/.
Inputs: <what it reads>  Outputs: <what it returns/writes>
Constraints: <the 2-3 golden rules that apply here>
Include a module docstring stating the design decision.
Then write tests/unit/test_<module>.py and run `make test` until green.
```

The order that worked for this project (generalize to yours):
1. `config.py` (settings) → 2. the client/abstraction layer (llm_client) →
3. the data contract (state) → 4. pure-logic utilities (chunker, router) →
5. infrastructure adapters (cache, persistence) → 6. domain logic (rag/) →
7. the agents (simplest first: supervisor → fetcher → extractor →
synthesizer → critic) → 8. the orchestration (graph) → 9. the API →
10. the UI → 11. CI/CD.

**Per-session hygiene:**
- One module (or one tightly-related pair) per session. `/clear` between.
- After each module: `make test`. After prompt changes: `make eval`.
- Invoke the code-reviewer subagent before each commit:
  "Use the code-reviewer agent on my staged diff."

---

## PART 4 — The Chore Sessions (Haiku)

`/model haiku` for: README updates, docstring polish, .env.example files,
Makefile help text, renaming. Prompt shape: "Update <file> to reflect
<change>. Keep the existing tone and structure."

---

## PART 5 — The Maintenance Loop

When you (or Claude) discover a new lesson:
1. Is it an invariant? → add to CLAUDE.md golden rules
2. Is it a procedure? → new/updated SKILL.md
3. Is it a standard? → rules/*.md
4. Is it a repeated request? → commands/*.md

Use `#` in Claude Code to quickly append a memory to CLAUDE.md mid-session.
Review CLAUDE.md monthly: delete anything stale — bloated memory degrades
every future session.

---

## PART 6 — Cheat-Sheet of the Whole Flow

```
EMPTY FOLDER
  └─ Opus + Plan Mode
      Prompt 1: seed (architecture + golden rules)   ← think hardest here
      Prompt 2: CLAUDE.md
      Prompt 3: settings.json
      Prompt 4: rules/
      Prompt 5: skills/
      Prompt 6: agents/ (subagents)
      Prompt 7: commands/
      Prompt 8: mcp.json
      COMMIT the constitution
  └─ Sonnet (build, dependency order, test-as-you-go)
      config → client → contract → utils → adapters →
      domain → agents → graph → api → ui → ci
      code-reviewer subagent before every commit
  └─ Haiku (docs, chores)
  └─ Forever: lessons → CLAUDE.md / rules / skills / commands
```

The meta-lesson: **the constitution is the project**. Code written inside
good guardrails by a mid-tier model beats code written without guardrails
by the best model. That's why planning gets Opus and typing gets Sonnet.
