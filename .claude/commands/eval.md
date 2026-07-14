---
description: Run RAGAS smoke evaluation and summarize results vs targets
---

Run `make eval`, capture the output, and report:
1. Each metric vs its target (pass/fail)
2. If any metric fails: which file most likely caused it (use the table in
   .claude/skills/run-evals/SKILL.md)
3. Whether it is safe to commit
