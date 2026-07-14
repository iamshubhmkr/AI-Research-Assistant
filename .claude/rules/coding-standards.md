# Coding Standards

- Python 3.11+. Type hints on all public functions.
- Module docstrings state the DESIGN DECISION the module embodies (why it
  exists, what trade-off it makes). These double as interview answers.
- No magic numbers: every tunable goes in config.py with a comment.
- Logging: logger.info for state transitions, logger.warning for degraded
  paths (retries, fallbacks), logger.error before any raise/default.
- LLM prompts are module-level UPPER_CASE constants — easy to find, easy to
  diff, easy to eval after changes.
- Error handling philosophy: agents degrade gracefully (supervisor → END,
  critic → approve, router → empty string) rather than crash the pipeline.
