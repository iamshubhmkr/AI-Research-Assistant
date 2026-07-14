# Testing Rules

- tests/unit: fast, deterministic, NO network, NO LLM calls. Pure logic
  (chunker boundaries, state reducers, cost math, router fallbacks).
- tests/ragas: golden-dataset evaluation. Runs in CI before deploy.
  Any metric dropping > 0.03 vs baseline FAILS the build.
- tests/smoke: post-deploy health checks against the live endpoint.
- After changing ANY prompt: run `make eval` and compare to the last scores.
  Prompts are code — they regress like code.
- New agent = new unit test for its pure-logic parts + a routing case in
  the supervisor rules.
