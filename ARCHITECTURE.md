# ARCHITECTURE.md — How This Project Works, Hop by Hop

> A complete walkthrough of the AI Research Assistant v3.1: every script, every
> function hop, every database and cache that gets created, traced with two
> concrete examples — one full pipeline run and one cache hit.
>
> Reading guide: Part 1 = the cast and the map. Part 2 = what gets created on
> disk. Part 3 = Example A, a question traced end to end, function by function.
> Part 4 = Example B, the same question asked again (cache hit). Part 5 = the
> supporting scripts that don't sit on the request path.

---

## PART 1 — The cast and the map

### The components (one job each)

| Piece | One job | Analogy |
|---|---|---|
| **Streamlit** (`ui/app.py`) | the web page the user clicks — sends HTTP, renders JSON | the front desk |
| **uvicorn** | the ASGI server: listens on :8000, runs the async event loop | the engine |
| **FastAPI** (`api/main.py`) | the app: routes each URL to the right endpoint function | the office with 3 doors |
| **LangGraph** (`agents/graph.py`) | the state machine: runs 6 agents in order, pauses for humans | the project manager |
| **Bedrock** (via `llm_client.py`) | ALL AI thinking: Sonnet (reason), Haiku (classify), Titan (embed) | the brains for hire (the only bill) |
| **Postgres** (Docker) | LangGraph's checkpoint store — the durable "save file" | the save file |
| **Redis** (Docker) | caches: embeddings + answer payloads | sticky-note memory |
| **Chroma** (local files) | vector DB: paper chunks + summaries, searchable by meaning | the smart library index |

### The whole map

```
        USER (browser)
          │  type question, click
          ▼
   ┌─────────────────────┐     HTTP      ┌───────────────────────────────────────────┐
   │ Streamlit  :8501    │ ───POST─────► │ uvicorn ─► FastAPI app :8000               │
   │ ui/app.py           │ ◄──JSON────── │ api/main.py (3 async endpoints)            │
   └─────────────────────┘               │      │                                     │
                                         │      ▼                                     │
                                         │ LangGraph graph  ──checkpoints──► Postgres │
                                         │ agents/graph.py                   (Docker) │
                                         │  supervisor + 5 worker agents              │
                                         │      │                                     │
                                         │      ├─ every AI call ► llm_client.py ─► AWS BEDROCK ($)
                                         │      ├─ vector search ────────────────► Chroma (./data)
                                         │      └─ caches ───────────────────────► Redis (Docker)
                                         └───────────────────────────────────────────┘
```

### One question = THREE HTTP requests

Because the pipeline pauses twice for a human, a single research job is three
separate calls, glued together by one **session id** (created once by
`/research/start`, carried back by the client on the next two calls). That
session id doubles as LangGraph's **thread_id** — the name of the save file in
Postgres.

```
REQ 1  POST /research/start            → fetch → extract → retrieve → ⏸ PAUSE 1 (freeze to Postgres)
REQ 2  POST /research/approve_chunks   → thaw  → synthesize → critique → ⏸ PAUSE 2 (freeze again)
REQ 3  POST /research/resolve_critique → thaw  → (maybe revise) → END → answer + cost
```

> **FAQ: where do `graph.ainvoke()` and `graph.aget_state()` live?**
> They are not in any file of this repo — they are methods of the compiled
> graph object (`CompiledStateGraph`) that `build_workflow().compile(...)`
> returns, defined inside the **langgraph library**. We *call* them from
> `api/main.py`. `ainvoke(initial)` = run from the start; `ainvoke(None)` =
> resume the saved thread; `aget_state()` = peek at the save file;
> `aupdate_state()` = let the human edit the save file.

---

## PART 2 — What gets created on disk (DBs, tables, caches)

Everything below is created lazily, on first use:

```
POSTGRES  (docker container, host port 5433)
  └─ created by: agents/graph.py get_graph() → AsyncPostgresSaver.setup()
     tables: checkpoints, checkpoint_blobs, checkpoint_writes (LangGraph's own schema)
     content: one serialized snapshot of the ResearchState per step, keyed by thread_id
     purpose: durable HITL state — pauses survive restarts and long waits

CHROMA  (local folders, no server, no credentials)
  ├─ ./data/chroma  → collection "research_papers"  (HNSW index, cosine distance)
  │     created by: rag/vector_store.py get_collection()
  │     rows: L0 raw chunks (level:0), L1 section summaries (level:1),
  │           L2 paper summaries (level:2) — all embedded by Titan
  └─ ./data/qcache  → collection "qa"
        created by: cache/semantic_cache.py SemanticQueryCache.__init__
        rows: one embedded QUESTION per cached answer (vector + pointer id "cid")

REDIS  (docker container, port 6379)
  ├─ emb:<sha256-of-text>  = JSON embedding vector      TTL 7 days   (L2 embedding cache)
  └─ qa:<uuid>             = JSON {answer,sources,...}  TTL 24 hours (L1 answer payloads)

AWS (optional / prod only)
  ├─ DynamoDB "research-sessions"  — session history (enable_dynamo=False locally → no-op)
  └─ S3 bucket — golden evaluation dataset (harvested high-quality answers)
```

---

## PART 3 — EXAMPLE A: one question traced end to end

**Input:** the user types into Streamlit:

> *"What are the limitations of RAG for multi-hop reasoning?"*

We will follow it hop by hop. Format for each hop:
`file → function` · what it does (high level) · what it touches ($ = Bedrock spend).

---

### REQUEST 1 — `/research/start`

#### Hop 0 · `ui/app.py` (Streamlit, stage "ask")
The button handler collects the question and options and makes a plain
blocking HTTP call: `requests.post("http://localhost:8000/research/start", json={...})`.
Streamlit runs **no pipeline logic** — it is a thin client.

#### Hop 1 · `api/main.py → start_research(req)`
The async endpoint (created by the `@app.post("/research/start")` decorator;
`async` only means it can yield during waits so uvicorn's event loop can serve
other users). It does five things:

```
start_research
 ├─ 1. graph = await get_graph()          → build/compile the graph (first call only)
 ├─ 2. reset_token_usage()                → zero the cost meter (llm_client.py)
 ├─ 3. semantic_cache().get(query)        → "asked before?" check (see Hop 2)
 ├─ 4. session_id = uuid4()               → e.g. "3f2a9b..." — the save-slot name
 └─ 5. await graph.ainvoke(_initial_state(...), config={"thread_id": session_id})
```

`_initial_state()` builds the empty **shared notebook** — the `ResearchState`
TypedDict from `agents/state.py` (query, empty lists for papers/chunks, flags
like `chunks_approved: False`, counters like `revision_count: 0`). Every agent
reads and writes this one object; no agent ever calls another agent.

#### Hop 1a · `agents/graph.py → get_graph()`  *(first request only)*
- `build_workflow()` — declares the topology: 6 nodes; entry point = supervisor;
  a **conditional edge** from supervisor that reads `state["next"]` to pick the
  next node; and an edge from every worker node **back to supervisor** (the
  heartbeat).
- Opens a psycopg connection pool → `AsyncPostgresSaver.setup()` → **creates
  the checkpoint tables in Postgres** (first run).
- `compile(checkpointer=..., interrupt_before=["synthesizer"], interrupt_after=["critic"])`
  — the two human pauses are baked into the graph itself.

#### Hop 2 · `cache/semantic_cache.py → SemanticQueryCache.get(question)`
The front door. It embeds the *question* and searches the `qa` Chroma
collection for a past question with cosine similarity ≥ **0.92**.

```
get(question)
 ├─ _embed(question) → rag/embedder.embed_texts()
 │      ├─ cache/embedding_cache.py EmbeddingCache.get()  → Redis "emb:<sha256>" (miss first time)
 │      └─ rag/embedder.py TitanEmbedder.encode()         → Bedrock Titan $ (tiny)
 ├─ chroma qa.query(query_embeddings=[vec])  → nearest past question
 └─ similarity < 0.92 or empty → return None  (MISS — run the pipeline)
```

First time asked → **MISS**. On to the graph.

#### Hop 3 · `agents/supervisor.py → supervisor_node(state)`  — routing hop #1
Builds a compact summary of the notebook (`raw_papers_count: 0`,
`sections_populated: False`, ...), sends it with a priority rulebook to
**Haiku $** via `llm_client.call_llm()`, parses the JSON reply, validates it
against `VALID_ROUTES`, and returns `{"next": "fetcher"}`. Any error → `END`
(fail safe — a glitch can never loop forever).

> `llm_client.py → call_llm()` is THE single choke point for every AI call in
> the project: it resolves the model id per provider, retries with backoff,
> trips a per-provider circuit breaker, and records tokens/latency per agent
> (`_track`) so `estimate_cost()` can price the run at the end.

#### Hop 4 · `agents/fetcher.py → fetcher_node(state)`
The **ReAct loop** (Reason + Act), max 8 iterations:

```
fetcher_node
 └─ loop: call_llm(Sonnet $, tools=ARXIV_TOOLS)
      │   model replies with stop_reason "tool_use" and a Thought
      ├─ run_arxiv_tool("search_arxiv") → _arxiv_search() → free arXiv API (HTTP)
      ├─ run_arxiv_tool("fetch_pdf")    → _fetch_sync(url) → download bytes
      │      └─ rag/document_router.py extract_text(url, bytes)
      │           ├─ _detect_kind(): sniff magic bytes (b"%PDF-" → pdf), never trust the suffix
      │           └─ _parse_pdf(): pymupdf.open(stream=bytes) → markdown text
      └─ append tool results to messages; loop until model stops or cap hits
 └─ return {"raw_papers": [{"url","text"},...], "paper_ids": [stable_id(url)...]}
```

`utils.py → stable_id()` = sha256-based short id (Python's built-in `hash()` is
salted per process and would break Chroma identity across restarts/workers).

*Example state after this hop:* `raw_papers` holds 1 paper,
`paper_id = "9c1d2e3f4a5b6c7d"`. Edge: fetcher → supervisor.

#### Hop 5 · supervisor again → `{"next": "extractor"}`  (papers exist, no sections yet) — Haiku $

#### Hop 6 · `agents/extractor.py → extractor_node(state)`
Reads each paper and **builds the searchable library**. Five jobs per paper:

```
extractor_node
 ├─ split_into_sections(text)          regex on headings → {"abstract":..., "methods":...}   (free)
 ├─ extract_facts_cot(text)            call_llm(Sonnet $) → structured facts JSON (CoT prompt)
 ├─ rag/chunker.py chunk_paper_sections()
 │     └─ semantic_chunk(): ~400-token chunks, cut at paragraph boundaries,
 │        50-token overlap so no fact is sliced in half; skips references     (free)
 ├─ rag/embedder.py embed_texts(chunk texts, EmbeddingCache)
 │     └─ per chunk: Redis "emb:" hit? else Titan.encode() $ then cache it
 ├─ collection.upsert(documents, embeddings, metadatas{level:0}, ids)
 │     → CHROMA WRITE: rows like "9c1d2e3f4a5b6c7d_L0_introduction_0"
 └─ rag/raptor.py RAPTORIndexer.build_tree(sections, paper_id)
       ├─ _summarize_section() call_llm(Haiku $) → upsert L1 "…_sec_introduction" (level:1)
       └─ _summarize_paper()   call_llm(Haiku $) → upsert L2 "…_summary"          (level:2)
 └─ rag/hyde.py generate_hypothetical_document(query)  call_llm(Sonnet $)
 └─ return {"sections", "extracted_facts", "hyde_docs"}
```

**After this hop the Chroma `research_papers` collection exists and is
populated** at three zoom levels (RAPTOR): raw chunks (L0), section summaries
(L1), paper summary (L2) — all embedded by the same Titan model, tagged by
`level` metadata. Edge: extractor → supervisor.

#### Hop 7 · supervisor again → `{"next": "retriever"}`  (sections exist, no chunks retrieved) — Haiku $

#### Hop 8 · `rag/retriever.py → retriever_node(state)` — the 8-stage heart
```
retriever_node → HybridRetriever(collection, embedder, EmbeddingCache)
 │   __init__: collection.get(where={"level":0}) → build BM25 keyword index in memory
 └─ retrieve(query):
     [1] _expand(query)          call_llm(Haiku $) → 2 alternative phrasings
     [2] per query: hyde_embed(q)  rag/hyde.py
           └─ cached? Redis "hyde:<q>" via EmbeddingCache : else
              generate fake academic ANSWER (Sonnet $) → Titan.encode() $ → cache 7d
           WHY: a question never embeds near an answer; a fake answer does
     [3] col.query(query_embeddings=[vec], where={"level":0})   ← DENSE (Chroma HNSW+cosine)
     [4] bm25.get_scores(q.split())                             ← SPARSE (exact keywords)
     [5] rrf_fuse(dense_all, sparse_all)   score = Σ 1/(60+rank) per list a chunk appears in
           → merge by RANK, not score (scores aren't comparable); agreement wins
     [6] _expand_parent(top 20)  swap each chunk for its L1 section summary
           → "search with a scalpel, read with a wide lens"
     [7] _get_reranker() → None (cross-encoder optional, off) → keep RRF order
     [8] _compress_parallel(top 5)  5× call_llm(Haiku $) in a thread pool
           → trim each chunk to its 2-3 relevant sentences, exact wording kept
 └─ return {"retrieved_chunks": [~5 tight chunks]}
```
Edge: retriever → supervisor.

#### Hop 9 · supervisor → `{"next": "synthesizer"}` … but ⏸ **PAUSE 1**
The graph was compiled with `interrupt_before=["synthesizer"]`, so instead of
running the synthesizer, LangGraph **freezes**: `AsyncPostgresSaver` writes the
entire notebook to Postgres under `thread_id`, and `graph.ainvoke(...)`
**returns** to the endpoint.

#### Hop 10 · back in `api/main.py → start_research`
```
snap = await graph.aget_state(config)        → snap.next == ("synthesizer",)  (proof it paused)
return {"status": "awaiting_chunk_approval", "session_id": "3f2a9b...",
        "retrieved_chunks": [...], "token_usage": {...}}
```
Streamlit stores the session id, switches to the chunk-review screen.
**Request 1 is over — nothing is in memory; the job lives in Postgres.**

---

### REQUEST 2 — `/research/approve_chunks`

The human unticks one weak chunk and types guidance: *"focus on multi-hop"*.

#### Hop 11 · `api/main.py → approve_chunks(req)`
```
approve_chunks
 ├─ current = await graph.aget_state(config)      → load the frozen notebook by session id
 ├─ filtered = drop the unticked chunk
 ├─ await graph.aupdate_state(config, {retrieved_chunks: filtered,
 │                                     chunks_approved: True,
 │                                     human_chunk_feedback: "focus on multi-hop"})
 │        ← the HUMAN edits the save file directly
 └─ state = await graph.ainvoke(None, config)     → None = "resume where frozen"
```
Note: approval does **not** re-run retrieval (re-retrieving the same query
would deterministically return the same chunks). The human curates the chunk
set by hand; the feedback steers the *writing*, not the search.

#### Hop 12 · `agents/synthesizer.py → synthesizer_node(state)`
```
synthesizer_node
 ├─ _build_context(chunks)  → "[paper_id: section]\ntext" blocks;
 │       prepends "USER GUIDANCE: focus on multi-hop"  (the Pause-1 feedback)
 ├─ simple query  → _cot()             call_llm(Sonnet $)
 │       forced 6-step <reasoning>: UNDERSTAND → INVENTORY (every fact + citation)
 │       → CONFLICTS → PATTERNS → GAPS (say what sources DON'T cover — never fill
 │       from memory: the anti-hallucination move) → STRUCTURE → then the answer
 │   complex query → _tree_of_thoughts(): 3 parallel Sonnet $ drafts
 │       (chronological/thematic/conflict-first) + Haiku $ vote
 ├─ _self_consistency()  3× call_llm(Haiku $) "uses ONLY context facts? YES/NO" → majority
 └─ return {synthesis, sc_verdicts, critique: None, revision_count: +1}
```
Edge: synthesizer → supervisor → `{"next": "critic"}` (Haiku $).

#### Hop 13 · `agents/critic.py → critic_node(state)`
```
critic_node
 ├─ call_llm(Sonnet $, FEW_SHOT_CRITIC)    5 worked examples DEFINE the severity
 │       ladder (fabricated stat=critical, over-broad claim=moderate, clean=approve)
 │       → {approved, issues, severity, reflexion}
 ├─ _faithfulness_score()  call_llm(Haiku $)  independent 0-1 grounding score
 │       (native judge, NOT ragas — ragas runs offline only)
 ├─ needs_revision = budget_left AND (critical OR (moderate AND score < 0.85))
 │       ← two signals must agree before paying for a rewrite
 ├─ if needs_revision: _build_reflexion() call_llm(Sonnet $)
 │       → 3 bullets of the SPECIFIC mistakes (the memo that makes the retry learn)
 └─ return {critique, needs_revision, faithfulness_score,
            final_answer: None if revising else synthesis, sources}
```

#### Hop 14 · ⏸ **PAUSE 2** — `interrupt_after=["critic"]`
The graph freezes again right after the critic. Back in `approve_chunks`:
```
snap.next truthy → return {"status": "awaiting_critique_review",
                           synthesis, critique, faithfulness_score, needs_revision}
```
Streamlit shows the draft + the critic's findings. **Request 2 over.**

---

### REQUEST 3 — `/research/resolve_critique`

Say the critic found a moderate issue (`needs_revision: True`) and the human
clicks "Accept / Revise per critic".

#### Hop 15 · `api/main.py → resolve_critique(req)`
```
resolve_critique
 ├─ await graph.aupdate_state(config, {human_override: False, human_revision_note?})
 ├─ state = await graph.ainvoke(None, config)     → resume
 │      supervisor (Haiku $): needs_revision & revision_count < 2 → "synthesizer"
 │      … interrupt_before synthesizer fires again → pause
 ├─ while snap.next == ("synthesizer",):          ← auto-step THROUGH the revision
 │      state = await graph.ainvoke(None, config)    pause (chunks were already
 │      snap  = await graph.aget_state(config)       approved by the human once)
 │      → synthesizer REWRITES with the reflexion memo injected:
 │           "<previous_failure_memory> …avoid these mistakes…"
 │        different prompt → different output → the retry LEARNS, not repeats
 │      → critic re-checks (fresh Sonnet $ + Haiku $) → approves → pause again
 ├─ snap.next still truthy → return "awaiting_critique_review" round 2 → human approves
 │      (second resolve_critique call: supervisor → END, graph truly finishes)
 └─ on END:
      ├─ usage = get_token_usage()  → per-agent tokens collected by _track()
      ├─ semantic_cache().set(query, answer, ...)    ← cache for 24h
      │     Redis "qa:<uuid>" = answer payload; qcache chroma gets the embedded question
      │     (SKIPPED if the human overrode the critic — no cache poisoning)
      ├─ persistence/dynamo.py SessionStore.add_query()   (no-op locally)
      ├─ evaluation/golden_dataset.py maybe_collect_from_production()
      │     faithfulness ≥ 0.92 → append to the S3 golden dataset (best-effort)
      └─ return {"status": "complete", final_answer, sources,
                 faithfulness_score, token_usage,
                 estimated_cost_usd: estimate_cost(usage)}   ← tokens × config prices
```

#### The final result the user sees

```json
{
  "status": "complete",
  "final_answer": "RAG systems face three main limitations in multi-hop reasoning:
                   single-step retrieval insufficiency [9c1d2e3f: introduction], ...
                   Gaps: the sources do not address latency trade-offs.",
  "sources": ["https://arxiv.org/pdf/2401.01234"],
  "faithfulness_score": 0.93,
  "token_usage": {"supervisor": {...}, "fetcher": {...}, "synthesizer": {...}, ...},
  "estimated_cost_usd": 0.14
}
```

#### What now exists on disk (created by this one question)

```
Postgres : ~10 checkpoint rows for thread "3f2a9b..." (one per step, latest = END state)
Chroma   : research_papers → L0 chunks + L1 section summaries + L2 paper summary
           qcache/qa       → 1 embedded question pointing at the cached answer
Redis    : emb:<sha256> keys (chunk + hyde embeddings, 7d) · qa:<uuid> (answer, 24h)
S3       : + 1 golden-dataset candidate (only because faithfulness ≥ 0.92)
```

---

## PART 4 — EXAMPLE B: a similar question five minutes later

**Input:** another user asks *"What problems does RAG have with multi-hop questions?"*
— different words, same meaning.

```
ui/app.py → POST /research/start
api/main.py → start_research
 └─ semantic_cache().get(question)
      ├─ embed_texts(question)      → Titan $ (~$0.000002 — the ONLY spend)
      ├─ qcache.query(vector)       → nearest stored question: similarity 0.94 ≥ 0.92 ✓
      ├─ Redis GET "qa:<cid>"       → the stored answer payload
      └─ return {answer, sources, from_cache: True, similarity: 0.94}
 └─ endpoint returns immediately: {"status": "complete", "from_cache": true, ...}
```

**No graph, no agents, no papers, no Sonnet.** The cache is keyed on *meaning*
(an embedding), not exact words — which is why a paraphrase hits. A ~$0.14
pipeline run became a ~$0.000002 lookup. This is also why an answer the human
had to force past the critic is never cached: it would be served to strangers.

---

## PART 5 — Scripts that support the request path

| Script | Role (high level) |
|---|---|
| `config.py` | every tunable as a validated Pydantic `Settings` object loaded from `.env` — model ids, prices, cache TTLs, loop caps, thresholds. No magic numbers in code. |
| `llm_client.py` | the single choke point: `call_llm()` (retry + backoff, per-provider circuit breaker, Bedrock/Anthropic dispatch, `_to_plain` serialization), `_track()` per-agent token/latency, `estimate_cost()` prices by model used. |
| `utils.py` | `stable_id()` — sha256 short ids (process-stable, unlike `hash()`). |
| `agents/state.py` | `ResearchState` TypedDict — the shared-notebook *contract* every agent reads/writes (typing contract, not runtime-validated). |
| `cache/llm_cache.py` | L3 response cache for temp-0 calls — **defined but not wired in** (honest gap). |
| `persistence/postgres.py` | sync checkpointer helper for scripts/notebooks; the API path uses the async saver in `agents/graph.py`. |
| `persistence/dynamo.py` | optional session history (`enable_dynamo=False` locally; every call degrades to a warning, never fails a request). |
| `persistence/s3.py` | `PaperStore` — golden-dataset JSON in S3 (append/load). |
| `evaluation/ragas_eval.py` | OFFLINE quality lane: RAGAS 5 metrics judged by **Bedrock** (Sonnet judge + Titan embeddings via `bedrock_judge()`), `--cases` for real pipeline output, `regression_check()` blocks deploys on a >0.03 drop. |
| `evaluation/golden_dataset.py` | `maybe_collect_from_production()` — harvests high-faithfulness answers into the golden set; `generate_per_paper()` — synthesizes Q/A pairs from papers. |
| `evaluation/token_tracker.py` | pretty per-agent cost report over `get_token_usage()`. |
| `scripts/e2e_local.py` | full-pipeline E2E with **stubbed Bedrock**: real HTTP fetch, real Redis/Postgres/Chroma, both HITL pauses, the revision loop — 13 checks, costs $0. |
| `scripts/bedrock_smoke.py` | 3 tiny live calls (Sonnet/Haiku/Titan) to verify AWS credentials + model access. |
| `scripts/collect_eval_case.py` | runs a REAL query through the live graph (auto-approving pauses) and appends a RAGAS test case to `evaluation/cases.json`. |
| `mcp/arxiv_server.py` + `.claude/mcp.json` | reference MCP server exposing `search_arxiv`/`fetch_pdf`; registered for the Claude Code assistant at dev time. **Not called by the runtime pipeline** — the fetcher uses equivalent inline tools (promoting it to a real MCP client is a planned upgrade). |
| `tests/unit/` | fast, no-network tests: chunker boundaries, RRF fusion (incl. the overlap-prefix dedup bug), cost math, state contract, document router, stable ids. |
| `Dockerfile` / `docker-compose.yml` / `Makefile` | 1-worker uvicorn image; Redis+Postgres (host port **5433** — many dev machines run a local Postgres on 5432); `make up/test/lint/eval/run/ui`. |

### Design rules the whole codebase obeys

1. **Shared notebook** — agents communicate only through `ResearchState`; no agent calls another.
2. **Supervisor heartbeat** — every node edges back to the supervisor; routing is decided fresh each step; errors route to END.
3. **One LLM choke point** — every AI call goes through `call_llm()`: retries, circuit breaking, and cost tracking for free, everywhere.
4. **Degrade, don't crash** — cache failure = miss; parse failure = safe default; every loop hard-capped (`max_revision_count`, ReAct cap, graph recursion limit).
5. **Model routing** — Sonnet only where real reasoning happens; Haiku for routing/summarizing/compressing/checking; Titan for all embeddings. All on one AWS bill.
