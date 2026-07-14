# INTERVIEW.md — The Complete Interview Pack

> How to use this file:
> **Part 0** — 2-minute elevator version (for "give me the gist").
> **Part 1** — the 15-minute one-go walkthrough. Rehearse this aloud 3–4 times;
> the goal is breadth with momentum — name each term, give ONE clause of why,
> move on. All depth lives in Part 2.
> **Part 2** — depth bank for follow-up questions (architecture + concepts).
> **Part 3** — prompting-techniques bank (zero-shot, few-shot, CoT, ToT, ...).
> **Part 4** — production scenario Q&A (consulting style).
> **Part 5** — pocket one-liners if you blank.

---

# PART 0 — The 2-Minute Elevator Version

"I built a production-grade AI research assistant. You ask a research question;
six specialized agents — orchestrated by a LangGraph state machine — find
academic papers, read and index them, retrieve the best evidence, write a
cited answer, and fact-check it. A human approves the evidence *before*
writing and reviews the fact-check *after* — and those pauses are durable,
because the whole state checkpoints to Postgres, so a paused job survives
restarts.

Retrieval is the showpiece: HyDE closes the question-vs-answer embedding gap,
dense and keyword search cover each other's blind spots, and rank fusion
merges them. Generation is guarded: forced citations, explicit gap-admission,
a separate critic with a faithfulness judge, and a revision loop that learns
from a memo of its own mistakes — hard-capped at two rounds.

Everything runs on AWS Bedrock — the strong model only where reasoning
happens, the cheap model everywhere else, embeddings included — so it's one
bill, and a semantically-cached repeat question costs essentially nothing.
Quality is measured, not hoped: a runtime judge gates every answer, and an
offline RAGAS regression gate blocks deploys if quality drops. The theme is
engineering *trust* around an LLM."

---

# PART 1 — The 15-Minute One-Go Walkthrough

*(~2,000 words ≈ 13–15 min spoken. Bold terms = your coverage checklist.
Discipline: name it → one clause of why → move on. Never dive.)*

### [0:00] Framing

"Let me walk you through a project I'm proud of — an AI research assistant.
I'll start with the two or three ideas it's built on, then take you through
the whole architecture end to end, and I'm happy to go deep on any piece
afterward.

The foundation: a large language model is brilliant at language but has one
dangerous flaw for serious work — when it doesn't know something, it doesn't
say 'I don't know,' it makes something up that sounds right. That's
**hallucination**. The standard fix is **RAG — Retrieval-Augmented
Generation**: before the model answers, you retrieve real documents and make
it answer *only* from those — grounding it in evidence instead of memory.
That's the core idea the whole project is organized around.

Second idea: instead of one giant prompt doing everything, I use **multiple
specialized agents** — small focused steps, each doing one job well — run by a
**state machine**: an orchestrator with a checklist that decides what runs
next based on where things stand. So the project is a **multi-agent RAG
pipeline run by a state machine**. That sentence is the whole thing; let me
unpack it."

### [2:00] What it is

"Concretely: you ask a research question, and six agents — a supervisor, a
fetcher, an extractor, a retriever, a synthesizer, and a critic — find
academic papers, read them, and write a **cited, fact-checked answer**. Two
things make it production-grade rather than a demo: a human approves the
evidence and reviews the fact-check at two checkpoints, and the whole thing
runs on AWS, one cloud bill. Let me trace a question through it."

### [3:00] The front door — endpoints and async

"The user clicks in a **Streamlit** frontend — a thin web page. It sends an
HTTP request to my backend, built with **FastAPI**. An **endpoint** is one
function tied to one URL — I have three: start research, approve chunks,
resolve critique. The engine that runs the app and its async event loop is
**uvicorn** — it receives each request and FastAPI routes it to the matching
endpoint.

Those endpoints are **async**, and the reason matters: a research request
spends almost all its time *waiting* — on the AI model, on the database.
Async lets one server serve *other users* during one user's waits. It doesn't
speed up a single request — the six steps are sequential anyway — it's about
concurrency across users.

One important detail: the pipeline pauses twice for the human, so one research
job is actually **three separate HTTP calls over time**, glued together by a
**session ID** created on the first call and passed back by the client on the
next two."

### [4:30] The orchestration — the state machine

"The endpoint hands control to **LangGraph**, my state machine. Two design
rules run everything. First, all six agents share **one notebook** — a single
state object — and never talk to each other directly; each reads the
notebook, does its job, writes back. That means I can swap any agent without
breaking the others. Second, after *every* agent, control returns to the
**supervisor**, which looks at the notebook and decides who goes next — I
call it the heartbeat. The flow isn't hard-wired; it's decided fresh at each
step, and on any error the supervisor just stops, so it can never loop
forever."

### [5:30] The agents, in order

"The **fetcher** searches academic databases and downloads papers, working in
a reason-act loop — think, search, look at results, think again — like a
person researching. Everything it downloads is normalized to plain text, so
the rest of the system only ever deals with text.

The **extractor** reads each paper and builds a searchable index: it splits
the paper into chunks — careful not to cut a fact in half — turns each chunk
into a numeric 'meaning vector' with an embedding model, and stores them in a
**vector database**. It also pre-builds a small summary tree — chunk level,
section level, paper level — so different questions can be answered at the
right zoom level; that technique is called **RAPTOR**.

Then the **retriever** — the heart of the system, where most RAG systems
quietly fail. It finds the best handful of chunks, layering several
techniques that each fix a specific failure. It rephrases the question a
couple of ways. It uses **HyDE**: it writes a *fake* ideal answer and
searches with *that*, because a question and its answer don't sit near each
other in meaning-space — but two answers do. It runs **two searches in
parallel** — one by meaning, one by exact keywords — because they have
opposite blind spots, and fuses them by rank with a method called **RRF**.
Then it trims each result to just the relevant sentences. The output is about
five tight, highly relevant chunks. Happy to go as deep as you like on any of
those."

### [8:00] Human checkpoint one — and durable state

"Now the first **human-in-the-loop** checkpoint. Before writing a single
word, the system *pauses* and shows the human the evidence it's about to use
— remove bad chunks, add guidance. A person controls the evidence before any
answer exists.

The engineering behind that pause matters: the human might take an hour, or
the server might restart. So at every step LangGraph **saves the entire state
to Postgres** — an auto-saving save file, keyed by the session ID. The pause
survives restarts and long waits; the next request just loads the save and
resumes."

### [9:30] Writing and fact-checking

"After approval, the **synthesizer** writes the answer through a structured
reasoning process; the two key moves are: list every fact *with its
citation*, and explicitly state what the documents *don't* cover — never fill
that gap from memory. That's the anti-hallucination move: it gives the model
a legal way to say 'the sources don't say.'

Then the **critic** — a separate agent — fact-checks the draft against the
source text with a calibrated severity scale, plus an independent faithfulness
score, so two signals must agree before a rewrite. If it sends the draft
back, it writes a memo of the *specific* mistakes that gets fed into the
retry — so the rewrite actually learns instead of repeating itself. The loop
is hard-capped at two rounds, because an AI judge can always find one more
nitpick. Then the second **human checkpoint**: approve, revise, or override."

### [11:00] Guardrails and infrastructure

"Pulling the **guardrails** together: every loop has a hard cap; every agent
degrades gracefully instead of crashing — supervisor stops, critic approves,
parser falls back; and every AI call goes through *one* function — one choke
point — giving me retries, a circuit breaker for provider outages, and
per-agent cost tracking for free.

Underneath: all AI — reasoning, classification, embeddings — runs on **AWS
Bedrock**, one bill, with deliberate **model routing**: the strong expensive
model only where real reasoning happens, the cheap fast model for sorting and
checking. **Postgres** holds the save files, **Redis** caches recent answers
so a repeated question is nearly free, the **vector database** holds
embeddings, and the supporting databases run in **Docker** locally so
development is free."

### [12:30] Quality and close

"Quality has two layers: at runtime the critic gates every answer; offline, a
standard evaluation framework — **RAGAS**, running on a Bedrock judge —
scores five metrics against a golden dataset and *blocks a deploy* if quality
drops past a threshold. Prompts are code; they regress like code.

Zooming out: the whole project is about engineering **trust** around a
language model. Ground it in retrieved evidence, force it to cite and admit
gaps, fact-check it with a second model, put a human at the two
highest-leverage moments, and make the whole thing pausable and restart-safe
— all on one Bedrock bill. That's the system — I'd love to go deeper on
whichever part interests you."

### Rehearsal spine (say it from just this list by pass 4)

RAG/hallucination → multi-agent + state machine → Streamlit → FastAPI
endpoints / async / uvicorn → session ID (3 calls) → LangGraph notebook +
supervisor heartbeat → fetcher (ReAct) → extractor (chunks, embeddings,
RAPTOR) → retriever (HyDE, hybrid, RRF, compression) → HITL 1 → Postgres save
file → synthesizer (cite + gaps) → critic (severity + faithfulness + reflexion,
cap 2) → HITL 2 → guardrails (caps, degrade, one LLM client) → Bedrock +
model routing → Postgres/Redis/vector DB/Docker → eval gate → "trust around
an LLM."

---

# PART 2 — Depth Bank (follow-up questions)

### "How do endpoints / async / uvicorn actually fit together?"
The decorator (`@app.post(...)`) makes a function an endpoint — `async` does
not. uvicorn is the **ASGI server** (ASGI = Asynchronous Server Gateway
Interface, the async successor to WSGI): it receives the HTTP request and
runs the event loop; FastAPI is the app that routes the URL to the matching
endpoint; the endpoint is the function that runs. Both are "async" but
differently: FastAPI is async-capable *application code*; uvicorn is the
async *runtime* that owns the loop. Nuance: the AI calls are synchronous
(boto3), but LangGraph runs each agent in a background thread — so a blocking
Bedrock call ties up its thread while the event loop stays free for other
users. *Async shell for concurrency, threads underneath to absorb blocking.*
And async only helps across users — one user's six steps are inherently
sequential; the only within-session parallelism is thread pools on
independent sub-tasks (3 ToT drafts, 5 compressions).

### "How does the session ID work?"
`/research/start` generates it once (a UUID) and returns it; Streamlit stores
it and sends it back on the next two calls. Three separate HTTP calls, three
endpoints, one shared ID — and that ID is also LangGraph's **thread_id**, the
name of the saved state in Postgres, which is how each call resumes the right
paused job. (Dry-cleaner ticket: shop issues it once, you present it on every
visit.)

### "Walk me through retrieval in detail."
Eight stages, each fixing a named failure: (1) **query expansion** — 3
phrasings cover more vocabulary; (2) **HyDE** — Hypothetical Document
Embeddings: generate a fake academic *answer* (not a rephrased question!) and
embed that, because documents are answer-shaped statements and a question
never embeds near a statement — the fake answer's facts can be wrong, it's a
search probe, discarded after embedding; raised recall 0.71→0.89; (3)
**dense** search — Titan embeds the probe, Chroma finds nearest neighbors
(Titan = the embedding model, Chroma = the store-and-search half of ONE dense
searcher — not two rivals); (4) **BM25** — exact keywords on the *raw query
words* (not the fake answer), opposite blind spot; (5) **RRF** — fuse by rank,
`score = Σ 1/(60+rank)` per list a chunk appears in — no score normalization
needed since cosine (0–1) and BM25 (unbounded) aren't comparable; agreement
between two different judges is the strongest signal; (6) **parent
expansion** — match on the small chunk, hand the writer its section summary:
"search with a scalpel, read with a wide lens"; (7) optional
**cross-encoder** rerank — reads query+chunk *together* through one
transformer (accurate, slow) vs the bi-encoder that embedded them separately
(fast, approximate) — classic two-stage: fast skim shortlists, slow read
ranks; (8) **compression** — Haiku trims each finalist to its 2–3 relevant
sentences, exact wording, so the expensive model reads only signal and
nothing gets "lost in the middle."

### "How does the vector search find neighbors fast?"
Chroma uses an **HNSW** index with **cosine** distance — set in one line
(`hnsw:space: cosine`). They answer different questions: HNSW = *how to
search fast* (a multi-layer graph — highways on top for big jumps, local
streets below to home in; greedy walk); cosine = *what "near" means* (angle
between vectors; we store normalized vectors so it's the natural metric).
HNSW is **approximate** nearest neighbor — it trades a sliver of accuracy for
sublinear speed, the right trade at scale.

### "How does HITL survive a restart? What happens at each pause?"
LangGraph checkpoints the entire state to Postgres after every step (save
file keyed by session ID); the graph interrupts *before* the synthesizer and
*after* the critic. At a pause the program actually ends and the HTTP
response returns; the next request passes the same ID, LangGraph loads the
save, resumes. **Pause 1 does not re-retrieve** — deliberately: retrieval is
deterministic (same query + same index = same chunks), so a re-retrieval loop
would repeat forever. Instead the human *directly curates* the chunk list and
their feedback steers the *synthesis* prompt. **Pause 2** can trigger a
rewrite — and the retry differs because its prompt changed: the critic's
**Reflexion memo** of the specific mistakes (plus any human note) is
injected. Different prompt → different output → the loop *converges* instead
of cycling; and a hard cap of 2 guarantees it ends regardless.

### "How do you stop hallucination?"
Four layers: retrieval grounds the answer in real text; the synthesizer must
cite every claim and explicitly state gaps (never fill them); the critic
fact-checks against the source — calibrated severity + a separate
faithfulness judge, two signals must agree before a rewrite; and a human
reviews at two checkpoints. Plus Reflexion makes failed retries learn.

### "Is the critic using RAGAS?"
No — deliberately. The critic runs two **native** LLM-judge functions: a
few-shot-calibrated Sonnet severity check and a one-shot Haiku faithfulness
score (one call, milliseconds). RAGAS runs **offline only** — five metrics
(faithfulness, answer relevancy, context precision/recall, correctness) on a
**Bedrock judge**, gating deploys on a >0.03 regression. Backstory worth
telling: v3 called RAGAS inline in the critic; its default judge is OpenAI,
which wasn't configured, so it silently returned 0.0 and forced needless
rewrites — I replaced it with a cheap dedicated judge and moved the heavy
framework offline. *A full eval framework doesn't belong in the request path.*

### "Why multi-agent instead of one big prompt?"
Separation of concerns and controllability: each agent is small, testable,
swappable; the shared-notebook rule means changing one can't break another;
the supervisor gives one place to control routing and fail safe. A
mega-prompt is hard to debug, hard to test, and hallucinates more.

### "Where does the money go and how do you control it?"
Only Bedrock costs money, and every call flows through one client that tracks
tokens per agent. Control = **model routing** (Sonnet only for
reasoning/writing; Haiku for routing, summarizing, compressing, checking) +
**three cache layers** (semantic answers, embeddings, and an L3 response
cache) + compression (fewer input tokens) + gating the N×-cost techniques
(ToT, self-consistency) to cases that earn them. Full uncached query ≈ $0.15;
semantic-cache hit ≈ $0.000002.

### "What does Pydantic do here?"
Guards the two doors where outside data enters: `BaseSettings` in `config.py`
loads `.env`, coerces string env-vars to typed values (`"2"`→int,
`"false"`→bool), validates choices (`Literal`) and **fails at startup** on a
typo; FastAPI request models parse and validate incoming JSON — bad requests
get a 422 before my code runs, defaults fill in, and handlers see clean typed
data. Nuance: the internal `ResearchState` is a `TypedDict` — a typing
contract, not runtime validation.

### "Where's MCP in this?"
Honest framing: the tool layer is *designed* MCP-style — a reference arXiv
MCP server exposes `search_arxiv`/`fetch_pdf` and is registered in `mcp.json`
for the Claude Code assistant at dev time — but the running fetcher currently
calls equivalent inline tools. The value of promoting it to a real MCP
client: swap arXiv for PubMed/Semantic Scholar by changing only the server,
and the broader security pattern — agents call *named tools*; credentials
live server-side and never enter a prompt, so even prompt injection can't
exfiltrate them.

### "Hardest bugs / real engineering decisions?"
(1) The HITL pause originally conflicted — the supervisor was told to END at
the pause point while the graph relied on an interrupt; both can't pause, so
the supervisor now routes forward and the interrupt owns the pause. (2) IDs
were built with Python's `hash()`, which is salted per process — broke vector
IDs across workers/restarts; switched to sha256. (3) The inline-RAGAS
failure above. (4) PDF parsing: arXiv URLs have no `.pdf` extension, so
routing by suffix fed PDFs to the HTML parser — fixed with magic-byte
sniffing (`%PDF-`). (5) Dependency matrix: the eval library needed an older
langchain ecosystem than the newest orchestrator — pinned versions and
documented why.

---

# PART 3 — Prompting Techniques Bank

*Opening line:* "I can talk about these abstractly, but it's easier to point
at where I actually used each one — almost every major technique shows up
somewhere in the pipeline, because each agent needed a different one."

### Decision table (the framework)

| Technique | What | Reach for it when | Cost |
|---|---|---|---|
| Zero-shot | instructions only | simple, well-specified tasks | cheapest |
| Few-shot | add worked examples | "good" is subjective — calibrate to YOUR bar | + input tokens |
| CoT | force step-by-step reasoning | multi-step tasks where jumping to answers fails | + output tokens |
| Self-consistency | sample N times, majority vote | one pass is noisy, need stability | N× |
| ToT | explore branches, judge picks | structure matters; one path = local optimum | N× |
| ReAct | interleave reasoning + tool calls | agent must ACT (search, APIs), not just think | loop cost |
| Reflexion | reflect on failure, memo, retry | iterative loops where retries must LEARN | + 1 call |

### In my project (lead with these)

- **Zero-shot** — supervisor routing, query expansion, compression,
  faithfulness judge, RAPTOR summaries. *Default; add complexity only when it
  underperforms.*
- **Few-shot** — the critic: five labeled examples define the severity ladder
  (fabricated stat = critical; over-broad claim = moderate; clean = approve),
  so the judge grades against anchors, not vibes.
- **CoT** — the synthesizer's mandatory 6 steps (understand → inventory facts
  with citations → conflicts → patterns → **gaps** → structure) and the
  extractor's fact extraction. Inventory = traceable claims; gaps = a legal
  way to say "sources don't cover this" — the anti-hallucination move.
- **ReAct** — the fetcher: Thought → Action (search/download) → Observation →
  Thought. The core agentic pattern for using tools/live data.
- **ToT** — synthesizer on complex queries: 3 parallel drafts
  (chronological/thematic/conflict-first) + a cheap judge vote. Gated to
  complex query types so I don't pay N× every time.
- **Self-consistency** — 3× "uses only context facts? YES/NO" on the cheap
  model, majority vote. Cheap insurance against single-sample noise.
- **Reflexion** — critic writes 3 bullets of specific mistakes; injected into
  the retry. *The retry learns, or it's a slot machine.*
- **Role prompting** — "You are the supervisor / a research fact-checker."
  Cheap; sets behavior and vocabulary.
- **Structured output** — "Return JSON only: {...}" + regex-extract +
  fallback so malformed output degrades instead of crashing. (Modern API
  alternative: native structured-output mode.)
- **Delimiters/tags** — `<reasoning>`, `<previous_failure_memory>`:
  unambiguous sections, less injection surface.
- **Negative constraints** — "NEVER fill gaps yourself", "only cite provided
  papers", "never repeat a search query": close known failure modes.
- **Prompt chaining** — the whole pipeline: small focused prompts feeding
  each other beat one mega-prompt for debuggability, testability, cost.

### Not in my project (know for completeness)

**One-shot** (single example); **zero-shot CoT** ("Let's think step by step" —
mine is the explicit structured version); **few-shot CoT** (examples include
their reasoning); **least-to-most** (decompose into ordered sub-problems —
my architecture does this at the pipeline level); **step-back** (general
question first, then specific); **generated-knowledge** (generate facts, then
answer — HyDE is a retrieval cousin); **assistant prefill** (seeding the
reply — worth knowing it's deprecated on newer models in favor of structured
outputs).

### Meta-questions consultants ask

- **"How do you know a prompt change is better?"** — "Prompts are code; they
  regress like code. Every change runs against a golden dataset offline
  (RAGAS) with a regression gate; runtime signals (faithfulness,
  human-override rate) confirm. No prompt ships on vibes — it ships on an
  eval delta."
- **"Reliability/maintainability?"** — prompts as named module-level
  constants (findable, diffable, evaluable); JSON outputs with parse
  fallbacks; temperature 0 for anything classification-like.
- **"Cost/latency in a prompt-heavy system?"** — model routing first; context
  discipline (compress before the expensive model — cost AND
  lost-in-the-middle); caching; gate the N× techniques.
- **"Prompt injection / bad output?"** — delimiters + role separation, JSON
  parsing with fallbacks, hard caps on every loop, human checkpoints.
  *Degrade, don't crash.*

---

# PART 4 — Production Scenario Q&A (consulting style)

### The framework — walk it aloud every time

> **1. Clarify** (one sharp question) → **2. Diagnose** (localize with data)
> → **3. Options** (2–3, with tradeoffs) → **4. Recommend** (pick, say why)
> → **5. Measure & prevent** (verify; stop recurrence).
> Always name the business metric: latency SLA, cost/query, accuracy,
> override rate. And: *latency, cost, quality are a triangle — ask which one
> the business will trade.*

*Reusable opener:* "Before fixing anything I'd find *where* the problem is —
I instrument per-agent latency and token cost, so I start from data, not
guesses."

### 1 · "An agent is stuck / a query takes forever"
Clarify: one query or all? hung or slow? Diagnose per-agent latency. Usual
suspects: hung external call with no timeout; a non-terminating loop; huge
retrieval; or it's not stuck — it's *paused* for a human (my checkpointing
makes those distinguishable). Fixes: timeouts on every external call; hard
caps on every loop (ReAct 8, revisions 2, graph recursion limit); circuit
breaker for provider throttling. *Every wait bounded, every loop capped — one
slow request degrades, never hangs the system.*

### 2 · "1 million documents arrive. What breaks?"
Be honest about limits: my BM25 index is rebuilt in memory per query (fatal
at 1M); local single-node Chroma won't serve 1M at low latency; corpus
embedding must be offline. Evolution: **decouple ingestion from serving**
(offline batch chunk+embed, ~50% cheaper via batch API) → managed distributed
vector DB (OpenSearch/pgvector/Qdrant) with HNSW → persistent sparse index in
OpenSearch → **metadata pre-filtering** before vector search → keep the
two-stage retrieve-then-rerank pattern. *Retrieval latency stays flat whether
it's a thousand docs or a million.*

### 3 · "60s latency; business needs <10s"
Dominated by sequential model calls. Levers, biggest first: **streaming** (SSE
exists — perceived latency drops immediately); **semantic cache** (repeats
~instant); **cut calls** — my LLM supervisor is deterministic rules, replacing
it with code removes ~6 round-trips (flag the tradeoff); route more to the
fast model; parallelize independent work (already: ToT drafts, compressions);
compress harder (shorter prompts = faster). Verify quality held with the eval
gate.

### 4 · "The model bill tripled"
Per-agent token tracking finds the offender fast. Levers: model routing
(structural), three cache layers, compression, batch API for offline work,
prompt caching for static prefixes, gate ToT/self-consistency. Business
framing: quantify cost/query and tie to ROI — 15¢ replacing an hour of
analyst time is a bargain; cached for 100 users it's ~free.

### 5 · "A user reports a hallucinated answer"
Key split: right evidence **not retrieved** (retrieval problem → recall, HyDE,
expansion) vs **retrieved but embellished** (synthesis/critic problem). I can
tell which — every session is checkpointed, so I pull the exact chunks and
faithfulness score it shipped with. Then: tune the failing layer, and add the
case to the golden dataset. *A production failure becomes a permanent test.*

### 6 · "Quality dropped after a prompt change — catch it BEFORE shipping?"
"Prompts are code — same gate as code": RAGAS on the golden set in CI blocks
the deploy on a >0.03 drop; canary/A-B risky changes and watch faithfulness +
override rate. Golden set grows from real high-quality production answers so
the benchmark tracks reality.

### 7 · "Bedrock has an outage / a dependency dies"
Know what degrades vs fails: provider down → circuit breaker + optional
fallback (the single LLM-client choke point is why all six agents get this
free); Redis down → cache fail-open to a miss (slower, pricier, alive);
vector DB down → graceful "can't answer now," never a hallucinated answer;
**Postgres down → the serious one** — it holds durable HITL state, so it's
the dependency to run multi-AZ. *Degrade, don't crash — and know which single
dependency is load-bearing.*

### 8 · "Traffic spikes 10×"
API is stateless (state in shared Postgres/Redis) → scale horizontally, more
containers not bigger ones (1 worker/container by design — token tracking and
local vector store aren't multi-process-safe). Real ceiling = provider
**tokens-per-minute quota** and Postgres — so capacity planning is about TPM
quotas + queuing with backpressure + circuit breaker shedding load
gracefully, not web servers.

### 9 · "Humans-in-the-loop are the bottleneck"
Make the human selective, not mandatory: route on confidence — auto-approve
high-faithfulness answers, escalate only the risky slice. Durable checkpoints
mean queued sessions wait safely for hours. *Spend human attention only where
it changes the outcome.*

### 10 · "How do you measure success in production?"
Three tiers — Quality: runtime faithfulness, offline RAGAS, human-override
rate (rising = early drift warning). Efficiency: cost/query, cache-hit rate,
p95 latency. Business: time saved, answer acceptance. *"Faithfulness,
cost-per-query, p95, and override-rate on one dashboard tell me if the system
is healthy, expensive, slow, or drifting."*

### Meta-points to land in ANY scenario
1. Diagnose with data before fixing. 2. Name the triangle (latency/cost/
quality) and ask what the business trades. 3. Degrade, don't crash.
4. Failures become tests. 5. Know your one load-bearing dependency (here:
Postgres).

---

# PART 5 — Pocket One-Liners

- **async** → "one waiter juggling many tables — helps across users, not within one request."
- **uvicorn/FastAPI/endpoint** → "engine / app / door: uvicorn receives and runs the loop, FastAPI routes, the endpoint runs."
- **session ID** → "issued once at start, carried by the client, and it's the name of the save file in Postgres."
- **LangGraph + Postgres** → "a project manager with an auto-saving save file."
- **HyDE** → "search for what the answer looks like, not the question — the fake answer is a probe, discarded after embedding."
- **hybrid** → "dense catches paraphrase, BM25 catches exact strings — opposite blind spots."
- **RRF** → "fuse ranks, not scores — agreement between two different judges wins."
- **RAPTOR** → "pre-compute the zoom-out at index time."
- **parent expansion** → "search with a scalpel, read with a wide lens."
- **cross-encoder** → "fast skim shortlists; slow careful read ranks the shortlist."
- **gaps step** → "a legal way to say 'the sources don't say' — exactly where hallucination otherwise happens."
- **Reflexion** → "the retry learns, or it's a slot machine."
- **revision cap** → "an LLM judge can always find one more issue — every loop has a ceiling."
- **evaluation** → "runtime judge gates each answer; RAGAS regression gates each deploy. Prompts are code — they regress like code."
- **reliability** → "degrade, don't crash."
- **the theme** → "engineering trust around an LLM."
