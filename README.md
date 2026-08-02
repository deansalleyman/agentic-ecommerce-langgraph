## Agentic Ecommerce LangGraph

A shopping assistant for an online camping store. It asks about the trip, searches the
store catalog, and narrows the options down with the customer — remembering who they are
and what they have ruled out, across conversations.

| Module | Role |
| --- | --- |
| [app/schemas.py](app/schemas.py) | The router struct and the memory records |
| [app/catalog.py](app/catalog.py) | Catalog + the `search_products` / `get_product` tools |
| [app/memory.py](app/memory.py) | trustcall extractors, change reporting, prompt rendering |
| [app/graph.py](app/graph.py) | The graph: agent, tools, memory nodes, weight review |
| [app/main.py](app/main.py) | FastAPI routes |

### Setup
Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `XAI_API_KEY` | yes | Authenticates the Grok model. The graph raises at runtime without it. |
| `XAI_MODEL` | no | Model name, defaults to `grok-4`. |
| `MAX_CARGO_WEIGHT_KG` | no | Weight above which a run pauses for human review, defaults to `100`. |
| `LANGSMITH_TRACING` | for tracing | Set to `true` to send traces to LangSmith. |
| `LANGSMITH_API_KEY` | for tracing / Studio | From https://smith.langchain.com → Settings → API Keys. |
| `LANGSMITH_PROJECT` | no | Project the traces land in, defaults to `default`. |
| `LANGSMITH_ENDPOINT` | no | LangSmith API host. |

`.env` is gitignored. Real environment variables take precedence over it.

### Local Development
Run the FastAPI app locally:

```bash
uv run fastapi dev
```

### Calling the API

The API follows the LangGraph Platform resource shape: create a thread, then post runs to it.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/threads` | Create a thread for a customer, returns `{"thread_id", "user_id"}` |
| `POST` | `/threads/{thread_id}/runs` | Run the graph on that thread, streams SSE |
| `GET` | `/threads/{thread_id}/state` | Non-streaming snapshot of the thread |
| `GET` | `/users/{user_id}/memory` | Everything remembered about a customer |
| `DELETE` | `/users/{user_id}/memory` | Forget a customer |

**1. Create a thread**

Threads belong to a customer. Memory is keyed by `user_id`, not by thread, so the same
customer picks up where they left off in a new conversation.

```bash
THREAD_ID=$(curl -sX POST http://127.0.0.1:8000/threads \
  -H "Content-Type: application/json" \
  -d '{"user_id": "dean"}' | jq -r .thread_id)
```

**2. Start a run**

```bash
curl -N -X POST "http://127.0.0.1:8000/threads/$THREAD_ID/runs" \
  -H "Content-Type: application/json" \
  -d '{"input": "I am off to Big Bear, California for 3 nights in October, two of us, backpacking."}'
```

The response is a Server-Sent Events stream of five event types:

- `run_started` — echoes the thread id
- `message` — one per message a node produced, with `node`, `type`, `content` and any `tool_calls`
- `memory` — a record was written: `record`, the agent's `reason`, and `changes`, a list of
  the individual edits applied (see [Memory](#memory) below)
- `interrupt` — the graph paused for human input (see below); no further events until you resume
- `done` — carries the full `react` trace: `react.steps` and `react.final_answer`

Posting another `{"input": ...}` to the same thread continues the conversation — the checkpointer
keeps the message history, so follow-ups like *"the MSR is too pricey, drop it"* work.

**3. Resume a paused run**

When the calculated weight exceeds `MAX_CARGO_WEIGHT_KG` (default 100), the stream ends on an
`interrupt` event asking for adjusted values. Send them back as a command:

```bash
curl -N -X POST "http://127.0.0.1:8000/threads/$THREAD_ID/runs" \
  -H "Content-Type: application/json" \
  -d '{"command": {"resume": {"item_count": 4, "unit_weight": 10}}}'
```

Values that are missing, non-numeric, or not greater than zero do **not** resume the graph. The
node re-prompts: you get another `interrupt` event, this time with an `error` field explaining
what was wrong, and the thread stays paused until usable numbers arrive.

**4. Inspect a thread, or a customer**

```bash
curl "http://127.0.0.1:8000/threads/$THREAD_ID/state"   # this conversation
curl "http://127.0.0.1:8000/users/dean/memory"          # what is remembered about them
```

The thread endpoint returns `status` (`idle` or `interrupted`), any pending `interrupts`,
`summary`, `original_query` and the same `react` trace. Both are browser-friendly, as is the
interactive documentation at `http://127.0.0.1:8000/docs`.

**Error responses**

| Status | Meaning |
| --- | --- |
| `404` | Unknown `thread_id` — create it with `POST /threads` first |
| `409` | Wrong mode for the thread's current state: a `command` sent to a thread with no pending interrupt, or an `input` sent to a paused one. The body of the paused case includes the pending `interrupts` |
| `422` | `input` was empty |

## Memory

Two things persist per customer, keyed by `user_id` and living beyond any single thread:

| Record | Store namespace | Holds |
| --- | --- | --- |
| `UserProfile` | `("profile", user_id)` | Activities they do, experience, gear owned, budget, brands, constraints like *sleeps cold* |
| `TripPlan` | `("trip", user_id)` | Destination, dates, nights, party size, season, conditions |
| `GearNeed` | `("gear_needs", user_id)` | **One document per thing they are shopping for**, each owning its candidate products |

`GearNeed` is the collection that gets narrowed. Each candidate carries a `status`
(`candidate` / `shortlisted` / `eliminated`), a `fit_reason`, and — once ruled out — an
`eliminated_reason`. Eliminated candidates are never deleted: why an option was rejected is
the useful part of the record, and stops the assistant re-suggesting it.

### How a record gets written

The agent decides. It has a **router struct**, `MemoryRouter`, bound as a tool but never
executed as one — a conditional edge reads `update_type` off the call and sends the run to
the matching node:

```
                ┌ MemoryRouter(profile)    → update_profile    ─┐
                ├ MemoryRouter(trip)       → update_trip       ─┤
START → agent ──┼ MemoryRouter(gear_needs) → update_gear_needs ─┼→ agent
                ├ catalog / cargo tool → tools → weight_review ─┘
                └ no tool call ────────────────────────────────→ END
```

Each update node hands the conversation to a [trustcall](https://github.com/hinthornw/trustcall)
extractor. trustcall asks the model for a **JSON patch** against the existing record and
validates it against the schema, rather than asking it to re-emit the whole document. That
distinction is the point: when a customer says *"the MSR is too pricey, drop it"*, only that
one candidate changes —

```
replace /candidates/3/status = eliminated
add     /candidates/3/eliminated_reason = The MSR is too pricey for me
```

— while the other three candidates keep their `fit_reason` text verbatim. Asking a model to
reproduce a large nested document is where detail quietly goes missing.

### Knowing what changed

Each extractor runs with a `Spy` listener attached (`with_listeners(on_end=...)`). trustcall
reaches the model through nested runs, so the patches are not in the return value; the spy
walks the run tree afterwards and recovers them, including trustcall's `planned_edits` — the
model's own sentence describing what it meant to change.

Those lines go two places: into the `ToolMessage` the agent sees, so its reply can tell the
customer specifically what was recorded, and into the `memory` SSE event for the UI. If the
spy comes back empty, a plain before/after diff of the stored record is used instead, so the
customer is never told "updated" with no detail.

Setting `MemoryRouter` aside, nothing is extracted on turns where the agent does not ask for
it, so an ordinary chat turn costs no extra model call.

Threads live in memory (`InMemorySaver`) and are lost when the server restarts.

## Using LangSmith

LangSmith gives you two things on this project:

- **Tracing** — a recorded, inspectable timeline of every graph run (what the model was sent, what
  it replied, which tools it called, how long each step took, token counts).
- **LangGraph Studio** — a visual debugger where you step through the graph node by node, edit
  state mid-run, and re-run from any point.

Both are already wired up. You only need an API key.

### 1. Get a LangSmith API key

1. Sign in at https://smith.langchain.com (the free tier is enough for local development).
2. Open **Settings → API Keys** and create a key. It starts with `lsv2_`.
3. Paste it into `.env`:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...your_key...
LANGSMITH_PROJECT=agentic-ecommerce-langgraph
```

`LANGSMITH_PROJECT` is just a bucket name — LangSmith creates it on the first trace, so it does
not need to exist beforehand. If you leave it unset, traces land in a project called `default`.

To turn tracing off again, set `LANGSMITH_TRACING=false` (or remove it). Nothing else breaks —
the graph runs exactly the same, it just stops reporting.

### 2. Tracing

No code changes are needed. With the variables above set, **every** graph run is traced: the
FastAPI routes, LangGraph Studio, and any script that imports `app.main`.

Start the API and send a request:

```bash
uv run fastapi dev
```

```bash
THREAD_ID=$(curl -sX POST http://127.0.0.1:8000/threads | jq -r .thread_id)
curl -N -X POST "http://127.0.0.1:8000/threads/$THREAD_ID/runs" \
  -H "Content-Type: application/json" \
  -d '{"input": "Calculate the cargo weight for 10 boxes at 3.5 kg each"}'
```

Then open https://smith.langchain.com, find the `agentic-ecommerce-langgraph` project in the
tracing projects list, and open the newest run. You get a root run for the graph with a child
span per step:

- `agent` → the `ChatXAI` call, with the full message list sent to Grok — including the
  rendered memory prompt — the raw reply, latency and token usage. Requested `tool_calls`
  appear here, whether for a catalog tool or the `MemoryRouter`.
- `tools` → the `search_products` / `get_product` / `calculate_cargo_weight` invocation, with
  the arguments Grok chose and the value returned.
- `update_profile` / `update_trip` / `update_gear_needs` → the trustcall extractor. Expand it
  to see the `PatchDoc` call: the `planned_edits` sentence and the individual JSON patches.
  This is the trace to open when a record ends up wrong.
- `agent` again → the follow-up call where Grok turns the result into a reply.

That loop is the whole point of tracing here: when an answer is wrong, it shows you whether
Grok picked the wrong tool arguments, patched the wrong field, or got good data and then
misread it.

A few things worth knowing about this project specifically:

- **Each run is its own trace, but runs share a thread.** The graph is compiled with an
  `InMemorySaver` and every route passes a `thread_id`, so message history carries across runs on
  the same thread — a resumed interrupt continues the conversation it paused.
- **Failed runs are traced too.** If `XAI_API_KEY` is missing or the model errors, the run shows
  up in red with the exception attached — often faster to read than the API response.
- **Memory writes are their own model calls.** A turn that saves a record costs an extra
  extractor call, visible as a separate child span. Turns where the agent does not call
  `MemoryRouter` cost nothing extra.
- **Import order matters.** Both entry points — [app/main.py](app/main.py) and
  [app/graph.py](app/graph.py) — call `load_env_file()` *before* importing
  langchain/langgraph, because the LangSmith SDK caches env-var reads on first access. If you
  rearrange the top of either file and tracing goes quiet, this is why.

### 3. LangGraph Studio

`langgraph.json` exposes the compiled graph (`./app/graph.py:graph`) to the LangGraph CLI under
the graph id `agent`. Start the dev server:

```bash
uv run langgraph dev
```

It serves the graph at `http://127.0.0.1:2024` and opens Studio at
`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`. Studio's UI is hosted by
LangSmith but talks to the server on your own machine — your keys and data stay local. Use
`--no-browser` to skip the auto-open, or `--port <n>` if 2024 is taken.

Check it came up:

```bash
curl http://127.0.0.1:2024/ok           # -> {"ok":true}
```

In Studio, pick the `agent` graph and submit a run. **Filling in `messages` is enough** — the
other `AgentState` fields (`current_topic`, `summary`, `original_query`, `input`, `output`,
`last_memory_update`) are all read with `.get()` and safely default to empty. A good first
input is a single human message:

```text
I'm off to Big Bear for 3 nights in October, two of us, backpacking.
```

From there you can watch the graph light up node by node, click any node to inspect the state
going in and out, edit that state and continue, or fork a thread to re-run from an earlier step
with a different input.

**Why [app/graph.py](app/graph.py) compiles the workflow twice.** The LangGraph API supplies
its own checkpointer and store, and *refuses to load* a graph that brings its own —
`GraphLoadError`, not a silent override. So `graph` (what `langgraph.json` exposes) compiles
with no persistence arguments, while `api_graph` (what FastAPI runs) compiles with the
`InMemorySaver` and `InMemoryStore` it needs to keep threads and memory alive between HTTP
requests. Same `workflow`, same nodes, two runtimes that persist differently.

Consequences worth knowing: Studio threads and FastAPI threads are separate, so a thread id
from one path means nothing to the other. And Studio runs carry no `user_id` in their config,
so memory written from Studio lands under the `anonymous` customer instead of mixing into a
real one.

The dev server is independent of `uv run fastapi dev` (port 8000), so you can run both at once.
Studio writes local state to `.langgraph_api/`, which is gitignored.

### Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| No traces show up in LangSmith | `LANGSMITH_TRACING` must be the exact lowercase string `true` — the SDK does a literal `== "true"` comparison, so `True`, `TRUE` and `1` all silently disable tracing. `LANGSMITH_API_KEY` must also be set. Restart the server after editing `.env`; it is only read at import. |
| `403 Forbidden` from `api.smith.langchain.com` in the logs | The API key is missing, a placeholder, or revoked. Regenerate it under Settings → API Keys. |
| Traces land in `default` instead of your project | `LANGSMITH_PROJECT` is not set in the environment the process actually sees. |
| `RuntimeError: XAI_API_KEY is not set` | Grok's key is missing from `.env`. This is unrelated to LangSmith — the graph refuses to run at all without it. |
| `langgraph dev` fails to bind | Port 2024 is in use; pass `--port 2025`. |
| Env changes seem ignored | The loader uses `os.environ.setdefault`, so a real shell environment variable wins over the `.env` file. Check with `echo $LANGSMITH_PROJECT`. |
