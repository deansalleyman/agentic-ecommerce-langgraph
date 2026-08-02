## Agentic Ecommerce LangGraph

### Setup
Copy the example environment file and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `XAI_API_KEY` | yes | Authenticates the Grok model. The graph raises at runtime without it. |
| `XAI_MODEL` | no | Model name, defaults to `grok-4`. |
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
| `POST` | `/threads` | Create a thread, returns `{"thread_id": ...}` |
| `POST` | `/threads/{thread_id}/runs` | Run the graph on that thread, streams SSE |
| `GET` | `/threads/{thread_id}/state` | Non-streaming snapshot of the thread |

**1. Create a thread**

```bash
THREAD_ID=$(curl -sX POST http://127.0.0.1:8000/threads | jq -r .thread_id)
```

**2. Start a run**

```bash
curl -N -X POST "http://127.0.0.1:8000/threads/$THREAD_ID/runs" \
  -H "Content-Type: application/json" \
  -d '{"input": "Calculate the cargo weight for 10 boxes at 3.5 kg each"}'
```

The response is a Server-Sent Events stream of four event types:

- `run_started` — echoes the thread id
- `message` — one per message a node produced, with `node`, `type`, `content` and any `tool_calls`
- `interrupt` — the graph paused for human input (see below); no further events until you resume
- `done` — carries the full `react` trace: `react.steps` and `react.final_answer`

Posting another `{"input": ...}` to the same thread continues the conversation — the checkpointer
keeps the message history, so follow-ups like *"what was that in pounds?"* work.

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

**4. Inspect a thread**

```bash
curl "http://127.0.0.1:8000/threads/$THREAD_ID/state"
```

Returns `status` (`idle` or `interrupted`), any pending `interrupts`, `summary`, `original_query`
and the same `react` trace. Browser-friendly, as is the interactive documentation at
`http://127.0.0.1:8000/docs`.

**Error responses**

| Status | Meaning |
| --- | --- |
| `404` | Unknown `thread_id` — create it with `POST /threads` first |
| `409` | Wrong mode for the thread's current state: a `command` sent to a thread with no pending interrupt, or an `input` sent to a paused one. The body of the paused case includes the pending `interrupts` |
| `422` | `input` was empty |

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

- `agent` → the `ChatXAI` call, with the full message list sent to Grok, the raw reply, latency
  and token usage. When Grok decides to use the tool, the requested `tool_calls` appear here.
- `tools` → the `calculate_cargo_weight` invocation, with the arguments Grok chose
  (`item_count`, `unit_weight`) and the value returned.
- `agent` again → the follow-up call where Grok turns the tool result into a final answer.

That `agent → tools → agent` loop is the whole point of tracing here: when an answer is wrong,
it shows you whether Grok picked the wrong tool arguments, or got good arguments and then
misread the result.

A few things worth knowing about this project specifically:

- **Each run is its own trace, but runs share a thread.** The graph is compiled with an
  `InMemorySaver` and every route passes a `thread_id`, so message history carries across runs on
  the same thread — a resumed interrupt continues the conversation it paused.
- **Failed runs are traced too.** If `XAI_API_KEY` is missing or the model errors, the run shows
  up in red with the exception attached — often faster to read than the API response.
- **Import order matters.** `app/main.py` loads `.env` *before* importing langchain/langgraph,
  because the LangSmith SDK caches env-var reads on first access. If you rearrange the top of
  that file and tracing goes quiet, this is why.

### 3. LangGraph Studio

`langgraph.json` exposes the compiled graph (`./app/main.py:graph`) to the LangGraph CLI under
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
other `AgentState` fields (`current_topic`, `summary`, `original_query`, `input`, `output`) are
all read with `.get()` and safely default to empty. A good first input is a single human message:

```text
Calculate the cargo weight for 10 boxes at 3.5 kg each
```

From there you can watch the graph light up node by node, click any node to inspect the state
going in and out, edit that state and continue, or fork a thread to re-run from an earlier step
with a different input. Studio threads are separate from the FastAPI ones: the dev server supplies
its own persistence layer and overrides the `InMemorySaver` compiled into `app/main.py`, so a
thread id from one path means nothing to the other.

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
