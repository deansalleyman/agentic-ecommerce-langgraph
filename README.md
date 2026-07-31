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
Once the server is running, you can call the graph endpoint with a JSON body:

```bash
curl -X POST "http://127.0.0.1:8000/run-graph" \
  -H "Content-Type: application/json" \
  -d '{"text": "Calculate the cargo weight for 10 boxes at 3.5 kg each"}'
```

The response includes:
- `summary`: a short summary of the run
- `original_query`: the original input text
- `react.steps`: the ReAct-style execution trace
- `react.final_answer`: the final answer produced by the model

You can also test the route directly in the browser by opening:

```text
http://127.0.0.1:8000/docs
```

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
curl -X POST "http://127.0.0.1:8000/run-graph" \
  -H "Content-Type: application/json" \
  -d '{"text": "Calculate the cargo weight for 10 boxes at 3.5 kg each"}'
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

- **Each API request is its own standalone trace.** The graph is compiled without a checkpointer
  and the routes call `graph.invoke(...)` without a thread id, so runs are not grouped into
  conversations. Nothing carries over between requests.
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
with a different input. Unlike the FastAPI path, Studio runs *do* persist as threads — the dev
server supplies its own in-memory checkpointer, which is why `workflow.compile()` in `app/main.py`
deliberately takes no checkpointer argument.

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
