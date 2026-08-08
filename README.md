## Song Artist Promotion App

An assistant for It's The Label, an independent record label. It serves two kinds of
visitor: **fans**, whose taste it draws out before searching the song catalog and narrowing
the options down with them, and **artists**, whose profile it builds up in conversation and
then publishes to the label's backend once they confirm. It remembers who someone is and
what they have ruled out, across conversations.

| Module | Role |
| --- | --- |
| [app/schemas.py](app/schemas.py) | The router struct and the memory records |
| [app/catalog.py](app/catalog.py) | Song catalog + the `search_songs` / `get_song` tools |
| [app/memory.py](app/memory.py) | trustcall extractors, change reporting, prompt rendering |
| [app/graph.py](app/graph.py) | The graph: agent, tools, memory nodes, publish gate |
| [app/graphql_backend_client.py](app/graphql_backend_client.py) | CRUD tools against the label's GraphQL backend |
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
| `SUMMARIZE_AFTER_MESSAGES` | no | Transcript length that triggers compaction, defaults to `12`. |
| `SUMMARY_KEEP_MESSAGES` | no | Recent messages kept when compacting, defaults to `4`. |
| `LANGSMITH_TRACING` | for tracing | Set to `true` to send traces to LangSmith. |
| `LANGSMITH_API_KEY` | for tracing / Studio | From https://smith.langchain.com → Settings → API Keys. |
| `LANGSMITH_PROJECT` | no | Project the traces land in, defaults to `default`. |
| `LANGSMITH_ENDPOINT` | no | LangSmith API host. |
| `ITL_GRAPHQL_URL` | no | The label's GraphQL backend, defaults to `https://itsthelabel.com/graphql`. |
| `ITL_ID_TOKEN` | no | Fallback Cognito **ID** token for scripts and Studio. The UI's per-run token wins over it — see below. |
| `ITL_AUTH_TYPE` | no | Value for the backend's mandatory `?type=` parameter when signed in, defaults to `artist`. |

`.env` is gitignored. Real environment variables take precedence over it.

### Signing backend calls

The label's backend gates publishing a song behind a signed-in artist, so the run has to
carry that user's Cognito **ID** token (`token_use: "id"` — not the access token). The UI
holds it and posts it with each run:

```bash
curl -X POST localhost:8000/threads -d '{"user_id": "dean", "id_token": "<id-token>"}'
curl -X POST localhost:8000/threads/$THREAD/runs -d '{"input": "publish my new single", "id_token": "<fresh-id-token>"}'
```

`id_token` is optional in both places — browsing the label as a fan needs no account. On a
run it replaces whatever the thread was holding, which is how a conversation outlives the
token's roughly one-hour life; omitting it keeps the previous one rather than signing the
user out mid-chat.

Resolution order is per-run token → Studio's `id_token` context field → `ITL_ID_TOKEN`. The
token is never a tool argument the model can see or invent: it travels on the run config, and
tools read it via `id_token_from_config` the same way they read `user_id`. It rides under a
key prefixed `__`, which is what keeps it out of LangSmith traces and checkpoint metadata —
see the comment in `app/identity.py` before renaming it.

### Local Development
Run the FastAPI app locally:

```bash
uv run fastapi dev
```

### Calling the API

The API follows the LangGraph Platform resource shape: create a thread, then post runs to it.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/threads` | Create a thread for a user, returns `{"thread_id", "user_id"}` |
| `POST` | `/threads/{thread_id}/runs` | Run the graph on that thread, streams SSE |
| `GET` | `/threads/{thread_id}/state` | Non-streaming snapshot of the thread |
| `GET` | `/users/{user_id}/memory` | Everything remembered about a user |
| `DELETE` | `/users/{user_id}/memory` | Forget a user |

**1. Create a thread**

Threads belong to a user. Memory is keyed by `user_id`, not by thread, so the same
user picks up where they left off in a new conversation.

```bash
THREAD_ID=$(curl -sX POST http://127.0.0.1:8000/threads \
  -H "Content-Type: application/json" \
  -d '{"user_id": "dean"}' | jq -r .thread_id)
```

**2. Start a run**

```bash
curl -N -X POST "http://127.0.0.1:8000/threads/$THREAD_ID/runs" \
  -H "Content-Type: application/json" \
  -d '{"input": "I want something quiet I can work to. I like shoegaze, nothing with shouty vocals."}'
```

The response is a Server-Sent Events stream of five event types:

- `run_started` — echoes the thread id
- `message` — one per message a node produced, with `node`, `type`, `content` and any `tool_calls`
- `memory` — a record was written: `record`, the agent's `reason`, and `changes`, a list of
  the individual edits applied (see [Memory](#memory) below)
- `compacted` — older turns were summarised and dropped from the transcript, with `dropped`
  giving the count (see [Compaction](#compaction) below)
- `interrupt` — the graph paused for human input (see below); no further events until you resume
- `done` — carries the full `react` trace: `react.steps` and `react.final_answer`

Posting another `{"input": ...}` to the same thread continues the conversation — the checkpointer
keeps the message history, so follow-ups like *"the Marisol one has vocals, skip it"* work.

**3. Resume a paused run**

The graph pauses when an artist profile is complete and the label's backend has not seen it
yet — creating an artist is a public write with no delete, so it never happens on the model's
say-so. The payload shown is built from the stored record:

```json
{ "question": "Publish 'The Quiet Hours' to the label? This creates a public artist profile, ...",
  "artist": {"name": "The Quiet Hours", "genre": "shoegaze", "bio": null, "image_url": null},
  "incomplete": ["bio", "image_url"] }
```

`incomplete` is advisory — the backend only requires a name — but it is worth a last look
before something public goes out half-filled. Resume with exactly one of three resolutions:

```bash
curl -N -X POST "http://127.0.0.1:8000/threads/$THREAD_ID/runs" \
  -H "Content-Type: application/json" \
  -d '{"command": {"resume": {"amend": {"bio": "Four people and one amp"}}}}'
```

| Resume value | Effect |
| --- | --- |
| `{"confirm": true}` | Send it. The returned id and slug are written to the record in code |
| `{"amend": {"genre": ..., "bio": ...}}` | Edit the profile and show it again; only backend-writable fields are accepted |
| `{"cancel": true}` | Leave it as a draft |

After an amend the profile is **re-shown** rather than silently sent. Anything else re-prompts
with an `error` field, and the thread stays paused until it gets a usable answer. If the
backend rejects the write, the ids stay unset and the gate offers again next turn.

**4. Inspect a thread, or a user**

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

Four records persist per user, keyed by `user_id` and living beyond any single thread:

| Record | Store namespace | Holds |
| --- | --- | --- |
| `UserProfile` | `("profile", user_id)` | Who they are as a listener: genres, artists they follow, what they have ruled out, how they listen |
| `ArtistProfile` | `("artist", user_id)` | Their own artist identity if they have one: name, genre, bio, image, influences, albums, songs, tour dates |
| `SongCollection` | `("found_music", user_id)` | Every song put in front of them, each with why it was offered and what became of it |
| `SongForecast` | `("forecast", user_id)` | Open-Meteo weather for where they are listening. Written by the weather tool in code, never by an extractor |

`SongCollection` is the collection that gets narrowed. Each song carries a `status`
(`suggested` / `liked` / `dismissed`), a `why_found`, and — once ruled out — a
`dismissed_reason`. Dismissed songs are never deleted: why something was rejected is the
useful part of the record, and stops the assistant offering it again.

`ArtistProfile` does double duty. Its first four fields are exactly the backend's
`CreateArtistInput`, so publishing is a projection (`to_create_input()`) rather than a
translation, and the fields below them are colour the label's API has no column for. Its last
two — `backend_artist_id` and `artist_slug` — are the only fields written in code after the
backend accepts the profile, and `update_artist` carries them over from the stored record
rather than trusting the extractor's output. The instruction tells the model not to invent
them, but an id is the one field where a hallucination gets acted on as real.

### Schemas per CRUD operation

The backend's write operations each have their own payload schema in
[app/schemas.py](app/schemas.py), kept separate from the memory records so that what we
*remember* about an artist and what we are *entitled to send* are never the same object:

| Schema | Backend operation |
| --- | --- |
| `CreateArtistInput` | `artists.createArtist` |
| `UploadSongInput` | `songs.getUploadUrl` |
| `CreateSongInput` | `songs.createSong` |

There is deliberately no password field anywhere. Records here are filled by an extraction
model reading the transcript and are served verbatim by `GET /users/{id}/memory`, so a
password field would be a standing instruction to copy a secret into a readable store.

### Two ways a record gets written

**Conversational facts are extracted.** What the user says has to be interpreted, so the
agent asks for a write and a model does the interpreting — the router path below.

**Data from a tool is written in code.** `SongForecast` arrives already structured from
Open-Meteo, so [weather_forcast_tool](app/weather_forcast.py) writes it to the store itself
via `InjectedStore` and returns it for the agent to talk about. No router call, no extractor,
no extra model turn — the write cannot be skipped or garbled. The same rule covers the
backend ids on `ArtistProfile`, which the publish gate writes after `createArtist` returns.

That is also why the forecast is its own record under `("forecast", user_id)` rather than a
field on another record. An extractor is handed the whole document as `existing` on every
update, so a field living there would be fair game to rewrite or drop on an unrelated turn.
Two writers on one field will eventually disagree, and the model writes last.

**Past the forecast horizon it falls back to history.** Open-Meteo needs no API key and its
forecast reaches about two weeks. Further out — which is exactly when someone is buying a
record — the tool pulls the *same calendar month a year earlier* from the historical
archive and stores its monthly average as typical conditions:

```
TYPICAL CONDITIONS FOR THE TRIP (PAST WEATHER, NOT A FORECAST):
- NOT A FORECAST. No forecast reaches 2026-12-01, so this is what Glasgow was
  actually like in December 2025 (31 days on record).
- Typical night 39.1F, typical day 53.7F
- Coldest night that month 23.5F, warmest day 65.3F — size warmth against the cold end
- 3.51 in of precipitation over the month
```

Which one you got is recorded as `basis: "forecast" | "historical"` on the record itself, not
just in prompt wording, so a stored value can never be read back as a prediction it never
was. The heading, the first line and the prompt rule all restate it.

Three details worth knowing: the archive has no chance-of-rain column (ERA5 does not model
one), so historical days carry measured precipitation only; a trip more than a year out steps
the sampled year back until it lands on a month that has actually finished; and the coldest
night on record is reported alongside the average, because a mean hides the night that
decides the bag.

### How a router-driven record gets written

The agent decides. It has a **router struct**, `MemoryRouter`, bound as a tool but never
executed as one — a conditional edge reads `update_type` off the call and sends the run to
the matching node:

```
                ┌ MemoryRouter(profile)     → update_profile     ────────┐
                ├ MemoryRouter(artist)      → update_artist → publish    ┤
START → agent ──┼ MemoryRouter(found_music) → update_found_music ────────┼→ agent
                ├ catalog / weather / backend tool → tools ──────────────┘
                └ no tool call ────────────────────────────────────────→ END
```

Each update node hands the conversation to a [trustcall](https://github.com/hinthornw/trustcall)
extractor. trustcall asks the model for a **JSON patch** against the existing record and
validates it against the schema, rather than asking it to re-emit the whole document. That
distinction is the point: when a user says *"the Marisol one has vocals, I can't work to
words"*, only that one song changes —

```
replace /songs/1/status = dismissed
add     /songs/1/dismissed_reason = The Marisol one has vocals, I can't work to words
```

— while the other songs keep their `why_found` text verbatim. Asking a model to reproduce a
large nested document is where detail quietly goes missing.

### The publish gate

`update_artist` is the one memory node that does not return straight to the agent. It routes
through `publish_review`, which interrupts before anything reaches the label's backend:

```
POST /threads/{id}/runs  {"command": {"resume": {"confirm": true}}}
                         {"command": {"resume": {"amend": {"bio": "Four people, one amp"}}}}
                         {"command": {"resume": {"cancel": true}}}
```

Creating an artist is a public write the backend offers no way to undo, so it never happens
on the model's say-so — `create_artist` is deliberately **not** in `backend_tools`, which is
what stops the agent routing around the gate. The payload shown to the user is built from the
stored record, not from anything the model composed that turn. An `amend` edits the record and
re-prompts; anything unrecognised re-prompts with an `error`, so the thread stays paused until
it gets a usable answer.

Resuming replays the node from the top, so only reads and idempotent amendments sit above the
`interrupt()` — the write itself is below the loop and runs once. If it fails, the backend ids
stay unset and the gate offers again next turn rather than leaving the user believing they are
on the roster.

### Knowing what changed

Each extractor runs with a `Spy` listener attached (`with_listeners(on_end=...)`). trustcall
reaches the model through nested runs, so the patches are not in the return value; the spy
walks the run tree afterwards and recovers them, including trustcall's `planned_edits` — the
model's own sentence describing what it meant to change.

Those lines go two places: into the `ToolMessage` the agent sees, so its reply can tell the
user specifically what was recorded, and into the `memory` SSE event for the UI. If the
spy comes back empty, a plain before/after diff of the stored record is used instead, so the
user is never told "updated" with no detail.

Setting `MemoryRouter` aside, nothing is extracted on turns where the agent does not ask for
it, so an ordinary chat turn costs no extra model call.

Threads live in memory (`InMemorySaver`) and are lost when the server restarts.

## Compaction

The whole transcript is re-sent to the model on every agent turn, so a long shopping
conversation gets steadily more expensive. Once it passes `SUMMARIZE_AFTER_MESSAGES`, the
`summarize_conversation` node folds the older turns into `state["summary"]`, drops them, and
that summary is prepended to later turns instead. The client sees a `compacted` event.

Two things make it safe rather than merely cheap:

**It runs only where a turn would otherwise end.** Compaction is checked *after* the router
and tool branches in `should_continue`. Intercepting a pending tool call to summarise
instead would leave an AI message whose tool call never gets a `ToolMessage`, and the next
model call fails on the malformed history.

**It cuts on a safe boundary.** `_safe_cut` moves the cut forward to a human turn or a plain
AI reply, never between a tool request and its result — either half surviving alone is
invalid. If that leaves nothing to drop, the node returns without calling the model at all,
rather than paying for a summary that removes zero messages.

Long-term memory is unaffected: profile, artist and found music live in the store, not the
transcript, so compaction never loses what was chosen or ruled out.

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
  -d '{"input": "Calculate the cargo weight for 10 boxes at 7.5 lb each"}'
```

Then open https://smith.langchain.com, find the `agentic-ecommerce-langgraph` project in the
tracing projects list, and open the newest run. You get a root run for the graph with a child

to run local LangSmith accesible 
```
uv run langgraph dev --no-browser
```

span per step:

- `agent` → the `ChatXAI` call, with the full message list sent to Grok — including the
  rendered memory prompt — the raw reply, latency and token usage. Requested `tool_calls`
  appear here, whether for a catalog tool or the `MemoryRouter`.
- `tools` → the `search_songs` / `get_song` / backend tool invocation, with
  the arguments Grok chose and the value returned.
- `update_profile` / `update_artist` / `update_found_music` → the trustcall extractor. Expand it
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
I want something quiet I can work to. I like shoegaze.
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
from one path means nothing to the other — and so are their stores, which is why deleting a
user through the API does not clear what Studio remembers.

### Starting from a clean slate

A new thread is **not** a fresh user. Threads hold the transcript; profile, artist and found
needs are keyed by `user_id` in the store and outlive any thread by design. Switch user,
and everything is new:

| | |
| --- | --- |
| In Studio | Open the run's **Configurable** panel and set `user_id`. The graph declares a context schema (`AssistantContext`), so the field is offered there. |
| For a whole session | `DEFAULT_USER_ID=test-2 uv run langgraph dev` — runs with no `user_id` fall under that name instead of `anonymous`. |
| Through the API | `curl -X DELETE http://127.0.0.1:8000/users/dean/memory`, or just post threads under a new `user_id`. |
| Wipe Studio entirely | Stop the server and `rm -rf .langgraph_api/` — its threads *and* store are pickled there, which is why restarting alone does not reset anything. |

`user_id` is read from `configurable` first, then the run context, then `DEFAULT_USER_ID`,
so the API and Studio can each supply it their own way.

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
