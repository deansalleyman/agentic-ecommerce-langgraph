"""Long-term memory: trustcall extractors, change reporting, and prompt rendering.

Memory is keyed by user rather than thread, so it outlives any one conversation:

    ("profile",     user_id) / "main"    -> UserProfile
    ("artist",      user_id) / "main"    -> ArtistProfile
    ("found_music", user_id) / "current" -> SongCollection
    ("forecast",    user_id) / "current" -> SongForecast

Records are *patched*, not rewritten. trustcall asks the model for a JSON patch against the
existing document and validates it against the schema, so narrowing a shortlist — flipping
one song to dismissed with a reason — leaves every other song untouched. Asking the model to
re-emit the whole document instead loses detail and burns tokens.

Two fields are deliberately out of the extractors' reach. `SongForecast` is written by the
weather tool, and `ArtistProfile.backend_artist_id` / `artist_slug` are written by the
publish gate once the label's backend has accepted the profile. Both arrive already
structured from an API, so passing them through a model could only lose or garble them.
"""

import datetime
import json
from typing import Any, Iterator

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tracers.schemas import Run
from langgraph.store.base import BaseStore
from trustcall import create_extractor

from app.llm import model
from app.schemas import (
    ArtistProfile,
    SongCollection,
    SongForecast,
    StoreRecord,
    UserProfile,
)

PROFILE_KEY = "main"
ARTIST_KEY = "main"
MUSIC_KEY = "current"
FORECAST_KEY = "current"


def _safe_label(user_id: str) -> str:
    """Encode a user id so it is legal as a store namespace label.

    Store namespaces reject periods and empty labels, but the most natural user id is
    an email address. Escaping rather than stripping keeps the mapping one-to-one: '%' is
    escaped first, so 'a.b@x.com' and a literal 'a%2Eb@x%2Ecom' can never collide on the
    same namespace. The user's real id is what the API reports back; this encoding is
    only ever seen inside the store.
    """
    encoded = user_id.strip().replace("%", "%25").replace(".", "%2E")
    return encoded or "anonymous"


def _namespace(record: StoreRecord, user_id: str) -> tuple[str, str]:
    return (record, _safe_label(user_id))


# -------------------------------------------------------------
# Extractors
# -------------------------------------------------------------
# Built once, only when a model is configured. Each user has exactly one profile, one artist
# identity and one collection of songs found for them, so none of these insert — a new song
# is a patch appending to SongCollection.songs, which keeps the dismissed ones alongside it.
if model is not None:
    profile_extractor = create_extractor(
        model, tools=[UserProfile], tool_choice="UserProfile"
    )
    artist_extractor = create_extractor(
        model, tools=[ArtistProfile], tool_choice="ArtistProfile"
    )
    music_extractor = create_extractor(
        model, tools=[SongCollection], tool_choice="SongCollection"
    )
else:  # pragma: no cover - import-time convenience for unconfigured machines
    profile_extractor = artist_extractor = music_extractor = None

TRUSTCALL_INSTRUCTION = """You maintain the label's records about this user.

Update the record from the conversation below. Preserve everything the conversation does \
not contradict — you are amending a record, not rewriting it.

Facts about the user, their own music, and their reactions record only what they actually \
said. A field with no evidence behind it stays null: an empty field is correct, a \
plausible guess is a false record the label will act on later. Do not infer a genre from \
one song they liked, or a dislike from one they skipped without saying why.

Null means null. Do not write an empty string or placeholder text in place of something \
you were not told.

Your own assessment is the exception, and is required rather than optional. Every song \
you record needs a why_found drawn from what the user has said and from the song's own \
tags, e.g. "shoegaze with the quiet vocals they asked for". That is your judgement to \
make, not something the user has to say first.

For the song collection specifically: keep every song that is already recorded, including \
dismissed ones. When the user rules a song out, set its status to "dismissed" and write \
dismissed_reason in their terms. When they like one, set its status to "liked".

For the artist record: never fill in backend_artist_id or artist_slug. Those come back \
from the label's own systems when a profile is published, and a value you invent would be \
taken for a real one."""


# -------------------------------------------------------------
# Spy: what did trustcall actually write?
# -------------------------------------------------------------
class Spy:
    """Collects the tool calls trustcall made, so a turn can report what changed.

    trustcall reaches the model through nested runs, and the patches it applied are not in
    the return value — only the merged result is. Attaching this as an on_end listener lets
    us walk the run tree afterwards and recover the individual edits, including the model's
    own `planned_edits` description of what it intended.
    """

    def __init__(self) -> None:
        self.called_tools: list[dict[str, Any]] = []

    def __call__(self, run: Run) -> None:
        queue: list[Run] = [run]
        while queue:
            current = queue.pop()
            if current.child_runs:
                queue.extend(current.child_runs)
            if current.run_type != "chat_model":
                continue
            self.called_tools.extend(_find_tool_calls(current.outputs))

    def unique_calls(self) -> list[dict[str, Any]]:
        """Tool calls with retries de-duplicated, in the order first seen."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for call in self.called_tools:
            fingerprint = json.dumps(
                [call.get("name"), call.get("args")], sort_keys=True, default=str
            )
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(call)
        return unique


def _find_tool_calls(payload: Any) -> Iterator[dict[str, Any]]:
    """Yield every tool call nested anywhere in a run's output payload.

    Deliberately structural rather than indexing a known path: this reads another library's
    internal run shape, and a reporting detail must never be what breaks a user's turn.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "tool_calls" and isinstance(value, list):
                for call in value:
                    if isinstance(call, dict) and "name" in call:
                        yield call
            else:
                yield from _find_tool_calls(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _find_tool_calls(item)


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, default=str)
        return rendered if len(rendered) <= 120 else rendered[:117] + "..."
    return str(value)


def summarize_spy(spy: Spy, schema_name: str) -> list[str]:
    """Turn captured tool calls into lines a person can read."""
    lines: list[str] = []
    for call in spy.unique_calls():
        name = call.get("name")
        args = call.get("args") or {}

        if name == "PatchDoc":
            edits = [
                f"{patch.get('op')} {patch.get('path')} = {_format_value(patch.get('value'))}"
                for patch in (args.get("patches") or [])
                if isinstance(patch, dict)
            ]
            if not edits:
                continue
            planned = (args.get("planned_edits") or "").strip().replace("\n", " ")
            doc_id = args.get("json_doc_id")
            headline = f"updated {doc_id}" if doc_id else "updated record"
            if planned:
                headline = f"{headline}: {planned}"
            lines.append(f"{headline} [{'; '.join(edits)}]")
        elif name == schema_name:
            # An insert: the whole record is new, so report what it was actually filled with
            # rather than just naming the schema.
            label = args.get("name")
            fields = [
                f"{key} = {_format_value(value)}"
                for key, value in args.items()
                if value not in (None, [], "", {})
            ]
            headline = f"created {schema_name}" + (f" '{label}'" if label else "")
            lines.append(f"{headline} [{'; '.join(fields)}]" if fields else headline)
        # PatchFunctionErrors is trustcall repairing its own invalid output — internal noise.
    return lines


# -------------------------------------------------------------
# Diff backstop
# -------------------------------------------------------------
def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}/{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}/{index}"))
    else:
        flat[prefix or "/"] = value
    return flat


def diff_records(before: dict[str, Any] | None, after: dict[str, Any]) -> list[str]:
    """Field-level diff of a stored record, used when the spy comes back empty.

    The spy produces better prose, but it depends on another library's run internals. This
    depends on nothing, so the user always gets specifics.
    """
    old, new = _flatten(before or {}), _flatten(after)
    changes: list[str] = []
    for path in sorted(set(old) | set(new)):
        was, now = old.get(path), new.get(path)
        if was == now:
            continue
        if was is None or was == [] or was == "":
            changes.append(f"set {path} = {_format_value(now)}")
        elif now is None:
            changes.append(f"cleared {path}")
        else:
            changes.append(f"{path}: {_format_value(was)} -> {_format_value(now)}")
    return changes


# -------------------------------------------------------------
# Reads
# -------------------------------------------------------------
def get_profile(store: BaseStore, user_id: str) -> UserProfile | None:
    item = store.get(_namespace("profile", user_id), PROFILE_KEY)
    return UserProfile.model_validate(item.value) if item else None


def get_artist(store: BaseStore, user_id: str) -> ArtistProfile | None:
    item = store.get(_namespace("artist", user_id), ARTIST_KEY)
    return ArtistProfile.model_validate(item.value) if item else None


def get_found_music(store: BaseStore, user_id: str) -> SongCollection | None:
    item = store.get(_namespace("found_music", user_id), MUSIC_KEY)
    return SongCollection.model_validate(item.value) if item else None


def get_forecast(store: BaseStore, user_id: str) -> SongForecast | None:
    item = store.get(_namespace("forecast", user_id), FORECAST_KEY)
    return SongForecast.model_validate(item.value) if item else None


# -------------------------------------------------------------
# Writes made in code, never by an extractor
# -------------------------------------------------------------
def set_forecast(store: BaseStore, user_id: str, forecast: SongForecast) -> None:
    """Record a forecast. Called by the weather tool, never by an extractor.

    The value arrives already structured from the API, so there is nothing for a model to
    infer — passing it through one could only lose or garble it.
    """
    store.put(_namespace("forecast", user_id), FORECAST_KEY, forecast.model_dump())


def mark_artist_published(
    store: BaseStore, user_id: str, backend_artist_id: str, artist_slug: str | None
) -> None:
    """Record the ids the label's backend gave this artist, after createArtist succeeded.

    Written here rather than left to an extractor for the same reason as the forecast: these
    come back from the API already correct, and they are also what stops the publish gate
    offering to publish the same profile twice.
    """
    artist = get_artist(store, user_id) or ArtistProfile()
    artist.backend_artist_id = backend_artist_id
    artist.artist_slug = artist_slug
    store.put(_namespace("artist", user_id), ARTIST_KEY, artist.model_dump())


def set_artist_fields(store: BaseStore, user_id: str, **fields: Any) -> ArtistProfile:
    """Amend the artist record in code. Used by the publish gate when the user corrects it.

    Only the fields the backend accepts are writable this way; the ids are not, so a user
    amending their own profile cannot claim an artist id that is not theirs.
    """
    artist = get_artist(store, user_id) or ArtistProfile()
    for key, value in fields.items():
        if key in ("name", "genre", "bio", "image_url") and value is not None:
            setattr(artist, key, value)
    store.put(_namespace("artist", user_id), ARTIST_KEY, artist.model_dump())
    return artist


def awaiting_publication(artist: ArtistProfile | None) -> bool:
    """Whether there is a complete artist profile that the backend has not seen yet.

    The gate's whole trigger condition, in one place: something worth sending, and no
    evidence it has already been sent.
    """
    return (
        artist is not None
        and not artist.is_published()
        and artist.to_create_input() is not None
    )


# -------------------------------------------------------------
# Reporting
# -------------------------------------------------------------
def liked_songs(collection: SongCollection | None) -> list[dict[str, Any]]:
    """The songs the user has actually said yes to, for the prompt and the API."""
    if collection is None:
        return []
    return [song.model_dump() for song in collection.songs if song.status == "liked"]


def read_all(store: BaseStore, user_id: str) -> dict[str, Any]:
    """Everything remembered about a user, for the memory endpoint."""
    profile = get_profile(store, user_id)
    artist = get_artist(store, user_id)
    collection = get_found_music(store, user_id)
    forecast = get_forecast(store, user_id)
    return {
        "user_id": user_id,
        "profile": profile.model_dump() if profile else None,
        "artist": artist.model_dump() if artist else None,
        "found_music": collection.model_dump() if collection else None,
        "forecast": forecast.model_dump() if forecast else None,
    }


def clear_all(store: BaseStore, user_id: str) -> None:
    for record, key in (
        ("profile", PROFILE_KEY),
        ("artist", ARTIST_KEY),
        ("found_music", MUSIC_KEY),
        ("forecast", FORECAST_KEY),
    ):
        namespace = _namespace(record, user_id)  # type: ignore[arg-type]
        if store.get(namespace, key):
            store.delete(namespace, key)


# -------------------------------------------------------------
# Writes
# -------------------------------------------------------------
def _extractor_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Conversation to extract from, prefixed with the update instruction.

    The caller strips the AI message carrying the router call before this: it has a tool
    call with no matching ToolMessage, which is a malformed history for another model call.
    """
    return [SystemMessage(content=TRUSTCALL_INSTRUCTION), *messages]


def _patch_record(
    extractor: Any,
    schema_name: str,
    before: dict[str, Any] | None,
    messages: list[AnyMessage],
) -> tuple[list[str], dict[str, Any]]:
    """Run one extractor over the conversation and report what it changed.

    All three records are single documents patched in place, so the mechanics are identical
    and live here rather than being repeated three times.
    """
    if extractor is None:
        raise RuntimeError("XAI_API_KEY is not set; cannot update memory.")

    spy = Spy()
    result = extractor.with_listeners(on_end=spy).invoke(
        {
            "messages": _extractor_messages(messages),
            "existing": {schema_name: before} if before else None,
        }
    )

    saved = result["responses"][0].model_dump()
    return summarize_spy(spy, schema_name) or diff_records(before, saved), saved


def update_profile(
    store: BaseStore, user_id: str, messages: list[AnyMessage]
) -> tuple[list[str], dict[str, Any]]:
    """Patch the user profile. Returns (change lines, the saved record)."""
    current = get_profile(store, user_id)
    before = current.model_dump() if current else None

    changes, saved = _patch_record(profile_extractor, "UserProfile", before, messages)
    store.put(_namespace("profile", user_id), PROFILE_KEY, saved)
    return changes, saved


def update_artist(
    store: BaseStore, user_id: str, messages: list[AnyMessage]
) -> tuple[list[str], dict[str, Any]]:
    """Patch the user's own artist identity. Returns (change lines, the saved record).

    The backend ids are carried over from the stored record rather than taken from the
    extractor's output. The instruction tells the model not to touch them, but an id is the
    one field where a hallucination would be acted on as though it were real, so it is
    enforced here as well.
    """
    current = get_artist(store, user_id)
    before = current.model_dump() if current else None

    changes, saved = _patch_record(artist_extractor, "ArtistProfile", before, messages)
    if current is not None:
        saved["backend_artist_id"] = current.backend_artist_id
        saved["artist_slug"] = current.artist_slug
    else:
        saved["backend_artist_id"] = saved["artist_slug"] = None

    store.put(_namespace("artist", user_id), ARTIST_KEY, saved)
    return changes, saved


def update_found_music(
    store: BaseStore, user_id: str, messages: list[AnyMessage]
) -> tuple[list[str], dict[str, Any]]:
    """Patch the collection of songs surfaced for this user."""
    current = get_found_music(store, user_id)
    before = current.model_dump() if current else None

    changes, saved = _patch_record(music_extractor, "SongCollection", before, messages)
    store.put(_namespace("found_music", user_id), MUSIC_KEY, saved)
    return changes, saved


# -------------------------------------------------------------
# Memory -> prompt
# -------------------------------------------------------------
def _describe_profile(profile: UserProfile | None) -> str:
    if profile is None:
        return "Nothing recorded yet."

    parts: list[str] = []
    if profile.name:
        parts.append(f"Name: {profile.name}")
    if profile.email:
        parts.append(f"Email: {profile.email}")
    if profile.account_type:
        parts.append(f"Here as: {profile.account_type}")
    if profile.favourite_genres:
        parts.append(f"Likes genres: {', '.join(profile.favourite_genres)}")
    if profile.favourite_artists:
        parts.append(f"Already listens to: {', '.join(profile.favourite_artists)}")
    if profile.dislikes:
        parts.append(f"Ruled out: {', '.join(profile.dislikes)}")
    if profile.constraints:
        parts.append(f"Constraints: {', '.join(profile.constraints)}")
    return "\n".join(f"- {part}" for part in parts) if parts else "Nothing recorded yet."


def _describe_artist(artist: ArtistProfile | None) -> str:
    """The user's own artist identity, and plainly whether the label has it yet."""
    if artist is None:
        return "Nothing recorded yet. This user has not presented themselves as an artist."

    parts: list[str] = []
    if artist.name:
        parts.append(f"Artist name: {artist.name}")
    if artist.genre:
        parts.append(f"Genre: {artist.genre}")
    if artist.bio:
        parts.append(f"Bio: {artist.bio}")
    if artist.image_url:
        parts.append(f"Image: {artist.image_url}")
    if artist.influences:
        parts.append(f"Influences: {', '.join(artist.influences)}")
    if artist.albums:
        parts.append(f"Albums: {', '.join(artist.albums)}")
    if artist.songs:
        parts.append(f"Songs: {', '.join(artist.songs)}")
    if artist.tour_dates:
        parts.append(f"Tour dates: {', '.join(artist.tour_dates)}")

    if artist.is_published():
        parts.append(
            f"PUBLISHED to the label — backend id {artist.backend_artist_id}"
            + (f", slug {artist.artist_slug}" if artist.artist_slug else "")
        )
    elif artist.to_create_input() is not None:
        parts.append(
            "NOT yet published. The profile has everything the label needs; the user will "
            "be asked to confirm before it is sent."
        )
    else:
        parts.append("NOT yet published, and still missing an artist name.")

    return "\n".join(f"- {part}" for part in parts) if parts else "Nothing recorded yet."


def _describe_found_music(collection: SongCollection | None) -> str:
    if collection is None or not collection.songs:
        return "Nothing recorded yet."

    lines: list[str] = []
    for song in collection.songs:
        line = f"- [{song.status}] {song.title}"
        if song.artist:
            line += f" — {song.artist}"
        if song.song_id:
            line += f" ({song.song_id})"
        reason = song.dismissed_reason or song.why_found
        if reason:
            line += f" — {reason}"
        lines.append(line)
    return "\n".join(lines)


def _forecast_heading(forecast: SongForecast | None) -> str:
    """The section heading, so a historical record is never filed under 'FORECAST'."""
    if forecast is not None and forecast.basis == "historical":
        return "TYPICAL CONDITIONS THERE (PAST WEATHER, NOT A FORECAST):"
    return "WEATHER WHERE THEY ARE LISTENING:"


def _describe_forecast(forecast: SongForecast | None) -> str:
    """The weather record, with its headline figures worked out here rather than by the model.

    Past weather standing in for a forecast is labelled as such on every line it appears, so
    there is no reading of this block in which last year's December becomes a prediction.
    """
    if forecast is None:
        return "Nothing fetched yet."

    if forecast.basis == "historical":
        average = forecast.month_average
        if average is None:
            return "Nothing fetched yet."

        month_name = datetime.date(
            int(average.month[:4]), int(average.month[5:7]), 1
        ).strftime("%B %Y")
        lines = [
            f"- NOT A FORECAST. No forecast reaches {forecast.start_date}, so this is what "
            f"{forecast.location} was actually like in {month_name} "
            f"({average.days_sampled} days on record).",
        ]
        if average.avg_low_f is not None and average.avg_high_f is not None:
            lines.append(
                f"- Typical night {average.avg_low_f}F, typical day {average.avg_high_f}F"
            )
        if average.coldest_low_f is not None:
            lines.append(
                f"- Coldest night that month {average.coldest_low_f}F, warmest day "
                f"{average.warmest_high_f}F"
            )
        if average.total_precip_in is not None:
            lines.append(f"- {average.total_precip_in} in of precipitation over the month")
        if average.max_wind_mph is not None:
            lines.append(f"- Strongest wind {average.max_wind_mph} mph")
        return "\n".join(lines)

    if not forecast.days:
        return "Nothing fetched yet."

    lows = [d.temp_min_f for d in forecast.days if d.temp_min_f is not None]
    highs = [d.temp_max_f for d in forecast.days if d.temp_max_f is not None]
    wet = [d for d in forecast.days if (d.precipitation_chance_pct or 0) >= 30]

    lines = [
        f"- {forecast.location}, {forecast.start_date} to {forecast.end_date} "
        f"(source: {forecast.source})"
    ]
    if lows and highs:
        lines.append(f"- Coldest night {min(lows)}F, warmest day {max(highs)}F")
    if wet:
        days = ", ".join(f"{d.date} {d.precipitation_chance_pct}%" for d in wet)
        lines.append(f"- Rain likely: {days}")
    else:
        lines.append("- No day above a 30% chance of rain")
    return "\n".join(lines)


def render_memory_prompt(store: BaseStore, user_id: str, summary: str = "") -> str:
    """The remembered context injected into every agent turn.

    Writing memory is pointless unless it comes back into the conversation; this is that
    half. It also tells the agent when to call the router, which is what keeps the records
    growing over a conversation.

    Two kinds of recall, deliberately assembled together so there is one place to look:
    the durable records from the store, and `summary` — the turns compaction has already
    dropped from the transcript. The summary is thread-scoped and transient; the records
    outlive the conversation.
    """
    # Only rendered once there is something to recall, so an uncompacted conversation
    # carries no empty heading.
    earlier = (
        f"\n\nEARLIER IN THIS CONVERSATION (older turns, summarised):\n{summary.strip()}"
        if summary.strip()
        else ""
    )

    # Without this the model dates "this week" from its training data, which quietly sends
    # the weather tool to the historical archive for a week that has already happened.
    today = datetime.date.today().isoformat()

    return f"""You are the assistant for It's The Label, an independent record label. You \
serve two kinds of visitor, and which one you are talking to changes what you do:

- A **fan** is here to find music. Ask what they listen to, search the catalog, and narrow \
it down with them until they have something they actually want to hear.
- An **artist** is here to be signed. Draw out who they are and what they make, build up \
their artist profile, and when it is complete offer to publish it to the label.

Work out which without interrogating them — most people say so in their first message or \
two. Someone can be both.

How to work:
- Recommend only songs returned by search_songs. Never invent songs, artists or releases.
- Ask about what actually drives the choice — what they already listen to, what they cannot \
stand, what they want it for — but no more than a couple of questions at a time.
- When you rule a song out, say why in plain terms, and record it.
- The catalog is the label's own back catalogue and works offline. The list_label_artists, \
get_label_artist and list_artist_songs tools read the live roster from the label's backend, \
which is a different and larger set. Use the catalog for discovery; use the backend tools \
when someone asks who is signed to the label right now. If a backend tool returns an error, \
say plainly that the label's systems are not answering and carry on with the catalog rather \
than pretending you found nothing.
- Weather is evidence about what to play them. If they mention where they are, or where \
they will be, call weather_forcast_tool — a wet grey week and a bright one call for \
different records, and it is one of the few things you can learn about someone without \
asking them another question. Feed it into the search: a cold wet week is an argument for \
`mood` of calm, melancholy or moody; a bright warm one for joyful, warm or euphoric. Say why \
you are doing it — "it's going to be grey there all week, so here is something to match" — \
so it reads as a reason rather than a guess, and drop the idea at once if they say their \
taste has nothing to do with the weather. What the tool finds is saved automatically and \
appears below; quote it from there rather than re-fetching. Today is {today} — work every \
date out from that, never from memory, and pass start_date as YYYY-MM-DD. "This week" means \
{today}, not a week from some other year. A real forecast only reaches about two weeks out; \
past that the tool returns what that month was actually like a year ago, marked NOT A \
FORECAST. Never present past weather as a prediction.
- Never ask for a password, and never repeat one back. Signing in happens outside this \
conversation; if someone offers a password, tell them not to and carry on.

Publishing an artist to the label:
- Build the artist record up in conversation first. The label needs a name at minimum, and \
a genre, bio and image make a better profile.
- You do not publish it yourself and you must not claim to have. Once the record is \
complete, the user is asked to confirm, and only then is it sent. You will be told what \
happened afterwards — report that, not what you expected to happen.
- Publishing a song is a separate two-step job once they are signed: get_song_upload_url \
first, the file is uploaded to what it returns, then register_song with the s3_key it gave \
you. You can only make one tool call at a time, so do these on separate turns and never \
call register_song before the upload has actually happened.

Keeping records (call the MemoryRouter tool):
- 'profile' when they reveal something durable about themselves as a listener: genres, \
artists they already follow, something they have ruled out, or how they will be listening.
- 'artist' when they tell you about their own music: their artist name, genre, bio, \
influences, albums, songs or tour dates.
- 'found_music' when you put songs in front of them, when they like one, or when they \
rule one out.

Record before you reply. You can only make one tool call at a time, so the order matters:

1. Search results just came back from search_songs? Your next action is a MemoryRouter \
call with update_type 'found_music' — recording every song the search returned, each with \
why you are offering it. Present them to the user on the turn after that.
2. The user just liked a song, or ruled one out? Same: MemoryRouter with 'found_music' \
first, then reply.
3. They told you something about themselves? MemoryRouter with 'profile'. About their own \
music? MemoryRouter with 'artist'. Then reply.

Never present or discuss songs you have not recorded. After a record is saved you are told \
exactly what changed — mention it to the user briefly, in your own words, as part of your \
reply.

=== What you already know about this user ==={earlier}

USER PROFILE:
{_describe_profile(get_profile(store, user_id))}

THEIR OWN ARTIST PROFILE:
{_describe_artist(get_artist(store, user_id))}

{_forecast_heading(get_forecast(store, user_id))}
{_describe_forecast(get_forecast(store, user_id))}

SONGS FOUND FOR THEM:
{_describe_found_music(get_found_music(store, user_id))}"""
