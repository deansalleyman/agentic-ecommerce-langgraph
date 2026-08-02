"""Long-term memory: trustcall extractors, change reporting, and prompt rendering.

Memory is keyed by user rather than thread, so it outlives any one conversation:

    ("profile",    user_id) / "main"    -> UserProfile
    ("trip",       user_id) / "current" -> TripPlan
    ("gear_needs", user_id) / need_id   -> GearNeed   (the collection)

Records are *patched*, not rewritten. trustcall asks the model for a JSON patch against the
existing document and validates it against the schema, so narrowing a shortlist — flipping
one candidate to eliminated with a reason — leaves every other candidate untouched. Asking
the model to re-emit the whole document instead loses detail and burns tokens.
"""

import json
from typing import Any, Iterator

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tracers.schemas import Run
from langgraph.store.base import BaseStore
from trustcall import create_extractor

from app.llm import model
from app.schemas import GearNeed, MemoryRecord, TripPlan, UserProfile

PROFILE_KEY = "main"
TRIP_KEY = "current"


def _namespace(record: MemoryRecord, user_id: str) -> tuple[str, str]:
    return (record, user_id)


# -------------------------------------------------------------
# Extractors
# -------------------------------------------------------------
# Built once, only when a model is configured. enable_inserts is on for gear needs alone:
# a customer accumulates needs over a conversation, but has exactly one profile and trip.
if model is not None:
    profile_extractor = create_extractor(
        model, tools=[UserProfile], tool_choice="UserProfile"
    )
    trip_extractor = create_extractor(model, tools=[TripPlan], tool_choice="TripPlan")
    gear_extractor = create_extractor(
        model, tools=[GearNeed], tool_choice="GearNeed", enable_inserts=True
    )
else:  # pragma: no cover - import-time convenience for unconfigured machines
    profile_extractor = trip_extractor = gear_extractor = None

TRUSTCALL_INSTRUCTION = """You maintain the store's records about this customer.

Update the record from the conversation below. Preserve everything the conversation does \
not contradict — you are amending a record, not rewriting it.

Facts about the customer, their trip, and their decisions record only what they actually \
said. A field with no evidence behind it stays null: an empty field is correct, a \
plausible guess is a false record the store will act on later. Do not infer budget from \
experience, or a preference from one product they happened to like.

Your own assessment is the exception, and is required rather than optional. Every \
candidate product needs a fit_reason drawn from its catalog specs — why this item suits \
this customer and this trip, e.g. "1.9kg, inside their 2kg limit, 2-person, 3-season". \
That is your judgement to make, not something the customer has to say first.

For gear needs specifically: keep every candidate product that is already recorded, \
including eliminated ones. When the customer rules something out, set that candidate's \
status to "eliminated" and write eliminated_reason in their terms. When they choose, set \
the need's status to "decided" and selected_product_id."""


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
    internal run shape, and a reporting detail must never be what breaks a customer's turn.
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
            label = args.get("need_id") or args.get("name")
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
    depends on nothing, so the customer always gets specifics.
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


def get_trip(store: BaseStore, user_id: str) -> TripPlan | None:
    item = store.get(_namespace("trip", user_id), TRIP_KEY)
    return TripPlan.model_validate(item.value) if item else None


def get_gear_needs(store: BaseStore, user_id: str) -> list[tuple[str, GearNeed]]:
    items = store.search(_namespace("gear_needs", user_id), limit=100)
    return [(item.key, GearNeed.model_validate(item.value)) for item in items]


def read_all(store: BaseStore, user_id: str) -> dict[str, Any]:
    """Everything remembered about a user, for the memory endpoint."""
    profile = get_profile(store, user_id)
    trip = get_trip(store, user_id)
    return {
        "user_id": user_id,
        "profile": profile.model_dump() if profile else None,
        "trip": trip.model_dump() if trip else None,
        "gear_needs": {key: need.model_dump() for key, need in get_gear_needs(store, user_id)},
    }


def clear_all(store: BaseStore, user_id: str) -> None:
    if store.get(_namespace("profile", user_id), PROFILE_KEY):
        store.delete(_namespace("profile", user_id), PROFILE_KEY)
    if store.get(_namespace("trip", user_id), TRIP_KEY):
        store.delete(_namespace("trip", user_id), TRIP_KEY)
    for key, _ in get_gear_needs(store, user_id):
        store.delete(_namespace("gear_needs", user_id), key)


# -------------------------------------------------------------
# Writes
# -------------------------------------------------------------
def _extractor_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Conversation to extract from, prefixed with the update instruction.

    The caller strips the AI message carrying the router call before this: it has a tool
    call with no matching ToolMessage, which is a malformed history for another model call.
    """
    return [SystemMessage(content=TRUSTCALL_INSTRUCTION), *messages]


def update_profile(
    store: BaseStore, user_id: str, messages: list[AnyMessage]
) -> tuple[list[str], dict[str, Any]]:
    """Patch the customer profile. Returns (change lines, the saved record)."""
    if profile_extractor is None:
        raise RuntimeError("XAI_API_KEY is not set; cannot update memory.")

    current = get_profile(store, user_id)
    before = current.model_dump() if current else None

    spy = Spy()
    result = profile_extractor.with_listeners(on_end=spy).invoke(
        {
            "messages": _extractor_messages(messages),
            "existing": {"UserProfile": before} if before else None,
        }
    )

    saved = result["responses"][0].model_dump()
    store.put(_namespace("profile", user_id), PROFILE_KEY, saved)
    return summarize_spy(spy, "UserProfile") or diff_records(before, saved), saved


def update_trip(
    store: BaseStore, user_id: str, messages: list[AnyMessage]
) -> tuple[list[str], dict[str, Any]]:
    """Patch the trip being planned. Returns (change lines, the saved record)."""
    if trip_extractor is None:
        raise RuntimeError("XAI_API_KEY is not set; cannot update memory.")

    current = get_trip(store, user_id)
    before = current.model_dump() if current else None

    spy = Spy()
    result = trip_extractor.with_listeners(on_end=spy).invoke(
        {
            "messages": _extractor_messages(messages),
            "existing": {"TripPlan": before} if before else None,
        }
    )

    saved = result["responses"][0].model_dump()
    store.put(_namespace("trip", user_id), TRIP_KEY, saved)
    return summarize_spy(spy, "TripPlan") or diff_records(before, saved), saved


def update_gear_needs(
    store: BaseStore, user_id: str, messages: list[AnyMessage]
) -> tuple[list[str], dict[str, Any]]:
    """Patch the gear-need collection. Returns (change lines, the saved records by key).

    Existing needs go in as (doc_id, schema_name, value) tuples; trustcall echoes the
    doc_id back in response_metadata as `json_doc_id`, or "New" for an insert. That
    mapping is what lets one need be amended while the others are left alone.
    """
    if gear_extractor is None:
        raise RuntimeError("XAI_API_KEY is not set; cannot update memory.")

    namespace = _namespace("gear_needs", user_id)
    before = {key: need.model_dump() for key, need in get_gear_needs(store, user_id)}

    spy = Spy()
    result = gear_extractor.with_listeners(on_end=spy).invoke(
        {
            "messages": _extractor_messages(messages),
            "existing": [(key, "GearNeed", value) for key, value in before.items()] or None,
        }
    )

    saved: dict[str, Any] = {}
    for response, metadata in zip(result["responses"], result["response_metadata"]):
        # trustcall types responses as bare BaseModel; re-validate to get the real shape.
        need = GearNeed.model_validate(response.model_dump())
        doc_id = metadata.get("json_doc_id")
        key = need.need_id if doc_id in (None, "New") else doc_id
        record = need.model_dump()
        store.put(namespace, key, record)
        saved[key] = record

    changes = summarize_spy(spy, "GearNeed")
    if not changes:
        for key, record in saved.items():
            changes.extend(f"{key}: {line}" for line in diff_records(before.get(key), record))
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
    if profile.home_base:
        parts.append(f"Based in: {profile.home_base}")
    if profile.activities:
        activities = "; ".join(
            ", ".join(
                filter(
                    None,
                    [
                        activity.activity,
                        activity.experience_level,
                        activity.frequency,
                        activity.notes,
                    ],
                )
            )
            for activity in profile.activities
        )
        parts.append(f"Activities: {activities}")
    if profile.owned_gear:
        parts.append(f"Already owns: {', '.join(profile.owned_gear)}")
    if profile.budget_band:
        parts.append(f"Budget: {profile.budget_band}")
    if profile.preferred_brands:
        parts.append(f"Likes brands: {', '.join(profile.preferred_brands)}")
    if profile.constraints:
        parts.append(f"Constraints: {', '.join(profile.constraints)}")
    return "\n".join(f"- {part}" for part in parts) if parts else "Nothing recorded yet."


def _describe_trip(trip: TripPlan | None) -> str:
    if trip is None:
        return "Nothing recorded yet."
    fields = {
        "Destination": trip.destination,
        "Starts": trip.start_date,
        "Nights": trip.nights,
        "Party size": trip.party_size,
        "Season": trip.season,
        "Travel mode": trip.travel_mode,
        "Conditions": trip.expected_conditions,
    }
    parts = [f"- {label}: {value}" for label, value in fields.items() if value is not None]
    return "\n".join(parts) if parts else "Nothing recorded yet."


def _describe_gear_needs(needs: list[tuple[str, GearNeed]]) -> str:
    if not needs:
        return "Nothing recorded yet."

    blocks: list[str] = []
    for _, need in needs:
        header = f"- {need.need_id} ({need.status})"
        if need.requirements:
            header += f" — needs: {need.requirements}"
        if need.selected_product_id:
            header += f" — chosen: {need.selected_product_id}"
        lines = [header]
        for candidate in need.candidates:
            detail = f"    - [{candidate.status}] {candidate.name} ({candidate.product_id})"
            if candidate.price_usd is not None:
                detail += f" £{candidate.price_usd:.2f}"
            reason = candidate.eliminated_reason or candidate.fit_reason
            if reason:
                detail += f" — {reason}"
            lines.append(detail)
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def render_memory_prompt(store: BaseStore, user_id: str) -> str:
    """The remembered context injected into every agent turn.

    Writing memory is pointless unless it comes back into the conversation; this is that
    half. It also tells the agent when to call the router, which is what keeps the records
    growing over a conversation.
    """
    return f"""You are a shopping assistant for an online camping and outdoor store. You \
help customers choose gear for real trips: ask about the trip and how they camp, search \
the catalog, and narrow the options down with them until they have decided.

How to work:
- Recommend only products returned by search_products. Never invent products or prices.
- Ask about what actually drives the choice — conditions, how far they carry it, whether \
they sleep cold, what they already own — but no more than a couple of questions at a time.
- When you rule a product out, say why in plain terms, and record it.
- Prices are in USD.

Keeping records (call the MemoryRouter tool):
- 'profile' when they reveal something durable about themselves: activities, experience, \
gear they own, budget, brands, or a constraint like sleeping cold.
- 'trip' when they mention where they are going, when, for how long, with how many people, \
or in what conditions.
- 'gear_needs' when they name something they need, when you rule a product out, or when \
they choose one.

Record before you reply. You can only make one tool call at a time, so the order matters:

1. Search results just came back from search_products? Your next action is a MemoryRouter \
call with update_type 'gear_needs' — recording the need and every product the search \
returned as candidates. Present them to the customer on the turn after that.
2. The customer just ruled a product out, or chose one? Same: MemoryRouter with \
'gear_needs' first, then reply.
3. They mentioned something about themselves or the trip? MemoryRouter with 'profile' or \
'trip' first, then reply.

Never present or discuss products you have not recorded. After a record is saved you are \
told exactly what changed — mention it to the customer briefly, in your own words, as part \
of your reply.

=== What you already know about this customer ===

CUSTOMER PROFILE:
{_describe_profile(get_profile(store, user_id))}

CURRENT TRIP:
{_describe_trip(get_trip(store, user_id))}

GEAR NEEDS AND PRODUCTS IN PLAY:
{_describe_gear_needs(get_gear_needs(store, user_id))}"""
