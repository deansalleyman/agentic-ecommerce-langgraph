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

from app.catalog import product_by_id
from app.llm import model
from app.schemas import CandidateProduct, GearNeed, MemoryRecord, TripPlan, UserProfile

PROFILE_KEY = "main"
TRIP_KEY = "current"


def _safe_label(user_id: str) -> str:
    """Encode a user id so it is legal as a store namespace label.

    Store namespaces reject periods and empty labels, but the most natural customer id is
    an email address. Escaping rather than stripping keeps the mapping one-to-one: '%' is
    escaped first, so 'a.b@x.com' and a literal 'a%2Eb@x%2Ecom' can never collide on the
    same namespace. The customer's real id is what the API reports back; this encoding is
    only ever seen inside the store.
    """
    encoded = user_id.strip().replace("%", "%25").replace(".", "%2E")
    return encoded or "anonymous"


def _namespace(record: MemoryRecord, user_id: str) -> tuple[str, str]:
    return (record, _safe_label(user_id))


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

Null means null. Do not write 0, an empty string, or placeholder text in place of \
something you were not told — a weight limit of 0 reads as "carry nothing" and will throw \
out every product in the catalog.

Your own assessment is the exception, and is required rather than optional. Every \
candidate product needs a fit_reason drawn from its catalog specs — why this item suits \
this customer and this trip, e.g. "4.2 lb, inside their 4.5 lb limit, 2-person, 3-season". \
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


def committed_totals(needs: list[tuple[str, GearNeed]]) -> dict[str, Any]:
    """What the customer has actually chosen: total weight, total cost, and the breakdown.

    Arithmetic the code owns rather than the model. A running total is a pure function of
    what is already recorded, so asking a model for it buys nothing and risks a dropped
    line or a mis-transcribed spec — and with one tool call per turn, the call is usually
    spent on recording anyway. These figures go straight into the prompt, always current.
    """
    items: list[dict[str, Any]] = []
    for _, need in needs:
        if not need.selected_product_id:
            continue
        product = product_by_id(need.selected_product_id)
        if product is None:
            continue
        items.append({
            "need_id": need.need_id,
            "product_id": product.product_id,
            "name": product.name,
            "weight_lb": product.weight_lb,
            "price_usd": product.price_usd,
        })

    return {
        "weight_lb": round(sum(i["weight_lb"] or 0 for i in items), 2),
        "cost_usd": round(sum(i["price_usd"] or 0 for i in items), 2),
        "items": items,
        # Weight is only trustworthy if every chosen product has one on file.
        "unweighed": [i["name"] for i in items if i["weight_lb"] is None],
    }


def carry_limit(trip: TripPlan | None, fallback: float) -> float:
    """The weight ceiling in force: the customer's own if they set one, else the fallback."""
    if trip is not None and trip.max_carry_weight_lb:
        return trip.max_carry_weight_lb
    return fallback


def drop_selection(store: BaseStore, user_id: str, need_id: str, reason: str) -> bool:
    """Un-choose a product, recording why. Returns False if the need is not found."""
    namespace = _namespace("gear_needs", user_id)
    item = store.get(namespace, need_id)
    if item is None:
        return False

    need = GearNeed.model_validate(item.value)
    dropped = need.selected_product_id
    need.selected_product_id = None
    need.status = "exploring"
    for candidate in need.candidates:
        if candidate.product_id == dropped:
            candidate.status = "eliminated"
            candidate.eliminated_reason = reason
    store.put(namespace, need_id, need.model_dump())
    return True


def swap_selection(store: BaseStore, user_id: str, need_id: str, product_id: str) -> bool:
    """Choose a different product for a need. Returns False if need or product is unknown."""
    namespace = _namespace("gear_needs", user_id)
    item = store.get(namespace, need_id)
    product = product_by_id(product_id)
    if item is None or product is None:
        return False

    need = GearNeed.model_validate(item.value)
    need.selected_product_id = product_id
    need.status = "decided"
    known = {c.product_id for c in need.candidates}
    if product_id not in known:
        need.candidates.append(
            CandidateProduct(
                product_id=product.product_id,
                name=product.name,
                price_usd=product.price_usd,
                status="shortlisted",
                fit_reason="Chosen to bring the pack under its weight limit.",
            )
        )
    for candidate in need.candidates:
        if candidate.product_id == product_id:
            candidate.status = "shortlisted"
            candidate.eliminated_reason = None
    store.put(namespace, need_id, need.model_dump())
    return True


def lighter_alternatives(needs: list[tuple[str, GearNeed]]) -> dict[str, list[dict[str, Any]]]:
    """Per need, the already-discussed candidates lighter than the chosen product.

    Offered with the interrupt so the customer can trade down without another search.
    """
    options: dict[str, list[dict[str, Any]]] = {}
    for _, need in needs:
        chosen = product_by_id(need.selected_product_id or "")
        if chosen is None or chosen.weight_lb is None:
            continue
        lighter = []
        for candidate in need.candidates:
            alt = product_by_id(candidate.product_id)
            if alt is None or alt.weight_lb is None or alt.product_id == chosen.product_id:
                continue
            if alt.weight_lb < chosen.weight_lb:
                lighter.append({
                    "product_id": alt.product_id,
                    "name": alt.name,
                    "weight_lb": alt.weight_lb,
                    "price_usd": alt.price_usd,
                    "saves_lb": round(chosen.weight_lb - alt.weight_lb, 2),
                })
        if lighter:
            options[need.need_id] = sorted(lighter, key=lambda o: -o["saves_lb"])
    return options


def set_carry_limit(store: BaseStore, user_id: str, limit_lb: float) -> None:
    """Raise or lower the trip's pack-weight limit."""
    trip = get_trip(store, user_id) or TripPlan()
    trip.max_carry_weight_lb = limit_lb
    store.put(_namespace("trip", user_id), TRIP_KEY, trip.model_dump())


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


def _describe_trip(
    trip: TripPlan | None, needs: list[tuple[str, GearNeed]], fallback_limit: float
) -> str:
    fields: dict[str, Any] = {}
    if trip is not None:
        fields = {
            "Destination": trip.destination,
            "Starts": trip.start_date,
            "Nights": trip.nights,
            "Party size": trip.party_size,
            "Season": trip.season,
            "Travel mode": trip.travel_mode,
            "Conditions": trip.expected_conditions,
            # A pack limit of 0 means "not given", never "carry nothing", so it is
            # filtered out below along with blanks rather than shown as a constraint.
            "Max pack weight": (
                f"{trip.max_carry_weight_lb} lb total" if trip.max_carry_weight_lb else None
            ),
        }

    # Totals are computed here, not asked of the model — see committed_totals.
    totals = committed_totals(needs)
    if totals["items"]:
        breakdown = ", ".join(
            f"{i['name']} {i['weight_lb']} lb" if i["weight_lb"] is not None
            else f"{i['name']} (weight unknown)"
            for i in totals["items"]
        )
        fields["Chosen so far"] = f"{breakdown} — ${totals['cost_usd']:.2f} total"

        limit = carry_limit(trip, fallback_limit)
        remaining = round(limit - totals["weight_lb"], 2)
        line = f"{totals['weight_lb']} lb of {limit} lb"
        line += f" — {remaining} lb still spare" if remaining >= 0 else f" — {abs(remaining)} lb OVER"
        if totals["unweighed"]:
            line += f" (excludes {', '.join(totals['unweighed'])}, no weight on file)"
        fields["Pack weight"] = line

    parts = [
        f"- {label}: {value}"
        for label, value in fields.items()
        if value is not None and value != ""
    ]
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
                detail += f" ${candidate.price_usd:.2f}"
            reason = candidate.eliminated_reason or candidate.fit_reason
            if reason:
                detail += f" — {reason}"
            lines.append(detail)
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def render_memory_prompt(store: BaseStore, user_id: str, fallback_limit: float) -> str:
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
- Prices are in USD, weights in lb.
- Do not do arithmetic yourself. The running pack weight, the total cost and the weight \
still spare are computed for you and shown under CURRENT TRIP below — quote those figures, \
do not recompute or estimate them. For a what-if over items that are not chosen catalog \
products ("what if I add 3 days of food?"), call calculate_gear_weight rather than adding \
up in your head.
- A max pack weight is a budget for the whole kit, not a limit on any single item. Only \
that trip's limit applies — if none is recorded, do not assume one, and do not carry one \
over from an earlier trip. When it is getting tight, say so and offer the lighter option \
rather than quietly dropping something they wanted.

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
{_describe_trip(get_trip(store, user_id), get_gear_needs(store, user_id), fallback_limit)}

GEAR NEEDS AND PRODUCTS IN PLAY:
{_describe_gear_needs(get_gear_needs(store, user_id))}"""
