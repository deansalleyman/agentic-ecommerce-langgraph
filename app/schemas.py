"""Schemas for the camping-store assistant.

Three groups live here:

1. `MemoryRouter` — the struct the model fills in to say which record it wants written.
   Bound as a tool but never executed as one; the graph routes on it.
2. The memory records themselves (`UserProfile`, `TripPlan`, `GearNeed`). trustcall patches
   these incrementally, so every field is optional or defaulted — a half-known profile must
   still validate. The `description` on each field is what the extraction model reads, so
   they are prompt text, not just documentation.
3. `Product` — a catalog row, deliberately distinct from `CandidateProduct` (the record of a
   product being considered).
"""

from typing import Literal

from pydantic import BaseModel, Field

# -------------------------------------------------------------
# Router
# -------------------------------------------------------------
# What the model may ask to have written. `forecast` is absent on purpose: it is written by
# the weather tool in code, so the router must never be able to route to it.
MemoryRecord = Literal["profile", "trip", "gear_needs"]

# Every record the store holds, router-writable or not.
StoreRecord = Literal["profile", "trip", "gear_needs", "forecast"]


class MemoryRouter(BaseModel):
    """Save durable information the customer has just given you.

    Call this whenever the conversation reveals something worth remembering after the
    chat ends: who they are and what they do outdoors, the trip they are planning, or
    which products are in play for a piece of gear.
    """

    update_type: MemoryRecord = Field(
        description=(
            "Which record to write. "
            "'profile' = who the customer is, the activities they do, gear they own, budget "
            "and brand preferences. "
            "'trip' = the trip they are kitting out: destination, dates, nights, party size, "
            "conditions. "
            "'gear_needs' = an item they are shopping for and the products under "
            "consideration for it, including ones being ruled out."
        )
    )
    reason: str = Field(
        description="What you are recording and why, in one sentence.",
    )


# -------------------------------------------------------------
# Memory records
# -------------------------------------------------------------
class OutdoorActivity(BaseModel):
    """A single outdoor pursuit the customer takes part in."""

    activity: str = Field(
        description="The activity, lowercase. e.g. 'backpacking', 'car camping', "
        "'winter hiking', 'bouldering', 'wild swimming'."
    )
    experience_level: Literal["beginner", "intermediate", "advanced"] | None = Field(
        default=None, description="How experienced they are at this specific activity."
    )
    frequency: str | None = Field(
        default=None,
        description="How often they do it, in their own words. e.g. 'most weekends', "
        "'a few times a year'.",
    )
    notes: str | None = Field(
        default=None,
        description="Anything else specific to this activity: terrain they favour, trips "
        "they mention, gear gripes.",
    )


class UserProfile(BaseModel):
    """Durable facts about the customer, carried across every conversation."""

    name: str | None = Field(default=None, description="The customer's name.")
    home_base: str | None = Field(
        default=None, description="Where they live or usually set out from."
    )
    activities: list[OutdoorActivity] = Field(
        default_factory=list,
        description="Every outdoor activity they have mentioned taking part in.",
    )
    owned_gear: list[str] = Field(
        default_factory=list,
        description="Gear they already own, so it is not recommended again. e.g. "
        "'Osprey Atmos 65 pack', '3-season down bag'.",
    )
    budget_band: Literal["value", "mid", "premium"] | None = Field(
        default=None,
        description="How they tend to spend on gear. Only set this if they have said "
        "something about price or budget. Do not infer it from their experience level or "
        "from a single product they liked — leave it null instead.",
    )
    preferred_brands: list[str] = Field(
        default_factory=list, description="Brands they favour or have spoken well of."
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Personal factors that affect what suits them. e.g. 'sleeps cold', "
        "'bad knees, wants light loads', 'needs tall sizing', 'flies with hold luggage only'.",
    )


class TripPlan(BaseModel):
    """The trip the customer is currently shopping for."""

    destination: str | None = Field(
        default=None, description="Where they are going. e.g. 'Big Bear', 'Cairngorms'."
    )
    start_date: str | None = Field(
        default=None,
        description="When the trip starts, as an ISO date (YYYY-MM-DD) if known, otherwise "
        "the month or period they said, verbatim.",
    )
    nights: int | None = Field(default=None, description="How many nights they are out.")
    party_size: int | None = Field(
        default=None, description="How many people are going, including the customer."
    )
    season: Literal["summer", "3-season", "winter"] | None = Field(
        default=None,
        description="The conditions the gear must cope with. Use 'winter' for anything "
        "involving snow, sustained sub-zero nights, or exposed mountain camping in winter.",
    )
    travel_mode: Literal["backpacking", "car camping", "basecamp"] | None = Field(
        default=None,
        description="How the gear will be carried. 'backpacking' means weight matters a lot; "
        "'car camping' means it barely matters.",
    )
    expected_conditions: str | None = Field(
        default=None,
        description="Weather and terrain they expect, in their own words. e.g. 'rain, "
        "near freezing at night, exposed ridges'.",
    )
    max_carry_weight_lb: float | None = Field(
        default=None,
        description="The heaviest loaded pack the customer is willing to carry on THIS "
        "trip, in lb. A total for everything they carry, not the weight of any one item. "
        "Only set it if they have stated a limit for this trip, e.g. 'I don't want to "
        "carry more than 30 lb' — it belongs to the trip, so do not carry one over from "
        "an earlier trip and do not infer one from their experience or fitness. If they "
        "have not given a limit, leave it null; never write 0.",
    )


class CandidateProduct(BaseModel):
    """One product being considered for a gear need, and where it stands."""

    product_id: str = Field(
        description="The catalog id, exactly as returned by search_products."
    )
    name: str = Field(description="The product name.")
    price_usd: float | None = Field(default=None, description="Price in USD.")
    status: Literal["candidate", "shortlisted", "eliminated"] = Field(
        default="candidate",
        description="'candidate' = found, not yet judged. 'shortlisted' = a serious "
        "contender. 'eliminated' = ruled out; always give eliminated_reason.",
    )
    fit_reason: str | None = Field(
        default=None,
        description="Always fill this in when adding a candidate: why this product suits "
        "this customer and this trip, citing its specs. e.g. '3.8 lb, under their 4.5 lb "
        "limit, 3-season'. This is your assessment, not something the customer said.",
    )
    eliminated_reason: str | None = Field(
        default=None,
        description="Why it was ruled out, in the customer's terms. e.g. 'over budget at "
        "$480', 'too heavy to carry for 3 days'.",
    )
    weight: float | None = Field(default=None, description="Weight of the product in lb.")


class GearNeed(BaseModel):
    """One item the customer needs for the trip, with the products in play for it.

    This is the collection record: there is one of these per thing they are shopping for,
    and it is narrowed over the conversation rather than rewritten.
    """

    need_id: str = Field(
        description="Stable lowercase slug identifying the need, e.g. 'tent', "
        "'sleeping_bag', 'stove'. Reuse the existing slug when updating a need."
    )
    category: str = Field(
        description="Catalog category this need shops in, e.g. 'tent', 'sleeping_bag'."
    )
    requirements: str | None = Field(
        default=None,
        description="What the item has to satisfy, e.g. '2-person, under 4.5 lb, 3-season, "
        "under $300'.",
    )
    status: Literal["exploring", "narrowed", "decided"] = Field(
        default="exploring",
        description="'exploring' = still gathering options. 'narrowed' = down to a "
        "shortlist. 'decided' = they have chosen; set selected_product_id.",
    )
    selected_product_id: str | None = Field(
        default=None, description="The chosen product's catalog id, once decided."
    )
    candidates: list[CandidateProduct] = Field(
        default_factory=list,
        description="Every product considered for this need, including eliminated ones — "
        "the history of the decision is the point, so never drop entries.",
    )
    weight: float | None = Field(default=None, description="Weight of the need in lb.")


# -------------------------------------------------------------
# Forecast — written by the weather tool, never by an extractor
# -------------------------------------------------------------
class ForecastDay(BaseModel):
    """One day of the forecast for a trip."""

    date: str
    temp_min_f: float | None = None
    temp_max_f: float | None = None
    precipitation_in: float | None = None
    precipitation_chance_pct: int | None = None
    wind_max_mph: float | None = None


class MonthAverage(BaseModel):
    """What one calendar month was actually like, averaged from the Open-Meteo archive.

    Used when a trip is beyond the forecast horizon. The archive carries measured
    precipitation but no chance-of-rain — ERA5 does not model one — so there is
    deliberately no percentage here to mistake for a forecast probability.
    """

    month: str = Field(description="The month sampled, as YYYY-MM.")
    days_sampled: int
    avg_low_f: float | None = None
    avg_high_f: float | None = None
    # What warmth decisions actually hinge on: the worst night that month, not the mean.
    coldest_low_f: float | None = None
    warmest_high_f: float | None = None
    total_precip_in: float | None = None
    max_wind_mph: float | None = None


class TripForecast(BaseModel):
    """Weather for a trip, as fetched from Open-Meteo.

    Deliberately not a field on `TripPlan`. The trip extractor is handed the whole
    `TripPlan` as `existing` on every trip update, so a field living there would be fair
    game for the model to rewrite or drop on some unrelated turn. Kept in its own record,
    written only by the weather tool, it cannot be touched by extraction at all.

    `basis` distinguishes a real forecast from past weather standing in for one. It lives on
    the record rather than only in prompt wording, so a stored value can never be read back
    as a prediction it never was.
    """

    location: str = Field(description="Resolved place name from geocoding.")
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    basis: Literal["forecast", "historical"] = "forecast"
    # Populated on the forecast path only. Per-day rows from another year would invite being
    # read as a day-by-day prediction, so the historical path leaves this empty.
    days: list[ForecastDay] = Field(default_factory=list)
    # Populated on the historical path only.
    years_sampled: list[int] = Field(default_factory=list)
    month_average: MonthAverage | None = None
    source: str = "open-meteo"


class GearItem(BaseModel):
    """One line in a pack list, for weighing up what a camper will carry.

    Typed rather than a loose dict so the model is given a schema for it: an untyped
    argument leaves it guessing the shape, and it guesses wrong.
    """

    name: str = Field(description="What the item is, e.g. 'tent', 'stove', 'day of food'.")
    weight_lb: float = Field(description="Weight of a single one of these, in lb.")
    quantity: int = Field(default=1, description="How many of this item are being carried.")


# -------------------------------------------------------------
# Catalog
# -------------------------------------------------------------
class Product(BaseModel):
    """A row in the store catalog."""

    product_id: str
    name: str
    brand: str
    category: str
    price_usd: float
    description: str
    activities: list[str] = Field(default_factory=list)
    in_stock: bool = True
    # Category-specific specs stay explicit rather than a loose dict, so search can filter
    # on them. Each is only meaningful for some categories.
    weight_lb: float | None = None
    season: Literal["summer", "3-season", "winter"] | None = None
    capacity: int | None = None
    temp_rating_c: float | None = None
