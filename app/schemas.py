"""Schemas for the label assistant.

Four groups live here:

1. `MemoryRouter` — the struct the model fills in to say which record it wants written.
   Bound as a tool but never executed as one; the graph routes on it.
2. The memory records themselves (`UserProfile`, `ArtistProfile`, `SongCollection`).
   trustcall patches these incrementally, so every field is optional or defaulted — a
   half-known profile must still validate. The `description` on each field is what the
   extraction model reads, so they are prompt text, not just documentation.
3. The backend CRUD payloads — one schema per write operation the GraphQL backend exposes.
   These are the shape the label's API actually accepts, kept separate from the memory
   records so that what we remember about an artist and what we are entitled to send are
   never the same object. `ArtistProfile.to_create_input()` is the only bridge between them.
4. `Song` — a catalog row, deliberately distinct from `FoundSong` (the record of a song
   surfaced to a user, with a note of why).

Note on passwords: there is deliberately no password field on `UserProfile`. Records here
are filled by an extraction model reading the transcript and are served verbatim by
GET /users/{id}/memory, so a password field is a standing instruction to copy a secret out
of the conversation into a readable store. Account credentials belong to the backend's
Cognito pool, and the only credential this app handles is the short-lived ID token, which
travels on the run config and is never stored — see app/identity.py.
"""

from typing import Literal

from pydantic import BaseModel, Field

# -------------------------------------------------------------
# Router
# -------------------------------------------------------------
# What the model may ask to have written. `forecast` is absent on purpose: it is written by
# the weather tool in code, so the router must never be able to route to it.
MemoryRecord = Literal["profile", "artist", "found_music"]

# Every record the store holds, router-writable or not.
StoreRecord = Literal["profile", "artist", "found_music", "forecast"]


class MemoryRouter(BaseModel):
    """Save durable information the user has just given you.

    Call this whenever the conversation reveals something worth remembering after the
    chat ends: who they are and what they listen to, their own artist identity if they
    are signing to the label, or which songs are in play for them.
    """

    update_type: MemoryRecord = Field(
        description=(
            "Which record to write. "
            "'profile' = who the user is: whether they are here as a fan browsing or as "
            "an artist seeking the label's patronage, and what they listen to. "
            "'artist' = the user's own artist identity in their own words: name, genre, "
            "bio, albums, songs, tour dates. "
            "'found_music' = songs surfaced for the user, each with a note of why it was "
            "put in front of them, and whether they liked or dismissed it."
        )
    )
    reason: str = Field(
        description="What you are recording and why, in one sentence.",
    )


# -------------------------------------------------------------
# Memory records
# -------------------------------------------------------------
class UserProfile(BaseModel):
    """Durable facts about the user, carried across every conversation."""

    name: str | None = Field(default=None, description="The user's name.")
    email: str | None = Field(
        default=None,
        description="The user's email address, only if they volunteer it.",
    )
    account_type: Literal["guest", "fan", "artist"] | None = Field(
        default="guest",
        description=(
            "How the user is here: 'guest' until they say, 'fan' if they are browsing and "
            "listening, 'artist' if they are presenting their own music to the label."
        ),
    )
    favourite_genres: list[str] = Field(
        default_factory=list,
        description="Genres the user has said they like, e.g. 'shoegaze', 'afrobeat'.",
    )
    favourite_artists: list[str] = Field(
        default_factory=list,
        description="Artists the user has named as ones they already listen to.",
    )
    dislikes: list[str] = Field(
        default_factory=list,
        description=(
            "Genres, sounds or artists the user has ruled out, in their own terms, "
            "e.g. 'nothing with heavy distortion'."
        ),
    )
    constraints: list[str] = Field(
        default_factory=list,
        description=(
            "Anything else that should shape what is put in front of them, e.g. "
            "'only wants tracks under 4 minutes', 'listens at work so nothing too loud'."
        ),
    )


class ArtistProfile(BaseModel):
    """The user's own artist identity, as they describe it over the conversation.

    The first four fields are deliberately the backend's CreateArtistInput, so publishing is
    a projection rather than a translation — see `to_create_input`. Everything below them is
    local colour the backend has no field for, and the last two are written in code once the
    backend has accepted the profile, never by an extractor.
    """

    name: str | None = Field(
        default=None, description="The artist or band name, as they want it shown."
    )
    genre: str | None = Field(
        default=None, description="Their main genre, e.g. 'rock', 'pop', 'jazz'."
    )
    bio: str | None = Field(
        default=None,
        description="A short description of the artist, in their own words where possible.",
    )
    image_url: str | None = Field(
        default=None, description="URL of a profile or poster image, if they have one."
    )

    albums: list[str] = Field(
        default_factory=list, description="Albums the artist says they have released."
    )
    songs: list[str] = Field(
        default_factory=list, description="Songs the artist has named as their own."
    )
    tour_dates: list[str] = Field(
        default_factory=list,
        description="Upcoming dates the artist has mentioned, as they said them.",
    )
    influences: list[str] = Field(
        default_factory=list,
        description="Artists they name as influences or comparisons for their sound.",
    )

    # Backend sync. Set by the publish tool after the label's API accepts the profile, so
    # their presence is what tells the graph this artist has already been published.
    backend_artist_id: str | None = Field(
        default=None,
        description="The label backend's id for this artist. Never fill this in yourself.",
    )
    artist_slug: str | None = Field(
        default=None,
        description="The label backend's slug for this artist. Never fill this in yourself.",
    )

    def is_published(self) -> bool:
        """Whether the label's backend has already accepted this profile."""
        return self.backend_artist_id is not None

    def to_create_input(self) -> "CreateArtistInput | None":
        """Project onto the backend's createArtist payload, or None if it is not ready.

        `name` is the one field the backend requires, so a profile without one is not a
        submission — it is a conversation still in progress.
        """
        if not self.name:
            return None
        return CreateArtistInput(
            name=self.name, genre=self.genre, bio=self.bio, image_url=self.image_url
        )


class FoundSong(BaseModel):
    """One song put in front of the user, and what came of it."""

    song_id: str | None = Field(
        default=None,
        description="The catalog id, exactly as returned by search_songs. Null if the song "
        "came up in conversation rather than from a search.",
    )
    title: str = Field(description="The song title.")
    artist: str | None = Field(default=None, description="Who performs it.")
    why_found: str | None = Field(
        default=None,
        description=(
            "Why this song was put in front of this user, drawn from what they have said "
            "they like and from the song's own tags, e.g. 'shoegaze with the quiet vocals "
            "they asked for'. This is your assessment to make, not something they have to "
            "say first."
        ),
    )
    status: Literal["suggested", "liked", "dismissed"] = Field(
        default="suggested",
        description="Where the song stands with the user.",
    )
    dismissed_reason: str | None = Field(
        default=None,
        description="If dismissed, why — in the user's own terms.",
    )


class SongCollection(BaseModel):
    """Every song surfaced for this user, with what became of each.

    Narrowed over the conversation rather than rewritten: a song the user rules out stays on
    the list as dismissed, so it is neither offered again nor silently forgotten.
    """

    songs: list[FoundSong] = Field(
        default_factory=list,
        description="Every song put in front of the user, including dismissed ones.",
    )


# -------------------------------------------------------------
# Backend CRUD payloads — one per write operation
# -------------------------------------------------------------
# These mirror the GraphQL backend's own input types. They exist as schemas rather than loose
# arguments so that a payload is validated before it reaches the network, and so the
# publish gate has something concrete to show the user before anything is sent.
class CreateArtistInput(BaseModel):
    """Payload for `artists.createArtist`. Built up over the conversation, sent once."""

    name: str = Field(description="The artist or band name. The backend requires this.")
    genre: str | None = None
    bio: str | None = None
    image_url: str | None = None


class UploadSongInput(BaseModel):
    """Payload for `songs.getUploadUrl` — step one of publishing a song."""

    artist_id: str = Field(description="The label backend's id for the artist.")
    filename: str = Field(description="The audio file's name, e.g. 'midnight-drive.mp3'.")


class CreateSongInput(BaseModel):
    """Payload for `songs.createSong` — step two, once the file is uploaded."""

    artist_id: str = Field(description="The label backend's id for the artist.")
    title: str = Field(description="The song title, as it should be displayed.")
    s3_key: str = Field(description="The s3Key handed back by getUploadUrl.")


# -------------------------------------------------------------
# Forecast — written by the weather tool, never by an extractor
# -------------------------------------------------------------
class ForecastDay(BaseModel):
    """One day of the forecast, used to match music to the weather someone is listening in."""

    date: str
    temp_min_f: float | None = None
    temp_max_f: float | None = None
    precipitation_in: float | None = None
    precipitation_chance_pct: int | None = None
    wind_max_mph: float | None = None


class MonthAverage(BaseModel):
    """What one calendar month was actually like, averaged from the Open-Meteo archive.

    Used when a date is beyond the forecast horizon. The archive carries measured
    precipitation but no chance-of-rain — ERA5 does not model one — so there is
    deliberately no percentage here to mistake for a forecast probability.
    """

    month: str = Field(description="The month sampled, as YYYY-MM.")
    days_sampled: int
    avg_low_f: float | None = None
    avg_high_f: float | None = None
    coldest_low_f: float | None = None
    warmest_high_f: float | None = None
    total_precip_in: float | None = None
    max_wind_mph: float | None = None


class SongForecast(BaseModel):
    """Weather for a place and date range, as fetched from Open-Meteo.

    Kept so the assistant can pick music against the weather someone is actually in — a wet
    grey week and a bright one call for different records.
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


# -------------------------------------------------------------
# Song catalog
# -------------------------------------------------------------
class Song(BaseModel):
    """A row in the local song catalog."""

    song_id: str
    name: str
    artist: str
    genre: str
    description: str
    mood: str | None = None
    year: int | None = None
    duration_sec: int | None = None
    tags: list[str] = Field(default_factory=list)
