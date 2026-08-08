"""The label's song catalog and the tools the agent uses to search it.

The catalog is seed data in app/data/songs.json. `search_songs` is the seam the label's
own backend slots into — the agent only ever sees the tool signature. It is kept local and
offline on purpose: discovery has to work when the backend does not, and the backend tools
in app/graphql_backend_client.py cover the live roster separately.
"""

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from app.schemas import Song

_CATALOG_PATH = Path(__file__).resolve().parent / "data" / "songs.json"

CATALOG: list[Song] = [
    Song.model_validate(row) for row in json.loads(_CATALOG_PATH.read_text())
]
_BY_ID: dict[str, Song] = {song.song_id: song for song in CATALOG}
GENRES: list[str] = sorted({song.genre for song in CATALOG})
MOODS: list[str] = sorted({song.mood for song in CATALOG if song.mood})

# Results per search. Small on purpose: the whole list would crowd out the conversation,
# and a shortlist the user can actually hold in their head is the point.
_MAX_RESULTS = 5


def song_by_id(song_id: str) -> Song | None:
    """Plain lookup, for code that needs a song without going through the tool."""
    return _BY_ID.get(song_id)


def _summarize(song: Song) -> dict[str, Any]:
    """The view the model sees — the full record comes from get_song."""
    summary: dict[str, Any] = {
        "song_id": song.song_id,
        "name": song.name,
        "artist": song.artist,
        "genre": song.genre,
        "description": song.description,
    }
    for field in ("mood", "year", "duration_sec"):
        value = getattr(song, field)
        if value is not None:
            summary[field] = value
    return summary


@tool
def search_songs(
    genre: str | None = None,
    artist: str | None = None,
    mood: str | None = None,
    tag: str | None = None,
    max_duration_sec: int | None = None,
    released_since: int | None = None,
) -> list[dict[str, Any]]:
    """Search the label's song catalog for music matching what a listener has asked for.

    Use this before recommending anything, so recommendations only ever cite real songs on
    the label. Every filter is optional — calling it with none returns a cross-section of
    the catalog, which is a reasonable way to start with someone who has not said much yet.

    Args:
        genre: One of ambient, afrobeat, country, electronic, folk, jazz, post-rock, punk,
            shoegaze, soul.
        artist: Restrict to one artist by name.
        mood: How the song feels, e.g. calm, melancholy, joyful, angry, euphoric.
        tag: A single descriptor to match, e.g. 'instrumental', 'quiet', 'danceable',
            'guitar', 'long'.
        max_duration_sec: Longest acceptable track, in seconds.
        released_since: Only songs released in this year or later.

    Returns:
        Up to 5 matching songs, newest first. Empty list if nothing matches — say so
        rather than inventing an alternative.
    """
    matches: list[Song] = []
    for song in CATALOG:
        if genre is not None and song.genre.lower() != genre.lower():
            continue
        if artist is not None and artist.lower() not in song.artist.lower():
            continue
        if mood is not None and (song.mood or "").lower() != mood.lower():
            continue
        if tag is not None and tag.lower() not in [t.lower() for t in song.tags]:
            continue
        if max_duration_sec is not None and (
            song.duration_sec is None or song.duration_sec > max_duration_sec
        ):
            continue
        if released_since is not None and (song.year is None or song.year < released_since):
            continue
        matches.append(song)

    matches.sort(key=lambda s: s.year or 0, reverse=True)
    return [_summarize(song) for song in matches[:_MAX_RESULTS]]


@tool
def get_song(song_id: str) -> dict[str, Any]:
    """Look up the full details of one catalog song by its id.

    Use when comparing a shortlist, or when the user asks about a specific track.

    Args:
        song_id: The catalog id, exactly as returned by search_songs.

    Returns:
        The song's full record, or an error message if the id is not in the catalog.
    """
    song = _BY_ID.get(song_id)
    if song is None:
        return {"error": f"No song with id {song_id!r} in the catalog."}
    return song.model_dump()


catalog_tools = [search_songs, get_song]
