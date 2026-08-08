"""CRUD tools for the its-the-label GraphQL backend.

Two things about this API shape the whole module.

First, **every operation is namespaced one level deep**. `Query` and `Mutation` expose only
domain containers; the real fields hang off `ArtistsQuery`, `SongsMutation` and friends. So a
document is never `mutation { createArtist(...) }` but always
`mutation { artists { createArtist(...) } }`. Getting this wrong fails at validation.

Second, **API Gateway needs two identity sources**: an `Authorization` header *and* a `?type=`
query-string parameter. Miss either and the request is rejected with a plain
`401 {"message": "Unauthorized"}` before any Lambda runs — which is not a GraphQL error and
does not come back in a GraphQL envelope. `type=pageview` takes the authorizer's guest path,
where the token is not inspected at all and `Bearer public` is what the web UI sends. Any
other `type` value makes it decode the token, which must be a Cognito **ID** token.

That mandatory query string is why this talks to the endpoint with `requests` directly rather
than through a GraphQL client library: the parameter belongs to the URL, not the payload, and
transports fight you about it. It also keeps the module dependency-free, matching
`app/weather_forcast.py`.

As in that module, the network half is split out from the `@tool` wrappers so it can be
exercised without a graph or a store, and failures come back as `{"error": ...}` dicts rather
than exceptions — the agent can read an error and say something useful about it; it cannot
catch one.

The caller's ID token is threaded through as an argument rather than read from a global. The
UI holds the signed-in user's token and posts it with each run, so it arrives on the run's
config and reaches the tools through `id_token_from_config` — see app/identity.py for why the
key it travels under is named the way it is. One process serves many users, and a token cached
in module state would be the wrong user's before long.
"""

import json
import os
from typing import Any

import requests
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.identity import id_token_from_config

# The public deployed endpoint. Overridable so the client can be pointed at a local Yoga
# server or a different stage without a code change.
GRAPHQL_URL = os.getenv("ITL_GRAPHQL_URL", "https://itsthelabel.com/graphql")
_TIMEOUT = 15

# The authorizer's guest path: `type=pageview` short-circuits before the token is looked at,
# so the literal string "public" is a valid bearer token there. This is what the web UI uses.
_PUBLIC_TOKEN = "public"
_PUBLIC_TYPE = "pageview"

# Any value other than "pageview" sends the authorizer down the JWT branch instead.
_AUTHED_TYPE = os.getenv("ITL_AUTH_TYPE", "artist")


def execute(
    document: str,
    variables: dict[str, Any] | None = None,
    *,
    id_token: str | None = None,
    require_auth: bool = False,
) -> dict[str, Any]:
    """POST one GraphQL document. Returns the `data` object, or `{"error": ...}`.

    The only place in this module that knows about HTTP.

    `id_token` is the caller's own Cognito ID token, passed down from the run rather than read
    from anywhere global — two users on one server must not share a token. Given one, the
    request takes the authorizer's JWT branch; without one it falls back to a public read.
    `require_auth` marks an operation the backend gates behind an artist account, and fails
    fast with a readable message instead of sending `Bearer public` at a gated field and
    earning an unexplained "Forbidden".
    """
    if id_token:
        token, type_ = id_token, _AUTHED_TYPE
    elif require_auth:
        return {
            "error": "This operation needs a signed-in artist, but the run carried no "
            "Cognito ID token. Sign in and retry."
        }
    else:
        token, type_ = _PUBLIC_TOKEN, _PUBLIC_TYPE

    try:
        response = requests.post(
            GRAPHQL_URL,
            params={"type": type_},
            json={"query": document, "variables": variables or {}},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            timeout=_TIMEOUT,
        )
    except requests.RequestException as exc:
        return {"error": f"Could not reach the label backend: {exc}"}

    try:
        payload = response.json()
    except json.JSONDecodeError:
        return {
            "error": f"The label backend returned a non-JSON response "
            f"(HTTP {response.status_code}): {response.text[:200]!r}"
        }

    # Three different failure envelopes come back from this stack and they look nothing alike.
    # API Gateway rejects before GraphQL is reached and answers in its own shape — no `data`,
    # no `errors`, just `message` — so check for that first.
    if "data" not in payload and "errors" not in payload:
        return {"error": _gateway_error(response, payload)}

    if payload.get("errors"):
        errors = payload["errors"]
        # `maskedErrors` is off on the server, so these messages are the resolver's own words
        # ("Forbidden", "Artist ... already exists.") and are worth passing through verbatim.
        return {"error": errors[0].get("message", "Unknown GraphQL error."), "errors": errors}

    return payload.get("data") or {}


def _gateway_error(response: requests.Response, payload: dict[str, Any]) -> str:
    """Turn an API Gateway rejection into something a reader can act on."""
    # The single most useful diagnostic available: it means API Gateway could not invoke the
    # custom authorizer Lambda at all, so every request fails whatever it asks for.
    if response.headers.get("x-amzn-errortype") == "AuthorizerConfigurationException":
        return (
            "The label backend's API Gateway authorizer is misconfigured and is rejecting "
            "every request (AuthorizerConfigurationException). This is a fault in the "
            "backend deployment, not in the query."
        )

    if response.status_code == 401:
        return (
            "The label backend rejected the credentials. Public reads need "
            "'Bearer public' with type=pageview; anything else needs a valid Cognito "
            "ID token in ITL_ID_TOKEN."
        )

    message = payload.get("message") or response.text[:200]
    return f"The label backend refused the request (HTTP {response.status_code}): {message}"


def _unwrap(data: dict[str, Any], *path: str) -> Any:
    """Walk the namespaced response, tolerating a null at any level.

    A resolver that throws leaves its whole branch null, so `data["artists"]["listArtists"]`
    would raise on exactly the responses worth reporting on.
    """
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


# -------------------------------------------------------------
# Artists
# -------------------------------------------------------------
_LIST_ARTISTS = """
query ListArtists($input: ListArtistsInput!) {
  artists {
    listArtists(input: $input) {
      data { id name artistSlug genre bio imageUrl }
      meta { limit hasNextPage }
    }
  }
}
"""

_GET_ARTIST = """
query GetArtist($artistSlug: String) {
  artists {
    getArtist(artistSlug: $artistSlug) {
      data { id name artistSlug genre bio imageUrl createdAt }
    }
  }
}
"""

_GET_ARTIST_BY_ID = """
query GetArtistById($id: ID!) {
  artists {
    getArtistById(id: $id) {
      data { id name artistSlug genre bio imageUrl createdAt }
    }
  }
}
"""

_CREATE_ARTIST = """
mutation CreateArtist($input: CreateArtistInput!) {
  artists {
    createArtist(input: $input) { id name artistSlug genre bio imageUrl createdAt }
  }
}
"""


def fetch_artists(
    limit: int = 10, genre: str | None = None
) -> dict[str, Any]:
    """List artists signed to the label. Returns `{"artists": [...], "meta": {...}}` or an error.

    `meta.hasNextPage` is reported by the backend but there is deliberately no cursor here:
    the resolver ignores the `cursor` argument, so offering paging would be a promise the
    server does not keep.
    """
    variables: dict[str, Any] = {"input": {"limit": limit}}
    if genre:
        variables["input"]["genre"] = genre

    data = execute(_LIST_ARTISTS, variables)
    if "error" in data:
        return data

    result = _unwrap(data, "artists", "listArtists")
    if result is None:
        return {"error": "The label backend returned no artist list."}
    return {"artists": result.get("data") or [], "meta": result.get("meta") or {}}


def fetch_artist(artist_slug: str) -> dict[str, Any]:
    """Fetch one artist by slug. Returns the artist record or an error."""
    data = execute(_GET_ARTIST, {"artistSlug": artist_slug})
    if "error" in data:
        return data

    artist = _unwrap(data, "artists", "getArtist", "data")
    if artist is None:
        return {"error": f"No artist found with slug {artist_slug!r}."}
    return artist


def fetch_artist_by_id(artist_id: str) -> dict[str, Any]:
    """Fetch one artist by id. Returns the artist record or an error."""
    data = execute(_GET_ARTIST_BY_ID, {"id": artist_id})
    if "error" in data:
        return data

    artist = _unwrap(data, "artists", "getArtistById", "data")
    if artist is None:
        return {"error": f"No artist found with id {artist_id!r}."}
    return artist


def create_artist(
    name: str,
    genre: str | None = None,
    bio: str | None = None,
    image_url: str | None = None,
    id_token: str | None = None,
) -> dict[str, Any]:
    """Create an artist profile. Returns the created record or an error.

    Only `name` is required by the schema; the optional fields are omitted rather than sent
    as nulls so a later update cannot be mistaken for a deliberate blanking.

    The backend puts no guard on this field, so a token is used when the caller has one but
    is not demanded — signing up is the step that happens *before* there is an account.
    """
    artist_input: dict[str, Any] = {"name": name}
    if genre:
        artist_input["genre"] = genre
    if bio:
        artist_input["bio"] = bio
    if image_url:
        artist_input["imageUrl"] = image_url

    data = execute(_CREATE_ARTIST, {"input": artist_input}, id_token=id_token)
    if "error" in data:
        return data

    artist = _unwrap(data, "artists", "createArtist")
    if artist is None:
        return {"error": f"The label backend did not create an artist named {name!r}."}
    return artist


# -------------------------------------------------------------
# Songs
# -------------------------------------------------------------
_GET_SONGS = """
query GetSongs($artistId: ID!, $limit: Int) {
  songs {
    getSongs(artistId: $artistId, limit: $limit) {
      id artistId title s3Key url createdAt
    }
  }
}
"""

_GET_UPLOAD_URL = """
mutation GetUploadUrl($artistId: ID!, $filename: String!) {
  songs {
    getUploadUrl(artistId: $artistId, filename: $filename) { url fields s3Key }
  }
}
"""

_CREATE_SONG = """
mutation CreateSong($artistId: ID!, $title: String!, $s3Key: String!) {
  songs {
    createSong(artistId: $artistId, title: $title, s3Key: $s3Key) {
      id artistId title s3Key url createdAt
    }
  }
}
"""


def fetch_songs(artist_id: str, limit: int = 20) -> list[dict[str, Any]] | dict[str, Any]:
    """List an artist's songs, newest first. Returns a list, or a dict with an error.

    The backend caps `limit` at 50 whatever is asked for.
    """
    data = execute(_GET_SONGS, {"artistId": artist_id, "limit": limit})
    if "error" in data:
        return data

    songs = _unwrap(data, "songs", "getSongs")
    return songs if songs is not None else []


def request_song_upload(
    artist_id: str, filename: str, id_token: str | None = None
) -> dict[str, Any]:
    """Ask for a presigned S3 target to upload an audio file to.

    Returns `{"url", "fields", "s3Key"}` — a presigned POST, not a PUT: the file has to be
    sent as multipart form data with every entry of `fields` included alongside it. Gated on
    the artist role, so it needs the caller's ID token.
    """
    data = execute(
        _GET_UPLOAD_URL,
        {"artistId": artist_id, "filename": filename},
        id_token=id_token,
        require_auth=True,
    )
    if "error" in data:
        return data

    target = _unwrap(data, "songs", "getUploadUrl")
    if target is None:
        return {"error": f"The label backend gave no upload target for {filename!r}."}
    return target


def publish_song(
    artist_id: str, title: str, s3_key: str, id_token: str | None = None
) -> dict[str, Any]:
    """Record an already-uploaded file as a song. Returns the created record or an error.

    `s3_key` must be the `s3Key` handed back by `request_song_upload`, and the file must
    already be at that key — the backend checks it exists and rejects anything else.

    Named apart from the `register_song` tool below on purpose: `@tool` rebinds the module
    global, so a tool sharing its wrapped function's name would call itself.
    """
    data = execute(
        _CREATE_SONG,
        {"artistId": artist_id, "title": title, "s3Key": s3_key},
        id_token=id_token,
        require_auth=True,
    )
    if "error" in data:
        return data

    song = _unwrap(data, "songs", "createSong")
    if song is None:
        return {"error": f"The label backend did not create a song titled {title!r}."}
    return song


# -------------------------------------------------------------
# Tools
# -------------------------------------------------------------
@tool
def list_label_artists(limit: int = 10, genre: str | None = None) -> dict[str, Any]:
    """Lists the music artists signed to the label.

    Use when the user wants to browse who is on the label, or to find artists in a genre.
    This is the live backend roster, which is different from the local song catalog.

    Args:
        limit: How many artists to return at most. Defaults to 10.
        genre: Optional genre to filter by, e.g. 'rock', 'pop', 'jazz'.

    Returns:
        The matching artists and the list metadata, or an error message explaining why not.
    """
    return fetch_artists(limit=limit, genre=genre)


@tool
def get_label_artist(artist_slug: str) -> dict[str, Any]:
    """Looks up one artist on the label by their slug.

    Use when the user asks about a specific artist and you have their slug from
    list_label_artists. Returns more detail than the list does.

    Args:
        artist_slug: The artist's slug, exactly as returned by list_label_artists,
            e.g. 'the-quiet-hours'.

    Returns:
        The artist's full record, or an error message if there is no such artist.
    """
    return fetch_artist(artist_slug)


@tool
def list_artist_songs(artist_id: str, limit: int = 20) -> list[dict[str, Any]] | dict[str, Any]:
    """Lists the songs an artist has published, newest first.

    Use after finding an artist to see what they have released. Each song comes with a
    playable url.

    Args:
        artist_id: The artist's id, as returned by list_label_artists or get_label_artist.
        limit: How many songs to return at most. Defaults to 20, capped at 50 by the backend.

    Returns:
        The artist's songs, or an error message explaining why they could not be fetched.
    """
    return fetch_songs(artist_id=artist_id, limit=limit)


@tool
def get_song_upload_url(
    artist_id: str, filename: str, config: RunnableConfig
) -> dict[str, Any]:
    """Gets a presigned upload target for a new song file. Step 1 of 2.

    Publishing a song takes three steps and you can only make one tool call per turn, so do
    them in order: call this to get an upload target, tell the user to upload the file to the
    returned url as a multipart form POST including every field in `fields`, and only then
    call register_song with the s3_key this returned. Requires a signed-in artist account.

    Args:
        artist_id: The id of the artist the song belongs to.
        filename: The audio file's name, e.g. 'midnight-drive.mp3'.

    Returns:
        The upload url, the form fields that must accompany the file, and the s3_key to pass
        to register_song — or an error message.
    """
    return request_song_upload(
        artist_id=artist_id, filename=filename, id_token=id_token_from_config(config)
    )


@tool
def register_song(
    artist_id: str, title: str, s3_key: str, config: RunnableConfig
) -> dict[str, Any]:
    """Publishes an uploaded song file as a song on the label. Step 2 of 2.

    Only call this once the file has actually been uploaded to the target from
    get_song_upload_url — the backend checks the file is there and rejects the call if it is
    not. Requires a signed-in artist account.

    Args:
        artist_id: The id of the artist the song belongs to.
        title: The song title, as it should be displayed.
        s3_key: The s3_key returned by get_song_upload_url for this file.

    Returns:
        The published song record including its playable url, or an error message.
    """
    return publish_song(
        artist_id=artist_id,
        title=title,
        s3_key=s3_key,
        id_token=id_token_from_config(config),
    )


# What the agent may call directly. `create_artist` is deliberately absent: creating an
# artist is a public, irreversible write with no delete on the backend, so it goes through
# the publish gate in app/graph.py, which shows the user the payload and waits for a yes.
# Giving the model the same power as a plain tool would route around that gate entirely.
#
# `register_song` needs no such gate: it cannot fire meaningfully without an s3_key from an
# upload the user physically performed, so the upload is itself the deliberate human step.
backend_tools = [
    list_label_artists,
    get_label_artist,
    list_artist_songs,
    get_song_upload_url,
    register_song,
]


if __name__ == "__main__":
    # Smoke test against whatever ITL_GRAPHQL_URL points at: uv run python -m app.graphql_backend_client
    from app.env import load_env_file

    load_env_file()
    print(f"POST {GRAPHQL_URL}?type={_PUBLIC_TYPE}")
    print(json.dumps(fetch_artists(limit=3), indent=2))
