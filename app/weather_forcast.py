"""Trip weather, from Open-Meteo.

The forecast is written straight to the store by the tool itself rather than being handed
to an extractor to transcribe. It arrives structured from the API, so putting a model in
the write path could only lose or garble it — and it would make the write conditional on
the model choosing to call MemoryRouter that turn.

Open-Meteo needs no API key. Its daily forecast reaches roughly two weeks ahead. Past that
the tool falls back to the historical archive for the same calendar month a year earlier and
records it as *typical conditions* — never as a forecast, which `TripForecast.basis` marks
on the record itself.
"""

import calendar
import datetime
from typing import Annotated, Any

import pytz
import requests
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore

from app import memory
from app.identity import user_id_from_config
from app.schemas import ForecastDay, MonthAverage, TripForecast

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_TIMEOUT = 10
_MAX_NIGHTS = 14
# The archive lags real time by a few days, so a month that has only just ended may be
# incomplete. Stepping back a month costs nothing and avoids a half-empty average.
_ARCHIVE_LAG_DAYS = 10


def _geocode(location: str) -> dict[str, Any] | None:
    response = requests.get(
        GEOCODE_URL, params={"name": location, "count": 1}, timeout=_TIMEOUT
    )
    results = response.json().get("results")
    return results[0] if results else None


def _parse_start(start_date: str) -> datetime.date | dict[str, Any]:
    try:
        return datetime.date.fromisoformat(start_date)
    except (TypeError, ValueError):
        return {"error": f"start_date must be an ISO date like 2026-08-14, got {start_date!r}."}


def fetch_forecast(
    location: str, start_date: str, nights: int = 1, place: dict[str, Any] | None = None
) -> TripForecast | dict[str, Any]:
    """Call Open-Meteo. Returns a TripForecast, or a dict with an `error` explaining why not.

    Split out from the tool so the network half can be exercised without a graph or a store.
    `place` may be supplied by a caller that has already geocoded, to save a round trip.
    """
    start = _parse_start(start_date)
    if isinstance(start, dict):
        return start

    end = start + datetime.timedelta(days=max(0, min(nights, _MAX_NIGHTS)))

    try:
        place = place or _geocode(location)
        if place is None:
            return {"error": f"Could not find a location matching '{location}'."}

        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": ",".join([
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                    "wind_speed_10m_max",
                ]),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": place.get("timezone") or "auto",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
            timeout=_TIMEOUT,
        )
        payload = response.json()
    except requests.RequestException as exc:
        return {"error": f"Could not reach the weather service: {exc}"}

    # Out-of-range dates come back as a 400 with a readable reason — pass it on rather than
    # inventing a forecast the API declined to give.
    if payload.get("error"):
        return {
            "error": f"No forecast available for {location} on {start_date}: "
            f"{payload.get('reason', 'out of range')}."
        }

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    if not dates:
        return {"error": f"No forecast data returned for {location}."}

    def column(name: str) -> list[Any]:
        values = daily.get(name) or []
        return list(values) + [None] * (len(dates) - len(values))

    days = [
        ForecastDay(
            date=date,
            temp_min_f=lo,
            temp_max_f=hi,
            precipitation_in=rain,
            precipitation_chance_pct=chance,
            wind_max_mph=wind,
        )
        for date, lo, hi, rain, chance, wind in zip(
            dates,
            column("temperature_2m_min"),
            column("temperature_2m_max"),
            column("precipitation_sum"),
            column("precipitation_probability_max"),
            column("wind_speed_10m_max"),
        )
    ]

    return TripForecast(
        location=place["name"],
        latitude=place["latitude"],
        longitude=place["longitude"],
        start_date=dates[0],
        end_date=dates[-1],
        basis="forecast",
        days=days,
    )


def _last_finished_occurrence(month: int, year: int, today: datetime.date) -> int:
    """Step `year` back until that month has finished and the archive should hold it.

    A trip more than a year out would otherwise ask the archive for a month that has not
    happened yet, and get an empty answer with no explanation.
    """
    cutoff = today - datetime.timedelta(days=_ARCHIVE_LAG_DAYS)
    while True:
        end_day = calendar.monthrange(year, month)[1]
        if datetime.date(year, month, end_day) <= cutoff:
            return year
        year -= 1


def _archive_month(place: dict[str, Any], year: int, month: int) -> dict[str, Any]:
    """Raw daily archive rows for one calendar month."""
    end_day = calendar.monthrange(year, month)[1]
    response = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            # No precipitation_probability_max: the archive returns it as all-null, since
            # ERA5 does not model a chance of rain. Only measured precipitation is real here.
            "daily": ",".join([
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
            ]),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": place.get("timezone") or "auto",
            "start_date": f"{year:04d}-{month:02d}-01",
            "end_date": f"{year:04d}-{month:02d}-{end_day:02d}",
        },
        timeout=_TIMEOUT,
    )
    return response.json()


def _numbers(values: Any) -> list[float]:
    """Drop gaps rather than letting one missing day poison an average."""
    return [v for v in (values or []) if isinstance(v, (int, float))]


def fetch_month_average(
    place: dict[str, Any], trip_start: datetime.date, today: datetime.date | None = None
) -> MonthAverage | dict[str, Any]:
    """What the trip's calendar month was like a year ago, averaged.

    Used when the trip is past the forecast horizon. This is climatology, not prediction —
    the caller must mark it as such on the record.
    """
    today = today or datetime.date.today()
    year = _last_finished_occurrence(trip_start.month, trip_start.year - 1, today)

    try:
        payload = _archive_month(place, year, trip_start.month)
        daily = payload.get("daily") or {}
        lows = _numbers(daily.get("temperature_2m_min"))
        highs = _numbers(daily.get("temperature_2m_max"))

        # Fallback within the fallback: if the whole month came back empty, sample a single
        # day from it rather than giving up entirely.
        if not lows and not highs:
            mid = f"{year:04d}-{trip_start.month:02d}-15"
            response = requests.get(
                ARCHIVE_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                    "precipitation_unit": "inch",
                    "timezone": place.get("timezone") or "auto",
                    "start_date": mid,
                    "end_date": mid,
                },
                timeout=_TIMEOUT,
            )
            daily = response.json().get("daily") or {}
            lows = _numbers(daily.get("temperature_2m_min"))
            highs = _numbers(daily.get("temperature_2m_max"))
    except requests.RequestException as exc:
        return {"error": f"Could not reach the weather archive: {exc}"}

    if not lows and not highs:
        return {
            "error": f"No historical weather on file for {place['name']} in "
            f"{year:04d}-{trip_start.month:02d}."
        }

    rain = _numbers(daily.get("precipitation_sum"))
    wind = _numbers(daily.get("wind_speed_10m_max"))

    def mean(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 1) if values else None

    return MonthAverage(
        month=f"{year:04d}-{trip_start.month:02d}",
        days_sampled=max(len(lows), len(highs)),
        avg_low_f=mean(lows),
        avg_high_f=mean(highs),
        coldest_low_f=min(lows) if lows else None,
        warmest_high_f=max(highs) if highs else None,
        total_precip_in=round(sum(rain), 2) if rain else None,
        max_wind_mph=max(wind) if wind else None,
    )


def fetch_trip_weather(
    location: str, start_date: str, nights: int = 1
) -> TripForecast | dict[str, Any]:
    """A real forecast if the trip is close enough, otherwise typical past conditions.

    Which one you get is decided by Open-Meteo rejecting the range, not by a hardcoded
    horizon here — the API stays authoritative about how far ahead it can see.
    """
    start = _parse_start(start_date)
    if isinstance(start, dict):
        return start

    try:
        place = _geocode(location)
    except requests.RequestException as exc:
        return {"error": f"Could not reach the weather service: {exc}"}
    if place is None:
        return {"error": f"Could not find a location matching '{location}'."}

    forecast = fetch_forecast(location, start_date, nights, place=place)
    if isinstance(forecast, TripForecast):
        return forecast

    average = fetch_month_average(place, start)
    if isinstance(average, dict):
        # Both routes failed. Report both reasons: showing only "out of range" would hide
        # that a fallback was attempted at all, and a transient archive outage would look
        # like the forecast horizon being the whole story.
        return {
            "error": f"{forecast.get('error', 'No forecast available.')} "
            f"Historical fallback also failed: {average.get('error', 'unknown reason')}"
        }

    end = start + datetime.timedelta(days=max(0, min(nights, _MAX_NIGHTS)))
    return TripForecast(
        location=place["name"],
        latitude=place["latitude"],
        longitude=place["longitude"],
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        basis="historical",
        years_sampled=[int(average.month[:4])],
        month_average=average,
    )


@tool
def weather_forcast_tool(
    location: str,
    start_date: str,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
    nights: int = 1,
) -> dict[str, Any]:
    """Fetches the weather for a trip and saves it to the trip record.

    Call this once you know where and when the customer is going. The result is saved
    automatically, so later turns can read it from the trip context rather than calling
    this again.

    Within about two weeks it returns a real day-by-day forecast (`basis: "forecast"`).
    Further out no forecast exists, so it returns what that month was actually like a year
    ago (`basis: "historical"`, with `month_average`). Treat that as typical conditions to
    plan against, never as a prediction, and tell the customer which one they are getting.

    Args:
        location: Place name, e.g. "Big Bear" or "Cairngorms".
        start_date: First day of the trip as an ISO date, YYYY-MM-DD.
        nights: How many nights they are out, so the whole trip window is covered.

    Returns:
        The weather record, or an `error` explaining why none is available.
    """
    result = fetch_trip_weather(location, start_date, nights)
    if isinstance(result, dict):
        return result

    # The deterministic write: straight to the store, no router call and no extractor, so
    # it cannot be skipped on a turn or garbled in transcription.
    memory.set_forecast(store, user_id_from_config(config), result)
    return result.model_dump()


def get_current_time_in_timezone(timezone: str) -> str:
    """Fetches the current local time in a specified timezone (e.g. 'America/New_York')."""
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


weather_tools = [weather_forcast_tool]
