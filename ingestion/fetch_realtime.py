"""Fetch one snapshot of the agency's GTFS-realtime feeds + Open-Meteo weather.

Writes three partitioned Parquet files under ``DATA_DIR``:
    raw/trip_updates/date=YYYY-MM-DD/trip_updates_<UTC>.parquet   <- predicted arrivals (delay label source)
    raw/vehicle_positions/date=YYYY-MM-DD/vehicle_positions_<UTC>.parquet
    raw/weather/date=YYYY-MM-DD/weather_<UTC>.parquet

Run: ``python ingestion/fetch_realtime.py`` (or ``python -m ingestion.fetch_realtime``).
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests
from google.transit import gtfs_realtime_pb2

try:
    from ingestion.feeds import DATA_DIR, HTTP_HEADERS, OPEN_METEO_URL, get_feeds
except ModuleNotFoundError:  # allow direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ingestion.feeds import DATA_DIR, HTTP_HEADERS, OPEN_METEO_URL, get_feeds

REQUEST_TIMEOUT = 30
RETRIES = 3


def _get(url: str, **kwargs) -> requests.Response:
    """GET with browser UA + exponential-backoff retry (SORTA feed is 'as is', no uptime SLA)."""
    headers = {**HTTP_HEADERS, **kwargs.pop("headers", {})}
    last_err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as err:
            last_err = err
            time.sleep(2**attempt)
    raise RuntimeError(f"GET failed after {RETRIES} attempts: {url}") from last_err


def _parse_feed(content: bytes) -> gtfs_realtime_pb2.FeedMessage:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(content)
    return feed


def fetch_trip_updates(url: str, snapshot_ts: int) -> pd.DataFrame:
    """Flatten TripUpdate.stop_time_update entries into one row per predicted stop arrival."""
    feed = _parse_feed(_get(url).content)
    rows = []
    for ent in feed.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        trip = tu.trip
        vehicle_id = tu.vehicle.id if tu.HasField("vehicle") else None
        for stu in tu.stop_time_update:
            arrival = stu.arrival.time if stu.HasField("arrival") and stu.arrival.time else None
            departure = (
                stu.departure.time if stu.HasField("departure") and stu.departure.time else None
            )
            rows.append(
                {
                    "snapshot_ts": snapshot_ts,
                    "trip_id": trip.trip_id or None,
                    "route_id": trip.route_id or None,
                    "start_date": trip.start_date or None,
                    "vehicle_id": vehicle_id,
                    "stop_id": stu.stop_id or None,
                    "stop_sequence": stu.stop_sequence or None,
                    "predicted_arrival": arrival,
                    "predicted_departure": departure,
                    "schedule_relationship": stu.schedule_relationship,
                }
            )
    return pd.DataFrame(rows)


def fetch_vehicle_positions(url: str, snapshot_ts: int) -> pd.DataFrame:
    """One row per vehicle with its current GPS position."""
    feed = _parse_feed(_get(url).content)
    rows = []
    for ent in feed.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        pos = v.position
        rows.append(
            {
                "snapshot_ts": snapshot_ts,
                "vehicle_id": v.vehicle.id or None,
                "trip_id": v.trip.trip_id or None,
                "route_id": v.trip.route_id or None,
                "latitude": pos.latitude or None,
                "longitude": pos.longitude or None,
                "bearing": pos.bearing or None,
                "speed": pos.speed or None,
                "current_stop_sequence": v.current_stop_sequence or None,
                "current_status": v.current_status,
                "vehicle_ts": v.timestamp or None,
            }
        )
    return pd.DataFrame(rows)


def fetch_weather(lat: float, lon: float, snapshot_ts: int) -> pd.DataFrame:
    """Current conditions from Open-Meteo (free, no API key) — second data source / model feature."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,rain,snowfall,wind_speed_10m,weather_code",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
    }
    current = _get(OPEN_METEO_URL, params=params).json().get("current", {})
    return pd.DataFrame(
        [
            {
                "snapshot_ts": snapshot_ts,
                "temperature_f": current.get("temperature_2m"),
                "precipitation_in": current.get("precipitation"),
                "rain_in": current.get("rain"),
                "snowfall_in": current.get("snowfall"),
                "wind_speed_mph": current.get("wind_speed_10m"),
                "weather_code": current.get("weather_code"),
            }
        ]
    )


def _write(df: pd.DataFrame, feed_name: str, snapshot_dt: datetime) -> Path:
    date_str = snapshot_dt.strftime("%Y-%m-%d")
    stamp = snapshot_dt.strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(DATA_DIR) / "raw" / feed_name / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{feed_name}_{stamp}.parquet"
    df.to_parquet(out_path, index=False)
    marker = "" if len(df) else "  [warn] 0 rows"
    print(f"  {feed_name}: {len(df)} rows -> {out_path}{marker}")
    return out_path


def main() -> None:
    feeds = get_feeds()
    now = datetime.now(UTC)
    snapshot_ts = int(now.timestamp())
    print(f"[{now.isoformat()}] agency={feeds.name}")

    trip_updates = fetch_trip_updates(feeds.trip_updates, snapshot_ts)
    vehicles = fetch_vehicle_positions(feeds.vehicle_positions, snapshot_ts)
    weather = fetch_weather(feeds.lat, feeds.lon, snapshot_ts)

    _write(trip_updates, "trip_updates", now)
    _write(vehicles, "vehicle_positions", now)
    _write(weather, "weather", now)

    if trip_updates.empty and vehicles.empty:
        raise SystemExit("No realtime data returned — feeds may be down.")
    print("snapshot complete.")


if __name__ == "__main__":
    main()
