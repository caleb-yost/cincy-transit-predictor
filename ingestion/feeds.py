"""Feed endpoints and app-wide constants. No API keys required.

Switch agencies with the AGENCY env var (``sorta`` default, ``mbta`` fallback).
MBTA is a documented drop-in because its feeds use the identical GTFS-realtime
format, so a demo never hard-fails if SORTA's "as is" feed goes down.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgencyFeeds:
    name: str
    static_gtfs: str
    vehicle_positions: str
    trip_updates: str
    alerts: str
    lat: float
    lon: float


SORTA = AgencyFeeds(
    name="sorta",
    static_gtfs="https://www.go-metro.com/wp-content/uploads/2024/12/google_transit-1.zip",
    vehicle_positions="https://tmgtfsprd.sorttrpcloud.com/TMGTFSRealTimeWebService/vehicle/vehiclepositions.pb",
    trip_updates="https://tmgtfsprd.sorttrpcloud.com/TMGTFSRealTimeWebService/tripupdate/tripupdates.pb",
    alerts="https://tmgtfsprd.sorttrpcloud.com/TMGTFSRealTimeWebService/alert/alerts.pb",
    lat=39.1031,
    lon=-84.5120,
)

MBTA = AgencyFeeds(
    name="mbta",
    static_gtfs="https://cdn.mbta.com/MBTA_GTFS.zip",
    vehicle_positions="https://cdn.mbta.com/realtime/VehiclePositions.pb",
    trip_updates="https://cdn.mbta.com/realtime/TripUpdates.pb",
    alerts="https://cdn.mbta.com/realtime/Alerts.pb",
    lat=42.3601,
    lon=-71.0589,
)

AGENCIES = {"sorta": SORTA, "mbta": MBTA}


def get_feeds() -> AgencyFeeds:
    """Return the active agency feed config (env AGENCY, default sorta)."""
    return AGENCIES[os.getenv("AGENCY", "sorta").lower()]


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# SORTA's realtime endpoints sit behind a WAF that 403s the default python-requests
# User-Agent, so every request presents a standard browser UA.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
HTTP_HEADERS = {"User-Agent": USER_AGENT}

# Root for the local data lake: raw/<feed>/date=YYYY-MM-DD/*.parquet + reference/*.parquet
DATA_DIR = os.getenv("DATA_DIR", "data")
