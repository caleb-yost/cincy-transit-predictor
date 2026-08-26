"""Download the static GTFS schedule zip and extract key tables to Parquet.

These reference tables (the *scheduled* times) are joined against the realtime
*predicted* arrivals to compute delay. Run weekly (schedule changes ~quarterly).

Transit agencies commonly publish next season's GTFS zip days or weeks before it takes
effect (SORTA does this too, e.g. shipping a schedule with every calendar.txt window
starting mid-month while the current season is still running). Blindly overwriting our
reference tables with that not-yet-effective schedule breaks the realtime<->schedule join
completely (trip_ids don't overlap at all), so before committing a refresh we check that
the new schedule's trip_ids actually appear in the vehicles currently on the road. If they
don't, we skip the write and keep the schedule that's actually in effect.

GTFS trip_ids are scoped to a schedule season (SORTA's look like "2608-...": the Aug 2026
season), so once a season *legitimately* rolls over, its trip_ids stop existing in any new
schedule pull. fct_arrivals inner-joins realtime predictions to trips/stop_times by trip_id,
so `trips` and `stop_times` accumulate across refreshes (union + dedupe, latest wins on a
key conflict) instead of being overwritten -- otherwise every rollover permanently orphans
all prior realtime history from the join, even though the raw data is still on disk. The
other tables (routes/stops/calendar/calendar_dates) aren't part of that join and are fine
as plain latest-wins snapshots.

Run: ``python ingestion/fetch_static_gtfs.py`` (or ``python -m ingestion.fetch_static_gtfs``).
Writes reference/<table>.parquet under ``DATA_DIR``.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

try:
    from ingestion.feeds import DATA_DIR, HTTP_HEADERS, get_feeds
except ModuleNotFoundError:  # allow direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ingestion.feeds import DATA_DIR, HTTP_HEADERS, get_feeds

# GTFS files we need -> output parquet table name. IDs kept as strings (leading zeros matter).
WANTED = {
    "stops.txt": "stops",
    "routes.txt": "routes",
    "trips.txt": "trips",
    "stop_times.txt": "stop_times",
    "calendar.txt": "calendar",
    "calendar_dates.txt": "calendar_dates",
}
REQUEST_TIMEOUT = 120
MIN_OVERLAP_FRACTION = 0.05  # if fewer than 5% of recently-seen realtime trip_ids are in the
# new schedule, treat it as not-yet-effective rather than as a broken/changed feed.
RECENT_SNAPSHOTS_TO_SAMPLE = 12
# Tables the realtime<->schedule join depends on (trip_id, or trip_id+stop_id+stop_sequence):
# accumulate across refreshes instead of overwriting, keyed by their natural key.
ACCUMULATE_KEYS = {
    "trips": ["trip_id"],
    "stop_times": ["trip_id", "stop_id", "stop_sequence"],
}


def recent_realtime_trip_ids() -> set[str]:
    """Sample trip_ids from the most recent realtime snapshots already on disk.

    Returns an empty set if there's no realtime history yet (e.g. first-ever run), in which
    case the caller has nothing to validate against and should just accept the new schedule.
    """
    raw_dir = Path(DATA_DIR) / "raw" / "trip_updates"
    if not raw_dir.exists():
        return set()
    date_dirs = sorted(raw_dir.glob("date=*"))
    if not date_dirs:
        return set()
    files = sorted(date_dirs[-1].glob("*.parquet"))[-RECENT_SNAPSHOTS_TO_SAMPLE:]
    if not files:
        return set()
    ids: set[str] = set()
    for f in files:
        try:
            ids.update(pd.read_parquet(f, columns=["trip_id"])["trip_id"].astype(str))
        except Exception as exc:
            print(f"  [warn] couldn't read {f}: {exc}")
    return ids


def main() -> None:
    feeds = get_feeds()
    print(f"Downloading static GTFS: {feeds.static_gtfs}")
    resp = requests.get(feeds.static_gtfs, timeout=REQUEST_TIMEOUT, headers=HTTP_HEADERS)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        available = set(zf.namelist())
        tables: dict[str, pd.DataFrame] = {}
        for fname, table in WANTED.items():
            if fname not in available:
                print(f"  [skip] {fname} not present in feed")
                continue
            with zf.open(fname) as handle:
                tables[table] = pd.read_csv(handle, dtype=str)

    live_ids = recent_realtime_trip_ids()
    if live_ids and "trips" in tables:
        new_ids = set(tables["trips"]["trip_id"].astype(str))
        overlap = len(live_ids & new_ids) / len(live_ids)
        if overlap < MIN_OVERLAP_FRACTION:
            print(
                f"::warning::New static GTFS shares only {overlap:.1%} of trip_ids with vehicles "
                "currently on the road (checked against the last "
                f"{len(live_ids)} realtime trip_ids). This looks like a not-yet-effective "
                "schedule published ahead of its start date, not a real update. Keeping the "
                "existing reference tables and skipping this refresh."
            )
            return
        print(f"  overlap check passed: {overlap:.1%} of recent realtime trip_ids found in new schedule.")
    elif not live_ids:
        print("  [info] no realtime history to validate against yet; accepting schedule as-is.")

    out_dir = Path(DATA_DIR) / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    for table, df in tables.items():
        out_path = out_dir / f"{table}.parquet"
        key = ACCUMULATE_KEYS.get(table)
        if key and out_path.exists():
            existing = pd.read_parquet(out_path)
            before = len(existing)
            # New rows win on a key conflict (a legitimate correction to an already-seen trip);
            # rows whose key isn't in the new pull (a prior season) are carried forward as-is.
            df = pd.concat([existing, df]).drop_duplicates(subset=key, keep="last")
            print(f"  {table}: merged {before} existing + new rows -> {len(df)} total (accumulated)")
        df.to_parquet(out_path, index=False)
        print(f"  {table}: {len(df)} rows -> {out_path}")
    print("static GTFS refresh complete.")


if __name__ == "__main__":
    main()
