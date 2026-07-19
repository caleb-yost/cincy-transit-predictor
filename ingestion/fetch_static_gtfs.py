"""Download the static GTFS schedule zip and extract key tables to Parquet.

These reference tables (the *scheduled* times) are joined against the realtime
*predicted* arrivals to compute delay. Run weekly (schedule changes ~quarterly).

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


def main() -> None:
    feeds = get_feeds()
    print(f"Downloading static GTFS: {feeds.static_gtfs}")
    resp = requests.get(feeds.static_gtfs, timeout=REQUEST_TIMEOUT, headers=HTTP_HEADERS)
    resp.raise_for_status()

    out_dir = Path(DATA_DIR) / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        available = set(zf.namelist())
        for fname, table in WANTED.items():
            if fname not in available:
                print(f"  [skip] {fname} not present in feed")
                continue
            with zf.open(fname) as handle:
                df = pd.read_csv(handle, dtype=str)
            out_path = out_dir / f"{table}.parquet"
            df.to_parquet(out_path, index=False)
            print(f"  {table}: {len(df)} rows -> {out_path}")
    print("static GTFS refresh complete.")


if __name__ == "__main__":
    main()
