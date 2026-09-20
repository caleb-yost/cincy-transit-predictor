"""Feature engineering shared by training and live prediction.

Keeping the feature list and single-row builder here means the Streamlit predictor constructs
exactly the columns the trained pipeline expects.
"""

from __future__ import annotations

import pandas as pd

CATEGORICAL = ["route_id", "sched_dow"]
NUMERIC = [
    "sched_hour",
    "stop_sequence",
    "is_weekend",
    "temperature_f",
    "precipitation_in",
    "snowfall_in",
    "wind_speed_mph",
    "route_recent_avg_delay",
]
FEATURES = CATEGORICAL + NUMERIC
REG_TARGET = "delay_minutes"
CLF_TARGET = "is_late"

# defaults used when a live weather value is missing at prediction time
WEATHER_DEFAULTS = {
    "temperature_f": 60.0,
    "precipitation_in": 0.0,
    "snowfall_in": 0.0,
    "wind_speed_mph": 5.0,
}


def load_labeled_frame(con) -> pd.DataFrame:
    """Pull only the columns training actually needs from the warehouse.

    ``select *`` was pulling every column, including several string/hash/timestamp ones
    (stop_delay_key, trip_id, vehicle_id, scheduled_at, predicted_at, weather_code) that
    training never touches. At a few million rows those add gigabytes of dead weight before
    the model even starts fitting.
    """
    cols = [*FEATURES, REG_TARGET, CLF_TARGET, "start_date"]
    return con.sql(f"select {', '.join(cols)} from mart_stop_delays").df()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, fill weather gaps, and drop noisy delay outliers for training."""
    df = df.copy()
    for col in ["temperature_f", "precipitation_in", "snowfall_in", "wind_speed_mph"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df["precipitation_in"] = df["precipitation_in"].fillna(0.0)
    df["snowfall_in"] = df["snowfall_in"].fillna(0.0)
    df["temperature_f"] = df["temperature_f"].fillna(df["temperature_f"].median())
    df["wind_speed_mph"] = df["wind_speed_mph"].fillna(df["wind_speed_mph"].median())

    df["is_weekend"] = df["is_weekend"].astype(int)
    df["sched_hour"] = pd.to_numeric(df["sched_hour"], errors="coerce").fillna(0).astype(int)
    df["sched_dow"] = pd.to_numeric(df["sched_dow"], errors="coerce").fillna(0).astype(int)
    df["stop_sequence"] = pd.to_numeric(df["stop_sequence"], errors="coerce").fillna(0).astype(int)
    df["route_id"] = df["route_id"].astype(str)
    if "route_recent_avg_delay" in df.columns:
        df["route_recent_avg_delay"] = pd.to_numeric(df["route_recent_avg_delay"], errors="coerce").fillna(0.0)
    else:
        # Column won't exist until the warehouse mart is rebuilt with it; neutral default until then.
        df["route_recent_avg_delay"] = 0.0

    df[REG_TARGET] = pd.to_numeric(df[REG_TARGET], errors="coerce")
    df = df.dropna(subset=[REG_TARGET])
    # buses are never realistically >1h off; clip the long tail of feed noise
    df = df[df[REG_TARGET].between(-30, 60)]

    # HistGradientBoosting's native categorical support (categorical_features="from_dtype")
    # reads this dtype directly, so the model consumes route_id/sched_dow as compact integer
    # codes instead of a one-hot expansion. At a few million rows, dense one-hot for ~60 dummy
    # columns was the single biggest memory cost in the whole pipeline (see git history:
    # 19GB locally at 7.9M rows, well past GitHub Actions' 7GB runner and the actual cause of
    # the daily transform-train OOM failures starting ~2026-09-17).
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    return df


def make_feature_row(
    route_id: str,
    sched_hour: int,
    sched_dow: int,
    stop_sequence: int,
    weather: dict | None = None,
    route_recent_avg_delay: float = 0.0,
) -> pd.DataFrame:
    """Build a single-row feature frame for on-demand prediction in the app.

    ``route_recent_avg_delay`` is a nowcast signal (how late this route has been running in the
    last ~90 min) and is only meaningful for near-term predictions; callers asking about a
    different day/hour than right now should leave it at the neutral default of 0.0.
    """
    weather = {**WEATHER_DEFAULTS, **(weather or {})}
    row = {
        "route_id": str(route_id),
        "sched_dow": int(sched_dow),
        "sched_hour": int(sched_hour),
        "stop_sequence": int(stop_sequence),
        "is_weekend": 1 if int(sched_dow) in (0, 6) else 0,
        "temperature_f": weather["temperature_f"],
        "precipitation_in": weather["precipitation_in"],
        "snowfall_in": weather["snowfall_in"],
        "wind_speed_mph": weather["wind_speed_mph"],
        "route_recent_avg_delay": float(route_recent_avg_delay),
    }
    out = pd.DataFrame([row])[FEATURES]
    # Must match training's dtype so the fitted model's categorical_features="from_dtype"
    # recognizes these columns; the model matches on category VALUES seen during training; a
    # single-row frame's own (trivial, one-value) category set doesn't need to match that.
    for col in CATEGORICAL:
        out[col] = out[col].astype("category")
    return out
