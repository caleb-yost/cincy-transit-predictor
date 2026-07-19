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
    """Pull the labeled stop-arrival table from the warehouse."""
    return con.sql("select * from mart_stop_delays").df()


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

    df[REG_TARGET] = pd.to_numeric(df[REG_TARGET], errors="coerce")
    df = df.dropna(subset=[REG_TARGET])
    # buses are never realistically >1h off; clip the long tail of feed noise
    df = df[df[REG_TARGET].between(-30, 60)]
    return df


def make_feature_row(
    route_id: str,
    sched_hour: int,
    sched_dow: int,
    stop_sequence: int,
    weather: dict | None = None,
) -> pd.DataFrame:
    """Build a single-row feature frame for on-demand prediction in the app."""
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
    }
    return pd.DataFrame([row])[FEATURES]
