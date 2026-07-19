"""Tests for feature engineering used by both training and live prediction."""

from __future__ import annotations

import pandas as pd

from ml.build_features import FEATURES, REG_TARGET, make_feature_row, prepare


def test_feature_row_has_expected_columns():
    row = make_feature_row("17", sched_hour=8, sched_dow=1, stop_sequence=20, weather={"temperature_f": 70})
    assert list(row.columns) == FEATURES
    assert len(row) == 1
    assert row.iloc[0]["is_weekend"] == 0
    assert row.iloc[0]["temperature_f"] == 70


def test_feature_row_weekend_flag():
    sunday = make_feature_row("17", sched_hour=8, sched_dow=0, stop_sequence=20)
    saturday = make_feature_row("17", sched_hour=8, sched_dow=6, stop_sequence=20)
    assert sunday.iloc[0]["is_weekend"] == 1
    assert saturday.iloc[0]["is_weekend"] == 1


def test_feature_row_uses_weather_defaults():
    row = make_feature_row("17", sched_hour=8, sched_dow=1, stop_sequence=20)
    assert row.iloc[0]["precipitation_in"] == 0.0
    assert row.iloc[0]["wind_speed_mph"] == 5.0


def test_prepare_fills_weather_and_clips_outliers():
    df = pd.DataFrame(
        {
            "route_id": ["1", "2", "3"],
            "sched_hour": [8, 9, 10],
            "sched_dow": [1, 2, 3],
            "stop_sequence": [3, 4, 5],
            "is_weekend": [0, 0, 0],
            "temperature_f": [None, 60.0, 55.0],
            "precipitation_in": [None, 0.1, 0.0],
            "snowfall_in": [None, 0.0, 0.0],
            "wind_speed_mph": [None, 5.0, 6.0],
            "delay_minutes": [3.0, 999.0, -2.0],  # 999 is feed noise, must be dropped
        }
    )
    out = prepare(df)
    assert out["precipitation_in"].isna().sum() == 0
    assert out["temperature_f"].isna().sum() == 0
    assert out[REG_TARGET].max() <= 60
    assert len(out) == 2  # the 999-minute outlier row is removed
