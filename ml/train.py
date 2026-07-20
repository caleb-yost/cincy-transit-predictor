"""Train the delay regressor + late-arrival classifier on mart_stop_delays.

Uses a TIME-BASED holdout (train on earlier arrivals, test on later) to avoid leakage, with a
random-split fallback while the dataset still spans too little time. Writes model.pkl + metrics.json.

Run: ``python ml/train.py`` (or ``python -m ml.train``).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import warehouse
    from ml.build_features import (
        CATEGORICAL,
        CLF_TARGET,
        FEATURES,
        REG_TARGET,
        load_labeled_frame,
        prepare,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import warehouse
    from ml.build_features import (
        CATEGORICAL,
        CLF_TARGET,
        FEATURES,
        REG_TARGET,
        load_labeled_frame,
        prepare,
    )

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MIN_ROWS = 50


def build_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def time_split(df, frac: float = 0.8):
    df = df.sort_values("scheduled_at")
    k = int(len(df) * frac)
    return df.iloc[:k], df.iloc[k:]


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    con = warehouse.connect()
    df = prepare(load_labeled_frame(con))
    n = len(df)
    n_days = int(df["start_date"].nunique()) if "start_date" in df.columns else 1
    n_hours = int(df["sched_hour"].nunique())
    print(f"labeled rows: {n}  |  service days: {n_days}  |  distinct hours: {n_hours}")

    # "meaningful" needs temporal spread, not just row count. One busy hour is still a smoke model.
    smoke = n < MIN_ROWS or n_days < 2 or n_hours < 6
    if smoke:
        print("[warn] limited temporal coverage; SMOKE model, metrics not yet meaningful.")

    train_df, test_df = time_split(df)
    split = "time-based"
    if test_df["scheduled_at"].nunique() < 3 or len(test_df) < 10:
        from sklearn.model_selection import train_test_split

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        split = "random (insufficient time span for a temporal split)"
    print(f"split: {split}  |  train={len(train_df)} test={len(test_df)}")

    x_tr, x_te = train_df[FEATURES], test_df[FEATURES]

    # ---- regression: delay in minutes ----
    reg = build_pipeline(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06))
    reg.fit(x_tr, train_df[REG_TARGET])
    pred = reg.predict(x_te)
    metrics = {
        "trained_at": datetime.now(UTC).isoformat(),
        "n_rows": int(n),
        "n_service_days": n_days,
        "n_distinct_hours": n_hours,
        "split": split,
        "is_smoke_model": bool(smoke),
        "regression": {
            "mae_minutes": round(float(mean_absolute_error(test_df[REG_TARGET], pred)), 3),
            "rmse_minutes": round(float(np.sqrt(mean_squared_error(test_df[REG_TARGET], pred))), 3),
            "r2": round(float(r2_score(test_df[REG_TARGET], pred)), 3),
            "baseline_mae_minutes": round(
                float(mean_absolute_error(test_df[REG_TARGET], np.full(len(test_df), train_df[REG_TARGET].mean()))),
                3,
            ),
        },
    }

    # ---- classification: is the bus >5 min late ----
    clf = None
    if train_df[CLF_TARGET].nunique() > 1 and test_df[CLF_TARGET].nunique() > 1:
        clf = build_pipeline(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06))
        clf.fit(x_tr, train_df[CLF_TARGET])
        proba = clf.predict_proba(x_te)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics["classification"] = {
            "roc_auc": round(float(roc_auc_score(test_df[CLF_TARGET], proba)), 3),
            "accuracy": round(float(accuracy_score(test_df[CLF_TARGET], preds)), 3),
            "precision": round(float(precision_score(test_df[CLF_TARGET], preds, zero_division=0)), 3),
            "recall": round(float(recall_score(test_df[CLF_TARGET], preds, zero_division=0)), 3),
            "late_rate": round(float(df[CLF_TARGET].mean()), 3),
        }
    else:
        metrics["classification"] = {"note": "only one class present so far, classifier skipped."}

    joblib.dump(
        {
            "regressor": reg,
            "classifier": clf,
            "features": FEATURES,
            "trained_at": metrics["trained_at"],
            "n_rows": int(n),
            "is_smoke_model": bool(smoke),
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nsaved -> {MODEL_PATH}\nsaved -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
