"""Train the delay regressor + late-arrival classifier on mart_stop_delays.

Evaluates with rolling-origin (walk-forward) time-series cross-validation once enough distinct
service days exist: each fold trains on every day strictly before a held-out test day, so no fold
ever sees its own future. Metrics are pooled across folds rather than averaged, since R^2 isn't
meaningfully averageable per-fold. Falls back to a single time-based holdout (or a random split)
while the dataset still spans too few days for that. The shipped model is refit on ALL data;
only the reported metrics come from the held-out folds.

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
MIN_TRAIN_DAYS = 3  # a fold's training window must span at least this many days before its test day
MAX_FOLDS = 5  # cap walk-forward folds so runtime stays flat as months of data accrue


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


def walk_forward_days(df):
    """Yield (train_df, test_df) per fold: train on every day strictly before the held-out day."""
    days = sorted(df["start_date"].unique())
    test_days = days[MIN_TRAIN_DAYS:][-MAX_FOLDS:]
    for test_day in test_days:
        yield df[df["start_date"] < test_day], df[df["start_date"] == test_day], test_day


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

    folds = [] if smoke else list(walk_forward_days(df))

    reg_true, reg_pred, reg_baseline_pred = [], [], []
    clf_true, clf_proba = [], []
    fold_log = []

    if len(folds) >= 2:
        split = f"rolling-origin walk-forward CV ({len(folds)} folds, pooled)"
        for train_df, test_df, test_day in folds:
            x_tr, x_te = train_df[FEATURES], test_df[FEATURES]

            reg = build_pipeline(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06))
            reg.fit(x_tr, train_df[REG_TARGET])
            reg_true.extend(test_df[REG_TARGET].tolist())
            reg_pred.extend(reg.predict(x_te).tolist())
            reg_baseline_pred.extend([train_df[REG_TARGET].mean()] * len(test_df))

            if train_df[CLF_TARGET].nunique() > 1 and test_df[CLF_TARGET].nunique() > 1:
                clf = build_pipeline(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06))
                clf.fit(x_tr, train_df[CLF_TARGET])
                clf_true.extend(test_df[CLF_TARGET].tolist())
                clf_proba.extend(clf.predict_proba(x_te)[:, 1].tolist())

            fold_log.append({"test_day": str(test_day), "train_rows": len(train_df), "test_rows": len(test_df)})
        print(f"split: {split}")
        for f in fold_log:
            print(f"  fold test_day={f['test_day']}  train={f['train_rows']}  test={f['test_rows']}")
    else:
        # Not enough distinct days yet for walk-forward CV; fall back to a single holdout.
        train_df, test_df = time_split(df)
        split = "time-based (insufficient days for walk-forward CV)"
        if test_df["scheduled_at"].nunique() < 3 or len(test_df) < 10:
            from sklearn.model_selection import train_test_split

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            split = "random (insufficient time span for a temporal split)"
        print(f"split: {split}  |  train={len(train_df)} test={len(test_df)}")

        x_tr, x_te = train_df[FEATURES], test_df[FEATURES]
        reg = build_pipeline(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06))
        reg.fit(x_tr, train_df[REG_TARGET])
        reg_true = test_df[REG_TARGET].tolist()
        reg_pred = reg.predict(x_te).tolist()
        reg_baseline_pred = [train_df[REG_TARGET].mean()] * len(test_df)

        if train_df[CLF_TARGET].nunique() > 1 and test_df[CLF_TARGET].nunique() > 1:
            clf = build_pipeline(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06))
            clf.fit(x_tr, train_df[CLF_TARGET])
            clf_true = test_df[CLF_TARGET].tolist()
            clf_proba = clf.predict_proba(x_te)[:, 1].tolist()

    metrics = {
        "trained_at": datetime.now(UTC).isoformat(),
        "n_rows": int(n),
        "n_service_days": n_days,
        "n_distinct_hours": n_hours,
        "split": split,
        "n_folds": len(folds),
        "fold_log": fold_log,
        "is_smoke_model": bool(smoke),
        "regression": {
            "mae_minutes": round(float(mean_absolute_error(reg_true, reg_pred)), 3),
            "rmse_minutes": round(float(np.sqrt(mean_squared_error(reg_true, reg_pred))), 3),
            "r2": round(float(r2_score(reg_true, reg_pred)), 3),
            "baseline_mae_minutes": round(float(mean_absolute_error(reg_true, reg_baseline_pred)), 3),
        },
    }

    # ---- classification: is the bus >5 min late (pooled across folds where both classes present) ----
    if clf_true and len(set(clf_true)) > 1:
        clf_preds = [1 if p >= 0.5 else 0 for p in clf_proba]
        metrics["classification"] = {
            "roc_auc": round(float(roc_auc_score(clf_true, clf_proba)), 3),
            "accuracy": round(float(accuracy_score(clf_true, clf_preds)), 3),
            "precision": round(float(precision_score(clf_true, clf_preds, zero_division=0)), 3),
            "recall": round(float(recall_score(clf_true, clf_preds, zero_division=0)), 3),
            "late_rate": round(float(df[CLF_TARGET].mean()), 3),
        }
    else:
        metrics["classification"] = {"note": "only one class present so far, classifier skipped."}

    # Reported metrics come from held-out folds above; the SHIPPED model is refit on every row
    # collected so far, since there's no reason to throw away data once it's no longer being tested.
    x_all = df[FEATURES]
    reg = build_pipeline(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06))
    reg.fit(x_all, df[REG_TARGET])
    clf = None
    if df[CLF_TARGET].nunique() > 1:
        clf = build_pipeline(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06))
        clf.fit(x_all, df[CLF_TARGET])

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
