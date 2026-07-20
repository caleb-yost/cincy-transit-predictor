"""Cincinnati bus delay dashboard.

Live vehicle map (straight from the GTFS-realtime feed), route reliability, a delay predictor
backed by the trained model, and historical trends. Reads marts from the warehouse
(MotherDuck in production, local DuckDB in dev) and the model from ml/artifacts/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import warehouse  # noqa: E402
from ingestion.feeds import get_feeds  # noqa: E402
from ingestion.fetch_realtime import fetch_vehicle_positions, fetch_weather  # noqa: E402
from ml.build_features import make_feature_row  # noqa: E402

MODEL_PATH = ROOT / "ml" / "artifacts" / "model.pkl"
METRICS_PATH = ROOT / "ml" / "artifacts" / "metrics.json"
DOW = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

st.set_page_config(page_title="Cincinnati Bus Delay Predictor", page_icon="🚌", layout="wide")


# ---------------------------------------------------------------- data access
@st.cache_data(ttl=60, show_spinner=False)
def live_vehicles() -> pd.DataFrame:
    feeds = get_feeds()
    return fetch_vehicle_positions(feeds.vehicle_positions, int(time.time()))


@st.cache_data(ttl=600, show_spinner=False)
def live_weather() -> dict:
    feeds = get_feeds()
    try:
        return fetch_weather(feeds.lat, feeds.lon, int(time.time())).iloc[0].to_dict()
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def query(sql: str) -> pd.DataFrame:
    try:
        con = warehouse.connect()
        df = con.sql(sql).df()
        con.close()
        return df
    except Exception:
        # Degrade gracefully when the warehouse isn't reachable (e.g. MotherDuck not yet configured).
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


# ---------------------------------------------------------------- header + KPIs
st.title("🚌 Cincinnati Bus Delay Predictor")
st.caption(
    "Live pipeline over Cincinnati Metro (SORTA) GTFS-realtime + weather. "
    "Ingested every 15 min → dbt/DuckDB → MotherDuck → scikit-learn."
)

metrics = load_metrics()
kpi = query(
    "select count(*) as n, round(avg(is_on_time)*100,1) as on_time, "
    "round(avg(delay_minutes),2) as avg_delay from mart_stop_delays"
)
try:
    vehicles = live_vehicles()
except Exception:
    vehicles = pd.DataFrame()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Buses live now", len(vehicles))
if not kpi.empty and kpi.iloc[0]["n"]:
    row = kpi.iloc[0]
    c2.metric("Arrivals observed", f"{int(row['n']):,}")
    c3.metric("On-time rate", f"{row['on_time']:.0f}%")
    c4.metric("Avg delay", f"{row['avg_delay']:.1f} min")
else:
    c2.metric("Arrivals observed", "n/a")
    c3.metric("On-time rate", "n/a")
    c4.metric("Avg delay", "n/a")

if metrics.get("is_smoke_model", True):
    st.warning(
        "⚠️ Early build: the model is still a **smoke model** trained on limited data. "
        "Predictions and reliability stats become meaningful after ~a week of ingestion.",
        icon="⚠️",
    )

tab_map, tab_rel, tab_predict, tab_trends, tab_about = st.tabs(
    ["🗺️ Live map", "📊 Reliability", "🔮 Predict my bus", "📈 Trends", "ℹ️ About"]
)

# ---------------------------------------------------------------- live map
with tab_map:
    st.subheader("Buses currently in service")
    if vehicles.empty:
        st.info("No live vehicles right now (few buses run overnight) or the feed is unreachable.")
    else:
        st.map(vehicles.dropna(subset=["latitude", "longitude"]), latitude="latitude", longitude="longitude", size=40)
        active = (
            vehicles.groupby("route_id").size().reset_index(name="buses").sort_values("buses", ascending=False)
        )
        st.caption(f"{len(vehicles)} buses across {active.shape[0]} routes.")
        st.dataframe(active, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- reliability
with tab_rel:
    st.subheader("Route reliability")
    rel = query("select * from mart_route_reliability")
    if rel.empty:
        st.info("Reliability marts are empty. They fill in as data accrues.")
    else:
        board = (
            rel.assign(on_time_w=rel["on_time_pct"] * rel["n_arrivals"], delay_w=rel["avg_delay_minutes"] * rel["n_arrivals"])
            .groupby(["route_id", "route_short_name"], dropna=False)
            .agg(arrivals=("n_arrivals", "sum"), on_time_w=("on_time_w", "sum"), delay_w=("delay_w", "sum"))
            .reset_index()
        )
        board["on_time_pct"] = (board["on_time_w"] / board["arrivals"]).round(1)
        board["avg_delay_min"] = (board["delay_w"] / board["arrivals"]).round(2)
        board = board[["route_short_name", "arrivals", "on_time_pct", "avg_delay_min"]].sort_values("on_time_pct")
        left, right = st.columns([1, 1])
        with left:
            st.caption("Least reliable routes (lowest on-time %)")
            st.dataframe(board.head(15), use_container_width=True, hide_index=True)
        with right:
            st.caption("On-time % by route")
            chart = board.dropna(subset=["route_short_name"]).set_index("route_short_name")["on_time_pct"]
            st.bar_chart(chart)

# ---------------------------------------------------------------- predictor
with tab_predict:
    st.subheader("Will my bus be late?")
    model = load_model()
    routes = query("select distinct route_id, route_short_name, route_long_name from dim_route order by route_short_name")
    if model is None:
        st.info("Model not trained yet. Run `python ml/train.py` after some data has accrued.")
    elif routes.empty:
        st.info("Route dimension is empty. Load the schedule reference first.")
    else:
        wx = live_weather()
        col1, col2, col3 = st.columns(3)
        labels = routes["route_short_name"].fillna(routes["route_id"]).tolist()
        pick = col1.selectbox("Route", labels)
        route_id = routes.iloc[labels.index(pick)]["route_id"]
        hour = col2.slider("Hour of day", 0, 23, 8)
        dow_name = col3.selectbox("Day", DOW, index=1)
        stop_seq = st.slider("Stop number along the route", 1, 120, 20,
                             help="Roughly how far into the route the stop is.")
        feat = make_feature_row(route_id, hour, DOW.index(dow_name), stop_seq, wx)
        delay = float(model["regressor"].predict(feat)[0])
        late_p = None
        if model.get("classifier") is not None:
            late_p = float(model["classifier"].predict_proba(feat)[0][1])

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted delay", f"{delay:+.1f} min")
        m2.metric("Chance >5 min late", f"{late_p*100:.0f}%" if late_p is not None else "n/a")
        verdict = "🟢 Likely on time" if delay <= 5 else ("🟡 Running late" if delay <= 12 else "🔴 Very late")
        m3.metric("Verdict", verdict)
        if wx:
            st.caption(
                f"Live weather baked in: {wx.get('temperature_f','?')}°F, "
                f"precip {wx.get('precipitation_in',0)} in, wind {wx.get('wind_speed_mph','?')} mph."
            )

# ---------------------------------------------------------------- trends
with tab_trends:
    st.subheader("Delay patterns")
    td = query("select sched_hour, delay_minutes, is_on_time from mart_stop_delays")
    if td.empty:
        st.info("No history yet.")
    else:
        by_hour = (
            td.groupby("sched_hour")
            .agg(avg_delay=("delay_minutes", "mean"), on_time_pct=("is_on_time", "mean"), n=("delay_minutes", "size"))
            .reset_index()
        )
        by_hour["on_time_pct"] = (by_hour["on_time_pct"] * 100).round(1)
        by_hour["avg_delay"] = by_hour["avg_delay"].round(2)
        left, right = st.columns(2)
        with left:
            st.caption("Average delay (min) by scheduled hour")
            st.bar_chart(by_hour.set_index("sched_hour")["avg_delay"])
        with right:
            st.caption("On-time % by scheduled hour")
            st.bar_chart(by_hour.set_index("sched_hour")["on_time_pct"])

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown(
        """
        **How it works**

        1. A GitHub Actions cron polls SORTA's GTFS-realtime feeds + Open-Meteo weather every 15 minutes,
           landing partitioned Parquet on a `data` branch.
        2. **dbt** (DuckDB) models the raw snapshots into a labeled `mart_stop_delays` table, comparing each
           *predicted* arrival to the *scheduled* one, then publishes to a **MotherDuck** cloud warehouse.
        3. **scikit-learn** trains a gradient-boosting delay regressor + late-arrival classifier with a
           time-based holdout, retrained daily.
        4. This Streamlit app serves the live map, reliability, predictor, and trends.

        Data: [SORTA developer feeds](https://www.go-metro.com/about/developer-data/) ·
        [Open-Meteo](https://open-meteo.com/). Delays anchor the schedule's clock time to the observed
        prediction, which sidesteps SORTA's non-standard after-midnight service dates.
        """
    )
    if metrics:
        st.json(metrics)
