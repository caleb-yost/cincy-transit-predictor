"""Cincinnati transit dashboard.

A live vehicle map straight from the GTFS-realtime feed, and a delay predictor backed by the
trained model. Reliability and trends fold into panels under the map. Reads marts from the
warehouse (MotherDuck in production, local DuckDB in dev) and the model from ml/artifacts/.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import pydeck as pdk  # noqa: E402
import streamlit as st  # noqa: E402

import warehouse  # noqa: E402
from ingestion.feeds import get_feeds  # noqa: E402
from ingestion.fetch_realtime import fetch_vehicle_positions, fetch_weather  # noqa: E402
from ml.build_features import make_feature_row  # noqa: E402

MODEL_PATH = ROOT / "ml" / "artifacts" / "model.pkl"
METRICS_PATH = ROOT / "ml" / "artifacts" / "metrics.json"
DOW = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
HOURS = [f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}" for h in range(24)]
CINCY_LAT, CINCY_LON = 39.1031, -84.5120
APPLE_BLUE = [0, 122, 255, 230]

st.set_page_config(page_title="Cincinnati Transit", page_icon="🚌", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"], [data-testid] {
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    }
    #MainMenu, footer, header[data-testid="stHeader"] { display: none; }
    [data-testid="stMainBlockContainer"] { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1080px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    /* cards */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 12px 30px rgba(0,0,0,0.06);
    }
    /* segmented control: bigger, centered pills */
    [data-testid="stSegmentedControl"] { display: flex; justify-content: center; }
    [data-testid="stSegmentedControl"] button { padding: 0.3rem 1.6rem; font-weight: 600; }
    /* rounded map */
    [data-testid="stDeckGlJsonChart"] > div { border-radius: 18px; overflow: hidden; }
    .statusline { opacity: 0.55; font-size: 0.95rem; }
    .result-delay { font-size: 3.2rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1.1; }
    .result-sub { opacity: 0.6; font-size: 1.0rem; margin-top: 0.1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def stops_for_route(route_id: str) -> pd.DataFrame:
    safe = str(route_id).replace("'", "''")
    return query(
        "select stop_name, stop_sequence, stop_lat, stop_lon "
        f"from dim_route_stops where route_id = '{safe}' and stop_name is not null "
        "order by stop_name"
    )


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


# ---------------------------------------------------------------- shared state
metrics = load_metrics()
try:
    vehicles = live_vehicles()
except Exception:
    vehicles = pd.DataFrame()
kpi = query(
    "select count(*) as n, round(avg(is_on_time)*100,1) as on_time, "
    "round(avg(delay_minutes),2) as avg_delay from mart_stop_delays"
)

n_live = len(vehicles)
if not kpi.empty and kpi.iloc[0]["n"]:
    r = kpi.iloc[0]
    status = f"{n_live} buses live · {r['on_time']:.0f}% on-time · {r['avg_delay']:.1f} min avg delay"
else:
    status = f"{n_live} buses live · collecting delay data"

# ---------------------------------------------------------------- header
st.markdown(
    f"""
    <div style="margin-bottom:0.4rem;">
      <span style="font-size:2rem; font-weight:700; letter-spacing:-0.03em;">Cincinnati Transit</span>
      <span class="statusline" style="margin-left:0.6rem;">{status}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
if metrics.get("is_smoke_model", True):
    st.caption("Early build — the model is still learning; predictions sharpen as a week of data accrues.")

_, mid, _ = st.columns([1, 2, 1])
with mid:
    view = st.segmented_control("Nav", ["Map", "Predict"], default="Map", label_visibility="collapsed")
view = view or "Map"


# ---------------------------------------------------------------- map view
def render_map() -> None:
    if vehicles.empty:
        st.info("No live buses right now (few run overnight), or the feed is briefly unreachable.")
    else:
        pts = vehicles.dropna(subset=["latitude", "longitude"]).copy()
        pts["route_label"] = pts["route_id"].fillna("?")
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pts,
            get_position=["longitude", "latitude"],
            get_fill_color=APPLE_BLUE,
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=1.5,
            get_radius=90,
            radius_min_pixels=4,
            radius_max_pixels=11,
            pickable=True,
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=CINCY_LAT, longitude=CINCY_LON, zoom=11),
            map_style=None,
            tooltip={"text": "Route {route_label}"},
        )
        st.pydeck_chart(deck, use_container_width=True, height=520)
        st.caption(f"{n_live} buses in service across {pts['route_id'].nunique()} routes.")

    rel = query("select * from mart_route_reliability")
    with st.expander("Route reliability", expanded=False):
        if rel.empty:
            st.caption("Fills in as delay data accrues.")
        else:
            board = (
                rel.assign(ontime_w=rel["on_time_pct"] * rel["n_arrivals"], delay_w=rel["avg_delay_minutes"] * rel["n_arrivals"])
                .groupby(["route_id", "route_short_name"], dropna=False)
                .agg(arrivals=("n_arrivals", "sum"), ontime_w=("ontime_w", "sum"), delay_w=("delay_w", "sum"))
                .reset_index()
            )
            board["On-time %"] = (board["ontime_w"] / board["arrivals"]).round(1)
            board["Avg delay (min)"] = (board["delay_w"] / board["arrivals"]).round(2)
            board = board.rename(columns={"route_short_name": "Route", "arrivals": "Arrivals"})
            board = board[["Route", "Arrivals", "On-time %", "Avg delay (min)"]].sort_values("On-time %")
            st.dataframe(board, use_container_width=True, hide_index=True, height=320)

    td = query("select sched_hour, delay_minutes, is_on_time from mart_stop_delays")
    with st.expander("Delay by time of day", expanded=False):
        if td.empty:
            st.caption("No history yet.")
        else:
            by_hour = (
                td.groupby("sched_hour")
                .agg(avg_delay=("delay_minutes", "mean"), on_time=("is_on_time", "mean"))
                .reset_index()
            )
            by_hour["Avg delay (min)"] = by_hour["avg_delay"].round(2)
            by_hour["On-time %"] = (by_hour["on_time"] * 100).round(1)
            left, right = st.columns(2)
            left.caption("Average delay by hour")
            left.bar_chart(by_hour.set_index("sched_hour")["Avg delay (min)"], color="#007aff")
            right.caption("On-time % by hour")
            right.bar_chart(by_hour.set_index("sched_hour")["On-time %"], color="#34c759")


# ---------------------------------------------------------------- predict view
def render_predict() -> None:
    model = load_model()
    routes = query("select distinct route_id, route_short_name, route_long_name from dim_route order by route_short_name")
    if model is None:
        st.info("The model isn't trained yet. It comes online once enough data has accrued.")
        return
    if routes.empty:
        st.info("Route data hasn't loaded into the warehouse yet.")
        return

    with st.container(border=True):
        st.markdown("#### Will my bus be late?")
        labels = routes["route_short_name"].fillna(routes["route_id"]).astype(str).tolist()
        c1, c2 = st.columns(2)
        route_label = c1.selectbox("Route", labels)
        route_id = routes.iloc[labels.index(route_label)]["route_id"]

        stops = stops_for_route(route_id)
        stop_lat = stop_lon = None
        if not stops.empty:
            stop_name = c2.selectbox("Stop", stops["stop_name"].tolist())
            row = stops[stops["stop_name"] == stop_name].iloc[0]
            stop_seq = int(row["stop_sequence"])
            stop_lat, stop_lon = row["stop_lat"], row["stop_lon"]
        else:
            c2.selectbox("Stop", ["(loads after the next warehouse refresh)"], disabled=True)
            stop_seq = 20

        c3, c4 = st.columns(2)
        default_day = (datetime.now().weekday() + 1) % 7
        day = c3.selectbox("Day", DOW, index=default_day)
        hour_label = c4.selectbox("Time", HOURS, index=datetime.now().hour)
        hour = HOURS.index(hour_label)

        feat = make_feature_row(route_id, hour, DOW.index(day), stop_seq, live_weather())
        delay = float(model["regressor"].predict(feat)[0])
        late_p = None
        if model.get("classifier") is not None:
            late_p = float(model["classifier"].predict_proba(feat)[0][1])

        if delay <= 5:
            color, verdict = "#34c759", "Usually on time"
        elif delay <= 12:
            color, verdict = "#ff9500", "Often a few minutes late"
        else:
            color, verdict = "#ff3b30", "Frequently very late"
        # The late-probability only makes sense once the classifier is trained on real data; while
        # it's a smoke model it disagrees with the delay estimate, so hide it to avoid contradiction.
        smoke = metrics.get("is_smoke_model", True)
        prob = f" · {late_p*100:.0f}% chance over 5 min late" if (late_p is not None and not smoke) else ""

        st.markdown(
            f'<div class="result-delay" style="color:{color}">{delay:+.0f} min</div>'
            f'<div class="result-sub">{verdict}{prob}</div>',
            unsafe_allow_html=True,
        )
        wx = live_weather()
        if wx:
            st.caption(
                f"Live weather in the estimate: {wx.get('temperature_f','?')}°F, "
                f"precip {wx.get('precipitation_in',0)} in, wind {wx.get('wind_speed_mph','?')} mph."
            )

    if stop_lat is not None and pd.notna(stop_lat):
        pin = pd.DataFrame([{"lat": stop_lat, "lon": stop_lon}])
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pin,
            get_position=["lon", "lat"],
            get_fill_color=[255, 59, 48, 230],
            get_line_color=[255, 255, 255, 255],
            line_width_min_pixels=2,
            get_radius=70,
            radius_min_pixels=7,
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=float(stop_lat), longitude=float(stop_lon), zoom=14),
            map_style=None,
        )
        st.pydeck_chart(deck, use_container_width=True, height=240)


if view == "Map":
    render_map()
else:
    render_predict()
