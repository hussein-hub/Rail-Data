import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
import re
from datetime import datetime, date, timedelta
import os
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
load_dotenv()

DB = create_engine(os.getenv("DATABASE_URL"))
BASE = os.getenv("API_BASE_URL")

st.set_page_config(layout="wide", page_title="Rail Intelligence", page_icon="🚆")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        border: 1px solid #2e3350;
    }
    .metric-label { color: #8d93b0; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
    .metric-value { color: #ffffff; font-size: 2rem; font-weight: 700; line-height: 1.1; }
    .metric-sub   { color: #8d93b0; font-size: 0.78rem; margin-top: 4px; }
    .section-title { font-size: 1.05rem; font-weight: 600; color: #c9cfe8; margin-bottom: 4px; }
    .caption-text  { color: #6b7294; font-size: 0.78rem; margin-top: 4px; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #13151f;
        padding: 6px 8px;
        border-radius: 12px;
        border: 1px solid #2e3350;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-size: 0.88rem;
        font-weight: 500;
        color: #8d93b0;
    }
    .stTabs [aria-selected="true"] {
        background: #2e3350 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🚆 Rail — Operational Intelligence Dashboard")
st.markdown("<p style='color:#6b7294;font-size:0.85rem;margin-top:-10px;'>Live train performance, network map and journey drill-down</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────
@st.cache_data(ttl=15)
def get_running_trains():
    try:
        r = requests.get(f"{BASE}/getCurrentTrainsXML", timeout=10)
        return pd.read_xml(StringIO(r.text))
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_all_stations():
    try:
        r = requests.get(f"{BASE}/getAllStationsXML", timeout=10)
        return pd.read_xml(StringIO(r.text))
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=15)
def get_trains_by_type(train_type: str):
    """Fetch running trains filtered by type: D=DART, M=Mainline, S=Suburban."""
    try:
        r = requests.get(
            f"{BASE}/getCurrentTrainsXML_WithTrainType",
            params={"TrainType": train_type},
            timeout=10,
        )
        return pd.read_xml(StringIO(r.text))
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=10)
def get_train_movements(train_code: str, train_date: str):
    try:
        r = requests.get(
            f"{BASE}/getTrainMovementsXML",
            params={"TrainId": train_code, "TrainDate": train_date},
            timeout=10,
        )
        return pd.read_xml(StringIO(r.text))
    except Exception:
        return pd.DataFrame()

def parse_delay_from_message(msg):
    if not msg or not isinstance(msg, str):
        return None
    m = re.search(r"\((\d+) mins? late\)", msg)
    if m:
        return int(m.group(1))
    if "on time" in msg.lower():
        return 0
    return None

def hex_to_rgba(hex_color, alpha=0.15):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def delay_color(mins):
    if mins is None:  return "#5b6499"
    if mins == 0:     return "#22c55e"
    if mins <= 5:     return "#f59e0b"
    return "#ef4444"

def time_str_to_minutes(t):
    """Convert HH:MM or HH:MM:SS string to minutes since midnight, or None."""
    try:
        s = str(t).strip()
        if not s or s in ("00:00:00", "00:00", "nan"):
            return None
        parts = s.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return None

def minutes_to_hhmm(m):
    if m is None:
        return ""
    h, mins = divmod(int(m), 60)
    return f"{h:02d}:{mins:02d}"

def sri_label(x):
    if x < 3:  return "Reliable"
    elif x < 7: return "Moderate"
    else:       return "High Risk"

def build_day_options(date_series):
    available_dates = sorted(pd.Series(date_series).dropna().unique(), reverse=True)
    today = date.today()
    yesterday = today - timedelta(days=1)

    day_options = ["All days"]
    day_map = {}
    if today in available_dates:
        day_options.append("Today")
        day_map["Today"] = today
    if yesterday in available_dates:
        day_options.append("Yesterday")
        day_map["Yesterday"] = yesterday
    for d in available_dates:
        if d in (today, yesterday):
            continue
        label = d.strftime("%d %b %Y")
        day_options.append(label)
        day_map[label] = d
    return day_options, day_map

def build_sri_view(data):
    if data.empty:
        return pd.DataFrame()
    risk = data.groupby("traincode")["late"].agg(["mean", "count"])
    risk["delay_freq"] = (
        data[data["late"] > 5].groupby("traincode")["late"].count() / risk["count"]
    ).fillna(0)
    risk["sri"] = (0.5 * risk["mean"]) + (0.5 * risk["delay_freq"] * 10)
    risk = risk.reset_index()

    latest = data.sort_values("timestamp").groupby("traincode").tail(1)
    merged = latest.merge(risk[["traincode", "sri"]], on="traincode")
    merged["reliability"] = merged["sri"].apply(sri_label)
    return merged

# ─────────────────────────────────────────────
# LOAD HISTORICAL DB DATA
# ─────────────────────────────────────────────
df = pd.read_sql("train_logs", DB)
df.columns = df.columns.str.strip().str.lower()
if df.empty:
    st.warning("Waiting for data collection...")
    st.stop()

df["late"]      = pd.to_numeric(df["late"],  errors="coerce").fillna(0).clip(lower=0)
df["duein"]     = pd.to_numeric(df["duein"], errors="coerce").fillna(0)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["date"]      = df["timestamp"].dt.date

# SRI calculation
merged = build_sri_view(df)

# ─────────────────────────────────────────────
# SHARED LIVE DATA  (fetched once, used by Tab 2 and Tab 3)
# Both tabs must see the exact same snapshot so a train visible on
# the map is always available in the Journey Profiler dropdown.
# ─────────────────────────────────────────────
_shared_trains_df = get_running_trains()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Station Analysis",
    "🗺️  Live Network Map",
    "🔍  Journey Profiler",
    "🚄  Fleet Scorecard",
    "📌  Snapshot Overview",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — STATION ANALYSIS (existing dashboard)
# ════════════════════════════════════════════════════════════════════
with tab1:
    day_options, day_map = build_day_options(df["date"])
    filter_col, station_col = st.columns([1, 1], gap="large")
    with filter_col:
        selected_day = st.selectbox("🗓️ Filter by day", day_options, index=0)
    if selected_day == "All days":
        filtered_df = df
    else:
        filtered_df = df[df["date"] == day_map[selected_day]].copy()

    if filtered_df.empty:
        with station_col:
            st.selectbox("📍 Select Station", ["No stations available"], index=0)
        st.warning("No data available for the selected day yet.")
    else:
        filtered_merged = build_sri_view(filtered_df)
        if filtered_merged.empty:
            with station_col:
                st.selectbox("📍 Select Station", ["No stations available"], index=0)
            st.warning("No data available for the selected day yet.")
        else:
            with station_col:
                station = st.selectbox("📍 Select Station", sorted(filtered_merged["stationfullname"].unique()))
            view = filtered_merged[filtered_merged["stationfullname"] == station].copy().sort_values("sri", ascending=True)
            # KPI cards
            total_trains  = len(view)
            avg_delay     = view["late"].mean()
            high_risk_pct = (view["reliability"] == "High Risk").sum() / max(total_trains, 1) * 100
            avg_sri       = view["sri"].mean()

            k1, k2, k3 = st.columns(3)
            def metric_card(col, label, value, sub=""):
                col.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-sub">{sub}</div>
                    </div>
                """, unsafe_allow_html=True)

            metric_card(k1, "Avg. delay",       f"{avg_delay:.1f} min", "across all services")
            metric_card(k2, "High-risk trains", f"{high_risk_pct:.0f}%","SRI ≥ 7")
            metric_card(k3, "Avg. risk score",  f"{avg_sri:.1f}",       "Service Risk Index")
            st.markdown("<br>", unsafe_allow_html=True)

            # Row 1
            col_l, col_r = st.columns([1, 1], gap="large")
            with col_l:
                st.markdown('<p class="section-title">Train Service Risk Index</p>', unsafe_allow_html=True)
                st.markdown('<p class="caption-text">SRI combines average lateness and delay frequency — higher = less reliable. Showing top 12 highest-risk trains.</p>', unsafe_allow_html=True)
                fig_sri = go.Figure(go.Bar(
                    y=view.tail(12)["traincode"],
                    x=view.tail(12)["sri"],
                    orientation="h",
                    marker=dict(
                        color=view.tail(12)["sri"],
                        colorscale=[[0,"#22c55e"],[0.43,"#f59e0b"],[1,"#ef4444"]],
                        cmin=0, cmax=10, line=dict(width=0),
                    ),
                    text=view.tail(12)["reliability"], textposition="inside", insidetextanchor="middle",
                    customdata=view.tail(12)["destination"],
                    hovertemplate="<b>%{y}</b><br>SRI: %{x:.1f}<br>To: %{customdata}<extra></extra>",
                ))
                fig_sri.update_layout(
                    xaxis=dict(title="Service Risk Index", range=[0, max(view["sri"].max()*1.15, 10)], gridcolor="#2e3350"),
                    yaxis=dict(title="", tickfont=dict(size=10)),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c9cfe8", size=11), margin=dict(l=10,r=10,t=10,b=30),
                    height=max(250, min(len(view), 12)*24), showlegend=False,
                )
                st.plotly_chart(fig_sri, width="stretch")

            with col_r:
                st.markdown('<p class="section-title">Delay Trend — Selected Day</p>', unsafe_allow_html=True)
                st.markdown('<p class="caption-text">15-minute rolling average delay, filtered by the day selector above.</p>', unsafe_allow_html=True)
                df_station = filtered_df[filtered_df["stationfullname"] == station].copy()
                ts = df_station.set_index("timestamp").resample("15min")["late"].mean().dropna().reset_index()
                ts.columns = ["time", "avg_delay"]
                fig_ts = go.Figure(go.Scatter(
                    x=ts["time"], y=ts["avg_delay"], mode="lines",
                    fill="tozeroy", line=dict(color="#6366f1", width=2.5),
                    fillcolor="rgba(99,102,241,0.15)",
                    hovertemplate="%{x|%H:%M}<br>Avg delay: %{y:.1f} min<extra></extra>",
                ))
                fig_ts.update_layout(
                    xaxis=dict(title="Time", gridcolor="#2e3350", tickformat="%H:%M"),
                    yaxis=dict(title="Avg Delay (min)", gridcolor="#2e3350"),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c9cfe8", size=12), margin=dict(l=10,r=10,t=10,b=30),
                    height=280,
                )
                st.plotly_chart(fig_ts, width="stretch")

            # Row 2
            col2_l, col2_r = st.columns([1.4, 1], gap="large")
            with col2_l:
                st.markdown('<p class="section-title">Corridor Reliability Heatmap</p>', unsafe_allow_html=True)
                st.markdown('<p class="caption-text">SRI per Origin → Destination corridor for trains serving this station. Green = reliable, Red = high risk.</p>', unsafe_allow_html=True)

                # Filter to corridors involving trains that serve the selected station
                trains_at_station = view["traincode"].unique()
                df_station_corridors = filtered_df[filtered_df["traincode"].isin(trains_at_station)]

                corridor = df_station_corridors.groupby(["origin","destination"])["late"].agg(["mean","count"])
                corridor["delay_freq"] = (
                    df_station_corridors[df_station_corridors["late"]>5].groupby(["origin","destination"])["late"].count() / corridor["count"]
                ).fillna(0)
                corridor["sri"] = (0.5*corridor["mean"])+(0.5*corridor["delay_freq"]*10)
                corridor = corridor.reset_index()
                heatmap_df = corridor.pivot(index="origin", columns="destination", values="sri").fillna(0)
                TOP_N = 15
                top_origins = corridor.groupby("origin")["sri"].mean().nlargest(TOP_N).index
                top_dests   = corridor.groupby("destination")["sri"].mean().nlargest(TOP_N).index
                heatmap_df  = heatmap_df.loc[heatmap_df.index.isin(top_origins), heatmap_df.columns.isin(top_dests)]
                n_rows, _ = heatmap_df.shape
                fig_hm = px.imshow(
                    heatmap_df, text_auto=".1f",
                    color_continuous_scale=[[0,"#16a34a"],[0.5,"#ca8a04"],[1,"#dc2626"]],
                    zmin=0, labels=dict(color="SRI"), aspect="auto",
                )
                fig_hm.update_traces(textfont=dict(size=10, color="white"),
                    hovertemplate="Origin: %{y}<br>Dest: %{x}<br>SRI: %{z:.1f}<extra></extra>")
                fig_hm.update_layout(
                    xaxis=dict(title="Destination", tickangle=-35, tickfont=dict(size=10), side="bottom"),
                    yaxis=dict(title="Origin", tickfont=dict(size=10), autorange="reversed"),
                    coloraxis_colorbar=dict(title="SRI", tickfont=dict(size=10), thickness=12, len=0.8),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c9cfe8"), margin=dict(l=10,r=10,t=10,b=60),
                    height=max(280, n_rows*28+60),
                )
                st.plotly_chart(fig_hm, width="stretch")

            with col2_r:
                st.markdown('<p class="section-title">Station Congestion Risk</p>', unsafe_allow_html=True)
                st.markdown('<p class="caption-text">Total SRI load per station — higher means more at-risk services arriving.</p>', unsafe_allow_html=True)
                station_risk = filtered_merged.groupby("stationfullname")["sri"].sum().sort_values(ascending=True).reset_index()
                fig_cong = go.Figure(go.Bar(
                    x=station_risk["sri"], y=station_risk["stationfullname"], orientation="h",
                    marker=dict(color=station_risk["sri"],
                        colorscale=[[0,"#22c55e"],[0.5,"#f59e0b"],[1,"#ef4444"]], line=dict(width=0)),
                    hovertemplate="%{y}<br>Total SRI: %{x:.1f}<extra></extra>",
                ))
                fig_cong.update_layout(
                    xaxis=dict(title="Total SRI", gridcolor="#2e3350"),
                    yaxis=dict(title="", tickfont=dict(size=10)),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c9cfe8", size=12), margin=dict(l=10,r=10,t=10,b=30),
                    height=max(280, len(station_risk)*22+60), showlegend=False,
                )
                st.plotly_chart(fig_cong, width="stretch")

            # Row 3
            st.markdown('<p class="section-title">Most Unpredictable Train Services</p>', unsafe_allow_html=True)
            st.markdown('<p class="caption-text">Services with highest delay variability. High std. deviation = hard to plan around even if average SRI looks moderate.</p>', unsafe_allow_html=True)
            train_var = filtered_df.groupby("traincode")["late"].std().dropna().sort_values(ascending=False).head(12).reset_index()
            train_var.columns = ["traincode","std_delay"]
            fig_var = go.Figure(go.Bar(
                x=train_var["traincode"], y=train_var["std_delay"],
                marker=dict(color=train_var["std_delay"],
                    colorscale=[[0,"#6366f1"],[1,"#ef4444"]], line=dict(width=0)),
                hovertemplate="Train: %{x}<br>Std deviation: %{y:.1f} min<extra></extra>",
            ))
            fig_var.update_layout(
                xaxis=dict(title="Train Code", tickangle=-30),
                yaxis=dict(title="Delay Std. Deviation (min)", gridcolor="#2e3350"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9cfe8", size=12), margin=dict(l=10,r=10,t=10,b=60),
                height=280, showlegend=False,
            )
            st.plotly_chart(fig_var, width="stretch")


# ════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE NETWORK MAP
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Live Train & Station Network</p>', unsafe_allow_html=True)
    st.markdown('<p class="caption-text">All running trains plotted in real time. Colour = how late the train is. Size of station dot = congestion risk (SRI). Refreshes every 15 seconds.</p>', unsafe_allow_html=True)

    with st.spinner("Fetching live data…"):
        trains_df   = _shared_trains_df
        stations_df = get_all_stations()

    if trains_df.empty or stations_df.empty:
        st.error("Could not reach the Rail API. Check your connection.")
    else:
        trains_df.columns   = trains_df.columns.str.strip().str.lower()
        stations_df.columns = stations_df.columns.str.strip().str.lower()

        # Clean lat/lon
        for col in ["trainlatitude","trainlongitude"]:
            trains_df[col] = pd.to_numeric(trains_df[col], errors="coerce")
        for col in ["stationlatitude","stationlongitude"]:
            stations_df[col] = pd.to_numeric(stations_df[col], errors="coerce")

        # Only running trains (R) with valid coords
        running = trains_df[
            (trains_df["trainstatus"] == "R") &
            trains_df["trainlatitude"].notna() &
            (trains_df["trainlatitude"] != 0)
        ].copy()
        not_started = trains_df[trains_df["trainstatus"] == "N"].copy()

        running["delay_mins"] = running["publicmessage"].apply(parse_delay_from_message)
        running["dot_color"]  = running["delay_mins"].apply(delay_color)
        running["delay_label"] = running["delay_mins"].apply(
            lambda x: "On time" if x == 0 else f"{x} min late" if x is not None else "Unknown"
        )

        # Station SRI sizes
        station_sri = merged.groupby("stationfullname")["sri"].sum().reset_index()
        station_sri.columns = ["stationdesc", "total_sri"]

        # Normalise station name for merge (best-effort)
        stations_merged = stations_df.merge(station_sri, on="stationdesc", how="left")
        stations_merged["total_sri"] = (
            pd.to_numeric(stations_merged["total_sri"], errors="coerce")
            .fillna(1)
            .clip(lower=0)
        )
        _max_sri = stations_merged["total_sri"].max()
        if not _max_sri or _max_sri == 0:
            _max_sri = 1
        stations_merged["dot_size"] = (
            6 + 14 * (stations_merged["total_sri"] / _max_sri)
        ).round(1).clip(lower=6)

        fig_map = go.Figure()

        # ── Layer 1: All stations ──────────────────────────────
        fig_map.add_trace(go.Scattermap(
            lat=stations_merged["stationlatitude"],
            lon=stations_merged["stationlongitude"],
            mode="markers",
            marker=dict(
                size=stations_merged["dot_size"],
                color="#6366f1",
                opacity=0.65,
            ),
            text=stations_merged["stationdesc"],
            customdata=stations_merged["total_sri"].round(1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Congestion SRI: %{customdata}<extra></extra>"
            ),
            name="Stations",
        ))

        # ── Layer 2: Not-yet-started trains ───────────────────
        if not not_started.empty:
            not_started_valid = not_started.dropna(subset=["trainlatitude","trainlongitude"])
            not_started_valid = not_started_valid[not_started_valid["trainlatitude"] != 0]
            if not not_started_valid.empty:
                fig_map.add_trace(go.Scattermap(
                    lat=not_started_valid["trainlatitude"],
                    lon=not_started_valid["trainlongitude"],
                    mode="markers",
                    marker=dict(size=10, color="#5b6499", opacity=0.5),
                    text=not_started_valid["traincode"],
                    customdata=not_started_valid["direction"],
                    hovertemplate="<b>%{text}</b><br>Not yet departed<br>%{customdata}<extra></extra>",
                    name="Not departed",
                ))

        # ── Layer 3: Running trains ────────────────────────────
        for status, color, label in [
            ("On time",      "#22c55e", "On time"),
            ("Late ≤5 min",  "#f59e0b", "1–5 min late"),
            ("Late >5 min",  "#ef4444", ">5 min late"),
            ("Unknown",      "#5b6499", "Status unknown"),
        ]:
            if status == "On time":
                subset = running[running["delay_mins"] == 0]
            elif status == "Late ≤5 min":
                subset = running[(running["delay_mins"] > 0) & (running["delay_mins"] <= 5)]
            elif status == "Late >5 min":
                subset = running[running["delay_mins"] > 5]
            else:
                subset = running[running["delay_mins"].isna()]

            if subset.empty:
                continue

            # Parse a clean hover message: first two non-empty lines of PublicMessage
            def short_msg(msg):
                if not isinstance(msg, str):
                    return ""
                lines = [l.strip() for l in msg.replace("\\n", "\n").split("\n") if l.strip()]
                return "<br>".join(lines[:2])

            subset = subset.copy()
            subset["short_msg"] = subset["publicmessage"].apply(short_msg)

            fig_map.add_trace(go.Scattermap(
                lat=subset["trainlatitude"],
                lon=subset["trainlongitude"],
                mode="markers+text",
                marker=dict(
                    size=14,
                    color=color,
                    opacity=0.92,
                ),
                text=subset["traincode"],
                textfont=dict(size=8, color="white"),
                textposition="top center",
                customdata=list(zip(subset["delay_label"], subset["direction"], subset["short_msg"])),
                hovertemplate=(
                    "<b>Train %{text}</b><br>"
                    "%{customdata[2]}<br>"
                    "Direction: %{customdata[1]}<br>"
                    "Status: <b>%{customdata[0]}</b>"
                    "<extra></extra>"
                ),
                name=label,
            ))

        fig_map.update_layout(
            map=dict(
                style="carto-darkmatter",
                center=dict(lat=53.35, lon=-7.85),
                zoom=6.4,
            ),
            legend=dict(
                bgcolor="#1e2130",
                bordercolor="#2e3350",
                borderwidth=1,
                font=dict(color="#c9cfe8", size=12),
                orientation="v",
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
            ),
            paper_bgcolor="#13151f",
            margin=dict(l=0, r=0, t=0, b=0),
            height=620,
        )
        st.plotly_chart(fig_map, width="stretch")

        # Live summary KPIs below the map
        n_running  = len(running)
        n_ontime   = (running["delay_mins"] == 0).sum()
        n_late     = (running["delay_mins"] > 0).sum()
        n_late_bad = (running["delay_mins"] > 5).sum()

        m1, m2, m3, m4 = st.columns(4)
        def metric_card(col, label, value, sub=""):
            col.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{sub}</div>
                </div>
            """, unsafe_allow_html=True)

        metric_card(m1, "Trains running",  n_running,  "across the network right now")
        metric_card(m2, "On time",         n_ontime,   f"{n_ontime/max(n_running,1)*100:.0f}% of running trains")
        metric_card(m3, "Slightly late",   n_late - n_late_bad,  "1–5 minutes late")
        metric_card(m4, "Significantly late", n_late_bad, "more than 5 minutes late")


# ════════════════════════════════════════════════════════════════════
# TAB 3 — JOURNEY PROFILER
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<p class="section-title">Journey Delay Profiler</p>', unsafe_allow_html=True)
    st.markdown('<p class="caption-text">Pick any running train to see a stop-by-stop breakdown of scheduled vs actual times — showing exactly where and how delays build up (or recover) along the route.</p>', unsafe_allow_html=True)

    with st.spinner("Loading running trains…"):
        live_trains = _shared_trains_df

    if live_trains.empty:
        st.error("Could not reach the Rail API.")
    else:
        live_trains.columns = live_trains.columns.str.strip().str.lower()

        # Build dropdown options — only R (running) trains with a usable public message
        running_opts = live_trains[live_trains["trainstatus"] == "R"].copy()

        def option_label(row):
            msg = str(row.get("publicmessage",""))
            m = re.search(r"-\s(.+?)\s+\(", msg.replace("\\n","\n"))
            route = m.group(1) if m else row.get("direction","")
            # Include traincode in label to guarantee uniqueness
            return f"{row['traincode']}  —  {route}"

        if not running_opts.empty:
            running_opts = running_opts.copy()
            running_opts["label"] = running_opts.apply(option_label, axis=1)
            # Drop duplicate train codes (API occasionally returns the same train twice)
            running_opts = running_opts.drop_duplicates(subset="traincode", keep="first")
            # Keep dict keyed by traincode, not label, then build sorted label list
            code_to_label = dict(zip(running_opts["traincode"], running_opts["label"]))
            label_to_code = {v: k for k, v in code_to_label.items()}
            sorted_labels  = sorted(label_to_code.keys())
        else:
            label_to_code = {}
            sorted_labels  = []

        pcol1, pcol2 = st.columns([1, 3], gap="large")
        with pcol1:
            manual_code = st.text_input(
                "Enter train code directly",
                placeholder="e.g. A905",
                help="Type any train code from Tab 1 — works even if the train isn't in the live dropdown below.",
            ).strip().upper()

            if not running_opts.empty:
                selected_label = st.selectbox(
                    "Or pick a currently running train",
                    sorted_labels,
                )
                dropdown_code = label_to_code[selected_label]
            else:
                st.info("No trains currently running in live feed.")
                selected_label = None
                dropdown_code  = None

            # Manual entry takes priority over dropdown
            selected_code = manual_code if manual_code else dropdown_code
            today_str     = date.today().strftime("%d %b %Y")

            if not selected_code:
                st.info("Enter a train code or wait for the live feed to load.")
            else:
                # Show live status if we have it from the dropdown
                live_row = running_opts[running_opts["traincode"] == selected_code] \
                    if not running_opts.empty else pd.DataFrame()
                if not live_row.empty:
                    clean_msg = str(live_row.iloc[0].get("publicmessage","")).replace("\\n","\n")
                    st.markdown(f"""
                        <div style='background:#1e2130;border:1px solid #2e3350;border-radius:10px;padding:14px 16px;margin-top:8px;'>
                            <div style='color:#8d93b0;font-size:0.72rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;'>Live Status</div>
                            <div style='color:#c9cfe8;font-size:0.84rem;line-height:1.6;white-space:pre-line;'>{clean_msg}</div>
                        </div>
                    """, unsafe_allow_html=True)
                elif manual_code:
                    st.markdown(f"""
                        <div style='background:#1e2130;border:1px solid #2e3350;border-radius:10px;padding:14px 16px;margin-top:8px;'>
                            <div style='color:#8d93b0;font-size:0.72rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;'>Train code</div>
                            <div style='color:#c9cfe8;font-size:0.84rem;'>{manual_code} — loading movement history…</div>
                        </div>
                    """, unsafe_allow_html=True)

        with pcol2:
            if not selected_code:
                st.empty()
            else:
                with st.spinner(f"Fetching movements for {selected_code}…"):
                    moves = get_train_movements(selected_code, today_str)

                if moves.empty:
                    st.warning("No movement data available for this train yet.")
                else:
                    moves.columns = moves.columns.str.strip().str.lower()

                    # Normalise column names (API returns mixed-case variants)
                    col_map = {
                        "scheduledarrival":   "sched_arr",
                        "scheduleddeparture": "sched_dep",
                        "arrival":            "act_arr",
                        "departure":          "act_dep",
                        "locationfullname":   "stop",
                        "locationorder":      "order",
                        "locationtype":       "loc_type",
                        "stoptype":           "stop_type",
                    }
                    moves = moves.rename(columns={k: v for k, v in col_map.items() if k in moves.columns})

                    moves["order"] = pd.to_numeric(moves.get("order", range(len(moves))), errors="coerce")
                    moves = moves.sort_values("order").reset_index(drop=True)

                    # Convert times to minutes since midnight
                    for col in ["sched_arr","sched_dep","act_arr","act_dep"]:
                        if col in moves.columns:
                            moves[f"{col}_min"] = moves[col].apply(time_str_to_minutes)

                    # Use scheduled departure for origin, scheduled arrival for all others
                    moves["sched_min"] = moves.get("sched_arr_min", pd.Series([None]*len(moves)))
                    moves.loc[moves["sched_min"].isna(), "sched_min"] = moves.get("sched_dep_min")

                    moves["act_min"] = moves.get("act_arr_min", pd.Series([None]*len(moves)))
                    moves.loc[moves["act_min"].isna(), "act_min"] = moves.get("act_dep_min")

                    # Delay in minutes at each stop
                    moves["delay"] = moves["act_min"] - moves["sched_min"]

                    # Only stops where we have at least scheduled time
                    plot_df = moves[moves["sched_min"].notna()].copy()
                    passed  = plot_df[plot_df["act_min"].notna()].copy()
                    future  = plot_df[plot_df["act_min"].isna()].copy()

                    if plot_df.empty:
                        st.warning("No timetable data found for this service.")
                    else:
                        # ── Subplots ──────────────────────────────────
                        fig_journey = make_subplots(
                            rows=2, cols=1,
                            row_heights=[0.55, 0.45],
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=[
                                "Stop-by-Stop Timeline  (dotted = scheduled · solid = actual · marker colour = delay severity)",
                                "Delay at Each Stop (minutes late vs schedule)",
                            ],
                        )

                        stops_all     = plot_df["stop"].tolist()
                        stops_passed  = passed["stop"].tolist()
                        stops_future  = future["stop"].tolist()

                        # Convert minutes-since-midnight to a dummy datetime so Plotly
                        # renders a proper numeric time axis with even spacing
                        BASE_DATE = "2000-01-01 "
                        def mins_to_dt(m):
                            if m is None:
                                return None
                            h, mn = divmod(int(m), 60)
                            return f"{BASE_DATE}{h:02d}:{mn:02d}:00"

                        # Scheduled times (all stops)
                        fig_journey.add_trace(go.Scatter(
                            x=plot_df["stop"],
                            y=plot_df["sched_min"].apply(mins_to_dt),
                            mode="lines+markers",
                            name="Scheduled",
                            line=dict(color="#6366f1", width=2, dash="dot"),
                            marker=dict(size=6, color="#6366f1"),
                            hovertemplate="%{x}<br>Scheduled: %{customdata}<extra></extra>",
                            customdata=plot_df["sched_min"].apply(minutes_to_hhmm),
                        ), row=1, col=1)

                        # Actual times (completed stops)
                        if not passed.empty:
                            fig_journey.add_trace(go.Scatter(
                                x=passed["stop"],
                                y=passed["act_min"].apply(mins_to_dt),
                                mode="lines+markers",
                                name="Actual",
                                line=dict(color="#22c55e", width=2.5),
                                marker=dict(size=8, color=passed["delay"].apply(
                                    lambda d: "#22c55e" if (d is None or d <= 0)
                                              else "#f59e0b" if d <= 5
                                              else "#ef4444"
                                )),
                                hovertemplate="%{x}<br>Actual: %{customdata}<extra></extra>",
                                customdata=passed["act_min"].apply(minutes_to_hhmm),
                            ), row=1, col=1)

                        # Fill between scheduled and actual on passed stops
                        if not passed.empty:
                            sched_y  = passed["sched_min"].apply(mins_to_dt).tolist()
                            actual_y = passed["act_min"].apply(mins_to_dt).tolist()
                            x_fill   = passed["stop"].tolist() + passed["stop"].tolist()[::-1]
                            y_fill   = actual_y + sched_y[::-1]
                            # Determine fill color by max delay
                            max_d = passed["delay"].max()
                            fill_c = "rgba(34,197,94,0.12)" if max_d <= 0 else \
                                     "rgba(245,158,11,0.12)" if max_d <= 5 else \
                                     "rgba(239,68,68,0.12)"
                            fig_journey.add_trace(go.Scatter(
                                x=x_fill, y=y_fill,
                                fill="toself",
                                fillcolor=fill_c,
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                            ), row=1, col=1)

                        # Delay bars — completed stops
                        if not passed.empty:
                            bar_colors = passed["delay"].apply(
                                lambda d: "#22c55e" if (d is None or d <= 0)
                                          else "#f59e0b" if d <= 5
                                          else "#ef4444"
                            )
                            fig_journey.add_trace(go.Bar(
                                x=passed["stop"],
                                y=passed["delay"].clip(lower=0),
                                name="Delay (min)",
                                marker=dict(color=bar_colors, line=dict(width=0)),
                                hovertemplate="%{x}<br>Delay: %{y} min<extra></extra>",
                            ), row=2, col=1)

                        # Future stops (no actual data yet) — ghost bars
                        if not future.empty:
                            fig_journey.add_trace(go.Bar(
                                x=future["stop"],
                                y=[0]*len(future),
                                name="Future stop",
                                marker=dict(color="rgba(99,102,241,0.2)", line=dict(width=0)),
                                hovertemplate="%{x}<br>Not yet reached<extra></extra>",
                            ), row=2, col=1)

                        # Zero line on delay chart
                        fig_journey.add_hline(y=0, row=2, col=1, line=dict(color="#2e3350", width=1))

                        n_stops = len(plot_df)
                        fig_journey.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#c9cfe8", size=11),
                            height=580,
                            legend=dict(
                                bgcolor="#1e2130", bordercolor="#2e3350", borderwidth=1,
                                font=dict(color="#c9cfe8", size=11),
                                orientation="h", x=0, y=1.04,
                            ),
                            margin=dict(l=20, r=20, t=40, b=20),
                            bargap=0.3,
                        )
                        for i in [1, 2]:
                            fig_journey.update_xaxes(
                                tickangle=-40, tickfont=dict(size=9),
                                gridcolor="#2e3350", row=i, col=1,
                            )
                        fig_journey.update_yaxes(
                            title="Clock time",
                            tickformat="%H:%M",
                            gridcolor="#2e3350",
                            row=1, col=1,
                        )
                        fig_journey.update_yaxes(
                            title="Delay (min)", gridcolor="#2e3350", row=2, col=1,
                        )
                        # Style subplot titles
                        for ann in fig_journey.layout.annotations:
                            ann.update(font=dict(size=12, color="#c9cfe8"))

                        st.plotly_chart(fig_journey, width="stretch")

                        # Summary strip below chart
                        if not passed.empty:
                            max_delay_stop = passed.loc[passed["delay"].idxmax(), "stop"] if passed["delay"].notna().any() else "—"
                            max_delay_val  = passed["delay"].max()
                            final_delay    = passed.iloc[-1]["delay"] if not passed.empty else None
                            recovered      = (final_delay is not None and final_delay < max_delay_val and max_delay_val > 0)

                            s1, s2, s3 = st.columns(3)
                            def metric_card(col, label, value, sub=""):
                                col.markdown(f"""
                                    <div class="metric-card">
                                        <div class="metric-label">{label}</div>
                                        <div class="metric-value">{value}</div>
                                        <div class="metric-sub">{sub}</div>
                                    </div>
                                """, unsafe_allow_html=True)

                            metric_card(s1, "Peak delay",
                                f"{int(max_delay_val)} min" if pd.notna(max_delay_val) and max_delay_val > 0 else "On time",
                                f"at {max_delay_stop}")
                            metric_card(s2, "Current delay",
                                f"{int(final_delay)} min" if pd.notna(final_delay) and final_delay > 0 else "On time",
                                "at last recorded stop")
                            metric_card(s3, "Time recovery",
                                "Yes ✓" if recovered else "No",
                                "train making up lost time" if recovered else "delay not recovered")


# ════════════════════════════════════════════════════════════════════
# TAB 4 — FLEET PERFORMANCE SCORECARD
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Fleet Performance Scorecard</p>', unsafe_allow_html=True)
    st.markdown('<p class="caption-text">Historical comparison of DART, Mainline and Suburban services — aggregated from collected snapshots every minute, giving stable and meaningful rates rather than a fluctuating live count.</p>', unsafe_allow_html=True)

    FLEET_TYPES  = {"DART": "D", "Mainline": "M", "Suburban": "S"}
    FLEET_COLORS = {"DART": "#6366f1", "Mainline": "#f59e0b", "Suburban": "#22c55e"}

    # ── Load historical fleet data from DB ────────────────────────
    try:
        fl = pd.read_sql("fleet_logs", DB)
        fl.columns = fl.columns.str.strip().str.lower()
        fl["timestamp"] = pd.to_datetime(fl["timestamp"])
        has_fleet_history = not fl.empty
    except Exception:
        fl = pd.DataFrame()
        has_fleet_history = False

    # ── Build stats: aggregate over all stored snapshots ─────────
    fleet_stats = []

    if has_fleet_history:
        for name in FLEET_TYPES:
            subset = fl[fl["fleet"] == name]
            if subset.empty:
                continue
            # Total observations = sum of "known" (trains with parseable delay) across all snapshots
            total_obs        = subset["known"].sum()
            total_on_time    = subset["on_time"].sum()
            total_slightly   = subset["slightly_late"].sum()
            total_sig        = subset["significantly"].sum()
            avg_delay        = round(float(subset["avg_delay"].mean()), 1)
            punctuality      = round(total_on_time / total_obs * 100, 1) if total_obs > 0 else 0.0
            snapshots        = len(subset)
            avg_running      = round(float(subset["total"].mean()), 1)
            avg_sig          = round(float(subset["significantly"].mean()), 1)  # avg per snapshot

            fleet_stats.append({
                "fleet":          name,
                "total":          avg_running,           # avg trains running per snapshot
                "snapshots":      snapshots,
                "on_time":        int(total_on_time),
                "slightly_late":  int(total_slightly),
                "significantly":  int(total_sig),
                "avg_sig":        avg_sig,               # avg >5 min late per snapshot
                "avg_delay":      avg_delay,
                "punctuality":    punctuality,
                "color":          FLEET_COLORS[name],
            })

    # ── Fallback: use a single live snapshot if no history yet ────
    if not fleet_stats:
        with st.spinner("No historical data yet — fetching live snapshot as fallback…"):
            raw_fleets = {name: get_trains_by_type(code) for name, code in FLEET_TYPES.items()}

        for name, df_fleet in raw_fleets.items():
            if df_fleet.empty:
                continue
            df_fleet = df_fleet.copy()
            df_fleet.columns = df_fleet.columns.str.strip().str.lower()
            running = df_fleet[df_fleet.get("trainstatus", pd.Series(dtype=str)) == "R"].copy() \
                if "trainstatus" in df_fleet.columns else df_fleet.copy()
            running["delay_mins"] = running["publicmessage"].apply(parse_delay_from_message) \
                if "publicmessage" in running.columns else None

            total        = len(running)
            known        = running["delay_mins"].notna().sum()
            on_time      = (running["delay_mins"] == 0).sum()
            slightly     = ((running["delay_mins"] > 0) & (running["delay_mins"] <= 5)).sum()
            significantly = (running["delay_mins"] > 5).sum()
            avg_delay    = running["delay_mins"].mean() if known > 0 else 0
            punctuality  = on_time / known * 100 if known > 0 else 0

            fleet_stats.append({
                "fleet":          name,
                "total":          total,
                "snapshots":      1,
                "on_time":        int(on_time),
                "slightly_late":  int(slightly),
                "significantly":  int(significantly),
                "avg_sig":        float(significantly),   # same as significantly for single snapshot
                "avg_delay":      round(float(avg_delay), 1),
                "punctuality":    round(float(punctuality), 1),
                "color":          FLEET_COLORS[name],
            })
        if fleet_stats:
            st.info("Showing a single live snapshot — restart the collector to begin building history. Values will stabilise over time.", icon="ℹ️")

    if not fleet_stats:
        st.error("Could not fetch fleet data from the Rail API.")
    else:
        stats_df = pd.DataFrame(fleet_stats)

        # ── ROW: KPI cards per fleet ──────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns(len(fleet_stats), gap="large")

        for col, row in zip(cols, fleet_stats):
            pct_color = "#22c55e" if row["punctuality"] >= 80 else \
                        "#f59e0b" if row["punctuality"] >= 60 else "#ef4444"
            snapshots_label = f"{row['snapshots']} snapshots" if row["snapshots"] > 1 else "live snapshot"
            col.markdown(f"""
                <div style='background:#1e2130;border:1px solid {row["color"]}55;border-radius:14px;
                            padding:20px 18px;text-align:center;'>
                    <div style='color:{row["color"]};font-size:0.72rem;text-transform:uppercase;
                                letter-spacing:.1em;font-weight:600;margin-bottom:8px;'>
                        {row["fleet"]}
                    </div>
                    <div style='color:{pct_color};font-size:2.6rem;font-weight:800;line-height:1;'>
                        {row["punctuality"]:.0f}%
                    </div>
                    <div style='color:#8d93b0;font-size:0.75rem;margin-top:4px;'>on-time rate ({snapshots_label})</div>
                    <hr style='border-color:#2e3350;margin:12px 0;'>
                    <div style='display:flex;justify-content:space-between;'>
                        <div>
                            <div style='color:#c9cfe8;font-size:1rem;font-weight:600;'>{row["total"]}</div>
                            <div style='color:#6b7294;font-size:0.7rem;'>avg running</div>
                        </div>
                        <div>
                            <div style='color:#f59e0b;font-size:1rem;font-weight:600;'>{row["avg_delay"]}</div>
                            <div style='color:#6b7294;font-size:0.7rem;'>avg delay (min)</div>
                        </div>
                        <div>
                            <div style='color:#ef4444;font-size:1rem;font-weight:600;'>{row["avg_sig"]}</div>
                            <div style='color:#6b7294;font-size:0.7rem;'>avg &gt;5 min late</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROW: Delay severity + Average delay side by side ──────
        chart2_l, chart2_r = st.columns([1, 1], gap="large")

        with chart2_l:
            st.markdown('<p class="section-title">Average Delay by Fleet Type</p>', unsafe_allow_html=True)
            st.markdown('<p class="caption-text">Mean minutes late across all collected snapshots (average of per-snapshot averages).</p>', unsafe_allow_html=True)

            fig_avg = go.Figure(go.Bar(
                x=stats_df["fleet"],
                y=stats_df["avg_delay"],
                marker=dict(
                    color=stats_df["avg_delay"],
                    colorscale=[[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
                    line=dict(width=0),
                ),
                text=stats_df["avg_delay"].apply(lambda x: f"{x} min"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Avg delay: %{y:.1f} min<extra></extra>",
            ))
            fig_avg.update_layout(
                xaxis=dict(title="Fleet Type"),
                yaxis=dict(title="Avg Delay (min)", gridcolor="#2e3350"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9cfe8", size=12),
                margin=dict(l=10, r=10, t=10, b=10),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig_avg, width="stretch")

        with chart2_r:
            st.markdown('<p class="section-title">Delay Severity Breakdown</p>', unsafe_allow_html=True)
            st.markdown('<p class="caption-text">Average number of trains per snapshot in each delay category — normalised by snapshot count so all fleets are comparable regardless of how long the collector has been running.</p>', unsafe_allow_html=True)

            stack_df = stats_df.copy()
            stack_df["avg_on_time"]  = stack_df["on_time"]       / stack_df["snapshots"]
            stack_df["avg_slightly"] = stack_df["slightly_late"]  / stack_df["snapshots"]

            fig_stack = go.Figure()
            for label, key, color in [
                ("On time",      "avg_on_time",  "#22c55e"),
                ("1–5 min late", "avg_slightly", "#f59e0b"),
                (">5 min late",  "avg_sig",       "#ef4444"),
            ]:
                fig_stack.add_trace(go.Bar(
                    name=label,
                    x=stack_df["fleet"],
                    y=stack_df[key].round(1),
                    marker=dict(color=color, line=dict(width=0)),
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.1f}} avg per snapshot<extra></extra>",
                ))
            fig_stack.update_layout(
                barmode="stack",
                xaxis=dict(title="Fleet Type"),
                yaxis=dict(title="Avg trains per snapshot", gridcolor="#2e3350"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9cfe8", size=12),
                legend=dict(
                    bgcolor="#1e2130", bordercolor="#2e3350", borderwidth=1,
                    orientation="h", x=0, y=1.12,
                    font=dict(color="#c9cfe8", size=11),
                ),
                margin=dict(l=10, r=10, t=30, b=10),
                height=280,
            )
            st.plotly_chart(fig_stack, width="stretch")

        # ── Insight banner ────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if len(fleet_stats) >= 2:
            worst  = min(fleet_stats, key=lambda x: x["punctuality"])
            best   = max(fleet_stats, key=lambda x: x["punctuality"])
            w_col  = "#ef4444" if worst["punctuality"] < 60 else "#f59e0b"
            b_col  = "#22c55e"
            gap    = best["punctuality"] - worst["punctuality"]

            st.markdown(f"""
                <div style='background:#1e2130;border:1px solid #2e3350;border-radius:12px;
                            padding:16px 22px;display:flex;gap:32px;align-items:center;'>
                    <div style='color:#8d93b0;font-size:0.75rem;text-transform:uppercase;
                                letter-spacing:.08em;white-space:nowrap;'>Key Insight</div>
                    <div style='color:#c9cfe8;font-size:0.9rem;line-height:1.6;'>
                        <span style='color:{w_col};font-weight:700;'>{worst["fleet"]}</span>
                        is the worst-performing fleet historically with a
                        <span style='color:{w_col};font-weight:700;'>{worst["punctuality"]:.0f}% on-time rate</span>
                        and an average delay of <span style='color:{w_col};font-weight:700;'>{worst["avg_delay"]} min</span> —
                        <span style='color:{b_col};font-weight:700;'>{gap:.0f} percentage points</span>
                        behind {best["fleet"]} ({best["punctuality"]:.0f}% on time).
                        Targeted investment in {worst["fleet"]} infrastructure or scheduling would yield the
                        highest network-wide punctuality improvement.
                    </div>
                </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# TAB 5 — SNAPSHOT OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<p class="section-title">Snapshot Overview — Late Services</p>', unsafe_allow_html=True)
    st.markdown('<p class="caption-text">Quick, actionable view of delays by day, station, and route. Default view shows averages across all days.</p>', unsafe_allow_html=True)

    day_options, day_map = build_day_options(df["date"])
    c1, c2, c3 = st.columns([1, 1, 1.4], gap="large")
    with c1:
        selected_day = st.selectbox("🗓️ Day", day_options, index=0, key="mgr_day")
    if selected_day == "All days":
        base_df = df.copy()
    else:
        base_df = df[df["date"] == day_map[selected_day]].copy()

    station_options = sorted(base_df["stationfullname"].dropna().unique()) if not base_df.empty else []
    with c2:
        if station_options:
            selected_station = st.selectbox("📍 Station", station_options, key="mgr_station")
        else:
            selected_station = st.selectbox("📍 Station", ["No stations available"], key="mgr_station")
            selected_station = None

    if selected_station:
        base_df = base_df[base_df["stationfullname"] == selected_station].copy()

    corridor_df = base_df[base_df["origin"].notna() & base_df["destination"].notna()].copy()
    if not corridor_df.empty:
        corridor_df["corridor"] = corridor_df["origin"] + " → " + corridor_df["destination"]
        corridor_options = ["All routes"] + sorted(corridor_df["corridor"].unique())
    else:
        corridor_options = ["All routes"]

    with c3:
        selected_corridor = st.selectbox(
            "➡️ Route (origin → destination)",
            corridor_options,
            key="mgr_route",
        )

    if selected_corridor != "All routes":
        base_df = base_df[base_df["origin"] + " → " + base_df["destination"] == selected_corridor]

    if base_df.empty:
        st.warning("No data available for the selected filters yet.")
    else:
        late_df = base_df[base_df["late"] > 0].copy()
        if late_df.empty:
            st.info("No late services found for the selected filters.")
        else:
            summary = late_df.groupby(
                ["traincode", "origin", "destination", "stationfullname"]
            ).agg(
                avg_late=("late", "mean"),
                max_late=("late", "max"),
                late_count=("late", "count"),
                late_days=("date", "nunique"),
                last_seen=("timestamp", "max"),
            ).reset_index()

            summary["avg_late"] = summary["avg_late"].round(1)
            summary["max_late"] = summary["max_late"].round(0).astype(int)
            summary["late_count"] = summary["late_count"].astype(int)
            summary["late_days"] = summary["late_days"].astype(int)
            summary["last_seen"] = summary["last_seen"].dt.strftime("%H:%M")
            summary = summary.sort_values(["avg_late", "late_count"], ascending=[False, False])

            if selected_day == "All days":
                st.markdown('<p class="caption-text">Average lateness across all days and how often each service was late.</p>', unsafe_allow_html=True)
                display_df = summary[[
                    "traincode",
                    "origin",
                    "destination",
                    "stationfullname",
                    "avg_late",
                    "late_days",
                ]]
            else:
                st.markdown('<p class="caption-text">Late services for the selected day with peak delay and last observed time.</p>', unsafe_allow_html=True)
                display_df = summary[[
                    "traincode",
                    "origin",
                    "destination",
                    "stationfullname",
                    "avg_late",
                    "max_late",
                    "late_count",
                    "last_seen",
                ]]

            display_df = display_df.rename(columns={
                "traincode": "Train",
                "origin": "Origin",
                "destination": "Destination",
                "stationfullname": "Station",
                "avg_late": "Avg late (min)",
                "max_late": "Max late (min)",
                "late_count": "Late count",
                "late_days": "Late days",
                "last_seen": "Last seen",
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        