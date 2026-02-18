import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
import re
from datetime import datetime, date
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
@st.cache_data(ttl=60)
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

@st.cache_data(ttl=60)
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

@st.cache_data(ttl=30)
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

# ─────────────────────────────────────────────
# LOAD HISTORICAL DB DATA
# ─────────────────────────────────────────────
df = pd.read_sql("train_logs", DB)
df.columns = df.columns.str.strip().str.lower()
if df.empty:
    st.warning("Waiting for data collection...")
    st.stop()

df["late"]      = pd.to_numeric(df["late"],  errors="coerce").fillna(0)
df["duein"]     = pd.to_numeric(df["duein"], errors="coerce").fillna(0)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# SRI calculation
risk = df.groupby("traincode")["late"].agg(["mean", "count"])
risk["delay_freq"] = (
    df[df["late"] > 5].groupby("traincode")["late"].count() / risk["count"]
).fillna(0)
risk["sri"] = (0.5 * risk["mean"]) + (0.5 * risk["delay_freq"] * 10)
risk = risk.reset_index()

latest = df.sort_values("timestamp").groupby("traincode").tail(1)
merged = latest.merge(risk[["traincode", "sri"]], on="traincode")

def sri_label(x):
    if x < 3:  return "Reliable"
    elif x < 7: return "Moderate"
    else:       return "High Risk"

merged["reliability"] = merged["sri"].apply(sri_label)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊  Station Analysis", "🗺️  Live Network Map", "🔍  Journey Profiler", "🚄  Fleet Scorecard"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — STATION ANALYSIS (existing dashboard)
# ════════════════════════════════════════════════════════════════════
with tab1:
    station = st.selectbox("📍 Select Station", sorted(merged["stationfullname"].unique()))
    view = merged[merged["stationfullname"] == station].copy().sort_values("sri", ascending=True)

    # KPI cards
    total_trains  = len(view)
    avg_delay     = view["late"].mean()
    high_risk_pct = (view["reliability"] == "High Risk").sum() / max(total_trains, 1) * 100
    avg_sri       = view["sri"].mean()

    k1, k2, k3, k4 = st.columns(4)
    def metric_card(col, label, value, sub=""):
        col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>
        """, unsafe_allow_html=True)

    metric_card(k1, "Trains in view",   total_trains,           f"at {station}")
    metric_card(k2, "Avg. delay",       f"{avg_delay:.1f} min", "across all services")
    metric_card(k3, "High-risk trains", f"{high_risk_pct:.0f}%","SRI ≥ 7")
    metric_card(k4, "Avg. risk score",  f"{avg_sri:.1f}",       "Service Risk Index")
    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1
    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        st.markdown('<p class="section-title">Train Service Risk Index</p>', unsafe_allow_html=True)
        st.markdown('<p class="caption-text">SRI combines average lateness and delay frequency — higher = less reliable.</p>', unsafe_allow_html=True)
        fig_sri = go.Figure(go.Bar(
            y=view["traincode"],
            x=view["sri"],
            orientation="h",
            marker=dict(
                color=view["sri"],
                colorscale=[[0,"#22c55e"],[0.43,"#f59e0b"],[1,"#ef4444"]],
                cmin=0, cmax=10, line=dict(width=0),
            ),
            text=view["reliability"], textposition="inside", insidetextanchor="middle",
            customdata=view["destination"],
            hovertemplate="<b>%{y}</b><br>SRI: %{x:.1f}<br>To: %{customdata}<extra></extra>",
        ))
        fig_sri.update_layout(
            xaxis=dict(title="Service Risk Index", range=[0, max(view["sri"].max()*1.15, 10)], gridcolor="#2e3350"),
            yaxis=dict(title="", tickfont=dict(size=11)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c9cfe8", size=12), margin=dict(l=10,r=10,t=10,b=30),
            height=max(250, len(view)*38), showlegend=False,
        )
        st.plotly_chart(fig_sri, use_container_width=True)

    with col_r:
        st.markdown('<p class="section-title">Delay Trend — Last 24 Hours</p>', unsafe_allow_html=True)
        st.markdown('<p class="caption-text">15-minute rolling average delay. Peaks signal recurring congestion windows.</p>', unsafe_allow_html=True)
        df_station = df[df["stationfullname"] == station].copy()
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
            height=max(250, len(view)*38),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # Row 2
    col2_l, col2_r = st.columns([1.4, 1], gap="large")
    with col2_l:
        st.markdown('<p class="section-title">Corridor Reliability Heatmap</p>', unsafe_allow_html=True)
        st.markdown('<p class="caption-text">SRI per Origin → Destination corridor for trains serving this station. Green = reliable, Red = high risk.</p>', unsafe_allow_html=True)

        # Filter to corridors involving trains that serve the selected station
        trains_at_station = view["traincode"].unique()
        df_station_corridors = df[df["traincode"].isin(trains_at_station)]

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
            height=max(320, n_rows*40+80),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with col2_r:
        st.markdown('<p class="section-title">Station Congestion Risk</p>', unsafe_allow_html=True)
        st.markdown('<p class="caption-text">Total SRI load per station — higher means more at-risk services arriving.</p>', unsafe_allow_html=True)
        station_risk = merged.groupby("stationfullname")["sri"].sum().sort_values(ascending=True).reset_index()
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
            height=max(320, len(station_risk)*32+80), showlegend=False,
        )
        st.plotly_chart(fig_cong, use_container_width=True)

    # Row 3
    st.markdown('<p class="section-title">Most Unpredictable Train Services</p>', unsafe_allow_html=True)
    st.markdown('<p class="caption-text">Services with highest delay variability. High std. deviation = hard to plan around even if average SRI looks moderate.</p>', unsafe_allow_html=True)
    train_var = df.groupby("traincode")["late"].std().dropna().sort_values(ascending=False).head(12).reset_index()
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
    st.plotly_chart(fig_var, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE NETWORK MAP
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="section-title">Live Train & Station Network</p>', unsafe_allow_html=True)
    st.markdown('<p class="caption-text">All running trains plotted in real time. Colour = how late the train is. Size of station dot = congestion risk (SRI). Refreshes every 60 seconds.</p>', unsafe_allow_html=True)

    with st.spinner("Fetching live data…"):
        trains_df   = get_running_trains()
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
        fig_map.add_trace(go.Scattermapbox(
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
                fig_map.add_trace(go.Scattermapbox(
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

            fig_map.add_trace(go.Scattermapbox(
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
            mapbox=dict(
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
        st.plotly_chart(fig_map, use_container_width=True)

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
        live_trains = get_running_trains()

    if live_trains.empty:
        st.error("Could not reach the Rail API.")
    else:
        live_trains.columns = live_trains.columns.str.strip().str.lower()

        # Build dropdown options — only R (running) trains with a usable public message
        running_opts = live_trains[live_trains["trainstatus"] == "R"].copy()
        if running_opts.empty:
            st.info("No trains currently running.")
        else:
            def option_label(row):
                msg = str(row.get("publicmessage",""))
                # Extract "Origin to Destination" from message line 2
                m = re.search(r"-\s(.+?)\s+\(", msg.replace("\\n","\n"))
                route = m.group(1) if m else row.get("direction","")
                return f"{row['traincode']}  —  {route}"

            running_opts["label"] = running_opts.apply(option_label, axis=1)
            label_to_code = dict(zip(running_opts["label"], running_opts["traincode"]))

            pcol1, pcol2 = st.columns([1, 3], gap="large")
            with pcol1:
                selected_label = st.selectbox("Select a train", list(label_to_code.keys()))
                selected_code  = label_to_code[selected_label]
                today_str      = date.today().strftime("%d %b %Y")

                full_msg = running_opts.loc[
                    running_opts["traincode"] == selected_code, "publicmessage"
                ].values[0] if not running_opts.empty else ""
                clean_msg = str(full_msg).replace("\\n","\n") if full_msg else ""

                st.markdown(f"""
                    <div style='background:#1e2130;border:1px solid #2e3350;border-radius:10px;padding:14px 16px;margin-top:8px;'>
                        <div style='color:#8d93b0;font-size:0.72rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;'>Live Status</div>
                        <div style='color:#c9cfe8;font-size:0.84rem;line-height:1.6;white-space:pre-line;'>{clean_msg}</div>
                    </div>
                """, unsafe_allow_html=True)

            with pcol2:
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
                            subplot_titles=["Scheduled vs Actual Times", "Delay at Each Stop (minutes)"],
                        )

                        stops_all     = plot_df["stop"].tolist()
                        stops_passed  = passed["stop"].tolist()
                        stops_future  = future["stop"].tolist()

                        # Scheduled times (all stops)
                        fig_journey.add_trace(go.Scatter(
                            x=plot_df["stop"],
                            y=plot_df["sched_min"].apply(minutes_to_hhmm),
                            mode="lines+markers",
                            name="Scheduled",
                            line=dict(color="#6366f1", width=2, dash="dot"),
                            marker=dict(size=6, color="#6366f1"),
                            hovertemplate="%{x}<br>Scheduled: %{y}<extra></extra>",
                        ), row=1, col=1)

                        # Actual times (completed stops)
                        if not passed.empty:
                            fig_journey.add_trace(go.Scatter(
                                x=passed["stop"],
                                y=passed["act_min"].apply(minutes_to_hhmm),
                                mode="lines+markers",
                                name="Actual",
                                line=dict(color="#22c55e", width=2.5),
                                marker=dict(size=8, color=passed["delay"].apply(
                                    lambda d: "#22c55e" if (d is None or d <= 0)
                                              else "#f59e0b" if d <= 5
                                              else "#ef4444"
                                )),
                                hovertemplate="%{x}<br>Actual: %{y}<extra></extra>",
                            ), row=1, col=1)

                        # Fill between scheduled and actual on passed stops
                        if not passed.empty:
                            sched_y  = passed["sched_min"].apply(minutes_to_hhmm).tolist()
                            actual_y = passed["act_min"].apply(minutes_to_hhmm).tolist()
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
                            title="Time", gridcolor="#2e3350", row=1, col=1,
                        )
                        fig_journey.update_yaxes(
                            title="Delay (min)", gridcolor="#2e3350", row=2, col=1,
                        )
                        # Style subplot titles
                        for ann in fig_journey.layout.annotations:
                            ann.update(font=dict(size=12, color="#c9cfe8"))

                        st.plotly_chart(fig_journey, use_container_width=True)

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
    st.markdown('<p class="caption-text">Live comparison of DART, Mainline and Suburban services — answers the strategic question: which fleet type needs the most attention?</p>', unsafe_allow_html=True)

    FLEET_TYPES = {"DART": "D", "Mainline": "M", "Suburban": "S"}
    FLEET_COLORS = {"DART": "#6366f1", "Mainline": "#f59e0b", "Suburban": "#22c55e"}

    with st.spinner("Fetching live fleet data across all service types…"):
        raw_fleets = {name: get_trains_by_type(code) for name, code in FLEET_TYPES.items()}

    # ── Build stats per fleet type ────────────────────────────────
    fleet_stats = []
    fleet_train_rows = []

    for name, df_fleet in raw_fleets.items():
        if df_fleet.empty:
            continue
        df_fleet = df_fleet.copy()
        df_fleet.columns = df_fleet.columns.str.strip().str.lower()

        for col in ["trainlatitude", "trainlongitude"]:
            if col in df_fleet.columns:
                df_fleet[col] = pd.to_numeric(df_fleet[col], errors="coerce")

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
            "known":          known,
            "on_time":        int(on_time),
            "slightly_late":  int(slightly),
            "significantly":  int(significantly),
            "avg_delay":      round(float(avg_delay), 1),
            "punctuality":    round(float(punctuality), 1),
            "color":          FLEET_COLORS[name],
        })

        running["fleet"] = name
        fleet_train_rows.append(running)

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
                    <div style='color:#8d93b0;font-size:0.75rem;margin-top:4px;'>on-time rate</div>
                    <hr style='border-color:#2e3350;margin:12px 0;'>
                    <div style='display:flex;justify-content:space-between;'>
                        <div>
                            <div style='color:#c9cfe8;font-size:1rem;font-weight:600;'>{row["total"]}</div>
                            <div style='color:#6b7294;font-size:0.7rem;'>running</div>
                        </div>
                        <div>
                            <div style='color:#f59e0b;font-size:1rem;font-weight:600;'>{row["avg_delay"]}</div>
                            <div style='color:#6b7294;font-size:0.7rem;'>avg delay (min)</div>
                        </div>
                        <div>
                            <div style='color:#ef4444;font-size:1rem;font-weight:600;'>{row["significantly"]}</div>
                            <div style='color:#6b7294;font-size:0.7rem;'>&gt;5 min late</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── ROW: Delay breakdown stacked bar + radar ──────────────
        chart_l, chart_r = st.columns([1.1, 1], gap="large")

        with chart_l:
            st.markdown('<p class="section-title">Delay Severity Breakdown</p>', unsafe_allow_html=True)
            st.markdown('<p class="caption-text">Distribution of on-time, slightly late and significantly late trains per fleet type right now.</p>', unsafe_allow_html=True)

            fig_stack = go.Figure()
            categories = [
                ("On time",         "on_time",       "#22c55e"),
                ("1–5 min late",    "slightly_late", "#f59e0b"),
                (">5 min late",     "significantly",  "#ef4444"),
            ]
            for label, key, color in categories:
                fig_stack.add_trace(go.Bar(
                    name=label,
                    x=stats_df["fleet"],
                    y=stats_df[key],
                    marker=dict(color=color, line=dict(width=0)),
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
                ))

            fig_stack.update_layout(
                barmode="stack",
                xaxis=dict(title="Fleet Type"),
                yaxis=dict(title="Number of Trains", gridcolor="#2e3350"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9cfe8", size=12),
                legend=dict(
                    bgcolor="#1e2130", bordercolor="#2e3350", borderwidth=1,
                    orientation="h", x=0, y=1.12,
                    font=dict(color="#c9cfe8", size=11),
                ),
                margin=dict(l=10, r=10, t=30, b=10),
                height=320,
            )
            st.plotly_chart(fig_stack, use_container_width=True)

        with chart_r:
            st.markdown('<p class="section-title">Multi-Dimension Performance Radar</p>', unsafe_allow_html=True)
            st.markdown('<p class="caption-text">Each axis is a performance dimension — larger area = better overall fleet performance.</p>', unsafe_allow_html=True)

            radar_dims = ["Punctuality", "Low avg delay", "Fleet size", "Reliability score"]

            fig_radar = go.Figure()
            for row in fleet_stats:
                total_max   = max(s["total"] for s in fleet_stats) or 1
                delay_max   = max(s["avg_delay"] for s in fleet_stats) or 1
                sig_max     = max(s["significantly"] for s in fleet_stats) or 1

                # Normalise each dimension to 0-100 (higher = better)
                punct_score    = row["punctuality"]                                    # already 0-100
                low_delay      = max(0, 100 - (row["avg_delay"] / max(delay_max, 0.1)) * 100)
                fleet_coverage = (row["total"] / total_max) * 100
                reliability    = max(0, 100 - (row["significantly"] / max(sig_max, 1)) * 100)

                values = [punct_score, low_delay, fleet_coverage, reliability]
                values_closed = values + [values[0]]
                dims_closed   = radar_dims + [radar_dims[0]]

                fig_radar.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=dims_closed,
                    fill="toself",
                    name=row["fleet"],
                    line=dict(color=row["color"], width=2),
                    fillcolor=hex_to_rgba(row["color"], 0.15),
                    hovertemplate="<b>" + row["fleet"] + "</b><br>%{theta}: %{r:.0f}<extra></extra>",
                ))

            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#1e2130",
                    radialaxis=dict(
                        visible=True, range=[0, 100],
                        gridcolor="#2e3350", tickfont=dict(size=9, color="#6b7294"),
                        tickvals=[25, 50, 75, 100],
                    ),
                    angularaxis=dict(
                        gridcolor="#2e3350",
                        tickfont=dict(size=11, color="#c9cfe8"),
                    ),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9cfe8", size=12),
                legend=dict(
                    bgcolor="#1e2130", bordercolor="#2e3350", borderwidth=1,
                    orientation="h", x=0.15, y=-0.1,
                    font=dict(color="#c9cfe8", size=11),
                ),
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── ROW: Average delay comparison bar + punctuality trend ──
        chart2_l, chart2_r = st.columns([1, 1], gap="large")

        with chart2_l:
            st.markdown('<p class="section-title">Average Delay by Fleet Type</p>', unsafe_allow_html=True)
            st.markdown('<p class="caption-text">Mean minutes late across all running trains with known delay status.</p>', unsafe_allow_html=True)

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
            st.plotly_chart(fig_avg, use_container_width=True)

        with chart2_r:
            st.markdown('<p class="section-title">On-Time Rate vs Fleet Size</p>', unsafe_allow_html=True)
            st.markdown('<p class="caption-text">Bubble size = number of trains running. Reveals whether larger fleets maintain punctuality.</p>', unsafe_allow_html=True)

            fig_bubble = go.Figure()
            for row in fleet_stats:
                fig_bubble.add_trace(go.Scatter(
                    x=[row["fleet"]],
                    y=[row["punctuality"]],
                    mode="markers+text",
                    marker=dict(
                        size=max(20, row["total"] * 2.5),
                        color=row["color"],
                        opacity=0.8,
                        line=dict(width=1.5, color="white"),
                    ),
                    text=[f"{row['punctuality']:.0f}%"],
                    textposition="middle center",
                    textfont=dict(size=13, color="white"),
                    name=row["fleet"],
                    hovertemplate=(
                        f"<b>{row['fleet']}</b><br>"
                        f"On-time: {row['punctuality']:.0f}%<br>"
                        f"Running trains: {row['total']}<br>"
                        f"Avg delay: {row['avg_delay']} min"
                        "<extra></extra>"
                    ),
                ))

            fig_bubble.add_hline(
                y=80, line=dict(color="#2e3350", dash="dash", width=1.5),
                annotation_text="80% target", annotation_font=dict(color="#6b7294", size=10),
                annotation_position="right",
            )
            fig_bubble.update_layout(
                xaxis=dict(title="Fleet Type", showgrid=False),
                yaxis=dict(title="On-Time Rate (%)", range=[0, 105], gridcolor="#2e3350"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c9cfe8", size=12),
                margin=dict(l=10, r=60, t=10, b=10),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

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
                        is the worst-performing fleet right now with a
                        <span style='color:{w_col};font-weight:700;'>{worst["punctuality"]:.0f}% on-time rate</span>
                        and an average delay of <span style='color:{w_col};font-weight:700;'>{worst["avg_delay"]} min</span> —
                        <span style='color:{b_col};font-weight:700;'>{gap:.0f} percentage points</span>
                        behind {best["fleet"]} ({best["punctuality"]:.0f}% on time).
                        Targeted investment in {worst["fleet"]} infrastructure or scheduling would yield the
                        highest network-wide punctuality improvement.
                    </div>
                </div>
            """, unsafe_allow_html=True)
