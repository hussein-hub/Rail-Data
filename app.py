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
@st.cache_data(ttl=5)  # Reduced from 15 to 5 seconds for quicker updates
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

def extract_traincode_from_message(msg):
    """Extract the correct train code from the public message."""
    if not msg or not isinstance(msg, str):
        return None
    # Try to extract from patterns like "D408" or "P213" at the start or after specific markers
    # Pattern: word boundary + 1-2 letters + 1-4 digits
    m = re.search(r'\b([A-Z]{1,2}\d{2,4})\b', str(msg))
    if m:
        return m.group(1)
    return None

def validate_train_coordinates(df):
    """
    Remove trains with suspicious coordinates:
    - 0.0 / 0.0 (not started yet)
    - Identical coordinates to other trains (corruption)
    - Invalid latitude/longitude ranges for Ireland
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Remove trains at 0,0
    df = df[(df["trainlatitude"] != 0.0) | (df["trainlongitude"] != 0.0)]
    
    # Ireland bounds (rough): 51.4 to 55.4 latitude, -5.3 to -10.6 longitude
    df = df[(df["trainlatitude"].between(51.0, 56.0)) & (df["trainlongitude"].between(-11.0, -5.0))]
    
    # Remove trains with identical coordinates to others (likely corrupted duplicates)
    # Group by coordinates and keep only the first train at each location
    coord_groups = df.groupby(["trainlatitude", "trainlongitude"]).size()
    duplicate_coords = coord_groups[coord_groups > 1].index
    
    for coords in duplicate_coords:
        lat, lon = coords
        matches = df[(df["trainlatitude"] == lat) & (df["trainlongitude"] == lon)]
        if len(matches) > 1:
            # Keep only the first one, remove others as likely corrupted duplicates
            keep_idx = matches.index[0]
            remove_indices = matches.index[1:]
            df = df.drop(remove_indices)
    
    return df

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
df = pd.read_sql("SELECT * FROM train_logs", DB)
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

# Load train_movements data for Tab 4
try:
    movements_df = pd.read_sql("SELECT * FROM train_movements", DB)
    movements_df.columns = movements_df.columns.str.strip().str.lower()
    if not movements_df.empty:
        movements_df["fetched_at"] = pd.to_datetime(movements_df["fetched_at"])
        movements_df["date"] = movements_df["fetched_at"].dt.date
        # Parse traindate string to date for filtering
        movements_df["traindate_parsed"] = pd.to_datetime(movements_df["traindate"], format="%d %b %Y", errors="coerce").dt.date
except Exception:
    movements_df = pd.DataFrame()  # Fallback to empty if table doesn't exist yet

# ─────────────────────────────────────────────
# SHARED LIVE DATA  (fetched once, used by Tab 2 and Tab 3)
# Both tabs must see the exact same snapshot so a train visible on
# the map is always available in the Journey Profiler dropdown.
# ─────────────────────────────────────────────
_shared_trains_df = get_running_trains()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Station Analysis",
    "🗺️  Live Network Map",
    "🔍  Journey Profiler",
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
                st.markdown('<p class="section-title">Most Severely Delayed Train Services</p>', unsafe_allow_html=True)
                st.markdown('<p class="caption-text">Services with highest average delays. These trains consistently accumulate the most lost time across all stops.</p>', unsafe_allow_html=True)
                train_delays = filtered_df.groupby("traincode")["late"].mean().dropna().sort_values(ascending=False).head(12).reset_index()
                train_delays.columns = ["traincode","avg_delay"]
                fig_var = go.Figure(go.Bar(
                    x=train_delays["traincode"], y=train_delays["avg_delay"],
                    marker=dict(color=train_delays["avg_delay"],
                        colorscale=[[0,"#22c55e"],[0.5,"#f59e0b"],[1,"#ef4444"]], line=dict(width=0)),
                    hovertemplate="Train: %{x}<br>Avg delay: %{y:.1f} min<extra></extra>",
                ))
                fig_var.update_layout(
                    xaxis=dict(title="Train Code", tickangle=-30),
                    yaxis=dict(title="Avg Delay (min)", gridcolor="#2e3350"),
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
    st.markdown('<p class="caption-text">All running trains plotted in real time. Colour = how late the train is. Size of station dot = average SRI per service (historical). Refreshes every 15 seconds.</p>', unsafe_allow_html=True)

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

        # ──── ADVANCED DATA VALIDATION ────────────────────────────────────
        # 1. Remove coordinate duplicates (corruption indicator)
        running = validate_train_coordinates(running)
        
        # 2. Fix corrupted train codes from API ────────────────────────────
        # The API sometimes returns incorrect traincode values, but publicmessage has the truth
        running["extracted_traincode"] = running["publicmessage"].apply(extract_traincode_from_message)
        # Use extracted code if available and different, otherwise keep API traincode
        running["traincode_clean"] = running.apply(
            lambda row: row["extracted_traincode"] if row["extracted_traincode"] else row["traincode"],
            axis=1
        )
        # Remove trains with None/invalid train codes
        running = running[running["traincode_clean"].notna()].copy()
        
        # 3. Remove duplicates keeping the first occurrence of each train code
        running = running.drop_duplicates(subset="traincode_clean", keep="first").copy()
        
        # 4. Final validation: ensure traincode matches the expected pattern
        running = running[running["traincode_clean"].str.match(r'^[A-Z]{1,2}\d{2,4}$', na=False)].copy()
        
        # Update the traincode column with cleaned version
        running["traincode"] = running["traincode_clean"]

        running["delay_mins"] = running["publicmessage"].apply(parse_delay_from_message)
        running["dot_color"]  = running["delay_mins"].apply(delay_color)
        running["delay_label"] = running["delay_mins"].apply(
            lambda x: "On time" if x == 0 else f"{x} min late" if x is not None else "Unknown"
        )

        # Station SRI sizes — use average SRI per service (mean) instead of sum
        station_sri = merged.groupby("stationfullname")["sri"].mean().reset_index()
        station_sri.columns = ["stationdesc", "total_sri"]

        # Normalise station name for merge (best-effort)
        stations_merged = stations_df.merge(station_sri, on="stationdesc", how="left")
        stations_merged["total_sri"] = (
            pd.to_numeric(stations_merged["total_sri"], errors="coerce")
            .fillna(0)
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
                "Avg SRI: %{customdata}<extra></extra>"
            ),
            name="Stations",
        ))

        # ── Layer 2: Not-yet-started trains ───────────────────
        if not not_started.empty:
            not_started = not_started.copy()
            # Apply coordinate validation
            not_started = validate_train_coordinates(not_started)
            
            # Apply same train code validation to not_started trains
            not_started["extracted_traincode"] = not_started["publicmessage"].apply(extract_traincode_from_message) if "publicmessage" in not_started.columns else None
            not_started["traincode_clean"] = not_started.apply(
                lambda row: row.get("extracted_traincode") if pd.notna(row.get("extracted_traincode")) else row.get("traincode"),
                axis=1
            )
            not_started["traincode"] = not_started["traincode_clean"]
            
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

        # ──── DATA VALIDATION: Fix corrupted train codes from API (same as Tab 2) ────
        live_trains = live_trains.copy()
        live_trains["extracted_traincode"] = live_trains["publicmessage"].apply(extract_traincode_from_message) if "publicmessage" in live_trains.columns else None
        live_trains["traincode_clean"] = live_trains.apply(
            lambda row: row.get("extracted_traincode") if pd.notna(row.get("extracted_traincode")) else row.get("traincode"),
            axis=1
        )
        # Validate traincode format
        live_trains = live_trains[live_trains["traincode_clean"].str.match(r'^[A-Z]{1,2}\d{2,4}$', na=False)].copy()
        live_trains["traincode"] = live_trains["traincode_clean"]

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
# TAB 4 — SNAPSHOT OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<p class="section-title">Snapshot Overview — Late Services</p>', unsafe_allow_html=True)
    st.markdown('<p class="caption-text">Quick, actionable view of delays by day, station, and route using detailed movement data. Default view shows averages across all days.</p>', unsafe_allow_html=True)

    # Use movements_df if available, otherwise fall back to train_logs
    if not movements_df.empty:
        tab4_df = movements_df.copy()
        # Use the pre-calculated delay columns from backfill script
        tab4_df["late"] = tab4_df["delay_arr_mins"].fillna(0).clip(lower=0)
        
        # Map columns to expected format
        tab4_df["stationfullname"] = tab4_df["locationfullname"]
        tab4_df["origin"] = tab4_df["trainorigin"]
        tab4_df["destination"] = tab4_df["traindestination"]
        tab4_df["timestamp"] = tab4_df["fetched_at"]
        
        # Use traindate_parsed for date filtering
        if "traindate_parsed" in tab4_df.columns:
            tab4_df["date"] = tab4_df["traindate_parsed"]
    else:
        tab4_df = df.copy()
        st.info("Using station logs data. Run backfill_movements.py to gather detailed movement data.")

    day_options, day_map = build_day_options(tab4_df["date"])
    c1, c2, c3 = st.columns([1, 1, 1.4], gap="large")
    with c1:
        selected_day = st.selectbox("🗓️ Day", day_options, index=0, key="mgr_day")
    if selected_day == "All days":
        base_df = tab4_df.copy()
    else:
        base_df = tab4_df[tab4_df["date"] == day_map[selected_day]].copy()

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
            
            # Calculate impact metrics
            summary["total_delay_impact"] = summary["avg_late"] * summary["late_days"]
            summary["consistency"] = summary["avg_late"] * (summary["late_days"] / summary["late_days"].max())
            
            # Sort differently based on view context
            if selected_day == "All days":
                # For all days: prioritize trains with highest cumulative delay impact
                # (avg delay × days late = total operational impact)
                summary = summary.sort_values(
                    ["total_delay_impact", "avg_late", "late_days"], 
                    ascending=[False, False, False]
                )
            else:
                # For selected day: prioritize by consistency and max delay
                summary = summary.sort_values(
                    ["avg_late", "consistency", "max_late"], 
                    ascending=[False, False, False]
                )

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
                    "last_seen",
                ]]

            display_df = display_df.rename(columns={
                "traincode": "Train",
                "origin": "Origin",
                "destination": "Destination",
                "stationfullname": "Station",
                "avg_late": "Avg late (min)",
                "max_late": "Max late (min)",
                "late_days": "Late days",
                "last_seen": "Last seen",
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        