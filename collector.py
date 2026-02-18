import requests
import pandas as pd
import schedule
import time
from sqlalchemy import create_engine
from datetime import datetime, timezone
from io import StringIO
import re
import os
from dotenv import load_dotenv

load_dotenv()

DB = create_engine(os.getenv("DATABASE_URL"))

STATIONS = os.getenv("STATIONS").split(",")

URL = os.getenv("URL")
FLEET_URL = os.getenv("FLEET_URL")
FLEET_TYPES = dict(item.split(":") for item in os.getenv("FLEET_TYPES").split(","))


def fetch_station(station):
    r = requests.get(URL, params={"StationCode": station, "NumMins": 60})
    xml_data = StringIO(r.text)
    df = pd.read_xml(xml_data)
    if df is None or df.empty:
        return None
    df["timestamp"] = datetime.now(timezone.utc)
    return df


def parse_delay_from_message(msg):
    """Extract delay minutes from PublicMessage string, same logic as app.py."""
    if not msg or not isinstance(msg, str):
        return None
    m = re.search(r"\((\d+) mins? late\)", msg)
    if m:
        return int(m.group(1))
    if "on time" in msg.lower():
        return 0
    return None


def fetch_fleet(fleet_name, fleet_code):
    """Fetch all running trains of a given fleet type and return a summary row."""
    try:
        r = requests.get(FLEET_URL, params={"TrainType": fleet_code}, timeout=10)
        df = pd.read_xml(StringIO(r.text))
        if df is None or df.empty:
            return None

        df.columns = df.columns.str.strip().str.lower()

        # Keep only running trains (TrainStatus == "R")
        if "trainstatus" in df.columns:
            df = df[df["trainstatus"] == "R"].copy()

        if df.empty:
            return None

        # Parse delay from PublicMessage
        if "publicmessage" in df.columns:
            df["delay_mins"] = df["publicmessage"].apply(parse_delay_from_message)
        else:
            df["delay_mins"] = None

        total        = len(df)
        known        = int(df["delay_mins"].notna().sum())
        on_time      = int((df["delay_mins"] == 0).sum())
        slightly     = int(((df["delay_mins"] > 0) & (df["delay_mins"] <= 5)).sum())
        significantly = int((df["delay_mins"] > 5).sum())
        avg_delay    = round(float(df["delay_mins"].mean()), 2) if known > 0 else 0.0
        punctuality  = round(on_time / known * 100, 2) if known > 0 else 0.0

        return {
            "fleet":           fleet_name,
            "total":           total,
            "known":           known,
            "on_time":         on_time,
            "slightly_late":   slightly,
            "significantly":   significantly,
            "avg_delay":       avg_delay,
            "punctuality":     punctuality,
            "timestamp":       datetime.now(timezone.utc),
        }
    except Exception as e:
        print(f"Fleet fetch error ({fleet_name}): {e}")
        return None


def job():
    # ── Station logs (existing) ──────────────────────────────────────
    frames = []
    for s in STATIONS:
        df = fetch_station(s)
        if df is not None:
            frames.append(df)
    if frames:
        data = pd.concat(frames)
        data.to_sql("train_logs", DB, if_exists="append", index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Station logs: {len(data)} rows")

    # ── Fleet snapshot logs (new) ────────────────────────────────────
    fleet_rows = []
    for name, code in FLEET_TYPES.items():
        row = fetch_fleet(name, code)
        if row is not None:
            fleet_rows.append(row)

    if fleet_rows:
        fleet_df = pd.DataFrame(fleet_rows)
        fleet_df.to_sql("fleet_logs", DB, if_exists="append", index=False)
        summary = ", ".join(f"{r['fleet']}={r['total']}" for r in fleet_rows)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fleet logs: {len(fleet_df)} rows ({summary})")


schedule.every(1).minutes.do(job)

print("Collector started...")
job()   # run immediately on startup so DB has data right away
while True:
    schedule.run_pending()
    time.sleep(30)
