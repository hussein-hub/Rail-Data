import requests
import pandas as pd
import schedule
import time
from sqlalchemy import create_engine
from datetime import datetime, timezone
from io import StringIO

DB = create_engine("sqlite:///irish_rail.db")

STATIONS = ["CNLLY", "HSTON", "PERSE"]

URL = "http://api.irishrail.ie/realtime/realtime.asmx/getStationDataByCodeXML_WithNumMins"

def fetch_station(station):
    r = requests.get(URL, params={"StationCode": station, "NumMins": 60})
    xml_data = StringIO(r.text)             
    df = pd.read_xml(xml_data)
    if df is None or df.empty:
        return None
    df["timestamp"] = datetime.now(timezone.utc) 
    return df

def job():
    frames = []
    for s in STATIONS:
        df = fetch_station(s)
        if df is not None:
            frames.append(df)
    if frames:
        data = pd.concat(frames)
        data.to_sql("train_logs", DB, if_exists="append", index=False)
        print("Logged:", len(data))

schedule.every(1).minutes.do(job)

print("🚆 Collector started...")
while True:
    schedule.run_pending()
    time.sleep(30)

