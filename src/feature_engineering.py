"""Create leakage-aware features available before the appointment."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
TARGET="no_show"; NUMERIC_FEATURES=["age","waiting_days","schedule_hour","chronic_count","handicap"]; BINARY_FEATURES=["scholarship","hypertension","diabetes","alcoholism","sms_received","same_day"]; CATEGORICAL_FEATURES=["gender","neighbourhood","appointment_dow","schedule_dow","appointment_month","age_group","waiting_bucket"]; MODEL_FEATURES=NUMERIC_FEATURES+BINARY_FEATURES+CATEGORICAL_FEATURES

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    data=frame.copy(); data["scheduled_datetime"]=pd.to_datetime(data.scheduled_datetime,utc=True); data["appointment_date"]=pd.to_datetime(data.appointment_date,utc=True)
    if "waiting_days" not in data: data["waiting_days"]=(data.appointment_date.dt.tz_localize(None)-data.scheduled_datetime.dt.tz_convert(None).dt.normalize()).dt.days
    data["schedule_hour"]=data.scheduled_datetime.dt.hour.astype("int8"); data["schedule_dow"]=data.scheduled_datetime.dt.day_name().str[:3]; data["appointment_dow"]=data.appointment_date.dt.day_name().str[:3]; data["appointment_month"]=data.appointment_date.dt.month.astype(str); data["same_day"]=(data.waiting_days==0).astype("int8"); data["chronic_count"]=data[["hypertension","diabetes","alcoholism"]].sum(axis=1).astype("int8")
    data["age_group"]=pd.cut(data.age,[-1,5,12,17,29,44,59,74,110],labels=["0-5","6-12","13-17","18-29","30-44","45-59","60-74","75+"]).astype("string")
    data["waiting_bucket"]=pd.cut(data.waiting_days,[-1,0,3,7,14,30,60,np.inf],labels=["same_day","1-3","4-7","8-14","15-30","31-60","61+"]).astype("string")
    for col in CATEGORICAL_FEATURES: data[col]=data[col].fillna("Unknown").astype(str)
    return data

def chronological_split(frame,validation_fraction=.15,test_fraction=.15):
    ordered=frame.sort_values("appointment_date").reset_index(drop=True); n=len(ordered); train_end=int(n*(1-validation_fraction-test_fraction)); val_end=int(n*(1-test_fraction)); return ordered.iloc[:train_end].copy(),ordered.iloc[train_end:val_end].copy(),ordered.iloc[val_end:].copy()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/processed/clean_appointments.csv"); p.add_argument("--output",default="data/processed/model_features.csv"); a=p.parse_args(); featured=engineer_features(pd.read_csv(a.input,parse_dates=["scheduled_datetime","appointment_date"])); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); featured.to_csv(out,index=False)
if __name__=="__main__": main()
