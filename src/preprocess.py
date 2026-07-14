"""Validate and clean raw medical appointment records."""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import pandas as pd
from utils import configure_logging, save_json
LOGGER=logging.getLogger(__name__)
REQUIRED={"PatientId","AppointmentID","Gender","ScheduledDay","AppointmentDay","Age","Neighbourhood","Scholarship","Hipertension","Diabetes","Alcoholism","Handcap","SMS_received","No-show"}
RENAME={"PatientId":"patient_id","AppointmentID":"appointment_id","Gender":"gender","ScheduledDay":"scheduled_datetime","AppointmentDay":"appointment_date","Age":"age","Neighbourhood":"neighbourhood","Scholarship":"scholarship","Hipertension":"hypertension","Diabetes":"diabetes","Alcoholism":"alcoholism","Handcap":"handicap","SMS_received":"sms_received","No-show":"no_show"}

def preprocess_dataframe(frame: pd.DataFrame):
    missing=REQUIRED.difference(frame.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    data=frame.copy().rename(columns=RENAME); input_rows=len(data); missing_cells=int(data.isna().sum().sum()); duplicates=int(data.duplicated().sum())
    data["scheduled_datetime"]=pd.to_datetime(data["scheduled_datetime"],utc=True,errors="coerce"); data["appointment_date"]=pd.to_datetime(data["appointment_date"],utc=True,errors="coerce").dt.normalize()
    data["gender"]=data.gender.astype("string").str.upper().str.strip(); data["neighbourhood"]=data.neighbourhood.astype("string").str.strip().str.upper(); data["no_show"]=data.no_show.astype("string").str.lower().str.strip().map({"yes":1,"no":0})
    for col in ["patient_id","appointment_id","age","scholarship","hypertension","diabetes","alcoholism","handicap","sms_received"]: data[col]=pd.to_numeric(data[col],errors="coerce")
    data=data.drop_duplicates(subset=["appointment_id"]).dropna(subset=["scheduled_datetime","appointment_date","age","gender","neighbourhood","no_show"]); data=data[data.gender.isin(["F","M"])]
    bad_age=int(((data.age<0)|(data.age>110)).sum()); data=data[data.age.between(0,110)]
    data["waiting_days"]=(data.appointment_date.dt.tz_localize(None)-data.scheduled_datetime.dt.tz_convert(None).dt.normalize()).dt.days; bad_wait=int((data.waiting_days<0).sum()); data=data[data.waiting_days>=0]
    for col in ["scholarship","hypertension","diabetes","alcoholism","sms_received"]: data[col]=data[col].fillna(0).clip(0,1).astype("int8")
    data["handicap"]=data.handicap.fillna(0).clip(lower=0).astype("int8"); data["age"]=data.age.astype("int16"); data["patient_id"]=data.patient_id.astype("int64"); data["appointment_id"]=data.appointment_id.astype("int64"); data["no_show"]=data.no_show.astype("int8")
    data=data.sort_values(["appointment_date","scheduled_datetime","appointment_id"]).reset_index(drop=True)
    quality={"input_rows":input_rows,"output_rows":len(data),"rows_removed":input_rows-len(data),"input_columns":frame.shape[1],"missing_cells_before":missing_cells,"exact_duplicate_rows_before":duplicates,"invalid_age_rows_removed":bad_age,"negative_wait_rows_removed":bad_wait,"target_positive_rows":int(data.no_show.sum()),"target_positive_rate":float(data.no_show.mean()),"date_min":str(data.appointment_date.min().date()),"date_max":str(data.appointment_date.max().date()),"unique_patients":int(data.patient_id.nunique()),"unique_neighbourhoods":int(data.neighbourhood.nunique())}
    return data,quality

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/raw/appointments.csv"); p.add_argument("--output",default="data/processed/clean_appointments.csv"); p.add_argument("--quality-output",default="reports/data_quality.json"); a=p.parse_args(); configure_logging()
    clean,quality=preprocess_dataframe(pd.read_csv(a.input,low_memory=False)); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); clean.to_csv(out,index=False); save_json(quality,a.quality_output); LOGGER.info("Saved %s rows",len(clean))
if __name__=="__main__": main()
