"""Run the complete reproducible modeling pipeline."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path

def run(command): print("$"," ".join(command)); subprocess.run(command,check=True)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default=""); p.add_argument("--demo-rows",type=int,default=30000); a=p.parse_args(); root=Path(__file__).resolve().parent; py=sys.executable; raw=Path(a.input) if a.input else root/"data/sample_data/demo_appointments.csv"
    if not a.input: run([py,str(root/"src/generate_demo_data.py"),"--rows",str(a.demo_rows),"--output",str(raw)])
    run([py,str(root/"src/preprocess.py"),"--input",str(raw),"--output",str(root/"data/processed/clean_appointments.csv"),"--quality-output",str(root/"reports/data_quality.json")]); run([py,str(root/"src/feature_engineering.py"),"--input",str(root/"data/processed/clean_appointments.csv"),"--output",str(root/"data/processed/model_features.csv")]); run([py,str(root/"src/train_model.py"),"--input",str(root/"data/processed/model_features.csv")]); run([py,str(root/"src/evaluate_model.py")]); run([py,str(root/"src/generate_visuals.py"),"--input",str(root/"data/processed/model_features.csv")])
if __name__=="__main__": main()
