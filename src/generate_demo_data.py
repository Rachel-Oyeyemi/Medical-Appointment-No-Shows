"""Generate deterministic non-identifying appointment data."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

NEIGHBOURHOODS = ["JARDIM CAMBURI","MARIA ORTIZ","RESISTÊNCIA","JARDIM DA PENHA","ITARARÉ","CENTRO","TABUAZEIRO","SANTA MARTHA","BONFIM","JESUS DE NAZARETH","SANTO ANTÔNIO","SÃO PEDRO","CARATOÍRA","ANDORINHAS","NOVA PALESTINA","DA PENHA","ROMÃO","GOIABEIRAS","ILHA DO PRÍNCIPE","MARUÍPE","BELA VISTA","FORTE SÃO JOÃO","SÃO CRISTÓVÃO","CONQUISTA","REDENÇÃO"]

def generate_demo(n_rows: int = 30000, seed: int = 42) -> pd.DataFrame:
    rng=np.random.default_rng(seed)
    patient_id=rng.choice(np.arange(10_000_000,10_012_000),n_rows)
    appointment_id=np.arange(5_000_000,5_000_000+n_rows)
    appointment_day=pd.Timestamp("2016-04-29")+pd.to_timedelta(rng.integers(0,42,n_rows),unit="D")
    waiting=np.clip(np.where(rng.random(n_rows)<.42,0,rng.gamma(2.2,8,n_rows).astype(int)),0,90)
    hour=np.clip(np.round(rng.normal(10.5,3,n_rows)).astype(int),6,19)
    scheduled=appointment_day-pd.to_timedelta(waiting,unit="D")+pd.to_timedelta(hour,unit="h")+pd.to_timedelta(rng.integers(0,60,n_rows),unit="m")
    gender=rng.choice(["F","M"],p=[.65,.35],size=n_rows)
    age=np.clip(np.round(rng.gamma(2.3,17,n_rows)).astype(int),0,102)
    hood=rng.choice(NEIGHBOURHOODS,n_rows)
    scholarship=rng.binomial(1,np.clip(.06+.07*((age>=18)&(age<=50)),0,1))
    hypertension=rng.binomial(1,1/(1+np.exp(-(-6.2+.09*age))))
    diabetes=rng.binomial(1,1/(1+np.exp(-(-7.2+.085*age))))
    alcoholism=rng.binomial(1,np.clip(.015+.04*((age>=25)&(age<=65)),0,.15))
    handicap=rng.choice([0,1,2],p=[.978,.020,.002],size=n_rows)
    sms=rng.binomial(1,np.clip(.06+.018*waiting+.10*(waiting>=7),.04,.82))
    effects={name:value for name,value in zip(NEIGHBOURHOODS,rng.normal(0,.38,len(NEIGHBOURHOODS)))}
    hood_effect=np.array([effects[x] for x in hood]); seg_a=np.isin(hood,NEIGHBOURHOODS[:6]); seg_b=np.isin(hood,NEIGHBOURHOODS[6:12])
    dow=appointment_day.dayofweek.to_numpy(); young=(age>=15)&(age<=35); older=age>=60
    logit=(-2.20+.030*waiting+.55*(waiting>=21)+.35*(waiting>=45)+.38*young-.35*older+.23*scholarship+.43*alcoholism+.20*(hour>=16)+.18*(dow==0)+hood_effect-.52*(waiting==0)+.55*((waiting>=14)&(sms==0))-.30*((waiting>=7)&(sms==1))+.30*((age<10)&(waiting>=30))+.95*(seg_a&young)-.65*(seg_a&older)+.75*(seg_b&(waiting>=15))+.60*(((dow==0)|(dow==4))&(waiting>=8)&(waiting<=30))+.55*((hour>=15)&(waiting>=21))-.18*hypertension-.12*diabetes+rng.normal(0,.72,n_rows))
    no_show=np.where(rng.random(n_rows)<1/(1+np.exp(-logit)),"Yes","No")
    return pd.DataFrame({"PatientId":patient_id,"AppointmentID":appointment_id,"Gender":gender,"ScheduledDay":scheduled.strftime("%Y-%m-%dT%H:%M:%SZ"),"AppointmentDay":appointment_day.strftime("%Y-%m-%dT00:00:00Z"),"Age":age,"Neighbourhood":hood,"Scholarship":scholarship,"Hipertension":hypertension,"Diabetes":diabetes,"Alcoholism":alcoholism,"Handcap":handicap,"SMS_received":sms,"No-show":no_show})

def main():
    p=argparse.ArgumentParser(); p.add_argument("--rows",type=int,default=30000); p.add_argument("--output",default="data/sample_data/demo_appointments.csv"); a=p.parse_args()
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); generate_demo(a.rows).to_csv(out,index=False)
    print(f"Saved {a.rows:,} synthetic appointments to {out}")
if __name__=="__main__": main()
