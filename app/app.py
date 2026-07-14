"""Streamlit no-show outreach decision-support app."""
from __future__ import annotations
import json,sys
from datetime import date,time
from pathlib import Path
import joblib,pandas as pd,streamlit as st
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src")); from feature_engineering import MODEL_FEATURES,engineer_features
st.set_page_config(page_title="Medical Appointment No-Show Intelligence",page_icon="🏥",layout="wide")
@st.cache_resource
def load_models(): return joblib.load(ROOT/"models/baseline_logistic_regression.joblib"),joblib.load(ROOT/"models/advanced_xgboost.joblib"),json.loads((ROOT/"models/threshold_config.json").read_text())
baseline,advanced,thresholds=load_models(); metrics=json.loads((ROOT/"reports/benchmark_metrics.json").read_text()); page=st.sidebar.radio("Navigate",["Home","Project Overview","Prediction Interface","Model Performance","Visualizations","Business Insights","About"]); st.sidebar.warning("Decision-support demonstration only. Never deny or deprioritize care using this score.")
if page=="Home":
    st.title("Medical Appointment No-Show Intelligence"); c=st.columns(4); c[0].metric("Official appointments","110,527"); c[1].metric("Official columns","14"); c[2].metric("No-show prevalence","≈20.2%"); c[3].metric("Primary metric","PR-AUC"); st.info("Rerun on the official Kaggle CSV before quoting final performance.")
elif page=="Project Overview": st.header("Project Overview"); st.markdown("Supportive reminder and care-navigation prioritization using chronological validation. Patient and appointment identifiers are excluded from modeling.")
elif page=="Prediction Interface":
    st.header("Single-Appointment Outreach Score"); label=st.selectbox("Model",["XGBoost","Logistic Regression"]); model=advanced if label=="XGBoost" else baseline
    with st.form("score"):
        c1,c2,c3=st.columns(3)
        with c1: appt=st.date_input("Appointment date",date(2016,6,10)); wait=st.number_input("Waiting days",0,180,14); sched_time=st.time_input("Scheduling time",time(10,0)); age=st.number_input("Age",0,110,35)
        with c2: gender=st.selectbox("Gender code",["F","M"]); hood=st.selectbox("Neighbourhood",["JARDIM CAMBURI","MARIA ORTIZ","RESISTÊNCIA","JARDIM DA PENHA","CENTRO","OTHER"]); scholarship=st.selectbox("Scholarship",[0,1]); sms=st.selectbox("SMS received",[0,1])
        with c3: hypertension=st.selectbox("Hypertension",[0,1]); diabetes=st.selectbox("Diabetes",[0,1]); alcoholism=st.selectbox("Alcoholism",[0,1]); handicap=st.number_input("Handicap count",0,4,0)
        submitted=st.form_submit_button("Score appointment",use_container_width=True)
    if submitted:
        appointment=pd.Timestamp(appt,tz="UTC"); scheduled=appointment-pd.Timedelta(days=int(wait))+pd.Timedelta(hours=sched_time.hour,minutes=sched_time.minute); record={"gender":gender,"scheduled_datetime":scheduled,"appointment_date":appointment,"age":age,"neighbourhood":hood,"scholarship":scholarship,"hypertension":hypertension,"diabetes":diabetes,"alcoholism":alcoholism,"handicap":handicap,"sms_received":sms,"waiting_days":wait}; featured=engineer_features(pd.DataFrame([record])); probability=float(model.predict_proba(featured[MODEL_FEATURES])[:,1][0]); threshold=float(thresholds[label]); st.metric("Estimated no-show probability",f"{probability:.1%}"); st.metric("Outreach threshold",f"{threshold:.1%}"); st.warning("Suggested workflow: supportive reminder or assistance—not denial of care.") if probability>=threshold else st.success("Suggested workflow: standard reminder process.")
elif page=="Model Performance":
    st.header("Model Performance"); st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if k!="confusion_matrix"} for r in metrics["models"]]),hide_index=True,use_container_width=True); st.markdown(f"**Validation-selected demonstration model:** {metrics['recommended_model']}"); st.caption(metrics["benchmark_scope"])
    for name in ["model_comparison.svg","precision_recall_curve.svg","confusion_matrix.svg"]: st.image(str(ROOT/"visuals"/name),use_container_width=True)
elif page=="Visualizations":
    st.header("Exploratory Visualizations")
    for name in ["class_distribution.svg","age_distribution.svg","waiting_time_distribution.svg","no_show_by_waiting_bucket.svg","no_show_by_sms.svg","no_show_by_age_group.svg","correlation_heatmap.svg","feature_importance.svg"]: st.image(str(ROOT/"visuals"/name),caption=name.replace("_"," ").replace(".svg","").title(),use_container_width=True)
elif page=="Business Insights": st.header("Business Insights"); st.markdown("1. Use risk bands for supportive outreach.
2. Choose thresholds from real capacity and costs.
3. Treat SMS as observational, not causal.
4. Monitor subgroup performance and patient experience.
5. Never restrict access using a score.")
else: st.header("About"); st.markdown("Built by **Rachel Oyeyemi** as a recruiter-ready healthcare machine-learning portfolio project.")
