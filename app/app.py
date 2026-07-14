"""Streamlit application for appointment no-show decision support."""
from __future__ import annotations

import json
import sys
from datetime import date, time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_model import choose_threshold
from feature_engineering import MODEL_FEATURES, engineer_features
from generate_demo_data import generate_demo
from preprocess import preprocess_dataframe
from train_model import train_models

st.set_page_config(page_title="Medical Appointment No-Show Intelligence", page_icon="🏥", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load committed artifacts or train deterministic fallbacks on first launch."""
    baseline_path = ROOT / "models/baseline_logistic_regression.joblib"
    advanced_path = ROOT / "models/advanced_xgboost.joblib"
    threshold_path = ROOT / "models/threshold_config.json"
    if baseline_path.exists() and advanced_path.exists() and threshold_path.exists():
        return {
            "baseline": joblib.load(baseline_path),
            "advanced": joblib.load(advanced_path),
            "thresholds": json.loads(threshold_path.read_text()),
            "scope": "Committed deterministic demonstration artifacts",
        }

    raw = generate_demo(8_000)
    clean, _ = preprocess_dataframe(raw)
    featured = engineer_features(clean)
    baseline, advanced, _, validation, _, _ = train_models(featured)
    thresholds = {}
    for label, model in [("Logistic Regression", baseline), ("XGBoost", advanced)]:
        probability = model.predict_proba(validation[MODEL_FEATURES])[:, 1]
        thresholds[label] = choose_threshold(validation["no_show"], probability)[0]
    return {
        "baseline": baseline,
        "advanced": advanced,
        "thresholds": thresholds,
        "scope": "Cached deterministic fallback",
    }


@st.cache_data
def load_json(path: Path):
    """Load a JSON artifact when it exists."""
    return json.loads(path.read_text()) if path.exists() else {}


ARTIFACTS = load_artifacts()
METRICS = load_json(ROOT / "reports/benchmark_metrics.json")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Project Overview", "Prediction Interface", "Model Performance", "Visualizations", "Business Insights", "About"],
)
st.sidebar.warning("Decision-support demonstration only. Never deny care or penalize a patient using this score.")
st.sidebar.caption(ARTIFACTS["scope"])

if page == "Home":
    st.title("Medical Appointment No-Show Intelligence")
    columns = st.columns(4)
    columns[0].metric("Official appointments", "110,527")
    columns[1].metric("Official columns", "14")
    columns[2].metric("No-show prevalence", "≈20.2%")
    columns[3].metric("Primary metric", "PR-AUC")
    st.markdown(
        "A production-style classification portfolio project that converts scheduling and patient context into "
        "**outreach prioritization**, with transparent thresholds and responsible-use controls."
    )
    st.info("Download the official Kaggle CSV and rerun the pipeline before quoting final performance.")

elif page == "Project Overview":
    st.header("Project Overview")
    st.markdown(
        """
        **Business question:** Which appointments may benefit from reminders, transportation support, or human outreach?

        **Modeling strategy:** class-weighted Logistic Regression versus XGBoost, evaluated on future appointment dates rather than a random holdout.

        **Leakage controls:** patient and appointment identifiers are excluded. The score is a workflow input—not a reason to deny, delay, or deprioritize care.
        """
    )

elif page == "Prediction Interface":
    st.header("Single-Appointment Outreach Score")
    model_label = st.selectbox("Model", ["XGBoost", "Logistic Regression"])
    model = ARTIFACTS["advanced"] if model_label == "XGBoost" else ARTIFACTS["baseline"]
    with st.form("appointment"):
        c1, c2, c3 = st.columns(3)
        with c1:
            appointment_date = st.date_input("Appointment date", date(2016, 6, 10))
            waiting_days = st.number_input("Days between scheduling and appointment", min_value=0, max_value=180, value=14)
            schedule_time = st.time_input("Scheduling time", time(10, 0))
            age = st.number_input("Age", min_value=0, max_value=110, value=35)
        with c2:
            gender = st.selectbox("Gender code", ["F", "M"])
            neighbourhood = st.selectbox(
                "Appointment neighbourhood",
                ["JARDIM CAMBURI", "MARIA ORTIZ", "RESISTÊNCIA", "JARDIM DA PENHA", "CENTRO", "OTHER"],
            )
            scholarship = st.selectbox("Scholarship", [0, 1])
            sms = st.selectbox("SMS received", [0, 1])
        with c3:
            hypertension = st.selectbox("Hypertension", [0, 1])
            diabetes = st.selectbox("Diabetes", [0, 1])
            alcoholism = st.selectbox("Alcoholism", [0, 1])
            handicap = st.number_input("Handicap count", min_value=0, max_value=4, value=0)
        submitted = st.form_submit_button("Score appointment", use_container_width=True)

    if submitted:
        appointment_timestamp = pd.Timestamp(appointment_date, tz="UTC")
        scheduled_timestamp = appointment_timestamp - pd.Timedelta(days=int(waiting_days)) + pd.Timedelta(
            hours=schedule_time.hour, minutes=schedule_time.minute
        )
        record = {
            "gender": gender,
            "scheduled_datetime": scheduled_timestamp,
            "appointment_date": appointment_timestamp,
            "age": int(age),
            "neighbourhood": neighbourhood,
            "scholarship": scholarship,
            "hypertension": hypertension,
            "diabetes": diabetes,
            "alcoholism": alcoholism,
            "handicap": handicap,
            "sms_received": sms,
            "waiting_days": int(waiting_days),
        }
        featured = engineer_features(pd.DataFrame([record]))
        probability = float(model.predict_proba(featured[MODEL_FEATURES])[:, 1][0])
        threshold = float(ARTIFACTS["thresholds"][model_label])
        st.metric("Estimated no-show probability", f"{probability:.1%}")
        st.metric("Demonstration outreach threshold", f"{threshold:.1%}")
        if probability >= threshold:
            st.warning("Suggested workflow: considerate reminder or support outreach—not denial of care.")
        else:
            st.success("Suggested workflow: standard reminder process.")

elif page == "Model Performance":
    st.header("Model Performance")
    if METRICS:
        display = pd.DataFrame(
            [{k: v for k, v in row.items() if k != "confusion_matrix"} for row in METRICS.get("models", [])]
        )
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.markdown(f"**Validation-selected demonstration model:** {METRICS.get('recommended_model')}")
        st.caption(METRICS.get("benchmark_scope", ""))
    for name in ["model_comparison.svg", "precision_recall_curve.svg", "confusion_matrix.svg"]:
        path = ROOT / "visuals" / name
        if path.exists():
            st.image(str(path), use_container_width=True)

elif page == "Visualizations":
    st.header("Exploratory and Model Visualizations")
    for name in [
        "class_distribution.svg",
        "age_distribution.svg",
        "waiting_time_distribution.svg",
        "no_show_by_waiting_bucket.svg",
        "no_show_by_sms.svg",
        "no_show_by_age_group.svg",
        "correlation_heatmap.svg",
        "feature_importance.svg",
    ]:
        path = ROOT / "visuals" / name
        if path.exists():
            st.image(str(path), caption=name.replace("_", " ").replace(".svg", "").title(), use_container_width=True)

elif page == "Business Insights":
    st.header("Business and Responsible-AI Insights")
    st.markdown(
        """
        1. Longer scheduling lead times are useful risk signals but should trigger support—not blame.
        2. Model thresholds must reflect outreach capacity and intervention cost.
        3. SMS status is observational; it does not prove reminders cause or prevent attendance.
        4. Monitor recall, precision, outreach volume, and outcomes by age, gender, and neighbourhood.
        5. Never use risk scores to deny appointments, reduce access, or penalize patients.
        """
    )

else:
    st.header("About")
    st.markdown(
        "Built by **Rachel Oyeyemi** as a recruiter-ready Data Analytics & AI portfolio project. "
        "It demonstrates healthcare data engineering, imbalanced classification, threshold economics, testing, Streamlit, executive communication, and responsible AI."
    )
