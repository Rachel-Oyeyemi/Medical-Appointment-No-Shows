# Project Charter — Medical Appointment No Shows

## Business Problem
Estimate appointment-level no-show risk early enough to prioritize considerate reminders, transportation support, rescheduling, or human care navigation.

## Objectives
Build a leakage-aware classification pipeline; compare Logistic Regression and XGBoost; select transparent outreach thresholds; deliver Streamlit, reports, tests, and presentation materials.

## Stakeholders
Clinic operations, schedulers, care-navigation teams, data teams, privacy/compliance leaders, and patients.

## Success Metrics
PR-AUC, recall, precision, F1, ROC-AUC, balanced accuracy, MCC, outreach volume, appointments recovered, and subgroup equity.

## Architecture
`Kaggle → validation → cleaning → features → chronological partitions → models → threshold policy → evaluation → Streamlit`

## Responsible Use
Never deny, delay, or deprioritize care based on a model score. Require privacy review, subgroup testing, human oversight, and intervention evaluation.
