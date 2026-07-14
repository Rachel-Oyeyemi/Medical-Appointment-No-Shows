# Medical Appointment No Shows

[![CI](https://github.com/Rachel-Oyeyemi/Medical-Appointment-No-Shows/actions/workflows/ci.yml/badge.svg)](https://github.com/Rachel-Oyeyemi/Medical-Appointment-No-Shows/actions)

A recruiter-ready machine-learning portfolio project for appointment no-show outreach prioritization using data engineering, chronological validation, Logistic Regression, XGBoost, cost-sensitive thresholds, Streamlit, tests, executive reporting, and responsible-AI controls.

> Committed metrics use deterministic synthetic appointments so the repository runs immediately. Download the official Kaggle CSV and rerun before quoting final performance.

## Dataset
Kaggle: `joniarroba/noshowappointments`. Widely reproduced profile: 110,527 Brazilian appointments, 14 columns, target `No-show`, and approximately 20.2% no-shows. Published reproductions report no missing values and no exact duplicate rows. Quality checks include an invalid negative age, non-binary `Handcap` values, date parsing, and possible negative waiting times.

## Methodology
1. Download and validate the CSV.
2. Standardize names, dates, target, and data types.
3. Engineer waiting time, calendar, age-group, waiting-bucket, and chronic-condition features.
4. Exclude PatientId and AppointmentID.
5. Split chronologically 70/15/15.
6. Train class-weighted Logistic Regression and XGBoost.
7. Select validation thresholds from transparent outreach costs.
8. Evaluate PR-AUC, recall, precision, F1, ROC-AUC, balanced accuracy, MCC, specificity, and outreach volume.

## Demonstration Results
| Model | Test PR-AUC | Precision | Recall | F1 | ROC-AUC | Outreach Rate |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.468 | 0.294 | 0.861 | 0.438 | 0.765 | 58.9% |
| XGBoost | 0.454 | 0.315 | 0.804 | 0.453 | 0.758 | 51.3% |

XGBoost is the validation-selected demo model by a very small margin. Mixed test results are retained honestly.

## Run
```bash
git clone https://github.com/Rachel-Oyeyemi/Medical-Appointment-No-Shows.git
cd Medical-Appointment-No-Shows
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app/app.py
pytest -q
```

Official data: `python src/download_data.py --source auto` then `python run_pipeline.py --input data/raw/appointments.csv`.

## Structure
`app/`, `data/`, `docs/`, `models/`, `notebooks/`, `presentation/`, `reports/`, `src/`, `tests/`, and `visuals/`, plus the requested charter, EDA, evaluation, recommendations, and career materials.

## Responsible Use
Never deny, delay, or deprioritize care using a model score. Production requires institutional validation, privacy and security review, calibration, subgroup testing, intervention experiments, monitoring, and human override.
