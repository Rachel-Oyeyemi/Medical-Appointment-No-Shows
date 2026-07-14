"""Generate EDA and model visualizations."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import joblib,matplotlib.pyplot as plt,numpy as np,pandas as pd
from sklearn.metrics import precision_recall_curve
from feature_engineering import MODEL_FEATURES,TARGET

def save(path):
    path.parent.mkdir(parents=True,exist_ok=True); plt.tight_layout(); plt.savefig(path.with_suffix(".png"),dpi=150,bbox_inches="tight"); plt.savefig(path.with_suffix(".svg"),bbox_inches="tight"); plt.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/processed/model_features.csv"); a=p.parse_args(); root=Path(__file__).resolve().parents[1]; frame=pd.read_csv(a.input,parse_dates=["scheduled_datetime","appointment_date"]); v=root/"visuals"
    counts=frame[TARGET].value_counts().reindex([0,1]); plt.figure(figsize=(7,4.5)); plt.bar(["Attended","No-show"],counts.values); plt.title("Appointment Outcome Distribution"); plt.ylabel("Appointments"); save(v/"class_distribution")
    plt.figure(figsize=(7,4.5)); plt.hist(frame.age,bins=30); plt.title("Age Distribution"); plt.xlabel("Age"); save(v/"age_distribution")
    plt.figure(figsize=(7,4.5)); plt.hist(frame.waiting_days.clip(upper=60),bins=30); plt.title("Waiting Time Distribution"); plt.xlabel("Days"); save(v/"waiting_time_distribution")
    rates=frame.groupby("waiting_bucket",observed=False)[TARGET].mean(); plt.figure(figsize=(8,4.5)); plt.bar(rates.index.astype(str),rates.values); plt.xticks(rotation=35); plt.title("No-show Rate by Waiting-Time Bucket"); save(v/"no_show_by_waiting_bucket")
    rates=frame.groupby("sms_received")[TARGET].mean().reindex([0,1]); plt.figure(figsize=(6.5,4.5)); plt.bar(["No SMS","SMS Received"],rates.values); plt.title("Observed No-show Rate by SMS Status"); save(v/"no_show_by_sms")
    rates=frame.groupby("age_group",observed=False)[TARGET].mean(); plt.figure(figsize=(8,4.5)); plt.bar(rates.index.astype(str),rates.values); plt.xticks(rotation=30); plt.title("No-show Rate by Age Group"); save(v/"no_show_by_age_group")
    corr=frame[["age","waiting_days","schedule_hour","chronic_count","sms_received",TARGET]].corr(); plt.figure(figsize=(7,5.5)); plt.imshow(corr,cmap="coolwarm",vmin=-1,vmax=1); plt.colorbar(); plt.xticks(range(len(corr)),corr.columns,rotation=45,ha="right"); plt.yticks(range(len(corr)),corr.columns); plt.title("Numeric Feature Correlation"); save(v/"correlation_heatmap")
    metrics=json.loads((root/"reports/benchmark_metrics.json").read_text()); test=[r for r in metrics["models"] if r["split"]=="test"]; labels=[r["model"] for r in test]; x=np.arange(2); plt.figure(figsize=(8,4.8)); plt.bar(x-.18,[r["pr_auc"] for r in test],.36,label="PR-AUC"); plt.bar(x+.18,[r["recall"] for r in test],.36,label="Recall"); plt.xticks(x,labels); plt.legend(); plt.title("Model Comparison on Chronological Test Set"); save(v/"model_comparison")
    parts=joblib.load(root/"models/evaluation_partitions.joblib"); test_frame=parts["test"]; plt.figure(figsize=(7,5))
    for name,file in [("Logistic Regression","baseline_logistic_regression.joblib"),("XGBoost","advanced_xgboost.joblib")]:
        model=joblib.load(root/"models"/file); prob=model.predict_proba(test_frame[MODEL_FEATURES])[:,1]; precision,recall,_=precision_recall_curve(test_frame[TARGET],prob); plt.plot(recall,precision,label=name)
    plt.axhline(test_frame[TARGET].mean(),linestyle="--",label="Prevalence"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curves"); plt.legend(); save(v/"precision_recall_curve")
    recommended=metrics["recommended_model"]; row=next(r for r in test if r["model"]==recommended); matrix=np.array(row["confusion_matrix"]); plt.figure(figsize=(5.5,5)); plt.imshow(matrix,cmap="Blues"); plt.xticks([0,1],["Attend","No-show"]); plt.yticks([0,1],["Attend","No-show"]); plt.xlabel("Predicted"); plt.ylabel("Actual");
    for i in range(2):
        for j in range(2): plt.text(j,i,f"{matrix[i,j]:,}",ha="center",va="center")
    plt.title(f"Confusion Matrix — {recommended}"); save(v/"confusion_matrix")
    advanced=joblib.load(root/"models/advanced_xgboost.joblib"); names=advanced.named_steps["preprocessor"].get_feature_names_out(); importance=advanced.named_steps["model"].feature_importances_; top=np.argsort(importance)[-15:]; plt.figure(figsize=(8,6)); plt.barh([names[i].replace("cat__","").replace("num__","").replace("bin__","") for i in top],importance[top]); plt.title("XGBoost Feature Importance"); save(v/"feature_importance")
if __name__=="__main__": main()
