"""Evaluate models and select transparent outreach thresholds."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score,average_precision_score,balanced_accuracy_score,confusion_matrix,f1_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score
from feature_engineering import MODEL_FEATURES,TARGET
from utils import save_json

def choose_threshold(y,p,fn_cost=5,fp_cost=1):
    rows=[]
    for threshold in np.linspace(.05,.90,172):
        pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); rows.append({"threshold":float(threshold),"cost":float(fn_cost*fn+fp_cost*fp),"true_negatives":int(tn),"false_positives":int(fp),"false_negatives":int(fn),"true_positives":int(tp),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0))})
    best=min(rows,key=lambda x:(x["cost"],-x["recall"],-x["precision"])); return best["threshold"],rows

def evaluate(model,frame,threshold):
    y=frame[TARGET].to_numpy(); p=model.predict_proba(frame[MODEL_FEATURES])[:,1]; pred=(p>=threshold).astype(int); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); return {"threshold":float(threshold),"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"roc_auc":float(roc_auc_score(y,p)),"pr_auc":float(average_precision_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"matthews_correlation":float(matthews_corrcoef(y,pred)),"specificity":float(tn/(tn+fp)),"false_positive_rate":float(fp/(tn+fp)),"predicted_outreach_rate":float(pred.mean()),"true_negatives":int(tn),"false_positives":int(fp),"false_negatives":int(fn),"true_positives":int(tp),"confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]]},p

def main():
    p=argparse.ArgumentParser(); p.add_argument("--models-dir",default="models"); p.add_argument("--reports-dir",default="reports"); a=p.parse_args(); md=Path(a.models_dir); rd=Path(a.reports_dir); rd.mkdir(parents=True,exist_ok=True); baseline=joblib.load(md/"baseline_logistic_regression.joblib"); advanced=joblib.load(md/"advanced_xgboost.joblib"); parts=joblib.load(md/"evaluation_partitions.joblib"); val,test=parts["validation"],parts["test"]; results=[]; thresholds={}; tables=[]
    for name,model in [("Logistic Regression",baseline),("XGBoost",advanced)]:
        threshold,rows=choose_threshold(val[TARGET],model.predict_proba(val[MODEL_FEATURES])[:,1]); thresholds[name]=threshold; tables += [{"model":name,**r} for r in rows]
        results += [{"model":name,"split":"validation",**evaluate(model,val,threshold)[0]},{"model":name,"split":"test",**evaluate(model,test,threshold)[0]}]
    recommended=max([r for r in results if r["split"]=="validation"],key=lambda r:r["pr_auc"])["model"]
    save_json({"benchmark_scope":"Deterministic synthetic appointment data; rerun on the official Kaggle CSV for final portfolio metrics.","primary_metric":"PR-AUC","models":results,"recommended_model":recommended,"threshold_cost_assumptions":{"missed_no_show":5,"unnecessary_outreach":1}},rd/"benchmark_metrics.json"); save_json(thresholds,md/"threshold_config.json"); pd.DataFrame(tables).to_csv(rd/"threshold_diagnostics.csv",index=False); print(recommended)
if __name__=="__main__": main()
