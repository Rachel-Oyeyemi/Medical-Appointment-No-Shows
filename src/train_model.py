"""Train Logistic Regression and XGBoost no-show classifiers."""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from feature_engineering import BINARY_FEATURES,CATEGORICAL_FEATURES,MODEL_FEATURES,NUMERIC_FEATURES,TARGET,chronological_split
from utils import configure_logging,save_json
LOGGER=logging.getLogger(__name__)

def preprocessor(scale=True):
    steps=[("impute",SimpleImputer(strategy="median"))]+([("scale",StandardScaler())] if scale else [])
    return ColumnTransformer([("num",Pipeline(steps),NUMERIC_FEATURES),("bin",SimpleImputer(strategy="most_frequent"),BINARY_FEATURES),("cat",OneHotEncoder(handle_unknown="ignore",min_frequency=5),CATEGORICAL_FEATURES)])

def build_baseline(): return Pipeline([("preprocessor",preprocessor(True)),("model",LogisticRegression(max_iter=2000,class_weight="balanced",C=.8,solver="liblinear",random_state=42))])
def build_advanced(weight): return Pipeline([("preprocessor",preprocessor(False)),("model",XGBClassifier(n_estimators=500,max_depth=6,learning_rate=.04,min_child_weight=4,subsample=.85,colsample_bytree=.85,reg_alpha=.05,reg_lambda=3,gamma=.05,scale_pos_weight=weight,objective="binary:logistic",eval_metric="aucpr",tree_method="hist",random_state=42,n_jobs=4))])

def train_models(frame):
    train,validation,test=chronological_split(frame); X=train[MODEL_FEATURES]; y=train[TARGET]; positives=int(y.sum()); weight=(len(y)-positives)/positives
    return build_baseline().fit(X,y),build_advanced(weight).fit(X,y),train,validation,test,weight

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",default="data/processed/model_features.csv"); p.add_argument("--models-dir",default="models"); a=p.parse_args(); configure_logging(); frame=pd.read_csv(a.input,parse_dates=["scheduled_datetime","appointment_date"]); baseline,advanced,train,val,test,weight=train_models(frame); out=Path(a.models_dir); out.mkdir(parents=True,exist_ok=True); joblib.dump(baseline,out/"baseline_logistic_regression.joblib",compress=3); joblib.dump(advanced,out/"advanced_xgboost.joblib",compress=3); joblib.dump({"validation":val,"test":test},out/"evaluation_partitions.joblib",compress=3); save_json({"features":MODEL_FEATURES,"target":TARGET,"split_strategy":"chronological 70/15/15","scale_pos_weight":weight,"random_seed":42,"training_rows":len(train),"validation_rows":len(val),"test_rows":len(test),"identifier_policy":"PatientId and AppointmentID excluded"},out/"model_metadata.json"); LOGGER.info("Saved models")
if __name__=="__main__": main()
