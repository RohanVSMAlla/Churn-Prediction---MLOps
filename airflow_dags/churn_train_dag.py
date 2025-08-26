from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn
import os

# Set MLflow tracking URI to local folder
mlflow.set_tracking_uri("file:///C:/Users/Rohan/OneDrive - Drexel University/Desktop/ChurnPrediction/mlruns")

def train_churn_model():
    # Load data
    import os
    df = pd.read_csv(os.path.join(os.environ["AIRFLOW_HOME"], "data", "telco_churn.csv"))



    # Preprocessing (same as notebook)
    df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)

    # Save model and columns
    joblib.dump(model, "C:/Users/Rohan/OneDrive - Drexel University/Desktop/ChurnPrediction/src/churn_xgb_model.pkl")
    joblib.dump(X_train.columns.tolist(), "C:/Users/Rohan/OneDrive - Drexel University/Desktop/ChurnPrediction/src/feature_columns.pkl")

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_metric("auc_roc", auc)
        mlflow.sklearn.log_model(model, "model")

# Define DAG
with DAG("churn_training_pipeline",
         start_date=datetime(2023, 1, 1),
         schedule_interval="@daily",
         catchup=False) as dag:

    task_train_model = PythonOperator(
        task_id="train_churn_model",
        python_callable=train_churn_model
    )
