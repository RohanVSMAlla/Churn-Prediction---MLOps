# End-to-End Customer Churn Prediction (MLOps Project)

This repository contains a **production-grade MLOps pipeline** for customer churn prediction, designed to demonstrate **FAANG/Tesla-level machine learning engineering practices**.

---

## Project Overview

The goal of this project is to **predict customer churn** for a telecom dataset and deploy the system in a way that is:

- **Scalable** → automated with Apache Airflow  
- **Reproducible** → experiment tracking with MLflow  
- **Explainable** → SHAP values for transparency  
- **Interactive** → business-facing Streamlit dashboard  
- **Portable** → containerized with Docker, ready for GCP/AWS deployment  

---

##  Architecture

```text
            ┌───────────────┐
            │   Dataset      │
            │ (Telco Churn)  │
            └───────┬───────┘
                    │
                    ▼
          ┌───────────────────┐
          │ Apache Airflow     │
          │  - Train DAG       │
          │  - Prediction DAG  │
          └───────┬───────────┘
                  │
                  ▼
         ┌─────────────────────┐
         │    MLflow           │
         │ Logs metrics, params│
         │ Models, artifacts   │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │      Streamlit       │
         │  Predictions + SHAP  │
         │  Dashboard           │
         └─────────┬───────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │       Docker         │
         │ Containerized App    │
         └─────────────────────┘


Setup Environment :
git clone https://github.com/<your-username>/churn-prediction-mlops.git
cd churn-prediction-mlops
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

Train the Model:
python src/train_model.py

Run Streamlit Dashboard:
streamlit run src/shap_dashboard.py --server.port 8501

How to Run (Docker):
Build Docker Image -   docker build -t churn-dashboard .
Run Container -  docker run --rm -p 8501:8501 \
  -v "$PWD/airflow/data:/app/airflow/data" \
  churn-dashboard

Airflow Quick Start:
Init Airflow  -  export AIRFLOW_HOME=$(pwd)/airflow
airflow db init

Start Webserver + Scheduler - airflow webserver --port 8080
airflow scheduler

Trigger a DAG

airflow dags trigger churn_training_pipeline
airflow dags trigger churn_prediction_pipeline



DATA USED IS TELCO CUSTOMER CHURN DATASET(IBM):
airflow/data/telco_churn.csv
airflow/data/new_customers.csv   # sample file for inference


Results

Model: XGBoost

ROC-AUC: ~0.81

Key Features: Tenure, Contract Type, Monthly Charges, Payment Method

