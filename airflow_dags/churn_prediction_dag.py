from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import sys
import os

# Ensure src is in path
sys.path.append(os.path.expanduser('~/ChurnPrediction/src'))

from prediction import predict_churn

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='Run churn prediction on new customers daily',
    schedule_interval='@daily',
    catchup=False
)

predict_task = PythonOperator(
    task_id='predict_churn',
    python_callable=predict_churn,
    dag=dag
)

predict_task
