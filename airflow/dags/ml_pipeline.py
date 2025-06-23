from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os
import sys

# S'assurer que Airflow voit le projet monté
sys.path.append("/opt/airflow/project")

# Définir les chemins vers les fichiers de configuration
CONFIG_PATH = "/opt/airflow/project/config/config.yaml"
PARAMS_PATH = "/opt/airflow/project/config/params.yaml"

def run_ingestion():
    subprocess.run(["python", "/opt/airflow/project/src/ingestion/main.py", "--config", CONFIG_PATH, "--params", PARAMS_PATH], check=True)

def run_preprocessing():
    subprocess.run(["python", "/opt/airflow/project/src/preprocessing/main.py", "--config", CONFIG_PATH, "--params", PARAMS_PATH], check=True)

def run_training():
    subprocess.run(["python", "/opt/airflow/project/src/training/main.py", "--config", CONFIG_PATH, "--params", PARAMS_PATH], check=True)

def run_evaluation():
    subprocess.run(["python", "/opt/airflow/project/src/evaluation/main.py", "--config", CONFIG_PATH, "--params", PARAMS_PATH], check=True)

default_args = {
    'owner': 'sarah',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

with DAG(
    dag_id='ml_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='Pipeline ML Rakuten orchestré avec Airflow',
    tags=['ml', 'rakuten', 'xgboost']
) as dag:

    task_ingest = PythonOperator(
        task_id='data_ingestion',
        python_callable=run_ingestion
    )

    task_preprocess = PythonOperator(
        task_id='data_preprocessing',
        python_callable=run_preprocessing
    )

    task_train = PythonOperator(
        task_id='model_training',
        python_callable=run_training
    )

    task_evaluate = PythonOperator(
        task_id='model_evaluation',
        python_callable=run_evaluation
    )

    task_ingest >> task_preprocess >> task_train >> task_evaluate
