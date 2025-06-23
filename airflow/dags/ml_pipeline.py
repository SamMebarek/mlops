from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os
import sys

# Ajouter le chemin vers le dossier src pour les imports dans le DAG (pas utile dans subprocess)
sys.path.append("/opt/airflow/project/src")

# Chemins vers les fichiers de configuration
CONFIG_PATH = "/opt/airflow/project/config/config.yaml"
PARAMS_PATH = "/opt/airflow/project/config/params.yaml"

# Fonctions appelées dans les tâches Airflow
def run_ingestion():
    subprocess.run([
        "python", "-m", "ingestion.components.data_ingestion",
        "--config", CONFIG_PATH,
        "--params", PARAMS_PATH
    ], check=True, cwd="/opt/airflow/project/src")

def run_preprocessing():
    subprocess.run([
        "python", "-m", "preprocessing.components.preprocess",
        "--config", CONFIG_PATH,
        "--params", PARAMS_PATH
    ], check=True, cwd="/opt/airflow/project/src")

def run_training():
    subprocess.run([
        "python", "-m", "training.components.train",
        "--config", CONFIG_PATH,
        "--params", PARAMS_PATH
    ], check=True, cwd="/opt/airflow/project/src")

def run_evaluation():
    subprocess.run([
        "python", "-m", "evaluation.components.evaluate",
        "--config", CONFIG_PATH,
        "--params", PARAMS_PATH
    ], check=True, cwd="/opt/airflow/project/src")

# Paramètres du DAG
default_args = {
    'owner': 'sarah',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

# Définition du DAG
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
