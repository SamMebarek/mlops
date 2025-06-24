from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os
import sys

# Chemins vers les fichiers de configuration et base source
CONFIG_PATH = "/opt/airflow/project/config/config.yaml"
PARAMS_PATH = "/opt/airflow/project/config/params.yaml"
BASE_SRC = "/opt/airflow/project/src"

# Ajouter le dossier src au PYTHONPATH si besoin pour les imports internes
sys.path.append(BASE_SRC)

# Tâches du pipeline

def run_ingestion():
    subprocess.run(
        ["python", "-m", "ingestion.components.data_ingestion", "--config", CONFIG_PATH, "--params", PARAMS_PATH],
        check=True,
        cwd=BASE_SRC
    )

def run_preprocessing():
    subprocess.run(
        ["python", "-m", "preprocessing.components.preprocess", "--config", CONFIG_PATH, "--params", PARAMS_PATH],
        check=True,
        cwd=BASE_SRC
    )

def run_training():
    try:
        result = subprocess.run(
            [
                "python", "-m", "training.components.train",
                "--config", CONFIG_PATH,
                "--params", PARAMS_PATH
            ],
            cwd="/opt/airflow/project/src",  # très important
            env={**os.environ, "PYTHONPATH": "/opt/airflow/project/src"},  # obligatoire pour les imports locaux
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Subprocess stdout:\n", result.stdout)
        print("✅ Subprocess stderr:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("❌ Subprocess failed!")
        print("⚠️ STDOUT:\n", e.stdout)
        print("⚠️ STDERR:\n", e.stderr)
        raise


def run_evaluation():
    try:
        result = subprocess.run(
            [
                "python", "-m", "evaluation.components.evaluate",
                "--config", CONFIG_PATH,
                "--params", PARAMS_PATH
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd="/opt/airflow/project/src"
        )
        print("✅ Subprocess stdout:\n", result.stdout)
        print("✅ Subprocess stderr:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("❌ Subprocess failed!")
        print("⚠️ STDOUT:\n", e.stdout)
        print("⚠️ STDERR:\n", e.stderr)
        raise


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

    # Orchestration des tâches
    task_ingest >> task_preprocess >> task_train >> task_evaluate
