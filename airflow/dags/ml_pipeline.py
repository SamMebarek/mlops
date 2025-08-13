#  airflow/dags/ml_pipeline.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os
import sys
import requests

# --- Constants (mounted by docker-compose) ---
PROJECT_ROOT = "/opt/airflow/project"
CONFIG_PATH = f"{PROJECT_ROOT}/config/config.yaml"
PARAMS_PATH = f"{PROJECT_ROOT}/config/params.yaml"
SRC_DIR = f"{PROJECT_ROOT}/src"

# In-docker service URL
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8002")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

# Evidently paths (current & reference)
CURRENT_CSV = f"{PROJECT_ROOT}/data/processed/clean_data.csv"
REFERENCE_CSV = f"{PROJECT_ROOT}/data/reference/clean_data_ref.csv"

# Ensure our src package imports work
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ---- Task callables ---------------------------------------------------------


def run_ingestion():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ingestion.components.data_ingestion",
            "--config",
            CONFIG_PATH,
            "--params",
            PARAMS_PATH,
        ],
        check=True,
        cwd=SRC_DIR,
        env={**os.environ, "PYTHONPATH": SRC_DIR},
    )


def run_preprocessing():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "preprocessing.components.preprocess",
            "--config",
            CONFIG_PATH,
            "--params",
            PARAMS_PATH,
        ],
        check=True,
        cwd=SRC_DIR,
        env={**os.environ, "PYTHONPATH": SRC_DIR},
    )


def run_training():
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.components.train",
            "--config",
            CONFIG_PATH,
            "--params",
            PARAMS_PATH,
        ],
        cwd=SRC_DIR,
        env={**os.environ, "PYTHONPATH": SRC_DIR},
        capture_output=True,
        text=True,
    )
    print("STDOUT:\n", res.stdout)
    print("STDERR:\n", res.stderr)
    if res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, res.args, res.stdout, res.stderr)


def run_evaluation():
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.components.evaluate",
            "--config",
            CONFIG_PATH,
            "--params",
            PARAMS_PATH,
        ],
        cwd=SRC_DIR,
        env={**os.environ, "PYTHONPATH": SRC_DIR},
        capture_output=True,
        text=True,
    )
    print("STDOUT:\n", res.stdout)
    print("STDERR:\n", res.stderr)
    if res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, res.args, res.stdout, res.stderr)


def run_evidently_batch():
    """
    Runs Evidently batch script to compute drift/summary and push to Pushgateway.
    Requires: evidently_batch.py, prometheus_client, and data files present.
    """
    script = f"{SRC_DIR}/monitoring/evidently_batch.py"
    cmd = [
        sys.executable,
        script,
        "--current",
        CURRENT_CSV,
        "--reference",
        REFERENCE_CSV,
        # Optionally, pass pushgateway via env PUSHGATEWAY_URL
        # "--pushgateway", os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091"),
        "--instance",
        os.getenv("INSTANCE", "airflow"),
    ]
    res = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": SRC_DIR},
        capture_output=True,
        text=True,
    )
    print("STDOUT:\n", res.stdout)
    print("STDERR:\n", res.stderr)
    if res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, res.args, res.stdout, res.stderr)


def reload_model_via_gateway():
    """
    Get admin JWT from gateway /login, then POST /reload-model.
    """
    login_payload = {"username": ADMIN_USER, "password": ADMIN_PASSWORD}
    r = requests.post(f"{GATEWAY_URL}/login", json=login_payload, timeout=10)
    if r.status_code != 200:
        raise RuntimeError(f"Login failed: {r.status_code} {r.text}")
    token = r.json()["access_token"]
    hdrs = {"Authorization": f"Bearer {token}"}
    r2 = requests.post(f"{GATEWAY_URL}/reload-model", headers=hdrs, timeout=15)
    if r2.status_code != 200:
        raise RuntimeError(f"Reload failed: {r2.status_code} {r2.text}")
    print("Reload response:", r2.json())


# ---- DAG --------------------------------------------------------------------

default_args = {
    "owner": "sarah",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    schedule_interval=None,  # manual for now
    catchup=False,
    description="Rakuten ML pipeline orchestrated by Airflow",
    tags=["ml", "rakuten", "xgboost"],
) as dag:

    task_ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=run_ingestion,
    )

    task_preprocess = PythonOperator(
        task_id="data_preprocessing",
        python_callable=run_preprocessing,
    )

    task_train = PythonOperator(
        task_id="model_training",
        python_callable=run_training,
    )

    task_evaluate = PythonOperator(
        task_id="model_evaluation",
        python_callable=run_evaluation,
    )

    task_evidently = PythonOperator(
        task_id="evidently_batch",
        python_callable=run_evidently_batch,
    )

    # Optional: flip to False to disable reload in DAG without code edits
    if os.getenv("RELOAD_AFTER_TRAIN", "true").lower() == "true":
        task_reload = PythonOperator(
            task_id="reload_model",
            python_callable=reload_model_via_gateway,
        )
        (
            task_ingest
            >> task_preprocess
            >> task_train
            >> task_evaluate
            >> task_evidently
            >> task_reload
        )
    else:
        task_ingest >> task_preprocess >> task_train >> task_evaluate >> task_evidently
