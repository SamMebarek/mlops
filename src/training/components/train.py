# src/training/components/train.py

import argparse
import logging
import os
import time
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone
import subprocess  # <-- [NOUVEAU] pour récupérer le SHA git
import hashlib  # <-- [NOUVEAU] fallback hash local
import yaml  # <-- [NOUVEAU] lire le .dvc si présent

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from dotenv import load_dotenv

from training.config.configuration import ConfigurationManager
from training.repository.repository import CsvModelRepository

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

load_dotenv()

logging.basicConfig(
    filename=Path("logs/training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://pushgateway:9091")
PUSH_GROUPING = {"instance": os.getenv("INSTANCE", "local"), "service": "training"}


def _push_metrics(
    status: int, duration_s: float, rows: int, r2: Optional[float], artifact_bytes: int
) -> None:
    try:
        reg = CollectorRegistry()
        g_status = Gauge("job_status", "1=success,0=failure", registry=reg)
        g_dur = Gauge("job_duration_seconds", "Job wall time in seconds", registry=reg)
        g_rows = Gauge("rows_processed_total", "Rows processed by the job", registry=reg)
        g_r2 = Gauge("train_r2_score", "R^2 on holdout set", registry=reg)
        g_art = Gauge("model_artifact_size_bytes", "Size of saved model artifact", registry=reg)

        g_status.set(float(status))
        g_dur.set(float(duration_s))
        g_rows.set(float(rows))
        if r2 is not None:
            g_r2.set(float(r2))
        g_art.set(float(artifact_bytes))

        push_to_gateway(PUSHGATEWAY_URL, job="training", grouping_key=PUSH_GROUPING, registry=reg)
        logger.info(
            f"Pushed metrics to Pushgateway ({PUSHGATEWAY_URL}) "
            f"status={status} duration_s={duration_s:.3f} rows={rows} r2={r2} artifact_bytes={artifact_bytes}"
        )
    except Exception as e:
        logger.warning(f"Failed to push metrics to Pushgateway: {e}")


def _unique_local_model_dir() -> Path:
    """
    Build a unique local directory for mlflow.sklearn.save_model.
    Prefer Airflow run_id, else MLflow run_id, else timestamp.
    """
    airflow_run_id = os.getenv("AIRFLOW_CTX_DAG_RUN_ID")
    mlflow_run_id = None
    try:
        active = mlflow.active_run()
        if active:
            mlflow_run_id = active.info.run_id
    except Exception:
        pass

    suffix = airflow_run_id or mlflow_run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("models") / f"xgb_model_mlflow_{suffix}"


def _dir_size_bytes(p: Path) -> int:
    if not p.exists():
        return 0
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


# ---------- [NOUVEAU] Utilitaires traçabilité données & code ----------


def _dvc_version_for(data_file: Path) -> Optional[str]:
    """
    Retourne l'identifiant de version DVC pour un fichier suivi.
    Stratégie:
      1) Lire le .dvc associé (ex: clean_data.csv.dvc) et extraire 'md5' / 'etag' / 'hash'
      2) Si non disponible, retourner None (on fera un fallback hash local)
    """
    dvc_ptr = data_file.with_suffix(data_file.suffix + ".dvc")
    if not dvc_ptr.exists():
        return None
    try:
        with open(dvc_ptr, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f) or {}
        outs = meta.get("outs") or meta.get("outs_no_cache") or []
        if not outs:
            return None
        entry = outs[0] or {}
        # champs possibles selon backends DVC
        return entry.get("md5") or entry.get("etag") or entry.get("hash") or None
    except Exception:
        return None


def _file_md5(p: Path) -> str:
    """
    Calcule un hash MD5 local (fallback si DVC non disponible).
    """
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> Optional[str]:
    """
    Retourne le commit SHA git courant (ou None si indisponible).
    """
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        return res.stdout.strip()
    except Exception:
        return None


# ---------------------------------------------------------------------


def run_training(config_path: str, params_path: str):
    t0 = time.perf_counter()
    status = 0
    rows = 0
    r2: Optional[float] = None
    artifact_bytes = 0

    try:
        # Load configurations
        cm = ConfigurationManager(config_path, params_path)
        train_cfg = cm.get_training_config()
        model_cfg = cm.get_model_config()

        # MLflow setup
        mlflow.set_tracking_uri(model_cfg.mlflow_tracking_uri)
        mlflow.set_registry_uri(model_cfg.mlflow_tracking_uri)
        mlflow.set_experiment(model_cfg.mlflow_experiment_name)

        # Load processed data
        data_path = train_cfg.processed_data_path
        if not data_path.exists():
            logger.error(f"Processed data not found: {data_path}")
            return
        df = pd.read_csv(data_path)
        rows = len(df)
        logger.info(f"Loaded processed data: {data_path} (shape={df.shape})")

        if "Prix" not in df.columns:
            logger.error("Target column 'Prix' missing.")
            return

        # -------- [NOUVEAU] Déterminer la version des données & le SHA git --------
        # 1) Essayer de lire la version DVC du CSV (sinon fallback MD5 local)
        data_version = _dvc_version_for(data_path) or _file_md5(data_path)
        # 2) SHA du code au moment de l'entraînement
        git_sha = _git_sha()
        # --------------------------------------------------------------------------

        # Features/target
        y = df["Prix"].values
        X = df.drop(columns=["Prix", "SKU", "Timestamp"], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=train_cfg.test_size, random_state=train_cfg.random_seed
        )
        logger.info(f"Split data: train_shape={X_train.shape}, test_shape={X_test.shape}")

        # Hyperparam space
        pdist = cm.params.training.param_dist
        param_dist = {
            "n_estimators": randint(pdist["n_estimators_min"], pdist["n_estimators_max"]),
            "learning_rate": uniform(
                pdist["learning_rate_min"],
                pdist["learning_rate_max"] - pdist["learning_rate_min"],
            ),
            "max_depth": randint(pdist["max_depth_min"], pdist["max_depth_max"]),
            "subsample": uniform(
                pdist["subsample_min"], pdist["subsample_max"] - pdist["subsample_min"]
            ),
            "colsample_bytree": uniform(
                pdist["colsample_bytree_min"],
                pdist["colsample_bytree_max"] - pdist["colsample_bytree_min"],
            ),
            "gamma": uniform(pdist["gamma_min"], pdist["gamma_max"] - pdist["gamma_min"]),
        }

        xgb = XGBRegressor(objective="reg:squarederror", random_state=train_cfg.random_seed)
        search = RandomizedSearchCV(
            xgb,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            verbose=1,
            n_jobs=-1,
            random_state=train_cfg.random_seed,
        )

        # Train + log
        with mlflow.start_run(run_name="XGBoost_RandSearch") as active_run:
            logger.info("Starting hyperparameter search and training.")

            # -------- [NOUVEAU] Journaliser la traçabilité avant l'entraînement -----
            # Ces paramètres permettent de relier modèle ↔ données ↔ code.
            mlflow.log_param("data_version", data_version)
            if git_sha:
                mlflow.log_param("git_sha", git_sha)
            # ------------------------------------------------------------------------

            search.fit(X_train, y_train)

            y_pred = search.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            best_params = search.best_params_
            logger.info(f"Best params: {best_params}")
            logger.info(f"R^2 score: {r2:.4f}")

            mlflow.log_metric("r2_score", r2)
            mlflow.log_params(best_params)

            # --------- DagsHub-friendly model logging & registration ----------
            local_model_dir = _unique_local_model_dir()
            if local_model_dir.exists():
                shutil.rmtree(local_model_dir)

            signature = infer_signature(X_train, search.best_estimator_.predict(X_train))
            mlflow.sklearn.save_model(
                sk_model=search.best_estimator_,
                path=str(local_model_dir),
                signature=signature,
                input_example=X_test.iloc[:1],
            )

            mlflow.log_artifacts(str(local_model_dir), artifact_path="xgb_model")

            run_id = active_run.info.run_id
            artifact_uri = mlflow.get_artifact_uri("xgb_model")
            client = MlflowClient()
            try:
                try:
                    client.create_registered_model(model_cfg.mlflow_model_name)
                except Exception:
                    pass
                client.create_model_version(
                    name=model_cfg.mlflow_model_name,
                    source=artifact_uri,
                    run_id=run_id,
                )
                logger.info(
                    f"Registered model '{model_cfg.mlflow_model_name}' from source={artifact_uri}, run_id={run_id}"
                )
            except Exception as reg_err:
                logger.warning(f"Model registry step failed (continuing): {reg_err}")
            # ------------------------------------------------------------------

        # Persist model locally (pickle) for simple loading paths
        model_path = train_cfg.model_dir / train_cfg.model_file_name
        repo = CsvModelRepository(model_path)
        repo.save(search.best_estimator_)

        if model_path.exists():
            artifact_bytes = model_path.stat().st_size
        artifact_bytes += _dir_size_bytes(local_model_dir)

        logger.info(
            f"Local MLflow dir: {local_model_dir} (size={_dir_size_bytes(local_model_dir)} bytes)"
        )
        logger.info(f"Model saved locally at: {model_path}")

        status = 1
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise
    finally:
        duration = time.perf_counter() - t0
        _push_metrics(
            status=status,
            duration_s=duration,
            rows=rows,
            r2=r2,
            artifact_bytes=artifact_bytes,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--params", required=True, help="Path to params.yaml")
    args = parser.parse_args()
    run_training(args.config, args.params)
