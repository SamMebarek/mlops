import argparse
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from dotenv import load_dotenv  # <— Ajout pour le .env

from training.config.configuration import ConfigurationManager
from training.repository.repository import CsvModelRepository

# Charger les variables du fichier .env
load_dotenv()

# Logger configuration
logging.basicConfig(
    filename=Path("logs/training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_training(config_path: str, params_path: str):
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
    logger.info(f"Loaded processed data: {data_path} (shape={df.shape})")

    # Verify target column
    if "Prix" not in df.columns:
        logger.error("Target column 'Prix' missing.")
        return

    # Prepare features and target
    y = df["Prix"].values
    X = df.drop(columns=["Prix", "SKU", "Timestamp"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_cfg.test_size, random_state=train_cfg.random_seed
    )
    logger.info(f"Split data: train_shape={X_train.shape}, test_shape={X_test.shape}")

    # Hyperparameter distributions
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

    # Model and search setup
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

    # Training and logging
    with mlflow.start_run(run_name="XGBoost_RandSearch"):
        logger.info("Starting hyperparameter search and training.")
        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        score = r2_score(y_test, y_pred)
        best_params = search.best_params_
        logger.info(f"Best params: {best_params}")
        logger.info(f"R^2 score: {score:.4f}")

        # Log to MLflow
        mlflow.log_metric("r2_score", score)
        mlflow.log_params(best_params)

        # Signature and model logging
        signature = infer_signature(X_train, search.best_estimator_.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=search.best_estimator_,
            artifact_path="xgb_model",
            registered_model_name=model_cfg.mlflow_model_name,
            signature=signature,
            input_example=X_test.iloc[:1],
        )

    # Persist model locally
    model_path = train_cfg.model_dir / train_cfg.model_file_name
    repo = CsvModelRepository(model_path)
    repo.save(search.best_estimator_)
    logger.info(f"Model saved locally at: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--params", required=True, help="Path to params.yaml")
    args = parser.parse_args()
    run_training(args.config, args.params)
