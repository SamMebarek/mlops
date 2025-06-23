import argparse
import logging
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
import mlflow
import json
from dotenv import load_dotenv

load_dotenv()


from evaluation.config.configuration import ConfigurationManager
from evaluation.utils.common import save_json

# Logger configuration
target_log = Path("logs/evaluation.log")
target_log.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=target_log,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_evaluation(config_path: str, params_path: str):
    # Load configuration
    cm = ConfigurationManager(config_path, params_path)
    cfg = cm.get_evaluation_config()

    # MLflow setup
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    # Load processed data
    if not cfg.processed_data_path.exists():
        logger.error(f"Processed data not found: {cfg.processed_data_path}")
        return 1
    df = pd.read_csv(cfg.processed_data_path)
    logger.info(f"Loaded processed data: {cfg.processed_data_path} (shape={df.shape})")

    # Load model
    if not cfg.model_path.exists():
        logger.error(f"Model file not found: {cfg.model_path}")
        return 1
    model = joblib.load(cfg.model_path)
    logger.info(f"Loaded model: {cfg.model_path}")

    # Prepare features and target
    if "Prix" not in df.columns:
        logger.error("Target column 'Prix' missing in data.")
        return 1
    y_true = df["Prix"].values
    X = df.drop(columns=["Prix", "SKU", "Timestamp"], errors="ignore")

    # Predictions
    y_pred = model.predict(X)

    # Compute metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    metrics = {"r2_score": r2, "mae": mae}
    logger.info(f"Computed metrics: {metrics}")

    # Save metrics locally
    save_json(cfg.metrics_output_path, metrics)
    logger.info(f"Metrics saved to: {cfg.metrics_output_path}")

    # Log metrics to MLflow
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        logger.info("Metrics logged to MLflow.")

    # Print summary
    print(f"Evaluation completed. R2: {r2:.4f}, MAE: {mae:.4f}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--params", required=True, help="Path to params.yaml")
    args = parser.parse_args()
    exit(run_evaluation(args.config, args.params))
