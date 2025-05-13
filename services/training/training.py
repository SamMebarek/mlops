import logging
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
import mlflow
import mlflow.sklearn
import yaml
from mlflow.models import infer_signature
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("training")


def charger_config():
    with open("params.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Remplace les ${VAR} par les vraies valeurs d'env
    for section in config:
        if isinstance(config[section], dict):
            for key, value in config[section].items():
                if isinstance(value, str) and "${" in value:
                    env_var = value.strip("${}").strip()
                    if env_var in os.environ:
                        config[section][key] = os.getenv(env_var)
    return config


def main():
    config = charger_config()
    train_params = config["train"]
    model_config = config["model_config"]

    # Config MLflow
    mlflow_tracking_uri = model_config["mlflow_tracking_uri"]
    if not mlflow_tracking_uri or not mlflow_tracking_uri.startswith("http"):
        raise ValueError("🚨 ERREUR : MLFLOW_TRACKING_URI invalide")

    print(f"📡 MLflow va utiliser l'URI : {mlflow_tracking_uri}")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_registry_uri(mlflow_tracking_uri)
    mlflow.set_experiment(model_config["mlflow_experiment_name"])

    # Chargement des données
    data_path = train_params["input"]
    if not os.path.exists(data_path):
        print(f"❌ Fichier prétraité introuvable : {data_path}")
        logger.error(f"Fichier prétraité introuvable : {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"📄 Données chargées : {df.shape}")
    logger.info(f"Données chargées, shape = {df.shape}")

    if "Prix" not in df.columns:
        print("❌ La colonne 'Prix' est manquante.")
        logger.error("La colonne 'Prix' est manquante.")
        return

    y = df["Prix"].values
    X = (
        df.drop(columns=["Prix", "SKU", "Timestamp"], errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_params["test_size"],
        random_state=train_params["random_state"],
    )

    # Hyperparamètres
    dist = train_params["param_dist"]
    param_dist = {
        "n_estimators": randint(dist["n_estimators_min"], dist["n_estimators_max"]),
        "learning_rate": uniform(dist["learning_rate_min"], dist["learning_rate_max"] - dist["learning_rate_min"]),
        "max_depth": randint(dist["max_depth_min"], dist["max_depth_max"]),
        "subsample": uniform(dist["subsample_min"], dist["subsample_max"] - dist["subsample_min"]),
        "colsample_bytree": uniform(dist["colsample_bytree_min"], dist["colsample_bytree_max"] - dist["colsample_bytree_min"]),
        "gamma": uniform(dist["gamma_min"], dist["gamma_max"] - dist["gamma_min"]),
    }

    model_xgb = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=train_params["random_state"]),
        param_distributions=param_dist,
        n_iter=10,
        cv=3,
        verbose=1,
        n_jobs=-1,
        random_state=train_params["random_state"],
    )

    with mlflow.start_run(run_name="XGBoost_RandSearch") as run:
        print("🚀 Entraînement du modèle en cours...")
        logger.info("Recherche des meilleurs hyperparamètres XGBoost.")
        model_xgb.fit(X_train, y_train)

        y_pred = model_xgb.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        best_params = model_xgb.best_params_

        logger.info(f"Meilleurs paramètres : {best_params}")
        logger.info(f"Score R² : {r2:.4f}")
        print(f"📈 Score R² : {r2:.4f}")
        print(f"🏅 Meilleurs paramètres : {best_params}")

        # Logging MLflow
        mlflow.log_metric("r2_score", r2)
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        input_example = X_test.iloc[:1]
        signature = infer_signature(X_train, model_xgb.best_estimator_.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model_xgb.best_estimator_,
            artifact_path="xgb_model",
            registered_model_name=model_config["mlflow_model_name"],
            signature=signature,
            input_example=input_example,
        )

        print(f"✅ Modèle entraîné et loggé avec succès. R² : {r2:.4f}")
        print(f"🧪 Voir l'expérience sur : {mlflow_tracking_uri}")

if __name__ == "__main__":
    main()
