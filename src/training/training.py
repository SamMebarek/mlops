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

# Chargement du fichier de configuration
with open("params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Remplacement des valeurs ${VAR} par les valeurs d'environnement
for section in config:
    if isinstance(config[section], dict):
        for key, value in config[section].items():
            if isinstance(value, str) and "${" in value:
                env_var = value.strip("${}").strip()
                if env_var in os.environ:
                    config[section][key] = os.getenv(env_var)

# Configuration de MLflow
mlflow_tracking_uri = config["model_config"]["mlflow_tracking_uri"]
if not mlflow_tracking_uri or not mlflow_tracking_uri.startswith("http"):
    raise ValueError("ERREUR : MLFLOW_TRACKING_URI invalide")

print(f"MLflow va utiliser l'URI : {mlflow_tracking_uri}")

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_registry_uri(mlflow_tracking_uri)

# Vérification que l'URI a bien été prise en compte
assert mlflow.get_tracking_uri() == mlflow_tracking_uri, "MLflow n'a pas pris l'URI en compte"

# Définition de l'expérience
mlflow.set_experiment(config["model_config"]["mlflow_experiment_name"])

# Configuration du logger
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", "training.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("training")


def main():
    train_params = config["train"]
    model_config = config["model_config"]

    logger.info("Début de l'entraînement du modèle.")

    # Chargement des données
    data_path = train_params["input"]
    if not os.path.exists(data_path):
        logger.error(f"Fichier prétraité introuvable : {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Données chargées, shape = {df.shape}")

    if "Prix" not in df.columns:
        logger.error("La colonne 'Prix' est manquante.")
        return

    # Séparation des features et de la target
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

    # Définition des distributions pour la recherche d'hyperparamètres
    dist_params = train_params["param_dist"]
    param_dist = {
        "n_estimators": randint(dist_params["n_estimators_min"], dist_params["n_estimators_max"]),
        "learning_rate": uniform(dist_params["learning_rate_min"], dist_params["learning_rate_max"] - dist_params["learning_rate_min"]),
        "max_depth": randint(dist_params["max_depth_min"], dist_params["max_depth_max"]),
        "subsample": uniform(dist_params["subsample_min"], dist_params["subsample_max"] - dist_params["subsample_min"]),
        "colsample_bytree": uniform(dist_params["colsample_bytree_min"], dist_params["colsample_bytree_max"] - dist_params["colsample_bytree_min"]),
        "gamma": uniform(dist_params["gamma_min"], dist_params["gamma_max"] - dist_params["gamma_min"]),
    }

    # Initialisation du modèle et entraînement
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
        logger.info("Recherche des meilleurs hyperparamètres XGBoost.")
        model_xgb.fit(X_train, y_train)

        y_pred = model_xgb.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        best_params = model_xgb.best_params_

        logger.info(f"Meilleurs paramètres : {best_params}")
        logger.info(f"Score R² : {r2:.4f}")

        # Log dans MLflow
        mlflow.log_metric("r2_score", r2)
        for param, value in best_params.items():
            mlflow.log_param(param, value)

        # Log du modèle
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


if __name__ == "__main__":
    main()
