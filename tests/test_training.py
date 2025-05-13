# tests/test_training.py

import os
import tempfile
import numpy as np
import pandas as pd
import yaml
import pytest
import mlflow
from src.training import training


@pytest.fixture
def fake_data(tmp_path):
    df = pd.DataFrame({
        "SKU": [f"sku_{i}" for i in range(30)],
        "Prix": np.random.rand(30) * 100,
        "PrixInitial": np.random.rand(30) * 100,
        "Timestamp": pd.date_range("2024-01-01", periods=30, freq="D"),
        "AgeProduitEnJours": np.random.randint(1, 100, 30),
        "QuantiteVendue": np.random.randint(1, 50, 30),
        "UtiliteProduit": np.random.rand(30),
        "ElasticitePrix": np.random.rand(30),
        "Remise": np.random.rand(30),
        "Qualite": np.random.rand(30),
        "Mois_sin": np.sin(2 * np.pi * np.random.randint(1, 13, 30) / 12),
        "Mois_cos": np.cos(2 * np.pi * np.random.randint(1, 13, 30) / 12),
        "Heure_sin": np.sin(2 * np.pi * np.random.randint(0, 24, 30) / 24),
        "Heure_cos": np.cos(2 * np.pi * np.random.randint(0, 24, 30) / 24),
    })

    file_path = tmp_path / "fake.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def fake_config(tmp_path, fake_data):
    config = {
        "train": {
            "input": fake_data,
            "test_size": 0.2,
            "random_state": 42,
            "param_dist": {
                "n_estimators_min": 10,
                "n_estimators_max": 50,
                "learning_rate_min": 0.01,
                "learning_rate_max": 0.3,
                "max_depth_min": 2,
                "max_depth_max": 5,
                "subsample_min": 0.5,
                "subsample_max": 1.0,
                "colsample_bytree_min": 0.5,
                "colsample_bytree_max": 1.0,
                "gamma_min": 0,
                "gamma_max": 0.1,
            },
        },
        "model_config": {
            "mlflow_tracking_uri": "http://localhost:5000",
            "mlflow_experiment_name": "TestExperiment",
            "mlflow_model_name": "BestModelTest",
        },
    }

    path = tmp_path / "params.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    return config


def test_training_pipeline(monkeypatch, fake_config):
    # Simule les fonctions MLflow pour ne pas faire de vrai tracking
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "set_registry_uri", lambda uri: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(mlflow, "log_metric", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_param", lambda k, v: None)

    class DummyRun:
        def __enter__(self): return self
        def __exit__(self, *args): pass

    monkeypatch.setattr(mlflow, "start_run", lambda run_name=None: DummyRun())
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda *args, **kwargs: None)

    # Remplace la config globale dans le module
    monkeypatch.setattr(training, "config", fake_config)

    # Appel du main (entraînement simulé)
    training.main()
