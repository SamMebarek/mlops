# tests/test_training.py

import pytest
import yaml
from pathlib import Path

from training.config.configuration import ConfigurationManager
from training.repository.repository import CsvModelRepository


def test_configuration_manager_creates_correct_paths(tmp_path):
    # Préparer config.yaml
    cfg = {
        "training": {
            "processed_data_path": "data/processed/clean_data.csv",
            "model_dir": "models",
            "model_file_name": "xgb_model.pkl",
        }
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump(cfg))

    # Préparer params.yaml
    params = {
        "training": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.1,
            "random_seed": 42,
            "test_size": 0.2,
        }
    }
    params_path = config_dir / "params.yaml"
    params_path.write_text(yaml.dump(params))

    # Instancier et récupérer la config d'entraînement
    cm = ConfigurationManager(str(config_path), str(params_path))
    tc = cm.get_training_config()

    # Chemins attendus
    expected_root = config_path.parent.parent  # tmp_path
    expected_data = expected_root / "data" / "processed" / "clean_data.csv"
    expected_model_dir = expected_root / "models"

    assert tc.processed_data_path == expected_data
    assert expected_model_dir.exists() and expected_model_dir.is_dir()
    assert tc.model_dir == expected_model_dir
    assert tc.model_file_name == "xgb_model.pkl"

    # Vérifier que les autres paramètres du training sont bien passés
    assert tc.epochs == 1
    assert tc.batch_size == 1
    assert pytest.approx(tc.learning_rate, 0.001) == 0.1
    assert tc.random_seed == 42
    assert pytest.approx(tc.test_size, 0.001) == 0.2


def test_csv_model_repository_save_and_load(tmp_path):
    # Chemin de sortie
    model_path = tmp_path / "models" / "my_model.pkl"
    repo = CsvModelRepository(model_path)

    # Exemple de "modèle" simple
    model = {"foo": "bar"}
    # Sauvegarde
    repo.save(model)
    # Le fichier doit exister
    assert model_path.exists()

    # Chargement
    loaded = repo.load()
    assert isinstance(loaded, dict)
    assert loaded == model
