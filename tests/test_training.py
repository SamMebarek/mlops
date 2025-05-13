import runpy
import shutil
import os
from pathlib import Path
import pytest
from unittest import mock
import sys
import types

def test_training_runs_without_crash(tmp_path, training_config):
    # Préparation : copie du fichier params.yaml
    shutil.copy(training_config, tmp_path / "params.yaml")
    os.chdir(tmp_path)
    os.environ["MLFLOW_TRACKING_URI"] = "https://fake-uri"

    # Création d'un faux module mlflow avec les attributs nécessaires
    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = mock.Mock()
    fake_mlflow.set_registry_uri = mock.Mock()
    fake_mlflow.set_experiment = mock.Mock()
    fake_mlflow.start_run = mock.Mock()
    fake_mlflow.log_metric = mock.Mock()
    fake_mlflow.log_param = mock.Mock()
    fake_mlflow.get_tracking_uri = mock.Mock(return_value="https://fake-uri")

    # Sous-module mlflow.sklearn
    fake_mlflow.sklearn = mock.Mock()
    fake_mlflow.sklearn.log_model = mock.Mock()

    # Sous-module mlflow.models
    fake_mlflow.models = mock.Mock()
    fake_mlflow.models.infer_signature = mock.Mock(return_value=None)

    training_script_path = Path(__file__).resolve().parents[1] / "src" / "training" / "training.py"

    with mock.patch.dict(sys.modules, {"mlflow": fake_mlflow, "mlflow.sklearn": fake_mlflow.sklearn, "mlflow.models": fake_mlflow.models}):
        try:
            runpy.run_path(str(training_script_path), run_name="__main__")
        except Exception as e:
            pytest.fail(f"Erreur lors de l'exécution du script training.py : {e}")
