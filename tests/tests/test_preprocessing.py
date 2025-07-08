# tests/test_preprocessing.py

import pytest
import pandas as pd
import yaml
from pathlib import Path

from preprocessing.components.preprocess import (
    convert_types,
    add_time_features,
    run_preprocessing,
)


def test_convert_and_add_time_features():
    # DataFrame minimal
    df = pd.DataFrame(
        {
            "SKU": [123],
            "Timestamp": ["2025-06-29 12:34:56"],
            "Prix": ["100"],  # sera converti en float
            "PrixInitial": [200],
            "AgeProduitEnJours": [10],
            "QuantiteVendue": [5],
            "UtiliteProduit": [0.8],
            "ElasticitePrix": [1],
            "Remise": [0.1],
            "Qualite": [5],
        }
    )
    # Conversion des types
    df_conv = convert_types(df)
    assert df_conv["SKU"].dtype == object
    assert pd.api.types.is_datetime64_any_dtype(df_conv["Timestamp"])
    for col in [
        "Prix",
        "PrixInitial",
        "AgeProduitEnJours",
        "QuantiteVendue",
        "UtiliteProduit",
        "ElasticitePrix",
        "Remise",
        "Qualite",
    ]:
        assert pd.api.types.is_float_dtype(df_conv[col])
    # Ajout des features temporelles
    df_time = add_time_features(df_conv)
    for col in ["Mois_sin", "Mois_cos", "Heure_sin", "Heure_cos"]:
        assert col in df_time.columns
        assert pd.api.types.is_float_dtype(df_time[col])


def test_run_preprocessing_writes_expected_columns(tmp_path, monkeypatch):
    # Préparer l'arborescence et le CSV source
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    source = raw_dir / "ingested_data.csv"
    df_in = pd.DataFrame(
        {
            "SKU": [1, 2],
            "Prix": [10.0, 20.0],
            "Timestamp": ["2025-06-29 00:00:00", "2025-06-29 01:00:00"],
            "AgeProduitEnJours": [1, 2],
            "QuantiteVendue": [1, 2],
            "UtiliteProduit": [0.5, 0.6],
            "ElasticitePrix": [1.1, 1.2],
            "Remise": [0.1, 0.2],
            "Qualite": [3, 4],
        }
    )
    df_in.to_csv(source, index=False)

    # Config YAML
    cfg = {
        "data_preprocessing": {
            "raw_data_path": "data/raw/ingested_data.csv",
            "processed_dir": "data/processed",
            "clean_file_name": "clean_data.csv",
        }
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.dump(cfg))

    # Params YAML
    keep_cols = list(df_in.columns) + ["Mois_sin", "Mois_cos", "Heure_sin", "Heure_cos"]
    params = {"preprocessing": {"columns_to_keep": keep_cols}}
    params_path = config_dir / "params.yaml"
    params_path.write_text(yaml.dump(params))

    # Exécuter le preprocessing
    run_preprocessing(str(config_path), str(params_path))

    # Vérifier le fichier sorti
    out_path = tmp_path / "data" / "processed" / "clean_data.csv"
    assert out_path.exists()
    df_out = pd.read_csv(out_path)
    assert list(df_out.columns) == keep_cols
    assert len(df_out) == len(df_in)
