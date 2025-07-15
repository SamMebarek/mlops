import pytest
import pandas as pd
import yaml
from pathlib import Path

from src.ingestion.components.data_ingestion import validate_schema, run_ingestion


def test_validate_schema_raises_on_missing_column():
    # DataFrame sans "Prix" doit lever ValueError
    df = pd.DataFrame({"SKU": [1, 2, 3]})
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        validate_schema(df)


def test_run_ingestion_writes_csv(tmp_path):
    # Préparer répertoires
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    # Fichier source
    source = tmp_path / "data" / "ingestion.csv"
    df_in = pd.DataFrame({"SKU": [1, 2], "Prix": [10.0, 20.5]})
    df_in.to_csv(source, index=False)
    # config.yaml
    cfg = {
        "data_ingestion": {
            "source_URL": str(source),
            "raw_data_dir": str(raw_dir),
            "ingested_file_name": "ingested_data.csv",
        }
    }
    config_path = tmp_path / "config" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.dump(cfg))
    # params.yaml
    params_path = tmp_path / "config" / "params.yaml"
    params_path.write_text(yaml.dump({"ingestion": {}}))
    # Appel de la fonction
    run_ingestion(str(config_path), str(params_path))
    # Vérifications
    out = raw_dir / "ingested_data.csv"
    assert out.exists()
    df_out = pd.read_csv(out)
    pd.testing.assert_frame_equal(df_in, df_out)
