"""Unit tests for dataset schema definitions and column-level constraints."""

from pathlib import Path
import pandas as pd
import pytest
import yaml

from src.data.validator import DatasetValidator


@pytest.fixture
def clean_synthetic_data(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    inp = pd.DataFrame({
        "isic_id": ["ISIC_001", "ISIC_002"],
        "lesion_id": ["L1", "L2"],
        "image_type": ["dermoscopic", "clinical"],
        "image_manipulation": ["none", "none"],
        "age_approx": [45, 60],
        "sex": ["male", "female"],
        "skin_tone_class": [2, 3],
    })
    inp.to_csv(data_dir / "training_input.csv", index=False)

    gt = pd.DataFrame({
        "lesion_id": ["L1", "L2"],
        "AKIEC": [0, 1], "BCC": [0, 0], "BEN_OTH": [0, 0], "BKL": [0, 0],
        "DF": [0, 0], "INF": [0, 0], "MAL_OTH": [0, 0], "MEL": [1, 0],
        "NV": [0, 0], "SCCKA": [0, 0], "VASC": [0, 0],
    })
    gt.to_csv(data_dir / "training_gt.csv", index=False)

    meta = pd.DataFrame({
        "isic_id": ["ISIC_001", "ISIC_002"],
        "lesion_id": ["L1", "L2"],
        "image_type": ["dermoscopic", "clinical"],
        "image_manipulation": ["none", "none"],
        "melanocytic": [True, False],
        "sex": ["male", "female"],
    })
    meta.to_csv(data_dir / "metadata.csv", index=False)
    return data_dir


def test_schema_required_columns(clean_synthetic_data):
    """Test 1: Validator verifies presence of all required schema columns."""
    validator = DatasetValidator(config="configs/data.yaml", data_root=clean_synthetic_data, strict=True)
    report = validator.run_validation()
    assert report["status"] == "PASSED"
    assert report["total_errors"] == 0


def test_schema_missing_required_identifier(clean_synthetic_data):
    """Test 2: Null identifier triggers error."""
    inp = pd.read_csv(clean_synthetic_data / "training_input.csv")
    inp.loc[0, "isic_id"] = None
    inp.to_csv(clean_synthetic_data / "training_input.csv", index=False)

    validator = DatasetValidator(config="configs/data.yaml", data_root=clean_synthetic_data, strict=False)
    report = validator.run_validation()
    assert report["status"] == "FAILED"
    assert any("contains 1 null/missing values" in e for e in report["errors"])


def test_schema_duplicate_identifiers(clean_synthetic_data):
    """Test 3: Duplicate primary keys trigger error."""
    inp = pd.read_csv(clean_synthetic_data / "training_input.csv")
    inp.loc[1, "isic_id"] = "ISIC_001"
    inp.to_csv(clean_synthetic_data / "training_input.csv", index=False)

    validator = DatasetValidator(config="configs/data.yaml", data_root=clean_synthetic_data, strict=False)
    report = validator.run_validation()
    assert report["status"] == "FAILED"
    assert any("duplicate values" in e for e in report["errors"])


def test_schema_binary_label_validity(clean_synthetic_data):
    """Test 5: Non-binary one-hot values trigger error."""
    gt = pd.read_csv(clean_synthetic_data / "training_gt.csv")
    gt.loc[0, "MEL"] = 99  # Invalid!
    gt.to_csv(clean_synthetic_data / "training_gt.csv", index=False)

    validator = DatasetValidator(config="configs/data.yaml", data_root=clean_synthetic_data, strict=False)
    report = validator.run_validation()
    assert report["status"] == "FAILED"
    assert any("invalid non-binary values" in e for e in report["errors"])
