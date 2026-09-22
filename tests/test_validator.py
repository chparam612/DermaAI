"""Unit tests for dataset schema and relational join validator."""

import json
from pathlib import Path
import pandas as pd
import pytest
import yaml

from src.data.validator import DatasetValidator, DataValidationError


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """Create a synthetic dataset directory matching ISIC schema with known characteristics."""
    data_dir = tmp_path / "mock_data"
    data_dir.mkdir()

    # 1. training_input.csv
    # 5 rows, 3 unique lesions, valid identifiers
    input_df = pd.DataFrame({
        "isic_id": ["ISIC_0001", "ISIC_0002", "ISIC_0003", "ISIC_0004", "ISIC_0005"],
        "lesion_id": ["LES_A", "LES_A", "LES_B", "LES_C", "LES_C"],
        "image_type": ["dermoscopic", "dermoscopic", "clinical", "tbp", "tbp"],
        "image_manipulation": ["none", "crop", "none", "none", "none"],
        "age_approx": [45, 45, 60, 35, 35],
        "sex": ["male", "male", "female", "female", "male"],
        "skin_tone_class": [2, 2, 3, 4, 4],
    })
    input_df.to_csv(data_dir / "training_input.csv", index=False)

    # 2. training_gt.csv
    # Matches lesion_id, contains the 11 diagnostic classes
    gt_df = pd.DataFrame({
        "lesion_id": ["LES_A", "LES_B", "LES_C"],
        "AKIEC": [0, 0, 1],
        "BCC": [0, 0, 0],
        "BEN_OTH": [0, 0, 0],
        "BKL": [0, 0, 0],
        "DF": [0, 0, 0],
        "INF": [0, 0, 0],
        "MAL_OTH": [0, 0, 0],
        "MEL": [1, 0, 0],
        "NV": [0, 1, 0],
        "SCCKA": [0, 0, 0],
        "VASC": [0, 0, 0],
    })
    gt_df.to_csv(data_dir / "training_gt.csv", index=False)

    # 3. metadata.csv
    metadata_df = pd.DataFrame({
        "isic_id": ["ISIC_0001", "ISIC_0002", "ISIC_0003", "ISIC_0004", "ISIC_0005"],
        "lesion_id": ["LES_A", "LES_A", "LES_B", "LES_C", "LES_C"],
        "image_type": ["dermoscopic", "dermoscopic", "clinical", "tbp", "tbp"],
        "image_manipulation": ["none", "crop", "none", "none", "none"],
        "melanocytic": [True, True, False, True, False],
        "sex": ["male", "male", "female", "female", "male"],
    })
    metadata_df.to_csv(data_dir / "metadata.csv", index=False)

    # 4. training_supp.csv (4 out of 5 present to test join coverage)
    supp_df = pd.DataFrame({
        "isic_id": ["ISIC_0001", "ISIC_0002", "ISIC_0003", "ISIC_0004"],
        "diagnosis_full": ["Melanoma in situ", "Melanoma invasive", "Melanocytic nevus", "Actinic keratosis"],
        "diagnosis_confirm_type": ["histopathology", "histopathology", "consensus", "histopathology"],
        "invasion_thickness_interval": ["0.0", "0.5-1.0", None, None],
    })
    supp_df.to_csv(data_dir / "training_supp.csv", index=False)

    return data_dir


@pytest.fixture
def config_path():
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "configs" / "data.yaml"


def test_validator_clean_dataset(mock_dataset_dir, config_path):
    """Test validation on a valid dataset produces 0 errors and generates JSON report."""
    report_json = mock_dataset_dir / "report.json"
    validator = DatasetValidator(
        config=config_path,
        data_root=mock_dataset_dir,
        strict=False,
    )
    report = validator.run_validation(output_json=report_json)

    assert report["status"] == "PASSED"
    assert report["total_errors"] == 0
    assert report_json.is_file()

    # Verify Joins
    assert "join_1" in report["joins"]
    assert report["joins"]["join_1"]["common_ids"] == 3
    assert report["joins"]["join_1"]["missing_in_right_count"] == 0

    assert "join_2" in report["joins"]
    assert report["joins"]["join_2"]["coverage_pct"] == 100.0

    assert "join_3" in report["joins"]
    # 4 out of 5 matched in supplementary
    assert report["joins"]["join_3"]["common_ids"] == 4
    assert report["joins"]["join_3"]["missing_in_right_count"] == 1


def test_validator_missing_required_column(mock_dataset_dir, config_path):
    """Test check 1: missing required column produces validation error."""
    # Remove 'sex' from training_input.csv
    inp_file = mock_dataset_dir / "training_input.csv"
    df = pd.read_csv(inp_file)
    df = df.drop(columns=["sex"])
    df.to_csv(inp_file, index=False)

    validator = DatasetValidator(config=config_path, data_root=mock_dataset_dir, strict=False)
    report = validator.run_validation()

    assert report["status"] == "FAILED"
    assert any("missing required columns: ['sex']" in e for e in report["errors"])


def test_validator_invalid_one_hot(mock_dataset_dir, config_path):
    """Test check 9: non-binary value in one-hot columns produces error."""
    gt_file = mock_dataset_dir / "training_gt.csv"
    df = pd.read_csv(gt_file)
    df.loc[0, "MEL"] = 5  # Invalid value!
    df.to_csv(gt_file, index=False)

    validator = DatasetValidator(config=config_path, data_root=mock_dataset_dir, strict=False)
    report = validator.run_validation()

    assert report["status"] == "FAILED"
    assert any("one-hot column 'MEL' contains invalid non-binary values" in e for e in report["errors"])


def test_validator_duplicate_primary_keys(mock_dataset_dir, config_path):
    """Test check 5: duplicate primary key values produce error."""
    inp_file = mock_dataset_dir / "training_input.csv"
    df = pd.read_csv(inp_file)
    df.loc[1, "isic_id"] = "ISIC_0001"  # Duplicate!
    df.to_csv(inp_file, index=False)

    validator = DatasetValidator(config=config_path, data_root=mock_dataset_dir, strict=False)
    report = validator.run_validation()

    assert report["status"] == "FAILED"
    assert any("primary key 'isic_id' contains 1 duplicate values" in e for e in report["errors"])


def test_validator_strict_mode_raises(mock_dataset_dir, config_path):
    """Verify strict mode raises DataValidationError when errors are present."""
    # Delete a required file
    (mock_dataset_dir / "training_gt.csv").unlink()

    validator = DatasetValidator(config=config_path, data_root=mock_dataset_dir, strict=True)
    with pytest.raises(DataValidationError):
        validator.run_validation()
