"""Unit tests for relational join integrity and split output structure."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.validator import DatasetValidator
from src.data.splitting import verify_split_leakage


def test_join_integrity_on_real_or_fixture(tmp_path):
    """Test 4: Relational joins calculate correct coverage and detect missing records."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # training_input with 3 records (L1, L2, L3)
    pd.DataFrame({
        "isic_id": ["I1", "I2", "I3"],
        "lesion_id": ["L1", "L2", "L3"],
        "image_type": ["c", "c", "c"],
        "image_manipulation": ["n", "n", "n"],
        "age_approx": [30, 40, 50],
        "sex": ["m", "f", "m"],
        "skin_tone_class": [1, 2, 3],
    }).to_csv(data_dir / "training_input.csv", index=False)

    # training_gt missing L3 (only L1, L2)
    pd.DataFrame({
        "lesion_id": ["L1", "L2"],
        "AKIEC": [0, 0], "BCC": [1, 0], "BEN_OTH": [0, 0], "BKL": [0, 0],
        "DF": [0, 0], "INF": [0, 0], "MAL_OTH": [0, 0], "MEL": [0, 1],
        "NV": [0, 0], "SCCKA": [0, 0], "VASC": [0, 0],
    }).to_csv(data_dir / "training_gt.csv", index=False)

    pd.DataFrame({
        "isic_id": ["I1", "I2", "I3"],
        "lesion_id": ["L1", "L2", "L3"],
        "image_type": ["c", "c", "c"],
        "image_manipulation": ["n", "n", "n"],
        "melanocytic": [True, True, False],
        "sex": ["m", "f", "m"],
    }).to_csv(data_dir / "metadata.csv", index=False)

    validator = DatasetValidator(config="configs/data.yaml", data_root=data_dir, strict=False)
    report = validator.run_validation()

    # Join 1 warning: L3 missing in gt
    assert "join_1" in report["joins"]
    assert report["joins"]["join_1"]["common_ids"] == 2
    assert report["joins"]["join_1"]["missing_in_left_count"] == 1


def test_split_file_schema():
    """Test 9 & 10: Verify split file schema and non-leakage on disk."""
    split_file = Path("data/splits/lesion_level_splits.csv")
    if split_file.is_file():
        df = pd.read_csv(split_file)
        # Required columns
        assert "lesion_id" in df.columns
        assert "isic_id" in df.columns
        assert "split" in df.columns

        # Verify no lesion leakage
        verify_split_leakage(df)
