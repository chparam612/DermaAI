"""Unit tests for lesion-level data splitting and data leakage prevention."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.splitting import (
    create_lesion_level_splits,
    verify_split_leakage,
    compute_split_statistics,
    DataLeakageError,
    VALID_SPLIT_NAMES,
)


@pytest.fixture
def mock_dataset_for_splitting(tmp_path):
    """Generates synthetic dataset directory with multiple images per lesion."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # 10 lesions, 25 total images
    # L1: 4 images, L2: 3 images, L3: 3 images, L4..L10: 1-3 images
    rows = []
    for i in range(1, 11):
        lesion_id = f"LES_{i:02d}"
        n_imgs = (i % 3) + 1
        for j in range(n_imgs):
            rows.append({
                "isic_id": f"ISIC_{i:02d}_{j:02d}",
                "lesion_id": lesion_id,
                "skin_tone_class": (i % 5),
                "image_type": "dermoscopic",
                "sex": "male" if i % 2 == 0 else "female",
            })
    df_inp = pd.DataFrame(rows)
    df_inp.to_csv(data_dir / "training_input.csv", index=False)

    # Ground truth (1 row per lesion)
    gt_rows = []
    for i in range(1, 11):
        lesion_id = f"LES_{i:02d}"
        gt_rows.append({
            "lesion_id": lesion_id,
            "MEL": 1 if i <= 3 else 0,
            "NV": 1 if i > 3 else 0,
        })
    df_gt = pd.DataFrame(gt_rows)
    df_gt.to_csv(data_dir / "training_gt.csv", index=False)

    return data_dir


def test_split_no_lesion_overlap(mock_dataset_for_splitting):
    """TEST 1 — NO LESION OVERLAP: Assert strict disjointness between all partitions."""
    splits_df, stats = create_lesion_level_splits(
        data_root=mock_dataset_for_splitting,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_lesions = set(splits_df.loc[splits_df["split"] == "train", "lesion_id"].unique())
    validation_lesions = set(splits_df.loc[splits_df["split"] == "validation", "lesion_id"].unique())
    test_lesions = set(splits_df.loc[splits_df["split"] == "test", "lesion_id"].unique())

    assert train_lesions.isdisjoint(validation_lesions), "Lesion leakage between train and validation!"
    assert train_lesions.isdisjoint(test_lesions), "Lesion leakage between train and test!"
    assert validation_lesions.isdisjoint(test_lesions), "Lesion leakage between validation and test!"


def test_split_every_record_assigned_once(mock_dataset_for_splitting):
    """TEST 2 — EVERY RECORD ASSIGNED ONCE: No images missing, no duplicates."""
    raw_inp = pd.read_csv(mock_dataset_for_splitting / "training_input.csv")
    splits_df, _ = create_lesion_level_splits(
        data_root=mock_dataset_for_splitting,
        seed=42,
    )

    assert len(splits_df) == len(raw_inp), "Total split records do not match raw records count!"
    assert splits_df["isic_id"].is_unique, "Duplicate isic_id found in split output!"
    assert splits_df["isic_id"].isna().sum() == 0, "Null isic_id found in split output!"
    assert set(splits_df["isic_id"]) == set(raw_inp["isic_id"]), "Some images are missing from splits!"
    assert splits_df["split"].notna().all(), "Some records have unassigned split values!"


def test_split_every_lesion_assigned_once(mock_dataset_for_splitting):
    """TEST 3 — EVERY LESION ASSIGNED ONCE: Every lesion in exactly one split."""
    raw_inp = pd.read_csv(mock_dataset_for_splitting / "training_input.csv")
    splits_df, _ = create_lesion_level_splits(
        data_root=mock_dataset_for_splitting,
        seed=42,
    )

    raw_lesions = set(raw_inp["lesion_id"])
    split_lesions = set(splits_df["lesion_id"])
    assert raw_lesions == split_lesions, "Some lesions were dropped or added during split!"

    # Group by lesion_id and assert each lesion has exactly 1 unique split
    splits_per_lesion = splits_df.groupby("lesion_id")["split"].nunique()
    assert (splits_per_lesion == 1).all(), "Some lesions appear in multiple splits!"


def test_split_valid_split_values(mock_dataset_for_splitting):
    """TEST 4 — VALID SPLIT VALUES: Only 'train', 'validation', 'test' allowed."""
    splits_df, _ = create_lesion_level_splits(
        data_root=mock_dataset_for_splitting,
        seed=42,
    )
    actual_values = set(splits_df["split"].unique())
    assert actual_values.issubset(VALID_SPLIT_NAMES)


def test_split_reproducibility(mock_dataset_for_splitting):
    """TEST 5 — REPRODUCIBILITY: Running twice with same seed produces identical splits."""
    splits_df1, _ = create_lesion_level_splits(
        data_root=mock_dataset_for_splitting,
        seed=123,
    )
    splits_df2, _ = create_lesion_level_splits(
        data_root=mock_dataset_for_splitting,
        seed=123,
    )
    pd.testing.assert_frame_equal(splits_df1, splits_df2)


def test_split_approximate_ratios(mock_dataset_for_splitting):
    """TEST 6 — APPROXIMATE SPLIT RATIOS: Partition fractions within documented tolerance."""
    _, stats = create_lesion_level_splits(
        data_root=mock_dataset_for_splitting,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    # With 10 lesions, integer rounding allows +/- 15% tolerance on small mock fixture
    train_pct = stats["partitions"]["train"]["lesion_pct"]
    assert 50.0 <= train_pct <= 90.0


def test_leakage_detector_catches_overlap():
    """Verify that verify_split_leakage raises DataLeakageError when leakage exists."""
    leaky_df = pd.DataFrame({
        "isic_id": ["IMG1", "IMG2", "IMG3"],
        "lesion_id": ["L1", "L1", "L2"],
        "split": ["train", "test", "test"],  # L1 leaked into train AND test!
    })
    with pytest.raises(DataLeakageError) as exc:
        verify_split_leakage(leaky_df)
    assert "Leakage between TRAIN and TEST" in str(exc.value)


def test_actual_generated_splits_leakage():
    """Smoke test on actual generated lesion_level_splits.csv if it exists."""
    split_file = Path("data/splits/lesion_level_splits.csv")
    if split_file.is_file():
        df = pd.read_csv(split_file)
        verify_split_leakage(df)
        assert len(df) == 10480
        assert df["lesion_id"].nunique() == 5240
