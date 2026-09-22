"""Unit tests for EDA calculation and visualization module."""

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless tests
from pathlib import Path
import pandas as pd
import pytest

from src.data.eda import (
    load_dataset_tables,
    get_dataset_overview,
    get_class_distribution,
    plot_class_distribution,
    plot_images_per_lesion_distribution,
    DIAGNOSTIC_CLASSES_11,
)


@pytest.fixture
def sample_eda_data():
    """Generates synthetic DataFrame dictionary for testing EDA functions."""
    input_df = pd.DataFrame({
        "isic_id": ["ISIC_1", "ISIC_2", "ISIC_3", "ISIC_4", "ISIC_5"],
        "lesion_id": ["L1", "L1", "L1", "L2", "L3"],
        "sex": ["male", "male", "female", "female", "male"],
    })
    gt_df = pd.DataFrame({
        "lesion_id": ["L1", "L1", "L1", "L2", "L3"],
        "AKIEC": [0, 0, 0, 1, 0],
        "BCC": [0, 0, 0, 0, 0],
        "BEN_OTH": [0, 0, 0, 0, 0],
        "BKL": [0, 0, 0, 0, 0],
        "DF": [0, 0, 0, 0, 0],
        "INF": [0, 0, 0, 0, 0],
        "MAL_OTH": [0, 0, 0, 0, 0],
        "MEL": [1, 1, 1, 0, 0],
        "NV": [0, 0, 0, 0, 1],
        "SCCKA": [0, 0, 0, 0, 0],
        "VASC": [0, 0, 0, 0, 0],
    })
    return {"training_input": input_df, "training_gt": gt_df}


def test_get_dataset_overview(sample_eda_data):
    """Test overview calculation on row counts, image/lesion cardinality, and images per lesion."""
    overview = get_dataset_overview(sample_eda_data)
    assert overview["row_counts"]["training_input"] == 5
    assert overview["row_counts"]["training_gt"] == 5

    assert overview["images"]["training_input"]["unique_isic_ids"] == 5
    assert overview["lesions"]["training_input"]["unique_lesions"] == 3

    # Images per lesion: L1 has 3, L2 has 1, L3 has 1 -> max=3, min=1
    ipl = overview["images_per_lesion"]
    assert ipl["min"] == 1
    assert ipl["max"] == 3
    assert ipl["distribution_summary"]["1_image"] == 2
    assert ipl["distribution_summary"]["2_to_5_images"] == 1


def test_get_class_distribution(sample_eda_data):
    """Test 11-class diagnostic frequency calculation."""
    class_df, metrics = get_class_distribution(sample_eda_data["training_gt"])
    assert len(class_df) == 11
    assert "imbalance_ratio_max_to_min" in metrics
    # Check that MEL has 3 images and 1 unique lesion
    mel_row = class_df[class_df["class_name"] == "MEL"].iloc[0]
    assert mel_row["image_count"] == 3
    assert mel_row["percentage"] == 60.0
    assert mel_row["unique_lesions"] == 1

    # Check AKIEC has 1 image and 1 unique lesion
    akiec_row = class_df[class_df["class_name"] == "AKIEC"].iloc[0]
    assert akiec_row["image_count"] == 1
    assert akiec_row["percentage"] == 20.0
    assert akiec_row["unique_lesions"] == 1


def test_plot_generation(sample_eda_data):
    """Verify matplotlib plot functions return valid figures without errors."""
    class_df, _ = get_class_distribution(sample_eda_data["training_gt"])
    fig1 = plot_class_distribution(class_df)
    assert fig1 is not None

    overview = get_dataset_overview(sample_eda_data)
    fig2 = plot_images_per_lesion_distribution(overview["images_per_lesion"])
    assert fig2 is not None
