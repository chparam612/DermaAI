"""Tests for frontend F4 panel report summary JSON."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_frontend_report_summary_integrity():
    report_file = PROJECT_ROOT / "results" / "frontend_report_summary.json"
    assert report_file.is_file(), "results/frontend_report_summary.json does not exist"

    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Required top-level blocks
    for block in ["status", "provenance", "overall_test_metrics", "malignant_vs_benign_metrics", "per_class_metrics", "skin_tone_subgroup_metrics", "caveats"]:
        assert block in data, f"Missing block '{block}'"

    # Source traceability in each block
    for block in ["provenance", "overall_test_metrics", "malignant_vs_benign_metrics", "per_class_metrics", "skin_tone_subgroup_metrics", "caveats"]:
        assert "source" in data[block], f"Block '{block}' missing 'source' field"

    # Provenance fields
    prov = data["provenance"]
    assert prov["seed"] == 42
    assert prov["split_sha256"] == "a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36"
    assert prov["total_test_samples"] == 786

    # Overall test metrics
    overall = data["overall_test_metrics"]
    assert overall["macro_f1"] == 0.469
    assert overall["accuracy"] == 0.7214
    assert overall["balanced_accuracy"] == 0.4724

    # Malignant metrics
    mal = data["malignant_vs_benign_metrics"]
    assert mal["sensitivity"] == 0.8885
    assert mal["specificity"] == 0.6552
    assert mal["false_negative_count"] == 65
    assert mal["false_positive_count"] == 70
    assert mal["true_positive_count"] + mal["false_negative_count"] == 583
    assert mal["true_negative_count"] + mal["false_positive_count"] == 203

    # Per-class metrics
    per_cls = data["per_class_metrics"]["classes"]
    assert len(per_cls) == 11
    assert "BCC" in per_cls
    assert per_cls["BCC"]["support"] == 391
    assert per_cls["BCC"]["recall"] == 0.9105

    # Subgroups & low_confidence flag
    subgroups = data["skin_tone_subgroup_metrics"]["subgroups"]
    assert "skin_tone_1" in subgroups
    assert subgroups["skin_tone_1"]["sample_count"] == 12
    assert subgroups["skin_tone_1"]["low_confidence"] is True
    assert subgroups["skin_tone_3"]["sample_count"] == 486
    assert subgroups["skin_tone_3"]["low_confidence"] is False

    # Caveats
    cav = data["caveats"]
    assert "seed_variance" in cav
    assert "0.1054" in cav["seed_variance"] or "0.105" in cav["seed_variance"]
    assert "53.6%" in cav["published_benchmark_incomparability"]
    assert cav["tail_class_supports"] == {
        "BEN_OTH": 8,
        "DF": 5,
        "INF": 5,
        "MAL_OTH": 2,
        "VASC": 6
    }
