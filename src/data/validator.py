"""Dataset Schema and Relational Integrity Validator for SkinCancerTracker.

Implements rigorous 10-point schema integrity checks and multi-file relational
join validation across ISIC multimodal inputs, ground truth, metadata, and
supplementary tables.

Strict Rule:
Does not automatically rename columns. Reports spelling discrepancies and unexpected
columns as errors or warnings while preserving upstream data integrity.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("validator")


class DataValidationError(Exception):
    """Raised when strict validation criteria are violated."""
    pass


class DatasetValidator:
    """Validates schemas, data types, identifiers, and relational joins for dermatology datasets."""

    def __init__(
        self,
        config: Union[Dict[str, Any], str, Path],
        data_root: Optional[Union[str, Path]] = None,
        strict: bool = False,
    ):
        """Initialize the DatasetValidator.

        Args:
            config: Configuration dictionary or path to configs/data.yaml.
            data_root: Optional directory path containing dataset files (overrides config).
            strict: If True, validation failure raises DataValidationError and triggers exit code 1.
        """
        if isinstance(config, (str, Path)):
            cfg_path = Path(config).resolve()
            if not cfg_path.is_file():
                raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
            with open(cfg_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = dict(config)

        # Resolve dataset root directory
        configured_root = data_root or self.config.get("dataset", {}).get("data_root", "data")
        self.data_root = Path(configured_root).resolve()
        self.strict = strict or self.config.get("validation", {}).get("strict", False)

        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.file_summaries: Dict[str, Dict[str, Any]] = {}
        self.join_summaries: Dict[str, Dict[str, Any]] = {}
        self.loaded_dataframes: Dict[str, pd.DataFrame] = {}

    def run_validation(self, output_json: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Execute all 10 schema checks and all relational join validations.

        Args:
            output_json: Optional path to serialize machine-readable validation report.

        Returns:
            Dictionary containing the complete validation report.
        """
        self.errors.clear()
        self.warnings.clear()
        self.file_summaries.clear()
        self.join_summaries.clear()
        self.loaded_dataframes.clear()

        # Step 1: Validate file existence and load DataFrames
        self._check_and_load_files()

        # Step 2: Validate each file schema, types, identifiers, nulls, one-hot labels
        for file_key, df in self.loaded_dataframes.items():
            file_cfg = self.config.get("files", {}).get(file_key, {})
            self._validate_file_schema(file_key, df, file_cfg)

        # Step 3: Validate Relational Joins
        self._validate_joins()

        # Step 4: Compile report
        report = self._build_report()

        # Step 5: Save JSON report if output path is provided
        target_out = output_json or self.config.get("validation", {}).get("report_output")
        if target_out:
            out_path = Path(target_out).resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Validation report saved to: {out_path}")

        # Step 6: Enforce strict failure if errors detected
        if self.strict and len(self.errors) > 0:
            raise DataValidationError(
                f"Strict validation failed with {len(self.errors)} error(s):\n  - "
                + "\n  - ".join(self.errors)
            )

        return report

    def _check_and_load_files(self) -> None:
        """Check 10: Verify every required dataset file exists, and load CSVs into memory."""
        files_cfg = self.config.get("files", {})

        for file_key, spec in files_cfg.items():
            filename = spec.get("filename", f"{file_key}.csv")
            filepath = self.data_root / filename
            required = spec.get("required", True)

            if not filepath.exists():
                msg = f"Dataset file missing: '{filepath.name}' at path {filepath}"
                if required:
                    self.errors.append(msg)
                else:
                    self.warnings.append(f"Optional dataset file missing: '{filepath.name}'")
                continue

            try:
                df = pd.read_csv(filepath, low_memory=False)
                self.loaded_dataframes[file_key] = df
            except Exception as e:
                self.errors.append(f"Failed to read CSV '{filepath.name}': {e}")

    def _validate_file_schema(self, file_key: str, df: pd.DataFrame, spec: Dict[str, Any]) -> None:
        """Execute schema checks 1-9 for a loaded DataFrame."""
        actual_cols = list(df.columns)
        required_cols = spec.get("required_columns", [])
        one_hot_cols = spec.get("one_hot_columns", [])
        all_expected_cols = set(required_cols + one_hot_cols + spec.get("optional_columns", []))
        primary_key = spec.get("primary_key")
        group_key = spec.get("group_key")

        # 1. Required columns exist
        missing_required = [c for c in required_cols if c not in actual_cols]
        if missing_required:
            self.errors.append(
                f"File '{file_key}' missing required columns: {missing_required}"
            )

        missing_one_hot = [c for c in one_hot_cols if c not in actual_cols]
        if missing_one_hot:
            self.errors.append(
                f"File '{file_key}' missing one-hot label columns: {missing_one_hot}"
            )

        # 2. Unexpected columns are reported
        unexpected_cols = [c for c in actual_cols if c not in all_expected_cols]
        if unexpected_cols:
            self.warnings.append(
                f"File '{file_key}' contains unexpected/unregistered columns: {unexpected_cols}"
            )

        # 3. Column names spelled correctly (Check spelling, do NOT automatically rename)
        for act_col in actual_cols:
            if act_col not in all_expected_cols:
                close_matches = difflib.get_close_matches(act_col, list(all_expected_cols), n=1, cutoff=0.75)
                if close_matches:
                    self.warnings.append(
                        f"File '{file_key}' column '{act_col}' might be misspelled. Did you mean '{close_matches[0]}'? "
                        "(Note: Automatic renaming is disabled; please correct source data)."
                    )

        # 4 & 5. Required identifiers not missing & Uniqueness properties
        unique_pk_count = None
        duplicate_pk_count = 0
        null_pk_count = 0

        if primary_key and primary_key in df.columns:
            pk_series = df[primary_key]
            null_pk_count = int(pk_series.isna().sum())
            if null_pk_count > 0:
                self.errors.append(
                    f"File '{file_key}' primary key '{primary_key}' contains {null_pk_count} null/missing values."
                )

            # Check uniqueness
            duplicate_pk_count = int(pk_series.duplicated().sum())
            unique_pk_count = int(pk_series.nunique())
            if duplicate_pk_count > 0:
                self.errors.append(
                    f"File '{file_key}' primary key '{primary_key}' contains {duplicate_pk_count} duplicate values. "
                    f"(Total rows: {len(df)}, Unique: {unique_pk_count})."
                )

        # Group identifier (lesion_id) properties
        unique_group_count = None
        duplicate_group_count = 0
        null_group_count = 0
        if group_key and group_key in df.columns:
            grp_series = df[group_key]
            null_group_count = int(grp_series.isna().sum())
            if null_group_count > 0:
                self.errors.append(
                    f"File '{file_key}' group identifier '{group_key}' contains {null_group_count} null values."
                )
            unique_group_count = int(grp_series.nunique())
            duplicate_group_count = int(grp_series.duplicated().sum())

        # 6. Data types are reasonable
        type_anomalies: List[str] = []
        for col in actual_cols:
            dtype_str = str(df[col].dtype)
            # Check ID columns are not floats
            if col in [primary_key, group_key] and "float" in dtype_str:
                self.warnings.append(
                    f"File '{file_key}' identifier '{col}' has float dtype '{dtype_str}'. String or integer expected."
                )

        # 7. Duplicate rows
        exact_duplicate_rows = int(df.duplicated().sum())
        if exact_duplicate_rows > 0:
            self.warnings.append(
                f"File '{file_key}' contains {exact_duplicate_rows} exact duplicate rows."
            )

        # 8. Missing values summary
        missing_counts = {col: int(df[col].isna().sum()) for col in actual_cols}
        missing_pcts = {col: round(float(df[col].isna().mean() * 100), 2) for col in actual_cols}

        # 9. One-hot label columns contain valid binary values {0, 1}
        invalid_one_hot_cols: Dict[str, List[Any]] = {}
        if one_hot_cols:
            for oh_col in one_hot_cols:
                if oh_col in df.columns:
                    col_vals = set(df[oh_col].dropna().unique())
                    # Valid values are subset of {0, 1} or {True, False}
                    if not col_vals.issubset({0, 1, 0.0, 1.0, True, False}):
                        invalid_one_hot_cols[oh_col] = list(col_vals)
                        self.errors.append(
                            f"File '{file_key}' one-hot column '{oh_col}' contains invalid non-binary values: {list(col_vals)}"
                        )

            # Check mutually exclusive or multi-label properties
            available_oh = [c for c in one_hot_cols if c in df.columns]
            if available_oh:
                row_sums = df[available_oh].sum(axis=1)
                unlabeled_count = int((row_sums == 0).sum())
                multi_labeled_count = int((row_sums > 1).sum())
                if unlabeled_count > 0:
                    self.warnings.append(
                        f"File '{file_key}' has {unlabeled_count} rows with zero active one-hot labels."
                    )
                if multi_labeled_count > 0:
                    self.warnings.append(
                        f"File '{file_key}' has {multi_labeled_count} rows with multiple active one-hot labels."
                    )

        # Save file summary
        self.file_summaries[file_key] = {
            "row_count": len(df),
            "column_count": len(actual_cols),
            "columns": actual_cols,
            "primary_key": primary_key,
            "unique_primary_keys": unique_pk_count,
            "duplicate_primary_keys": duplicate_pk_count,
            "null_primary_keys": null_pk_count,
            "group_key": group_key,
            "unique_group_keys": unique_group_count,
            "duplicate_group_keys": duplicate_group_count,
            "null_group_keys": null_group_count,
            "exact_duplicate_rows": exact_duplicate_rows,
            "missing_value_counts": missing_counts,
            "missing_value_pct": missing_pcts,
            "one_hot_columns": one_hot_cols,
        }

    def _validate_joins(self) -> None:
        """Validate multi-file referential integrity for JOIN 1, JOIN 2, and JOIN 3."""
        # ----------------------------------------------------------------------
        # JOIN 1: training_gt.lesion_id must match training_input.lesion_id
        # ----------------------------------------------------------------------
        if "training_gt" in self.loaded_dataframes and "training_input" in self.loaded_dataframes:
            df_gt = self.loaded_dataframes["training_gt"]
            df_inp = self.loaded_dataframes["training_input"]

            if "lesion_id" in df_gt.columns and "lesion_id" in df_inp.columns:
                gt_lesions = set(df_gt["lesion_id"].dropna().unique())
                inp_lesions = set(df_inp["lesion_id"].dropna().unique())

                missing_in_inp = gt_lesions - inp_lesions
                missing_in_gt = inp_lesions - gt_lesions
                common_lesions = gt_lesions & inp_lesions
                total_lesions = gt_lesions | inp_lesions

                coverage = (len(common_lesions) / len(total_lesions) * 100) if total_lesions else 100.0

                join_1_info = {
                    "join_name": "JOIN 1: training_gt.lesion_id <-> training_input.lesion_id",
                    "left_file": "training_gt",
                    "right_file": "training_input",
                    "key": "lesion_id",
                    "left_unique_ids": len(gt_lesions),
                    "right_unique_ids": len(inp_lesions),
                    "common_ids": len(common_lesions),
                    "missing_in_right_count": len(missing_in_inp),
                    "missing_in_left_count": len(missing_in_gt),
                    "missing_in_right_sample": list(missing_in_inp)[:5],
                    "missing_in_left_sample": list(missing_in_gt)[:5],
                    "coverage_pct": round(coverage, 2),
                    "left_duplicates": int(df_gt["lesion_id"].duplicated().sum()),
                    "right_duplicates": int(df_inp["lesion_id"].duplicated().sum()),
                    "left_nulls": int(df_gt["lesion_id"].isna().sum()),
                    "right_nulls": int(df_inp["lesion_id"].isna().sum()),
                }
                self.join_summaries["join_1"] = join_1_info

                if missing_in_inp:
                    self.errors.append(
                        f"JOIN 1 Error: {len(missing_in_inp)} lesion IDs present in training_gt are missing from training_input."
                    )
                if missing_in_gt:
                    self.warnings.append(
                        f"JOIN 1 Warning: {len(missing_in_gt)} lesion IDs present in training_input are missing from training_gt."
                    )

        # ----------------------------------------------------------------------
        # JOIN 2: training_input.isic_id must match metadata.isic_id
        # ----------------------------------------------------------------------
        if "training_input" in self.loaded_dataframes and "metadata" in self.loaded_dataframes:
            df_inp = self.loaded_dataframes["training_input"]
            df_meta = self.loaded_dataframes["metadata"]

            if "isic_id" in df_inp.columns and "isic_id" in df_meta.columns:
                inp_ids = set(df_inp["isic_id"].dropna().unique())
                meta_ids = set(df_meta["isic_id"].dropna().unique())

                missing_in_meta = inp_ids - meta_ids
                missing_in_inp = meta_ids - inp_ids
                common_ids = inp_ids & meta_ids
                total_ids = inp_ids | meta_ids
                coverage = (len(common_ids) / len(total_ids) * 100) if total_ids else 100.0

                join_2_info = {
                    "join_name": "JOIN 2: training_input.isic_id <-> metadata.isic_id",
                    "left_file": "training_input",
                    "right_file": "metadata",
                    "key": "isic_id",
                    "left_unique_ids": len(inp_ids),
                    "right_unique_ids": len(meta_ids),
                    "common_ids": len(common_ids),
                    "missing_in_right_count": len(missing_in_meta),
                    "missing_in_left_count": len(missing_in_inp),
                    "missing_in_right_sample": list(missing_in_meta)[:5],
                    "missing_in_left_sample": list(missing_in_inp)[:5],
                    "coverage_pct": round(coverage, 2),
                    "left_duplicates": int(df_inp["isic_id"].duplicated().sum()),
                    "right_duplicates": int(df_meta["isic_id"].duplicated().sum()),
                    "left_nulls": int(df_inp["isic_id"].isna().sum()),
                    "right_nulls": int(df_meta["isic_id"].isna().sum()),
                }
                self.join_summaries["join_2"] = join_2_info

                if missing_in_meta:
                    self.errors.append(
                        f"JOIN 2 Error: {len(missing_in_meta)} image IDs in training_input are missing from metadata."
                    )
                if missing_in_inp:
                    self.warnings.append(
                        f"JOIN 2 Warning: {len(missing_in_inp)} image IDs in metadata are missing from training_input."
                    )

        # ----------------------------------------------------------------------
        # JOIN 3: training_input.isic_id must match training_supp.isic_id
        # ----------------------------------------------------------------------
        if "training_input" in self.loaded_dataframes and "training_supp" in self.loaded_dataframes:
            df_inp = self.loaded_dataframes["training_input"]
            df_supp = self.loaded_dataframes["training_supp"]

            if "isic_id" in df_inp.columns and "isic_id" in df_supp.columns:
                inp_ids = set(df_inp["isic_id"].dropna().unique())
                supp_ids = set(df_supp["isic_id"].dropna().unique())

                missing_in_supp = inp_ids - supp_ids
                missing_in_inp = supp_ids - inp_ids
                common_ids = inp_ids & supp_ids
                total_ids = inp_ids | supp_ids
                coverage = (len(common_ids) / len(total_ids) * 100) if total_ids else 100.0

                join_3_info = {
                    "join_name": "JOIN 3: training_input.isic_id <-> training_supp.isic_id",
                    "left_file": "training_input",
                    "right_file": "training_supp",
                    "key": "isic_id",
                    "left_unique_ids": len(inp_ids),
                    "right_unique_ids": len(supp_ids),
                    "common_ids": len(common_ids),
                    "missing_in_right_count": len(missing_in_supp),
                    "missing_in_left_count": len(missing_in_inp),
                    "missing_in_right_sample": list(missing_in_supp)[:5],
                    "missing_in_left_sample": list(missing_in_inp)[:5],
                    "coverage_pct": round(coverage, 2),
                    "left_duplicates": int(df_inp["isic_id"].duplicated().sum()),
                    "right_duplicates": int(df_supp["isic_id"].duplicated().sum()),
                    "left_nulls": int(df_inp["isic_id"].isna().sum()),
                    "right_nulls": int(df_supp["isic_id"].isna().sum()),
                }
                self.join_summaries["join_3"] = join_3_info

                if missing_in_supp:
                    self.warnings.append(
                        f"JOIN 3 Warning: {len(missing_in_supp)} image IDs in training_input do not have supplementary records."
                    )
                if missing_in_inp:
                    self.warnings.append(
                        f"JOIN 3 Warning: {len(missing_in_inp)} image IDs in training_supp are missing from training_input."
                    )

    def _build_report(self) -> Dict[str, Any]:
        """Synthesize file metrics, joins, errors, and warnings into report dictionary."""
        return {
            "status": "PASSED" if len(self.errors) == 0 else "FAILED",
            "strict_mode": self.strict,
            "data_root": str(self.data_root),
            "total_errors": len(self.errors),
            "total_warnings": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "files": self.file_summaries,
            "joins": self.join_summaries,
        }

    def print_console_report(self, report: Optional[Dict[str, Any]] = None) -> None:
        """Print clean, human-readable summary of validation results to console."""
        rep = report or self._build_report()

        print("\n" + "=" * 80)
        print("SKINCANCERTRACKER — DATASET VALIDATION REPORT")
        print("=" * 80)
        print(f"Data Root:   {rep['data_root']}")
        print(f"Status:      {rep['status']} (Errors: {rep['total_errors']}, Warnings: {rep['total_warnings']})")
        print(f"Strict Mode: {rep['strict_mode']}")
        print("-" * 80)

        # File Summaries
        print("\nFILE SUMMARIES:")
        for fname, fmeta in rep.get("files", {}).items():
            print(f"  • {fname}:")
            print(f"      Rows: {fmeta['row_count']} | Columns: {fmeta['column_count']} | Exact Duplicate Rows: {fmeta['exact_duplicate_rows']}")
            if fmeta.get("primary_key"):
                print(f"      Primary Key ('{fmeta['primary_key']}'): Unique={fmeta['unique_primary_keys']}, Dupes={fmeta['duplicate_primary_keys']}, Nulls={fmeta['null_primary_keys']}")
            if fmeta.get("group_key"):
                print(f"      Group Key ('{fmeta['group_key']}'): Unique={fmeta['unique_group_keys']}, Dupes={fmeta['duplicate_group_keys']}, Nulls={fmeta['null_group_keys']}")
            
            # Print top missing columns if any
            high_missing = {k: v for k, v in fmeta["missing_value_pct"].items() if v > 0}
            if high_missing:
                miss_str = ", ".join([f"{k}: {v}% ({fmeta['missing_value_counts'][k]})" for k, v in list(high_missing.items())[:4]])
                print(f"      Missing Values: {miss_str}")

        # Relational Join Summaries
        print("\nRELATIONAL JOIN INTEGRITY:")
        for jkey, jmeta in rep.get("joins", {}).items():
            print(f"  • {jmeta['join_name']}:")
            print(f"      Key: '{jmeta['key']}' | Coverage: {jmeta['coverage_pct']}% | Common IDs: {jmeta['common_ids']}")
            print(f"      Left Unique: {jmeta['left_unique_ids']} (Dupes: {jmeta['left_duplicates']}, Nulls: {jmeta['left_nulls']})")
            print(f"      Right Unique: {jmeta['right_unique_ids']} (Dupes: {jmeta['right_duplicates']}, Nulls: {jmeta['right_nulls']})")
            if jmeta["missing_in_right_count"] > 0:
                print(f"      Missing in Right: {jmeta['missing_in_right_count']} (Samples: {jmeta['missing_in_right_sample']})")
            if jmeta["missing_in_left_count"] > 0:
                print(f"      Missing in Left: {jmeta['missing_in_left_count']} (Samples: {jmeta['missing_in_left_sample']})")

        # Warnings & Errors
        if rep["warnings"]:
            print("\nWARNINGS:")
            for w in rep["warnings"]:
                print(f"  [WARN] {w}")

        if rep["errors"]:
            print("\nERRORS:")
            for e in rep["errors"]:
                print(f"  [ERROR] {e}")

        print("\n" + "=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dataset validator."""
    parser = argparse.ArgumentParser(
        description="SkinCancerTracker Dataset Schema and Join Validator"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
        help="Path to directory containing dataset files (e.g. data/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data.yaml",
        help="Path to dataset configuration YAML (default: configs/data.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/validation/validation_report.json",
        help="Path to output JSON validation report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any validation errors are detected",
    )
    return parser.parse_args()


def main() -> None:
    """CLI Entry Point."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    validator = DatasetValidator(
        config=args.config,
        data_root=args.data_root,
        strict=args.strict,
    )

    try:
        report = validator.run_validation(output_json=args.output)
        validator.print_console_report(report)
        if args.strict and report["total_errors"] > 0:
            sys.exit(1)
    except DataValidationError as e:
        validator.print_console_report()
        logger.error(f"Strict validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validator encountered unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
