"""Utility script to verify and package files for Kaggle migration."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import zipfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_kaggle_upload")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def verify_assets() -> bool:
    """Verify that all required assets exist locally before Kaggle migration."""
    logger.info("=" * 70)
    logger.info("VERIFYING LOCAL ASSETS FOR KAGGLE MIGRATION")
    logger.info("=" * 70)

    # 1. Weights
    ckpt = PROJECT_ROOT / "checkpoints" / "panderm_base.pth"
    if not ckpt.is_file():
        logger.error(f"[X] PanDerm weights missing: {ckpt}")
        return False
    logger.info(f"[OK] PanDerm weights: {ckpt} ({ckpt.stat().st_size / 1e6:.1f} MB)")

    # 2. Splits
    splits = PROJECT_ROOT / "data" / "splits" / "lesion_level_splits.csv"
    if not splits.is_file():
        logger.error(f"[X] Splits CSV missing: {splits}")
        return False
    logger.info(f"[OK] Splits CSV: {splits} ({splits.stat().st_size / 1e6:.1f} MB)")

    # 3. Label mapping
    label_map = PROJECT_ROOT / "configs" / "label_mapping.json"
    if not label_map.is_file():
        logger.error(f"[X] Label mapping JSON missing: {label_map}")
        return False
    logger.info(f"[OK] Label mapping: {label_map}")

    # 4. Images
    img_dir = PROJECT_ROOT / "data" / "images"
    if not img_dir.is_dir():
        logger.error(f"[X] Images directory missing: {img_dir}")
        return False
    img_count = len(list(img_dir.glob("*.jpg")))
    logger.info(f"[OK] Images directory: {img_dir} ({img_count} images found)")

    # 5. Notebook
    nb = PROJECT_ROOT / "notebooks" / "kaggle_exp2_panderm_v2.ipynb"
    if not nb.is_file():
        logger.error(f"[X] Kaggle notebook missing: {nb}")
        return False
    logger.info(f"[OK] Kaggle notebook: {nb}")

    # 6. Kaggle YAML config
    cfg = PROJECT_ROOT / "configs" / "experiments" / "exp2_panderm_kaggle.yaml"
    if not cfg.is_file():
        logger.error(f"[X] Kaggle config missing: {cfg}")
        return False
    logger.info(f"[OK] Kaggle config: {cfg}")

    logger.info("=" * 70)
    logger.info("ALL ASSETS VERIFIED SUCCESSFULLY.")
    logger.info("=" * 70)
    return True


def package_codebase(output_zip: Path) -> None:
    """Package the Python codebase (excluding large data/checkpoints/unrelated submodules) into a clean zip."""
    logger.info(f"Packaging codebase into {output_zip}...")
    exclude_dirs = {
        "__pycache__",
        "__snapshots__",
        ".pytest_cache",
        ".git",
        "wandb",
        "checkpoints",
        ".antigravity",
        "skincancertracker.egg-info",
        "dermfm-zero",
    }

    count = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in PROJECT_ROOT.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(PROJECT_ROOT)
                dir_parts = set(rel_path.parent.parts)
                # Exclude blacklisted directories or any dunder directory (__something__) at directory level
                if dir_parts.intersection(exclude_dirs) or any(p.startswith("__") and p.endswith("__") for p in dir_parts):
                    continue
                # Do not package zip files (e.g. codebase.zip or milk10k_dataset.zip) inside codebase.zip
                if file_path.suffix == ".zip":
                    continue
                # Exclude top-level data/ raw images and large arrays, but strictly keep data/splits/lesion_level_splits.csv
                # Crucial: Check rel_path.parts[0] so that src/data is NOT accidentally matched!
                if rel_path.parts[0] == "data":
                    if not (rel_path.as_posix().startswith("data/splits")):
                        continue
                # Exclude top-level results/ temporary model outputs, but keep Exp-1 and Exp-2 summary metrics JSONs
                # Crucial: Check rel_path.parts[0] so that any src/results would NOT accidentally be matched!
                if rel_path.parts[0] == "results":
                    if not (file_path.suffix == ".json" and ("exp1" in rel_path.as_posix() or "exp2" in rel_path.as_posix())):
                        continue
                # Exclude binary media/weights unless in external/panderm/classification or configs
                if file_path.suffix in {".pt", ".pth", ".wandb", ".jpg", ".png", ".tar", ".gz"}:
                    if "classification" not in rel_path.as_posix() and "configs" not in rel_path.as_posix():
                        continue
                zf.write(file_path, arcname=rel_path.as_posix())
                count += 1
    logger.info(f"Packaged {count} files ({output_zip.stat().st_size / 1e6:.2f} MB) to {output_zip}")


def verify_package_integrity(zip_path: Path) -> bool:
    """Extract zip to clean temporary directory and run pytest and import assertions against it."""
    logger.info("=" * 70)
    logger.info("VERIFYING EXTRACTED PACKAGE INTEGRITY & TESTS")
    logger.info("=" * 70)
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_root)

        # 1. Verify critical src packages and files exist
        required_modules = [
            temp_root / "src" / "__init__.py",
            temp_root / "src" / "data" / "__init__.py",
            temp_root / "src" / "data" / "datasets.py",
            temp_root / "src" / "data" / "loaders.py",
            temp_root / "src" / "data" / "transforms.py",
            temp_root / "src" / "data" / "validator.py",
            temp_root / "src" / "models" / "__init__.py",
            temp_root / "src" / "models" / "panderm.py",
            temp_root / "src" / "train" / "__init__.py",
            temp_root / "src" / "train" / "train_experiment.py",
            temp_root / "src" / "eval" / "__init__.py",
            temp_root / "src" / "eval" / "evaluate.py",
            temp_root / "src" / "losses" / "__init__.py",
            temp_root / "src" / "utils" / "__init__.py",
            temp_root / "configs" / "label_mapping.json",
            temp_root / "configs" / "experiments" / "exp2_panderm_kaggle.yaml",
            temp_root / "data" / "splits" / "lesion_level_splits.csv",
            temp_root / "results" / "exp1_baseline" / "summary_metrics.json",
            temp_root / "results" / "exp2_panderm" / "summary_metrics.json",
        ]
        for mod in required_modules:
            if not mod.is_file():
                logger.error(f"[X] Critical module missing in extracted package: {mod.relative_to(temp_root)}")
                return False
            logger.info(f"[OK] Verified packaged file: {mod.relative_to(temp_root)}")

        # 2. Check no __snapshots__ or dermfm-zero exists
        for p in temp_root.rglob("*"):
            if "__snapshots__" in p.parts:
                logger.error(f"[X] Reserved __snapshots__ directory found in package: {p}")
                return False
            if "dermfm-zero" in p.parts:
                logger.error(f"[X] Unrelated dermfm-zero file found in package: {p}")
                return False

        # 3. Run pytest inside extracted temporary directory
        logger.info("Running pytest against extracted package in isolated tempdir...")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(temp_root)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(temp_root / "tests")],
            cwd=str(temp_root),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(f"[X] PyTest failed inside extracted package:\n{result.stdout}\n{result.stderr}")
            return False
        logger.info("[OK] All tests PASSED inside extracted package!")
        logger.info(result.stdout.strip().split("\n")[-1])

    logger.info("=" * 70)
    logger.info("PACKAGE INTEGRITY VERIFIED: ZERO PACKAGING REGRESSIONS DETECTED.")
    logger.info("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and verify assets for Kaggle migration.")
    parser.add_argument("--package-code", action="store_true", help="Create a clean codebase.zip for Kaggle dataset upload")
    args = parser.parse_args()

    success = verify_assets()
    if success and args.package_code:
        zip_path = PROJECT_ROOT / "codebase.zip"
        package_codebase(zip_path)
        valid = verify_package_integrity(zip_path)
        if not valid:
            logger.error("Package verification FAILED.")
            exit(1)
