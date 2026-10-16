"""Generate and validate Kaggle execution notebooks for Exp-1, Exp-2c, and Exp-3b."""

import json
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_cell(source_lines, cell_type="code"):
    return {
        "cell_type": cell_type,
        "metadata": {},
        "outputs": [] if cell_type == "code" else undefined if False else [],
        "execution_count": None if cell_type == "code" else None,
        "source": [line + "\n" for line in source_lines],
    }


def build_exp1_notebook() -> dict:
    cells = []

    # Markdown Header
    cells.append(
        make_cell(
            [
                "# Exp-1: Standard CNN Baseline (ResNet-50) 30-Epoch Training (MILK10k Dataset)",
                "",
                "**Hardware Setting**:",
                "- **Accelerator**: GPU P100 (or GPU T4 x2)",
                "- **Internet**: ON",
                "- **Secrets**: Add `WANDB_API_KEY` under Notebook Settings -> Add-ons / Secrets",
                "",
                "---",
                "### Attached Kaggle Datasets (Required):",
                "1. `skincancertracker-code` -> `/kaggle/input/datasets/paramveerchaudhary/skincancertracker-code` (uploaded `codebase.zip`)",
                "2. `milk10k-dataset` -> `/kaggle/input/datasets/paramveerchaudhary/milk10k-dataset` (contains `images/` and `lesion_level_splits.csv`)",
            ],
            cell_type="markdown",
        )
    )

    # Cell 1: Hardware & Environment Verification (Gate: GPU Present)
    cells.append(
        make_cell(
            [
                "# Cell 1: Hardware & Environment Verification (Gate 1: GPU Present)",
                "import torch",
                "print('PyTorch Version:', torch.__version__)",
                "print('CUDA Available:', torch.cuda.is_available())",
                "assert torch.cuda.is_available(), 'GATE FAILED: CUDA GPU is not available! Please enable GPU accelerator in Kaggle notebook settings.'",
                "gpu_name = torch.cuda.get_device_name(0)",
                "vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)",
                "print(f'GPU Device: {gpu_name}')",
                "print(f'Total VRAM: {vram_gb:.2f} GB')",
                "assert vram_gb >= 14.0, f'GATE FAILED: Expected >= 14GB GPU (P100/T4), found {vram_gb:.2f} GB on {gpu_name}'",
                "print('[PASSED] GPU hardware gate verified.')",
            ]
        )
    )

    # Cell 2: Setup Codebase
    cells.append(
        make_cell(
            [
                "# Cell 2: Setup Codebase (Exclusively from uploaded dataset)",
                "import os, sys, shutil",
                "from pathlib import Path",
                "",
                'KAGGLE_USERNAME = "paramveerchaudhary"',
                'INPUT_BASE = Path(f"/kaggle/input/datasets/{KAGGLE_USERNAME}")',
                "",
                "# Fallback to flat /kaggle/input if dataset is attached in flat layout",
                'if not INPUT_BASE.exists() and (Path("/kaggle/input/skincancertracker-code").exists() or Path("/kaggle/input/milk10k-dataset").exists()):',
                '    INPUT_BASE = Path("/kaggle/input")',
                "",
                "REPO_DIR = Path('/kaggle/working/SkinCancerTracker')",
                'uploaded_code = INPUT_BASE / "skincancertracker-code"',
                "",
                "if not uploaded_code.exists():",
                "    raise FileNotFoundError(",
                '        f"Required dataset \'{uploaded_code}\' not found!\\n"',
                '        f"Checked: {uploaded_code}\\n"',
                '        "Please ensure \'skincancertracker-code\' is attached to this notebook under Data -> Add Data."',
                "    )",
                "",
                "if REPO_DIR.exists():",
                "    shutil.rmtree(REPO_DIR)",
                "",
                "zip_in_dataset = uploaded_code / 'codebase.zip'",
                "if zip_in_dataset.is_file():",
                "    import zipfile",
                "    REPO_DIR.mkdir(parents=True, exist_ok=True)",
                "    with zipfile.ZipFile(zip_in_dataset, 'r') as zf:",
                "        zf.extractall(REPO_DIR)",
                "    print(f'Extracted codebase from {zip_in_dataset} to {REPO_DIR}')",
                "else:",
                "    shutil.copytree(uploaded_code, REPO_DIR)",
                "    print(f'Copied codebase from {uploaded_code} to {REPO_DIR}')",
                "",
                "os.chdir(REPO_DIR)",
                "if str(REPO_DIR) not in sys.path:",
                "    sys.path.insert(0, str(REPO_DIR))",
                "print('INPUT_BASE resolved to:', INPUT_BASE)",
                "print('Active working directory:', os.getcwd())",
            ]
        )
    )

    # Cell 3: Install Required Dependencies
    cells.append(
        make_cell(
            [
                "# Cell 3: Install Required Dependencies",
                '!pip install -q "timm==1.0.29" "wandb>=0.18.0" "open_clip_torch>=2.20.0"',
                "import timm",
                "print('timm version:', timm.__version__)",
                "print('Dependencies installed successfully.')",
            ]
        )
    )

    # Cell 4: Weights & Biases Authentication
    cells.append(
        make_cell(
            [
                "# Cell 4: Weights & Biases Online Authentication",
                "import os",
                "import wandb",
                "from kaggle_secrets import UserSecretsClient",
                "",
                "try:",
                "    user_secrets = UserSecretsClient()",
                "    wandb_key = user_secrets.get_secret('WANDB_API_KEY')",
                "    assert wandb_key and len(wandb_key.strip()) > 0, 'WANDB_API_KEY secret is empty'",
                "    wandb.login(key=wandb_key)",
                "    os.environ['WANDB_API_KEY'] = wandb_key",
                "    os.environ['WANDB_MODE'] = 'online'",
                "    print('=' * 70)",
                "    print('[OK] W&B Authenticated successfully in ONLINE mode.')",
                "    print('Live training loss, LR schedule, and macro F1 curves will stream to W&B.')",
                "    print('=' * 70)",
                "except Exception as e:",
                "    print('\\n' + '!' * 70)",
                "    print('CRITICAL WARNING: W&B ONLINE AUTHENTICATION FAILED!')",
                "    print(f'Reason: {e}')",
                "    print('FALLING BACK TO OFFLINE LOGGING (WANDB_MODE=offline).')",
                "    print('NO LIVE DASHBOARD METRICS WILL BE STREAMED TO YOUR W&B ACCOUNT!')",
                "    print('To fix: Open Notebook Settings -> Add-ons -> Secrets -> Add \"WANDB_API_KEY\"')",
                "    print('!' * 70 + '\\n')",
                "    os.environ['WANDB_MODE'] = 'offline'",
            ]
        )
    )

    # Cell 5: Verify Dataset Paths & SHA-256 Hash Gate
    cells.append(
        make_cell(
            [
                "# Cell 5: Verify Dataset Paths & Split Hash Gate (Gate 2: SHA-256 Check)",
                "import hashlib",
                "from pathlib import Path",
                "",
                'IMAGE_DIR = INPUT_BASE / "milk10k-dataset" / "images"',
                'SPLITS_FILE = INPUT_BASE / "milk10k-dataset" / "lesion_level_splits.csv"',
                'EXPECTED_SPLIT_HASH = "a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36"',
                "",
                "if not SPLITS_FILE.exists():",
                "    repo_splits = Path('data/splits/lesion_level_splits.csv').resolve()",
                "    if repo_splits.exists():",
                "        SPLITS_FILE = repo_splits",
                "",
                "print(f'INPUT_BASE      : {INPUT_BASE}')",
                "print(f'Image Directory : {IMAGE_DIR} (Exists: {IMAGE_DIR.exists()})')",
                "if IMAGE_DIR.exists():",
                "    img_count = len(list(IMAGE_DIR.glob('*.jpg')))",
                "    print(f'  Found {img_count} images.')",
                "print(f'Splits File     : {SPLITS_FILE} (Exists: {SPLITS_FILE.exists()})')",
                "",
                "assert IMAGE_DIR.is_dir(), f'GATE FAILED: Image directory not found at {IMAGE_DIR}!'",
                "assert SPLITS_FILE.is_file(), f'GATE FAILED: Splits file not found at {SPLITS_FILE}!'",
                "",
                "# Strictly verify SHA-256 hash of splits file",
                "hasher = hashlib.sha256()",
                "with open(SPLITS_FILE, 'rb') as f:",
                "    while chunk := f.read(65536):",
                "        hasher.update(chunk)",
                "split_hash = hasher.hexdigest()",
                "print(f'Splits file SHA-256: {split_hash}')",
                "assert split_hash == EXPECTED_SPLIT_HASH, (",
                "    f'GATE FAILED: Splits file SHA-256 {split_hash} does not match expected {EXPECTED_SPLIT_HASH}!'",
                ")",
                "print('[PASSED] Split file SHA-256 hash gate verified: exact lesion-level partition confirmed.')",
            ]
        )
    )

    # Cell 6: 1-Batch Smoke Test Gate (ResNet-50)
    cells.append(
        make_cell(
            [
                "# Cell 6: 1-Batch Smoke Test Gate (ResNet-50)",
                "import subprocess",
                "import sys",
                "import timm",
                "import torch",
                "",
                "cmd = [",
                "    sys.executable, 'kaggle/run_exp1_resnet50.py',",
                "    '--config', 'configs/experiments/exp1_resnet50_kaggle.yaml',",
                "    '--image-dir', str(IMAGE_DIR),",
                "    '--splits-file', str(SPLITS_FILE),",
                "    '--smoke-test'",
                "]",
                "print('Executing smoke test command:', ' '.join(cmd))",
                "res = subprocess.run(cmd, capture_output=True, text=True)",
                "print(res.stdout)",
                "if res.stderr:",
                "    print(res.stderr)",
                "",
                "assert res.returncode == 0, f'GATE FAILED: Smoke test exited with error code {res.returncode}'",
                "",
                "# In-kernel model assertion: ResNet-50 instantiates cleanly with num_classes=11",
                "test_model = timm.create_model('resnet50', pretrained=True, num_classes=11)",
                "assert test_model is not None, 'GATE FAILED: Could not instantiate ResNet-50 via timm!'",
                "print('\\n' + '=' * 70)",
                "print('[PASSED] EXP-1 SMOKE TEST & MODEL INITIALIZATION GATES CLEARED SUCCESSFULLY.')",
                "print('=' * 70)",
            ]
        )
    )

    # Cell 7: Full 30-Epoch Training Execution
    cells.append(
        make_cell(
            [
                "# Cell 7: Launch Full 30-Epoch Training Run (Exp-1 ResNet-50 Baseline)",
                "# ==============================================================================",
                "# EXPERIMENT CONFIGURATION: SEED",
                "# ==============================================================================",
                "SEED = 42",
                "print('=' * 70)",
                "print(f'EXPERIMENT RANDOM SEED: {SEED}')",
                "print('=' * 70)",
                "",
                "suffix = f'_seed{SEED}' if SEED != 42 else ''",
                "CHECKPOINT_DIR = f'/kaggle/working/checkpoints/exp1_resnet50{suffix}'",
                "RESULTS_DIR = f'/kaggle/working/results/exp1_resnet50{suffix}'",
                "",
                "# Launch full 30-epoch training via standalone execution script",
                "!python kaggle/run_exp1_resnet50.py \\",
                "    --config configs/experiments/exp1_resnet50_kaggle.yaml \\",
                "    --image-dir {IMAGE_DIR} \\",
                "    --splits-file {SPLITS_FILE} \\",
                "    --checkpoint-dir {CHECKPOINT_DIR} \\",
                "    --results-dir {RESULTS_DIR} \\",
                "    --batch-size 64 \\",
                "    --lr 1.0e-4 \\",
                "    --epochs 30 \\",
                "    --seed {SEED} \\",
                "    --online",
            ]
        )
    )

    # Cell 8: Display Summary Metrics & Verify Seed
    cells.append(
        make_cell(
            [
                "# Cell 8: Display Summary Metrics & Verify Seed",
                "import json",
                "from pathlib import Path",
                "",
                "metrics_file = Path(RESULTS_DIR) / 'metrics.json'",
                "summary_file = Path(RESULTS_DIR) / 'summary_metrics.json'",
                "",
                "assert metrics_file.is_file(), f'GATE FAILED: metrics.json not found at {metrics_file}!'",
                "with open(metrics_file, 'r', encoding='utf-8') as f:",
                "    metrics_data = json.load(f)",
                "",
                "# Assert seed read back from metrics.json matches SEED",
                "recorded_seed = metrics_data.get('seed')",
                "print('=' * 70)",
                "print(f'Verifying random seed: Expected SEED={SEED}, Recorded seed={recorded_seed}')",
                "assert recorded_seed == SEED, f'SEED INTEGRITY ERROR: Expected seed {SEED}, but metrics.json recorded {recorded_seed}!'",
                "print(f'[PASSED] Seed integrity verified ({recorded_seed} == {SEED}).')",
                "print('=' * 70)",
                "",
                "if summary_file.is_file():",
                "    with open(summary_file, 'r', encoding='utf-8') as f:",
                "        summary_data = json.load(f)",
                "    print('EXP-1 RESNET-50 FINAL EVALUATION METRICS (Held-Out Test Split, N=786)')",
                "    print('=' * 70)",
                "    test_metrics = summary_data.get('test_overall', {})",
                "    for k in ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'malignant_recall', 'malignant_fnr', 'macro_auroc', 'malignant_auroc']:",
                "        val = test_metrics.get(k, 'N/A')",
                "        if isinstance(val, float):",
                "            print(f'{k:24s}: {val:.4f}')",
                "        else:",
                "            print(f'{k:24s}: {val}')",
                "    print('=' * 70)",
                "    if 'per_class' in test_metrics:",
                "        print('PER-CLASS PERFORMANCE BREAKDOWN:')",
                "        for cls, val in test_metrics['per_class'].items():",
                "            print(f'  {cls:10s} - Recall: {val.get(\"recall\", 0.0):.4f}, Precision: {val.get(\"precision\", 0.0):.4f}, F1: {val.get(\"f1\", 0.0):.4f} (Support: {val.get(\"support\", 0)})')",
            ]
        )
    )

    # Cell 9: List Output Folder & Package Lightweight Artifact Zip
    cells.append(
        make_cell(
            [
                "# Cell 9: List Output Folder & Package Lightweight Artifact Zip",
                "import os, zipfile",
                "from pathlib import Path",
                "",
                "print('=' * 70)",
                "print(f'OUTPUT DIRECTORY CONTENTS: {RESULTS_DIR}')",
                "print('=' * 70)",
                "for item in sorted(Path(RESULTS_DIR).glob('*')):",
                "    print(f'  {item.name:<30s} ({item.stat().st_size / 1e3:8.1f} KB)')",
                "",
                "print('\\n' + '=' * 70)",
                "print(f'CHECKPOINT DIRECTORY CONTENTS: {CHECKPOINT_DIR}')",
                "print('=' * 70)",
                "for item in sorted(Path(CHECKPOINT_DIR).glob('*')):",
                "    print(f'  {item.name:<30s} ({item.stat().st_size / 1e6:8.1f} MB)')",
                "",
                "# Package ONLY lightweight evaluation files into zip (no best_model.pt)",
                "zip_path = Path(f'/kaggle/working/exp1_resnet50{suffix}_artifacts.zip')",
                "required_files = [",
                "    'predictions_val.csv',",
                "    'predictions_test.csv',",
                "    'metrics.json',",
                "    'summary_metrics.json',",
                "    'training_curves.json',",
                "]",
                "with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:",
                "    for fname in required_files:",
                "        fpath = Path(RESULTS_DIR) / fname",
                "        assert fpath.is_file(), f'Required artifact file {fname} not found in {RESULTS_DIR}!'",
                "        zf.write(fpath, arcname=fname)",
                "",
                "print('\\n' + '=' * 70)",
                "print(f'[PASSED] Lightweight artifact package created successfully: {zip_path}')",
                "print(f'Size: {zip_path.stat().st_size / 1e3:.1f} KB (ready for immediate download)')",
                "print('=' * 70)",
            ]
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def build_panderm_notebook(exp_id: str, title: str, script_name: str, config_name: str, is_logit_adj: bool = False) -> dict:
    cells = []

    # Markdown Header
    cells.append(
        make_cell(
            [
                f"# {title}",
                "",
                "**Hardware Setting**:",
                "- **Accelerator**: GPU P100 (or GPU T4 x2)",
                "- **Internet**: ON",
                "- **Secrets**: Add `WANDB_API_KEY` under Notebook Settings -> Add-ons / Secrets",
                "",
                "---",
                "### Attached Kaggle Datasets (Required):",
                "1. `skincancertracker-code` -> `/kaggle/input/datasets/paramveerchaudhary/skincancertracker-code` (uploaded `codebase.zip`)",
                "2. `milk10k-dataset` -> `/kaggle/input/datasets/paramveerchaudhary/milk10k-dataset` (contains `images/` and `lesion_level_splits.csv`)",
                "3. `panderm-weights` -> `/kaggle/input/datasets/paramveerchaudhary/panderm-weights/panderm_base.pth`",
            ],
            cell_type="markdown",
        )
    )

    # Cell 1: Hardware Verification (Gate 1: GPU Present)
    cells.append(
        make_cell(
            [
                "# Cell 1: Hardware & Environment Verification (Gate 1: GPU Present)",
                "import torch",
                "print('PyTorch Version:', torch.__version__)",
                "print('CUDA Available:', torch.cuda.is_available())",
                "assert torch.cuda.is_available(), 'GATE FAILED: CUDA GPU is not available! Please enable GPU accelerator in Kaggle notebook settings.'",
                "gpu_name = torch.cuda.get_device_name(0)",
                "vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)",
                "print(f'GPU Device: {gpu_name}')",
                "print(f'Total VRAM: {vram_gb:.2f} GB')",
                "assert vram_gb >= 14.0, f'GATE FAILED: Expected >= 14GB GPU (P100/T4), found {vram_gb:.2f} GB on {gpu_name}'",
                "print('[PASSED] GPU hardware gate verified.')",
            ]
        )
    )

    # Cell 2: Setup Codebase
    cells.append(
        make_cell(
            [
                "# Cell 2: Setup Codebase (Exclusively from uploaded dataset)",
                "import os, sys, shutil",
                "from pathlib import Path",
                "",
                'KAGGLE_USERNAME = "paramveerchaudhary"',
                'INPUT_BASE = Path(f"/kaggle/input/datasets/{KAGGLE_USERNAME}")',
                "",
                "# Fallback to flat /kaggle/input if dataset is attached in flat layout",
                'if not INPUT_BASE.exists() and (Path("/kaggle/input/skincancertracker-code").exists() or Path("/kaggle/input/milk10k-dataset").exists()):',
                '    INPUT_BASE = Path("/kaggle/input")',
                "",
                "REPO_DIR = Path('/kaggle/working/SkinCancerTracker')",
                'uploaded_code = INPUT_BASE / "skincancertracker-code"',
                "",
                "if not uploaded_code.exists():",
                "    raise FileNotFoundError(",
                '        f"Required dataset \'{uploaded_code}\' not found!\\n"',
                '        f"Checked: {uploaded_code}\\n"',
                '        "Please ensure \'skincancertracker-code\' is attached to this notebook under Data -> Add Data."',
                "    )",
                "",
                "if REPO_DIR.exists():",
                "    shutil.rmtree(REPO_DIR)",
                "",
                "zip_in_dataset = uploaded_code / 'codebase.zip'",
                "if zip_in_dataset.is_file():",
                "    import zipfile",
                "    REPO_DIR.mkdir(parents=True, exist_ok=True)",
                "    with zipfile.ZipFile(zip_in_dataset, 'r') as zf:",
                "        zf.extractall(REPO_DIR)",
                "    print(f'Extracted codebase from {zip_in_dataset} to {REPO_DIR}')",
                "else:",
                "    shutil.copytree(uploaded_code, REPO_DIR)",
                "    print(f'Copied codebase from {uploaded_code} to {REPO_DIR}')",
                "",
                "os.chdir(REPO_DIR)",
                "if str(REPO_DIR) not in sys.path:",
                "    sys.path.insert(0, str(REPO_DIR))",
                "print('INPUT_BASE resolved to:', INPUT_BASE)",
                "print('Active working directory:', os.getcwd())",
            ]
        )
    )

    # Cell 3: Install Required Dependencies
    cells.append(
        make_cell(
            [
                "# Cell 3: Install Required Dependencies",
                '!pip install -q "timm==1.0.29" "wandb>=0.18.0" "open_clip_torch>=2.20.0"',
                "import open_clip",
                "print('open_clip version:', open_clip.__version__)",
                "print('Dependencies installed successfully.')",
            ]
        )
    )

    # Cell 4: Weights & Biases Authentication
    cells.append(
        make_cell(
            [
                "# Cell 4: Weights & Biases Online Authentication",
                "import os",
                "import wandb",
                "from kaggle_secrets import UserSecretsClient",
                "",
                "try:",
                "    user_secrets = UserSecretsClient()",
                "    wandb_key = user_secrets.get_secret('WANDB_API_KEY')",
                "    assert wandb_key and len(wandb_key.strip()) > 0, 'WANDB_API_KEY secret is empty'",
                "    wandb.login(key=wandb_key)",
                "    os.environ['WANDB_API_KEY'] = wandb_key",
                "    os.environ['WANDB_MODE'] = 'online'",
                "    print('=' * 70)",
                "    print('[OK] W&B Authenticated successfully in ONLINE mode.')",
                "    print('Live training loss, LR schedule, and macro F1 curves will stream to W&B.')",
                "    print('=' * 70)",
                "except Exception as e:",
                "    print('\\n' + '!' * 70)",
                "    print('CRITICAL WARNING: W&B ONLINE AUTHENTICATION FAILED!')",
                "    print(f'Reason: {e}')",
                "    print('FALLING BACK TO OFFLINE LOGGING (WANDB_MODE=offline).')",
                "    print('NO LIVE DASHBOARD METRICS WILL BE STREAMED TO YOUR W&B ACCOUNT!')",
                "    print('To fix: Open Notebook Settings -> Add-ons -> Secrets -> Add \"WANDB_API_KEY\"')",
                "    print('!' * 70 + '\\n')",
                "    os.environ['WANDB_MODE'] = 'offline'",
            ]
        )
    )

    # Cell 5: Verify Dataset and Pretrained Weights Paths & Split Hash Gate
    cells.append(
        make_cell(
            [
                "# Cell 5: Verify Dataset Paths & Split Hash Gate (Gate 2: SHA-256 Check)",
                "import hashlib",
                "from pathlib import Path",
                "",
                'IMAGE_DIR = INPUT_BASE / "milk10k-dataset" / "images"',
                'SPLITS_FILE = INPUT_BASE / "milk10k-dataset" / "lesion_level_splits.csv"',
                'CHECKPOINT_PATH = INPUT_BASE / "panderm-weights" / "panderm_base.pth"',
                'EXPECTED_SPLIT_HASH = "a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36"',
                "",
                "if not SPLITS_FILE.exists():",
                "    repo_splits = Path('data/splits/lesion_level_splits.csv').resolve()",
                "    if repo_splits.exists():",
                "        SPLITS_FILE = repo_splits",
                "",
                "print(f'INPUT_BASE      : {INPUT_BASE}')",
                "print(f'Image Directory : {IMAGE_DIR} (Exists: {IMAGE_DIR.exists()})')",
                "if IMAGE_DIR.exists():",
                "    img_count = len(list(IMAGE_DIR.glob('*.jpg')))",
                "    print(f'  Found {img_count} images.')",
                "print(f'Splits File     : {SPLITS_FILE} (Exists: {SPLITS_FILE.exists()})')",
                "print(f'PanDerm Weights : {CHECKPOINT_PATH} (Exists: {CHECKPOINT_PATH.exists()})')",
                "if CHECKPOINT_PATH.exists():",
                "    print(f'  Weight file size: {CHECKPOINT_PATH.stat().st_size / 1e6:.1f} MB')",
                "",
                "assert IMAGE_DIR.is_dir(), f'GATE FAILED: Image directory not found at {IMAGE_DIR}!'",
                "assert SPLITS_FILE.is_file(), f'GATE FAILED: Splits file not found at {SPLITS_FILE}!'",
                "assert CHECKPOINT_PATH.is_file(), f'GATE FAILED: Pretrained weights not found at {CHECKPOINT_PATH}!'",
                "",
                "# Strictly verify SHA-256 hash of splits file",
                "hasher = hashlib.sha256()",
                "with open(SPLITS_FILE, 'rb') as f:",
                "    while chunk := f.read(65536):",
                "        hasher.update(chunk)",
                "split_hash = hasher.hexdigest()",
                "print(f'Splits file SHA-256: {split_hash}')",
                "assert split_hash == EXPECTED_SPLIT_HASH, (",
                "    f'GATE FAILED: Splits file SHA-256 {split_hash} does not match expected {EXPECTED_SPLIT_HASH}!'",
                ")",
                "print('[PASSED] Split file SHA-256 hash gate verified: exact lesion-level partition confirmed.')",
            ]
        )
    )

    # Cell 6: 1-Batch Smoke Test & Model Architecture Gate (Gate 3: PanDerm Architecture Gate)
    cells.append(
        make_cell(
            [
                "# Cell 6: 1-Batch Smoke Test & Model Architecture Gate (Gate 3: PanDerm Gate)",
                "import subprocess",
                "import sys",
                "import torch",
                "",
                "cmd = [",
                f"    sys.executable, '{script_name}',",
                f"    '--config', '{config_name}',",
                "    '--image-dir', str(IMAGE_DIR),",
                "    '--splits-file', str(SPLITS_FILE),",
                "    '--checkpoint-path', str(CHECKPOINT_PATH),",
                "    '--smoke-test'",
                "]",
                "print('Executing smoke test command:', ' '.join(cmd))",
                "res = subprocess.run(cmd, capture_output=True, text=True)",
                "print(res.stdout)",
                "if res.stderr:",
                "    print(res.stderr)",
                "",
                "assert res.returncode == 0, f'GATE FAILED: Smoke test exited with error code {res.returncode}'",
                "",
                "combined_output = (res.stdout or '') + '\\n' + (res.stderr or '')",
                "",
                "# 1. Assert no generic timm fallback occurred",
                "assert 'Could not initialize official PanDerm' not in combined_output, \\",
                "    'GATE FAILED: \"Could not initialize official PanDerm\" detected in logs!'",
                "assert 'Falling back to timm' not in combined_output, \\",
                "    'GATE FAILED: \"Falling back to timm\" detected in logs!'",
                "",
                "# 2. Assert zero missing/unexpected backbone keys in execution log",
                "assert 'VERIFIED: Zero missing backbone keys and zero unexpected keys' in combined_output, \\",
                "    'GATE FAILED: Checkpoint was not cleanly loaded into PanDerm backbone!'",
                "",
                "# 3. In-kernel model assertion: verify loaded class is official VisionTransformer",
                "from src.models.panderm import PanDermClassifier",
                "test_classifier = PanDermClassifier(checkpoint_path=str(CHECKPOINT_PATH))",
                "model_mod = test_classifier.model.__class__.__module__",
                "model_cls = test_classifier.model.__class__.__name__",
                "print(f'\\nKernel verification: Loaded model class is {model_mod}.{model_cls}')",
                "assert model_mod == 'models.modeling_finetune' and model_cls == 'VisionTransformer', \\",
                "    f'GATE FAILED: Expected models.modeling_finetune.VisionTransformer, got {model_mod}.{model_cls}'",
                "",
                "# 4. In-kernel checkpoint key assertion: missing keys == 2 (head.weight, head.bias only) and unexpected keys == 0",
                "ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)",
                "sd = ckpt.get('model', ckpt.get('state_dict', ckpt))",
                "cleaned = {}",
                "for k, v in sd.items():",
                "    nk = k[7:] if k.startswith('module.') else (k[8:] if k.startswith('encoder.') else k)",
                "    if not (nk.startswith('decoder.') or nk.startswith('teacher.')):",
                "        if nk.startswith('norm.'):",
                "            nk = nk.replace('norm.', 'fc_norm.')",
                "        cleaned[nk] = v",
                "",
                "missing, unexpected = test_classifier.model.load_state_dict(cleaned, strict=False)",
                "print(f'Kernel checkpoint key verification: Missing={len(missing)} {missing}, Unexpected={len(unexpected)} {unexpected}')",
                "assert len(unexpected) == 0, f'GATE FAILED: Found {len(unexpected)} unexpected keys: {unexpected}'",
                "assert set(missing) == {'head.weight', 'head.bias'}, f'GATE FAILED: Expected only head keys missing, got: {missing}'",
                "",
                "print('\\n' + '=' * 70)",
                "print('[PASSED] ALL ARCHITECTURE & CHECKPOINT GATES CLEARED SUCCESSFULLY.')",
                "print('=' * 70)",
            ]
        )
    )

    # Cell 7: Full 30-Epoch Training Execution
    exec_lines = [
        f"# Cell 7: Launch Full 30-Epoch Fine-Tuning Run ({exp_id})",
        "# ==============================================================================",
        "# EXPERIMENT CONFIGURATION: SEED",
        "# ==============================================================================",
        "SEED = 42",
        "print('=' * 70)",
        "print(f'EXPERIMENT RANDOM SEED: {SEED}')",
        "print('=' * 70)",
        "",
        "suffix = f'_seed{SEED}' if SEED != 42 else ''",
        f"CHECKPOINT_DIR = f'/kaggle/working/checkpoints/{exp_id}{{suffix}}'",
        f"RESULTS_DIR = f'/kaggle/working/results/{exp_id}{{suffix}}'",
        "",
        f"!python {script_name} \\",
        f"    --config {config_name} \\",
        "    --image-dir {IMAGE_DIR} \\",
        "    --splits-file {SPLITS_FILE} \\",
        "    --checkpoint-path {CHECKPOINT_PATH} \\",
        "    --checkpoint-dir {CHECKPOINT_DIR} \\",
        "    --results-dir {RESULTS_DIR} \\",
        "    --batch-size 64 \\",
        "    --lr 2.5e-4 \\",
        "    --epochs 30 \\",
    ]
    if is_logit_adj:
        exec_lines.append("    --tau 1.0 \\")
    exec_lines.extend(
        [
            "    --seed {SEED} \\",
            "    --online",
        ]
    )
    cells.append(make_cell(exec_lines))

    # Cell 8: Display Summary Metrics & Verify Seed
    cells.append(
        make_cell(
            [
                "# Cell 8: Display Summary Metrics & Verify Seed",
                "import json",
                "from pathlib import Path",
                "",
                "metrics_file = Path(RESULTS_DIR) / 'metrics.json'",
                "summary_file = Path(RESULTS_DIR) / 'summary_metrics.json'",
                "",
                "assert metrics_file.is_file(), f'GATE FAILED: metrics.json not found at {metrics_file}!'",
                "with open(metrics_file, 'r', encoding='utf-8') as f:",
                "    metrics_data = json.load(f)",
                "",
                "# Assert seed read back from metrics.json matches SEED",
                "recorded_seed = metrics_data.get('seed')",
                "print('=' * 70)",
                "print(f'Verifying random seed: Expected SEED={SEED}, Recorded seed={recorded_seed}')",
                "assert recorded_seed == SEED, f'SEED INTEGRITY ERROR: Expected seed {SEED}, but metrics.json recorded {recorded_seed}!'",
                "print(f'[PASSED] Seed integrity verified ({recorded_seed} == {SEED}).')",
                "print('=' * 70)",
                "",
                "if summary_file.is_file():",
                "    with open(summary_file, 'r', encoding='utf-8') as f:",
                "        summary_data = json.load(f)",
                f"    print('{exp_id.upper()} FINAL EVALUATION METRICS (Held-Out Test Split, N=786)')",
                "    print('=' * 70)",
                "    test_metrics = summary_data.get('test_overall', {})",
                "    for k in ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'malignant_recall', 'malignant_fnr', 'macro_auroc', 'malignant_auroc']:",
                "        val = test_metrics.get(k, 'N/A')",
                "        if isinstance(val, float):",
                "            print(f'{k:24s}: {val:.4f}')",
                "        else:",
                "            print(f'{k:24s}: {val}')",
                "    print('=' * 70)",
                "    if 'per_class' in test_metrics:",
                "        print('PER-CLASS PERFORMANCE BREAKDOWN:')",
                "        for cls, val in test_metrics['per_class'].items():",
                "            print(f'  {cls:10s} - Recall: {val.get(\"recall\", 0.0):.4f}, Precision: {val.get(\"precision\", 0.0):.4f}, F1: {val.get(\"f1\", 0.0):.4f} (Support: {val.get(\"support\", 0)})')",
            ]
        )
    )

    # Cell 9: List Output Folder & Package Lightweight Artifact Zip
    cells.append(
        make_cell(
            [
                "# Cell 9: List Output Folder & Package Lightweight Artifact Zip",
                "import os, zipfile",
                "from pathlib import Path",
                "",
                "print('=' * 70)",
                "print(f'OUTPUT DIRECTORY CONTENTS: {RESULTS_DIR}')",
                "print('=' * 70)",
                "for item in sorted(Path(RESULTS_DIR).glob('*')):",
                "    print(f'  {item.name:<30s} ({item.stat().st_size / 1e3:8.1f} KB)')",
                "",
                "print('\\n' + '=' * 70)",
                "print(f'CHECKPOINT DIRECTORY CONTENTS: {CHECKPOINT_DIR}')",
                "print('=' * 70)",
                "for item in sorted(Path(CHECKPOINT_DIR).glob('*')):",
                "    print(f'  {item.name:<30s} ({item.stat().st_size / 1e6:8.1f} MB)')",
                "",
                "# Package ONLY lightweight evaluation files into zip (no best_model.pt)",
                f"zip_path = Path(f'/kaggle/working/{exp_id}{{suffix}}_artifacts.zip')",
                "required_files = [",
                "    'predictions_val.csv',",
                "    'predictions_test.csv',",
                "    'metrics.json',",
                "    'summary_metrics.json',",
                "    'training_curves.json',",
                "]",
                "with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:",
                "    for fname in required_files:",
                "        fpath = Path(RESULTS_DIR) / fname",
                "        assert fpath.is_file(), f'Required artifact file {fname} not found in {RESULTS_DIR}!'",
                "        zf.write(fpath, arcname=fname)",
                "",
                "print('\\n' + '=' * 70)",
                "print(f'[PASSED] Lightweight artifact package created successfully: {zip_path}')",
                "print(f'Size: {zip_path.stat().st_size / 1e3:.1f} KB (ready for immediate download)')",
                "print('=' * 70)",
            ]
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }


def validate_notebook(nb_path: Path, expected_script: str, expected_config: str):
    """Perform strict local validation of generated notebook structure, JSON validity, compile(), and paths."""
    print(f"Validating notebook: {nb_path.name}...")

    # 1. Valid JSON
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb, "Notebook missing 'cells' key"
    assert nb.get("nbformat") == 4, "Notebook must be nbformat 4"

    # 2. Check each code cell compiles when shell/magic commands are stripped
    code_cell_idx = 0
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code_cell_idx += 1
            src = "".join(cell["source"])

            # Strip bash shell commands starting with '!' or '%' including multi-line continuations
            clean_lines = []
            in_shell_cmd = False
            for line in src.split("\n"):
                stripped = line.strip()
                if in_shell_cmd:
                    clean_lines.append("# " + line)
                    if not stripped.endswith("\\"):
                        in_shell_cmd = False
                elif stripped.startswith("!") or stripped.startswith("%"):
                    clean_lines.append("# " + line)
                    if stripped.endswith("\\"):
                        in_shell_cmd = True
                else:
                    clean_lines.append(line)
            clean_code = "\n".join(clean_lines)

            try:
                compile(clean_code, f"{nb_path.name}_cell_{i}", "exec")
            except Exception as e:
                raise AssertionError(f"Cell {i} in {nb_path.name} failed compile(): {e}\nCode:\n{clean_code}")

    print(f"  [OK] All {code_cell_idx} code cells passed Python compile() successfully.")

    # 3. Check that referenced scripts and configs exist locally
    assert (PROJECT_ROOT / expected_script).is_file(), f"Expected script {expected_script} does not exist!"
    assert (PROJECT_ROOT / expected_config).is_file(), f"Expected config {expected_config} does not exist!"
    print(f"  [OK] Verified referenced script exists: {expected_script}")
    print(f"  [OK] Verified referenced config exists: {expected_config}")

    # 4. Check SEED variable is present, default 42, and passed via --seed
    full_text = json.dumps(nb)
    assert "SEED = 42" in full_text, "Missing SEED = 42 definition"
    assert "--seed {SEED}" in full_text, "Missing --seed {SEED} CLI passing"
    assert "recorded_seed == SEED" in full_text, "Missing post-training seed verification assertion"
    print("  [OK] SEED definition, execution flag, and post-training assertion verified.")


def main():
    notebooks_dir = PROJECT_ROOT / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    # 1. Exp-1: ResNet-50
    nb_exp1 = build_exp1_notebook()
    path_exp1 = notebooks_dir / "kaggle_exp1_resnet50.ipynb"
    with open(path_exp1, "w", encoding="utf-8") as f:
        json.dump(nb_exp1, f, indent=2)
    validate_notebook(path_exp1, "kaggle/run_exp1_resnet50.py", "configs/experiments/exp1_resnet50_kaggle.yaml")

    # 2. Exp-2c: PanDerm Uniform Control Arm
    nb_exp2c = build_panderm_notebook(
        exp_id="exp2c_panderm_uniform",
        title="Exp-2c: PanDerm Fine-Tuning Control Arm (ViT-Base, Uniform Shuffling, 30 Epochs)",
        script_name="kaggle/run_exp2c_panderm_uniform.py",
        config_name="configs/experiments/exp2c_panderm_uniform_kaggle.yaml",
        is_logit_adj=False,
    )
    path_exp2c = notebooks_dir / "kaggle_exp2c_panderm_uniform.ipynb"
    with open(path_exp2c, "w", encoding="utf-8") as f:
        json.dump(nb_exp2c, f, indent=2)
    validate_notebook(path_exp2c, "kaggle/run_exp2c_panderm_uniform.py", "configs/experiments/exp2c_panderm_uniform_kaggle.yaml")

    # 3. Exp-3b: PanDerm Logit Adjustment
    nb_exp3b = build_panderm_notebook(
        exp_id="exp3b_panderm_logit_adj",
        title="Exp-3b: PanDerm Fine-Tuning with Train-Time Logit Adjustment (ViT-Base, 30 Epochs)",
        script_name="kaggle/run_exp3b_panderm_logit_adj.py",
        config_name="configs/experiments/exp3b_panderm_logit_adj_kaggle.yaml",
        is_logit_adj=True,
    )
    path_exp3b = notebooks_dir / "kaggle_exp3b_panderm_logit_adj.ipynb"
    with open(path_exp3b, "w", encoding="utf-8") as f:
        json.dump(nb_exp3b, f, indent=2)
    validate_notebook(path_exp3b, "kaggle/run_exp3b_panderm_logit_adj.py", "configs/experiments/exp3b_panderm_logit_adj_kaggle.yaml")

    print("\n" + "=" * 70)
    print("ALL THREE KAGGLE NOTEBOOKS SUCCESSFULLY GENERATED & VALIDATED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
