# SkinCancerTracker Demo Runbook

**Service**: FastAPI Inference Backend (Phase 6-lite)  
**Model**: PanDerm ViT-Base (Exp-2c Uniform Control Arm, Seed 42)  
**Scope**: Research & Decision-Support Prototype (CPU-Ready)  

---

## 1. Quick Start Guide

### Prerequisites
- Python 3.10+ installed (`py` or `python` command available).
- Pinned backend dependencies installed (`pip install -r api/requirements.txt`).
- Exported slim checkpoint at `checkpoints/final/demo_model.pt` (327.4 MB).

### Starting the Server
From the repository root directory, run:

```powershell
.\run_demo.ps1
```

Or manually:
```powershell
py -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

The server will log:
```
[INFO] Initializing SkinCancerTracker inference backend...
[INFO] Initialized official PanDerm ViT (Base)
[INFO] InferenceEngine initialized successfully and set to eval mode.
[INFO] SkinCancerTracker inference backend ready to receive requests.
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Stopping the Server
- Press `Ctrl + C` in the running PowerShell terminal.
- If running in background or stuck:
  ```powershell
  Get-Process -Name python* | Where-Object { $_.MainWindowTitle -like "*uvicorn*" } | Stop-Process
  ```

---

## 2. Health & Verification Procedures

### Basic Health Check
Run in PowerShell:
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health"
```
Expected output:
```json
{
  "status": "ok",
  "model_version": "exp2c-panderm-vit-base-v1",
  "device": "cpu",
  "checkpoint_sha256": "137c79912024"
}
```

### Model Info & Provenance Check
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/model-info"
```

### Automated Smoke Test
Run the automated smoke test script to verify end-to-end inference and rollout across 3 diverse test lesions:
```powershell
py scripts/demo_api_smoke.py
```

---

## 3. Rehearsal Checklist (5 Honest Sample Lesions)

During the demo presentation, show these 5 specific test lesions to demonstrate realistic, honest model capabilities—including known clinical failure modes:

| Case | ISIC ID | Lesion ID | True Diagnosis | Model Prediction | Confidence | Clinical Note for Guide/Mentor |
|---|---|---|---|---|---|---|
| **1. True Positive Malignancy** | `ISIC_8325963` | `IL_0009339` | **BCC** (Basal Cell) | **BCC** | 89.6% | Strong true positive on common malignancy; attention rollout focuses cleanly on the central ulcerated nest. |
| **2. True Negative Benign** | `ISIC_8015399` | `IL_0076321` | **NV** (Melanocytic Nevus) | **NV** | 88.9% | True negative on common benign lesion; malignant probability is low (6.9%). |
| **3. False Alarm (Benign Mimicker)** | `ISIC_2531111` | `IL_0700399` | **BKL** (Benign Keratosis) | **BCC** | 87.4% | **False Positive Error**: Pigmented seborrheic keratosis incorrectly flagged as BCC with high confidence. Demonstrates benign-to-malignant false alarms. |
| **4. Missed Malignancy (Critical Miss)** | `ISIC_8667960` | `IL_0149726` | **MEL** (Melanoma) | **NV** | 91.0% | **Critical False Negative**: True melanoma predicted as benign nevus (Melanoma probability only 0.6%). Crucial teaching moment: demonstrates why raw softmax is uncalibrated and why AI cannot replace clinician review. |
| **5. Squamous Malignancy** | `ISIC_0881081` | `IL_0142779` | **SCCKA** (Squamous / KA) | **SCCKA** | 88.9% | Correct detection of keratinizing squamous malignancy; rollout highlights peripheral hyperkeratosis. |

---

## 4. What to Say About Limitations (Mentor Talking Points)

When presenting to guides, mentors, or reviewers, use these candid talking points:

1. **"This is a Decision-Support Prototype, Not an Autonomous Diagnostic Device"**:
   - The disclaimer *"RESEARCH / DECISION-SUPPORT PROTOTYPE ONLY. Not a diagnostic device."* is attached to every single prediction payload.
   - Point out that raw softmax outputs can be overconfident even when wrong (as shown in Case 4).

2. **"Single-Modality Evaluation (Dermoscopic Only)"**:
   - Although MILK10k provides paired clinical close-ups, this demo model evaluates dermoscopic images only. Clinical images and skin tones are validated but ignored by this baseline arm (as indicated in the `input_note` field).

3. **"Seed Sensitivity and Extreme Long-Tail Uncertainty"**:
   - The demo model achieves **0.4690 Test Macro-F1** on seed 42.
   - However, an identical recipe trained with seed 43 achieved **0.3636 Test Macro-F1** (a gap of ~0.105), showing high sensitivity to initialization on the extreme class imbalance of MILK10k.
   - Rare tail classes (`MAL_OTH`: $n=2$, `DF`: $n=5$, `INF`: $n=5$, `VASC`: $n=6$) have very low test support; their performance metrics have wide confidence intervals.

4. **"Benchmark Comparability"**:
   - Our internal held-out test split ($N=786$) is partitioned strictly at the lesion level to prevent data leakage.
   - Our numbers cannot be directly compared to the published 53.6% top-1 accuracy in the original MILK10k paper, which used a different cross-institutional split and multimodal ensemble.

---

## 5. Troubleshooting Guide

### Issue 1: `Missing demo_model.pt`
- **Symptom**: `FileNotFoundError: Checkpoint not found at: ... checkpoints/final/demo_model.pt`
- **Fix**: Run `py scripts/gate1_export_and_parity.py` to extract and verify the slim checkpoint from `checkpoints/exp2c_panderm_uniform_seed42/best_model.pt`.

### Issue 2: Port 8000 in Use
- **Symptom**: `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`
- **Fix**: Check what is using port 8000:
  ```powershell
  Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
  ```
  Or launch Uvicorn on another port:
  ```powershell
  py -m uvicorn api.app:app --host 127.0.0.1 --port 8080
  ```

### Issue 3: CORS Errors in Frontend Browser Console
- **Symptom**: `Access to fetch at 'http://127.0.0.1:8000/api/predict' from origin 'http://localhost:5173' has been blocked by CORS policy`
- **Fix**: The backend allows `http://localhost:5173` and `http://127.0.0.1:5173` by default. If your frontend Vite dev server runs on another port (e.g. 3000 or 5174), set the environment variable before starting:
  ```powershell
  $env:CORS_ORIGINS = "http://localhost:5173,http://localhost:3000,http://localhost:5174"
  .\run_demo.ps1
  ```
