# SkinCancerTracker Deployment & Hosting Analysis

**Document Scope**: Architecture options, measured compute/memory requirements, cloud free-tier limits, license compliance, and security recommendations for demonstrating the SkinCancerTracker inference backend.

---

## 1. Measured Resource Profile (From Step 1 CPU Benchmark)

All numbers are measured on the local host CPU environment (`scripts/gate1_export_and_parity.py` and `scripts/gate2_verify_rollout.py`):

| Metric | Measured Value | Practical Minimum Requirement |
|---|---|---|
| **Checkpoint Storage (`demo_model.pt`)** | **343,331,341 bytes (327.43 MB)** | >= 500 MB free disk space |
| **Python Process RAM (Pre-Load)** | **388.4 MB** | Baseline runtime |
| **Model In-Memory Allocation** | **1,065.7 MB** | PyTorch model weights on CPU |
| **Peak Working Set / RSS RAM** | **1,479.7 MB (~1.48 GB)** | **>= 2.0 GB dedicated RAM** |
| **CPU Model Load Time** | **4.0518 seconds** | One-time server startup |
| **Inference Latency (ViT Forward Pass)** | **206.96 ms - 234.13 ms (Median)** | ~250 ms |
| **Explainability Latency (Rollout)** | **216.11 ms (Median)** | ~220 ms |
| **Total HTTP Response Latency** | **450 ms - 520 ms per request** | Sub-second decision support |

---

## 2. Deployment Architecture Options

### Option A: Local-Only Demo (Recommended)
- **Architecture**: Uvicorn running on `127.0.0.1:8000` with the Vite frontend running on `http://localhost:5173`.
- **Hardware Needs**: 1.5 GB free RAM, multi-core CPU.
- **Pros**:
  - Zero deployment or cloud hosting costs.
  - Zero network latency or external internet dependency.
  - 100% compliant with medical dataset privacy rules (no images leave the machine).
- **Cons**: Requires physical laptop/screen sharing for mentor review.

### Option B: Temporary Tunnelled Demo (Best for Remote Mentor Sessions)
- **Architecture**: The local server runs on `127.0.0.1:8000`, exposed through a temporary secure tunnel:
  - **Cloudflare Tunnel (`cloudflared`)**:
    ```powershell
    cloudflared tunnel --url http://127.0.0.1:8000
    ```
  - **ngrok**:
    ```powershell
    ngrok http 8000
    ```
- **Hardware Needs**: Same as local (1.48 GB RAM).
- **Free-Tier Limits**:
  - **ngrok Free Tier**: 1 free static domain, 1 tunnel, 20,000 requests/month (Official Docs: [https://ngrok.com/pricing](https://ngrok.com/pricing)).
  - **Cloudflare Tunnel**: Free for web traffic via Cloudflare Zero Trust (Official Docs: [https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)).
- **Pros**: Allows the mentor to test the live API and web interface from their own tablet, phone, or browser without cloud deployment.
- **Cons**: Requires active local PC during the presentation.

### Option C: Public Cloud Hosting Platforms
If public cloud deployment is required, candidate free tiers were audited against our **1.48 GB RAM requirement**:

| Hosting Platform | Quoted Free-Tier Limits (Official Docs) | Viability for SkinCancerTracker |
|---|---|---|
| **Hugging Face Spaces** | **2 vCPU, 16 GB RAM, 50 GB disk** (Free Basic CPU Space). Official Docs: [https://huggingface.co/docs/hub/spaces-overview](https://huggingface.co/docs/hub/spaces-overview) | **FULLY VIABLE**. 16 GB RAM easily accommodates the 1.48 GB model memory footprint. |
| **Google Cloud Run** | **2,000,000 free requests/month, 360,000 GiB-seconds RAM, 180,000 vCPU-seconds**. Official Docs: [https://cloud.google.com/run/pricing](https://cloud.google.com/run/pricing) | **VIABLE WITH 2GB CONTAINER**. Requires setting container memory to 2.0 GiB. Cold start latency is ~8–12s on CPU. |
| **Render.com** | **512 MB RAM, 0.1 vCPU, spins down after 15 min**. Official Docs: [https://render.com/docs/free](https://render.com/docs/free) | **INSUFFICIENT (OOM CRASH)**. The 512 MB memory limit will trigger a kernel Out-Of-Memory kill during checkpoint load (requires ~1.48 GB). |
| **Railway.app** | **Free trial only; Hobby plan is $5/month for 8 GB RAM**. Official Docs: [https://docs.railway.com/reference/pricing](https://docs.railway.com/reference/pricing) | **REQUIRES PAID PLAN**. Free tier does not provide enough memory. |
| **Fly.io** | **Free tier deprecated; charges per GB-hour**. Official Docs: [https://fly.io/docs/about/pricing/](https://fly.io/docs/about/pricing/) | **PAID ONLY**. |

---

## 3. Licenses & Redistribution Terms of Use

### (i) MILK10k Dataset Images
- **License**: **Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0)**.
- **Official License Terms**: [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)
- **Attribution**: *"MILK study team"* (verified from `lesion_level_splits.csv`).
- **Permitted**: Academic research, educational demonstration, non-commercial adaptation with attribution.
- **Restrictions**: Commercial use strictly forbidden. Redistribution of medical images over public internet servers without authentication raises patient privacy concerns.

### (ii) PanDerm Pretrained Foundation Model Weights
- **License**: **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC-BY-NC-ND 4.0)**.
- **Official License Statement**: Yan et al., *"A multimodal vision foundation model for clinical dermatology"*, *Nature Medicine* 2025. Official Repo: [https://github.com/SiyuanYan1/PanDerm#license](https://github.com/SiyuanYan1/PanDerm#license)
- **Official Terms**: [https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- **Permitted**: Academic research, evaluation, fine-tuning for non-commercial research with attribution.
- **Restrictions**: Commercial deployment forbidden; redistributed derivative foundation models require author permission.

---

## 4. Final Recommendation

1. **Keep Dataset Images Local**:
   - Never upload the 10,480 MILK10k patient images to public cloud storage (e.g. S3, public buckets, or unprotected web servers).
   - Keep images in `data/images/` on the local development machine.
2. **Demo via Local or Tunnelled Session**:
   - For guide/mentor evaluation, run the local backend via `.\run_demo.ps1`.
   - If the guide wants remote access, spin up a temporary Cloudflare Tunnel or ngrok session during the meeting and terminate it immediately after.
3. **If Cloud Hosting Is Mandated**:
   - Deploy only to **Hugging Face Spaces (CPU Basic, 16 GB RAM)** with password/access protection enabled.
   - Do NOT bundle the full dataset into the repository; allow user-uploaded test images only.
