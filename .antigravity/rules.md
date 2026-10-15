# Antigravity Operating Rules for SkinCancerTracker

1. **Training Execution Policy**: NEVER run heavy training jobs locally. All multi-epoch model training must be executed exclusively on Kaggle GPU environments using standalone, config-driven execution scripts.
2. **Auto-Backgrounding & Command Execution Policy**:
   - Commands should be launched with the `-u` unbuffered flag.
   - If the CLI runtime auto-backgrounds a short command (such as `pytest` or evaluation diagnostics exceeding the synchronous threshold), do NOT terminate or cancel it with `manage_task kill`.
   - Never kill tasks. Await background task completion notifications, or poll to completion.
   - Report any auto-backgrounding event immediately to the user.
3. **Smoke Tests & DataLoader Safety**: DataLoader `num_workers` must be 0 for any local smoke-test invocation, enforced directly in Python to prevent worker subprocess deadlocks on Windows.
4. **Data Contract Integrity**: Never alter the lesion-level splitting contract (`data/splits/lesion_level_splits.csv`), class mapping indices (`configs/label_mapping.json`), or canonical class order across evaluation modules.
