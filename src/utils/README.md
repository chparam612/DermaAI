# Utilities Module (`src/utils`)

Shared infrastructure utilities for configuration parsing, hardware profiling, and experiment reproducibility.

## Files

- **`config.py`**:
  YAML configuration loading, schema validation against `REQUIRED_CONFIG_SCHEMA`, dot-accessible dictionary wrapper (`ConfigDict`), and command-line argument parsing with flag overrides.
- **`reproducibility.py`**:
  Deterministic seed management (`set_seed`), environment software version discovery (`get_environment_versions`), hardware profiling (`get_hardware_info`), Git commit hash tracking (`get_git_commit_hash`), and reproducibility manifest export (`capture_reproducibility_manifest`).
