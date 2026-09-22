# Notebooks (`notebooks/`)

This directory contains exploratory data analysis (EDA) notebooks, interactive visualizations, and workflow tutorials.

## Guidelines

- All heavy training, evaluation, and data manipulation logic must live in the `src/` Python package, not inline inside notebooks.
- Notebooks should import modular code from `src.data`, `src.models`, and `src.eval`.
- For Google Colab quickstart and environment verification, refer to the root `setup_colab.ipynb`.
