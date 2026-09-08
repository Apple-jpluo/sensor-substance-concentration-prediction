# Sensor-Based Substance Classification and Concentration Prediction

Official, code-only implementation of a two-stage sensor-modelling study. The
data span sequential acquisition periods whose distributions shift over time,
so model selection is based primarily on Leave-One-Period-Out (LOPO)
cross-validation rather than random row-level splitting.

This repository contains source code and usage documentation only. It does not
publish precomputed metrics, figures, or predictions. Every output is generated
locally under the Git-ignored `artifacts/` directory.

## Research design

The workflow has four parts:

1. Diagnose class imbalance, concentration structure and period drift.
2. Predict one of six substances from 120 sensor features.
3. Compare global concentration regression with regressors conditioned on the
   true substance.
4. Fit the final Soft Probability Specialist Model (SPSM): an MLP produces six
   class probabilities, six substance-specific random forests predict
   concentration, and the final concentration is their probability-weighted
   sum.

`period` is never used as an input feature. It is used only to define temporal
validation groups.

## Repository layout

```text
configs/final.json          SPSM hyperparameters
src/statsml/                data checks, models, metrics and LOPO evaluation
scripts/run_eda.py          compact data and drift analysis
scripts/evaluate_classifiers.py
scripts/evaluate_regressors.py
scripts/evaluate_spsm.py    final LOPO experiment with row-level OOF output
scripts/train_final.py      fit periods 1--7 and predict period 8
tests/                      schema and SPSM invariants
```

## Data setup

Place the three authorised CSV files in `data/raw/`:

```text
data/raw/train.csv
data/raw/public_test.csv
data/raw/private_test.csv
```

They are excluded from Git and are not distributed with the code. See
`data/README.md` for the schema. Never place private-test truth in this project.

## Environment

Install the pinned environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

## Reproduce the analysis

Run commands from the repository root:

```bash
python scripts/run_eda.py
python scripts/evaluate_classifiers.py
python scripts/evaluate_regressors.py
python scripts/evaluate_spsm.py
python scripts/train_final.py
```

The classification and regression comparison scripts are deliberately small:
they retain the principal baselines needed to explain the model choice instead
of every exploratory grid search. The SPSM evaluation is the authoritative
final-model implementation.

Full LOPO is computationally expensive because it trains seven MLPs and 42
random forests. Locally generated outputs include per-period metrics, pooled
out-of-fold metrics and an unweighted fold summary so that aggregation is
explicit. Row-level OOF predictions are not saved by default because they
contain target values; use
`--oof-output artifacts/spsm_oof.csv` only for local diagnostics.

## License

The code is released under the MIT License. The dataset is not covered by the
software license and is not distributed in this repository.
