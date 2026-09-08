# Data

The repository expects these local files under `data/raw/`:

| File | Rows | Columns | Role |
|---|---:|---:|---|
| `train.csv` | 5,933 | 123 | Labelled periods 1--6 |
| `public_test.csv` | 3,613 | 123 | Labelled period 7 |
| `private_test.csv` | 3,600 | 121 | Unlabelled period 8 |

Each input contains `period` and `feature_1` through `feature_120`. Labelled
files additionally contain `substance` and `concentration`. The modelling code
uses only the 120 feature columns as inputs; `period` is reserved for temporal
cross-validation.

Raw data are intentionally ignored by Git and are not distributed with this
code repository. Place an authorised local copy in `data/raw/` before running
the scripts.

Do not add private-test targets or answer-key files to the repository.
