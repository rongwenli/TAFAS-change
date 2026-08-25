# Prediction-Derived Closed-Form TTA scripts

This directory mirrors the experiment matrix in `scripts/`, but selects the
gradient-free `CLOSED_FORM` TTA method. Every leaf `run.sh` loads the same
source checkpoint used by TAFAS. If the checkpoint is missing, it invokes the
matching script in `train_scripts/` first.

Example:

```bash
bash closed_form_scripts/DLinear/ETTh1_96/run.sh
```

Outputs are written to:

```text
results/<MODEL>/<DATASET>_<PRED_LEN>/closed_form_tta/
```

The main hyperparameters can be changed with environment variables:

```bash
CLOSED_FORM_RANK=4 \
CLOSED_FORM_POGT_LEN=12 \
CLOSED_FORM_RIDGE_LAMBDA=0.01 \
bash closed_form_scripts/DLinear/ETTh1_96/run.sh
```

Any trailing arguments are appended as YACS overrides. For example:

```bash
bash closed_form_scripts/DLinear/ETTh1_96/run.sh \
    TTA.CLOSED_FORM.SUBSPACE_UPDATE_INTERVAL 16
```
