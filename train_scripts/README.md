# Training scripts

This directory mirrors every experiment under `scripts/`. Each leaf-level
`run.sh` trains the corresponding forecasting model and writes its best model
to:

```text
checkpoints/<MODEL>/<DATASET>_<PRED_LEN>/checkpoint_best.pth
```

For example:

```bash
bash train_scripts/DLinear/ETTh1_720/run.sh
bash scripts/DLinear/ETTh1_720/run.sh
```

The second command runs TTA. If its checkpoint does not exist or is empty, it
automatically invokes the matching training script first.

Both kinds of scripts can be launched from any working directory. They use the
currently active Python by default. Set `TAFAS_PYTHON` to select a specific
interpreter, or `TAFAS_CHECKPOINT_ROOT` and `TAFAS_RESULT_ROOT` to redirect
outputs. Extra arguments are appended as YACS configuration overrides:

```bash
bash train_scripts/DLinear/ETTh1_720/run.sh SOLVER.MAX_EPOCH 10
```

Run `python script_tools/generate_experiment_scripts.py --check` after changing
the experiment matrix. Run it without `--check` to regenerate the mirrored
training scripts, closed-form TTA scripts, and checkpoint guards.
