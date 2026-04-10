# Shayan's Final

This folder contains the lean shareable version of the current best honest BTC DCA model.

## Files

- `compact_best_dca_model.py`
  - compact one-file version of the winning model logic
- `lstm_helpers.py`
  - local LSTM helper functions used by the compact model
- `run_compact_best_dca_model.py`
  - main runner teammates should use
- `optimize_compact_best_dca_model.py`
  - small local optimization sweep around the winner

## Model Summary

This is the recovery-reweight + `10%` uniform anchor model.

Main structure:

- MA sleeve
- combined sentiment sleeve
  - Polymarket + FGI
- LSTM sleeve
- S&P sleeve
- conditional halving sleeve
- small EW-RSI confirmation sleeve
- recovery-aware reweighting
- `10%` uniform DCA anchor

## Key Results

Single strong run:

- `win_rate = 97.77%`
- `exp_decay_percentile = 39.65%`
- `score = 68.71%`

3-run robustness check:

- `score_mean = 68.35%`
- `score_std = 0.18`
- `win_rate_mean = 97.13%`
- `exp_decay_mean = 39.56%`

## How To Run

From the repo root:

```bash
MPLCONFIGDIR=/tmp/mpl_share_best ./.venv-tf/bin/python "Shayan's final/run_compact_best_dca_model.py"
```

## How To Optimize

From the repo root:

```bash
MPLCONFIGDIR=/tmp/mpl_share_best ./.venv-tf/bin/python "Shayan's final/optimize_compact_best_dca_model.py"
```

## Notes

- This folder is meant to be the clean handoff version for teammates.
- It still depends on the repo’s shared `template/` and `data/` folders.
- The compact model was verified against the packaged winner and matches its logic and weights.
