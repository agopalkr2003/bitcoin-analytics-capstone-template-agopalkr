# Compact Non-Crypto Polymarket + Halving Model

This is the compact shareable version of our strongest **pre-Erick** ensemble model.

Files:

- `compact_best_dca_model.py`
  - core model logic
- `run_compact_best_dca_model.py`
  - main runner
- `lstm_helpers.py`
  - local LSTM helper utilities

## Why This Model Matters

This was our best internal ensemble model before moving to the later Erick-style path.

It is the version that combined:

- non-crypto / macro-style Polymarket features
- halving-cycle logic
- LSTM timing
- MA trend signals
- combined sentiment
- S&P regime context
- confirmation logic
- recovery-aware reweighting
- a `10%` uniform DCA anchor

So if you want the version that best represents **our own multi-sleeve research path**, this is the one.

## Polymarket Usage

This model does **not** use BTC-only Polymarket attention the way Erick’s model does.

Instead, it uses the rolling open-question Polymarket feature pipeline built from non-crypto / macro-style questions. The idea was to add external macro and policy information that was not already contained in BTC price and trend signals.

## Halving Usage

This model includes a conditional halving sleeve rather than a simple always-on halving rule.

That was important because earlier experiments showed that halving information could help, but only when used carefully inside the broader ensemble.

## Main Structure

The compact model blends:

- MA sleeve
- combined sentiment sleeve
  - Polymarket + FGI
- LSTM sleeve
- S&P sleeve
- conditional halving sleeve
- small confirmation sleeve
- recovery-aware reweighting
- regime-aware confirmation switch
- `10%` uniform anchor

Key compact default weights:

- `MA = 0.33`
- `Sentiment = 0.15`
- `LSTM = 0.34`
- `S&P = 0.08`
- `Halving = 0.05`
- `confirmation_weight = 0.03`
- `anchor_weight = 0.10`

## Main Results

Regime-aware `3-seed` compact comparison result:

- `score = 69.00%`
- `win_rate = 98.47%`
- `exp_decay_percentile = 39.54%`

3-run robustness:

- `score_mean = 69.07%`
- `score_std = 0.08`
- `win_rate_mean = 98.59%`
- `exp_decay_mean = 39.55%`

This model was very stable and interpretable, even though its raw score was later surpassed by the Erick + Puell path.

## How To Run

From repo root:

```bash
MPLCONFIGDIR=/tmp/mpl_share_best ./.venv-tf/bin/python "eda/Different model tests/compact_noncrypto_halving_model/run_compact_best_dca_model.py"
```

## Dependency Note

This copied code still depends on shared repo assets, especially:

- `template/`
- `data/`
- the repo’s existing Polymarket-derived feature inputs

So this folder is meant for organization and explanation, not as a fully standalone export.
