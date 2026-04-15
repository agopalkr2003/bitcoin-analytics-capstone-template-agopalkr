# Polymarket + MA Only Model

This is the earlier model that combined:

- a `200`-day moving-average timing rule
- a rolling Polymarket overlay

Files:

- `model_development_polymarket_overlay.py`
  - core MA + Polymarket model logic
- `run_backtest.py`
  - backtest runner for the overlay experiment

## Why This Model Matters

This was one of the first clean Polymarket experiments in the project.

Its job was simple:

- keep a standard MA-based DCA engine
- test whether Polymarket can add any useful external information on top of that

So this model was important as an early research step before we built the bigger ensemble systems.

## Model Logic

The model uses:

- a primary BTC trend signal from price relative to the `200`-day MA
- a secondary Polymarket overlay built from rolling selected open-question features

The overlay combines lagged:

- mean return z-score
- mean question price
- mean absolute BTC correlation
- selected question count

Then the final signal is:

- MA signal
- plus `overlay_strength * Polymarket overlay`

The tested production-style setting used:

- `overlay_strength = 2.0`

That value was chosen from the active-window sweep because it gave the best tested win rate without over-amplifying the overlay.

## Polymarket Usage

Unlike Erick’s later BTC-only Polymarket sentiment, this model used the rolling open-question feature pipeline and treated Polymarket as an overlay on top of the MA baseline.

So it was an early step toward the later non-crypto / macro-style Polymarket research path.

## Main Results

Active-window result:

- start: `2025-03-12`
- end: `2025-12-31`
- `window_days = 180`
- `score = 58.59%`
- `win_rate = 77.39%`
- `exp_decay_percentile = 39.78%`

Best active-window overlay-strength result:

- `overlay_strength = 2.0`

MA-only baseline on the same active window:

- `score = 55.28%`
- `win_rate = 69.57%`
- `exp_decay_percentile = 40.99%`

Full-range run:

- `score = 51.53%`
- `win_rate = 61.24%`
- `exp_decay_percentile = 41.81%`
- `total_windows = 2557`

Interpretation:

- Polymarket helped this early model when the overlay was active
- the biggest gain was in win rate versus the MA-only baseline
- but the model was still much simpler and weaker than the later ensemble systems

## How To Run

From repo root:

```bash
MPLCONFIGDIR=/tmp/mpl_share_best ./.venv-tf/bin/python "eda/Different model tests/polymarket_ma_only_model/run_backtest.py"
```

## Dependency Note

This model depends on shared repo assets, especially:

- `template/`
- `data/`
- `eda/outputs/rolling_open_selector/rolling_open_question_features.csv`

So this folder is a clean organizational copy for review and writing, not a standalone package.
