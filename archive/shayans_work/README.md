# Shayan's Work

This folder collects the code and outputs created during the Polymarket + BTC
DCA exploration.

Contents:

- `example_polymarket_overlay/`
  MA + Polymarket overlay model, backtest runners, and strength sweep outputs.

- `eda/`
  Research scripts for:
  - non-BTC Polymarket correlation scans
  - candidate DCA feature generation
  - rolling open-question selection
  - baseline vs overlay weight comparison

- `eda/outputs/`
  Saved outputs from the research scripts:
  - `non_btc_polymarket_corr/`
  - `dca_polymarket_features/`
  - `rolling_open_selector/`
  - `weight_comparison/`

- `template/`
  Copies of the template files that were updated to support custom backtest
  start dates and shorter experimental window lengths.

Key decision captured in the overlay model:

- `POLYMARKET_OVERLAY_STRENGTH = 2.0`
  Chosen because it produced the highest tested win rate in the active
  Polymarket 180-day-window experiment.
