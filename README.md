# Bitcoin Analytics Capstone

A Bitcoin DCA (Dollar Cost Averaging) strategy model with comprehensive backtesting capabilities. This repository implements a dynamic weight allocation model based on MVRV (Market Value to Realized Value) indicators and price momentum signals.

## Overview

This project computes dynamic investment weights for Bitcoin DCA strategies, adjusting daily allocations based on:
- **MVRV Z-score**: Buy more when undervalued (low MVRV)
- **Price vs 200-day MA**: Buy more when price is below long-term trend
- **4-year Halving Cycle Percentile**: Context from Bitcoin's halving cycle
- **MVRV Momentum**: Acceleration/deceleration of MVRV trends
- **Signal Confidence**: Amplify signals when multiple indicators agree

## Repository Structure

```
.
├── model_development.py    # Core model logic and weight computation
├── backtest.py            # Backtesting framework and visualization
├── prelude.py             # Data loading and backtest utilities
├── model.md               # Model documentation
├── model_backtest.md      # Backtest documentation
├── requirements.txt       # Python dependencies
├── data/                  # Data directory
│   ├── download_data.py   # Data download script
│   ├── Coin Metrics/      # CoinMetrics BTC data
│   └── Polymarket/        # Polymarket data
└── tests/                  # Test suite
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd bitcoin-analytics-capstone-template
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Setup

### Option 1: Download via Script

Run the data download script:
```bash
python data/download_data.py
```

This will download data from Google Drive and organize it into the `data/` directory.

### Option 2: Manual Download

All data needed for this capstone can be accessed via:
[Capstone Data Google Drive](https://drive.google.com/drive/folders/1gizJ_n-QCnE8qrFM-BU3J_ZpaR3HCjn7?usp=sharing)

Download the data and place it in the `data/` directory, preserving the subfolder structure:
- `data/Coin Metrics/` - CoinMetrics BTC data
- `data/Polymarket/` - Polymarket data files

## Usage

### Running the Backtest

Run the full backtest analysis:
```bash
python backtest.py
```

This will:
1. Load BTC price and MVRV data from CoinMetrics
2. Precompute all model features
3. Run SPD (sats-per-dollar) backtest across rolling 1-year windows
4. Generate visualizations and export metrics to `output/` directory

### Using the Model

```python
from prelude import load_data
from model_development import precompute_features, compute_window_weights
import pandas as pd

# Load data
btc_df = load_data()

# Precompute features
features_df = precompute_features(btc_df)

# Compute weights for a date range
start_date = pd.Timestamp("2024-01-01")
end_date = pd.Timestamp("2024-12-31")
current_date = pd.Timestamp("2024-12-31")

weights = compute_window_weights(
    features_df, 
    start_date, 
    end_date, 
    current_date
)
```

## Output

Running `backtest.py` generates the following files in the `output/` directory:

- `performance_comparison.svg` - Line chart comparing dynamic vs uniform percentile over time
- `excess_percentile_distribution.svg` - Histogram of excess percentile distribution
- `win_loss_comparison.svg` - Bar chart showing wins/losses breakdown
- `cumulative_performance.svg` - Area chart of cumulative excess percentile
- `metrics_summary.svg` - Table visualization of key metrics
- `metrics.json` - Complete metrics data in JSON format

## Testing

Run the test suite:
```bash
pytest
```

The test suite includes:
- Backtest error handling tests
- Backtest performance tests
- Model edge case tests
- Weight stability tests
- Backtest visualization tests

## Documentation

- **Model Documentation**: See `model.md` for detailed explanation of the weight computation model
- **Backtest Documentation**: See `model_backtest.md` for backtesting framework details

## Key Features

- **Dynamic Weight Allocation**: Adjusts daily DCA amounts based on market signals
- **Comprehensive Backtesting**: Tests strategy across rolling 1-year windows from 2018 to present
- **No Forward-Looking Bias**: All features are lagged to prevent information leakage
- **Weight Stability**: Past weights are locked and never change as new data arrives
- **Performance Metrics**: Win rate, SPD percentile, and exponential-decay scoring

## Requirements

- Python 3.11+
- See `requirements.txt` for full dependency list

## License

[Add license information if applicable]
