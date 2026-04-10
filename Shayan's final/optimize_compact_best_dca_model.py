import logging
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parents[1]
for path in [str(THIS_DIR), str(REPO_DIR)]:
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(1, str(REPO_DIR))

from template.prelude_template import backtest_dynamic_dca, load_data
from compact_best_dca_model import (
    DEFAULT_CONFIG,
    compute_window_weights,
    precompute_features,
    train_lstm_model,
)


START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

CANDIDATES = [
    {"label": "default", **DEFAULT_CONFIG},
    {
        "label": "more_lstm",
        **DEFAULT_CONFIG,
        "base_ma_weight": 0.33,
        "base_lstm_weight": 0.32,
    },
    {
        "label": "more_anchor",
        **DEFAULT_CONFIG,
        "anchor_weight": 0.15,
    },
    {
        "label": "less_anchor",
        **DEFAULT_CONFIG,
        "anchor_weight": 0.05,
    },
    {
        "label": "more_ma_less_sent",
        **DEFAULT_CONFIG,
        "base_ma_weight": 0.38,
        "base_sentiment_weight": 0.12,
    },
    {
        "label": "more_sentiment",
        **DEFAULT_CONFIG,
        "base_ma_weight": 0.32,
        "base_sentiment_weight": 0.18,
    },
    {
        "label": "less_snp_more_lstm",
        **DEFAULT_CONFIG,
        "base_lstm_weight": 0.33,
        "base_snp_weight": 0.07,
    },
    {
        "label": "slightly_more_halving",
        **DEFAULT_CONFIG,
        "base_halving_weight": 0.08,
        "base_ma_weight": 0.32,
    },
]


def evaluate_candidate(btc_df, lstm_model, features_df, candidate: dict) -> dict:
    config = {k: v for k, v in candidate.items() if k != "label"}

    def compute_weights_fn(df_window, current_date=None):
        start_date = df_window.index.min()
        end_date = df_window.index.max()
        if current_date is None:
            current_date = end_date
        return compute_window_weights(
            lstm_model,
            features_df,
            start_date,
            end_date,
            current_date,
            config=config,
        )

    df_spd, exp_decay = backtest_dynamic_dca(
        btc_df,
        compute_weights_fn,
        features_df=features_df,
        strategy_label=f"Compact Best Model Optimization: {candidate['label']}",
        start_date=START_DATE,
        end_date=END_DATE,
    )
    win_rate = (df_spd["dynamic_percentile"] > df_spd["uniform_percentile"]).mean() * 100
    score = 0.5 * win_rate + 0.5 * exp_decay
    excess = df_spd["dynamic_percentile"] - df_spd["uniform_percentile"]
    return {
        "label": candidate["label"],
        "win_rate": float(win_rate),
        "exp_decay_percentile": float(exp_decay),
        "score": float(score),
        "mean_excess": float(excess.mean()),
        "median_excess": float(excess.median()),
        "total_windows": int(len(df_spd)),
        **config,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = THIS_DIR / "optimization_output_compact_best_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "compact_best_model_optimization.csv"
    top_path = output_dir / "compact_best_model_optimization_top.csv"

    btc_df = load_data()
    logging.info("Training compact optimizer LSTM once...")
    lstm_model = train_lstm_model(btc_df)
    logging.info("Precomputing compact optimizer features once...")
    features_df = precompute_features(btc_df)

    rows = []
    total = len(CANDIDATES)
    for idx, candidate in enumerate(CANDIDATES, start=1):
        logging.info("Evaluating candidate %s/%s: %s", idx, total, candidate["label"])
        rows.append(evaluate_candidate(btc_df, lstm_model, features_df, candidate))
        results = pd.DataFrame(rows).sort_values("score", ascending=False)
        results.to_csv(sweep_path, index=False)
        results.head(10).to_csv(top_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved optimization outputs to: {output_dir}")


if __name__ == "__main__":
    main()
