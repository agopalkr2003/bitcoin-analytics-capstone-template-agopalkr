import logging
import random
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_DIR = THIS_DIR.parent
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

LSTM_SEEDS = [7, 17, 29]


def set_global_seed(seed: int) -> None:
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        from tensorflow import keras

        keras.utils.set_random_seed(seed)
    except Exception:
        pass


def train_seeded_lstm_models(btc_df, seeds: list[int]):
    models = []
    for idx, seed in enumerate(seeds, start=1):
        logging.info("Training compact best-model LSTM seed %s/%s: %s", idx, len(seeds), seed)
        set_global_seed(seed)
        models.append(train_lstm_model(btc_df))
    return models


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    btc_df = load_data()
    logging.info("Training compact best-model 3-seed LSTM ensemble...")
    lstm_model = train_seeded_lstm_models(btc_df, LSTM_SEEDS)
    logging.info("Precomputing compact best-model features...")
    features_df = precompute_features(btc_df)

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
            config=DEFAULT_CONFIG,
        )

    df_spd, exp_decay = backtest_dynamic_dca(
        btc_df,
        compute_weights_fn,
        features_df=features_df,
        strategy_label="Compact Shareable Best DCA Model",
        start_date="2018-01-01",
        end_date="2025-12-31",
    )
    win_rate = (df_spd["dynamic_percentile"] > df_spd["uniform_percentile"]).mean() * 100
    score = 0.5 * win_rate + 0.5 * exp_decay
    excess = df_spd["dynamic_percentile"] - df_spd["uniform_percentile"]
    print(
        {
            "win_rate": float(win_rate),
            "exp_decay_percentile": float(exp_decay),
            "score": float(score),
            "mean_excess": float(excess.mean()),
            "median_excess": float(excess.median()),
            "total_windows": int(len(df_spd)),
            "model_variant": "compact_regime_confirmation_3seed",
            "lstm_seeds": LSTM_SEEDS,
            "config": DEFAULT_CONFIG,
        }
    )


if __name__ == "__main__":
    main()
