from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1] / "forex"
sys.path.insert(0, str(PROJECT_DIR))

from src.crypto_signals.api import ExchangeClient
from src.crypto_signals.data import load_or_download_history
from src.crypto_signals.model import SignalModel


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("axion_train")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AXION_ML signal model")
    parser.add_argument("--model-path", default="models/signal_model.pkl", help="Output model path")
    parser.add_argument("--model-type", default="xgboost", choices=["xgboost", "random_forest"], help="Model family to train")
    parser.add_argument("--timeframe", default="15m", help="OHLCV interval for training data")
    parser.add_argument("--training-pairs", type=int, default=80, help="Number of symbols to fetch for training")
    parser.add_argument("--lookback-bars", type=int, default=500, help="History bars to collect per symbol")
    parser.add_argument("--test-size", type=float, default=0.15, help="Validation split size")
    parser.add_argument("--horizon", type=int, default=6, help="Label horizon in bars")
    parser.add_argument("--risk-reward", type=float, default=2.0, help="Target risk/reward ratio for labels")
    parser.add_argument("--enable-smc", action="store_true", help="Enable SMC feature extraction")
    parser.add_argument("--smc-features", default=None, help="Comma-separated SMC features to include")
    parser.add_argument("--api-key", default=None, help="Exchange API key")
    parser.add_argument("--api-secret", default=None, help="Exchange API secret")
    parser.add_argument("--api-base-url", default=None, help="Optional exchange API base URL")
    parser.add_argument("--feature-importance", default="reports/feature_importance.csv", help="Path to save feature importance")
    parser.add_argument("--metrics-file", default="reports/training_metrics.json", help="Path to save training metrics")
    parser.add_argument("--log-file", default="logs/training.log", help="Path to save training log")
    return parser.parse_args()


def parse_smc_features(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    log_path = root / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_path)

    try:
        logger.info("Starting training run")
        client = ExchangeClient(api_key=args.api_key, api_secret=args.api_secret, base_url=args.api_base_url)
        symbols = client.get_top_symbols(limit=args.training_pairs)
        logger.info("Loaded %d candidate symbols", len(symbols))

        data_dir = root / "forex" / "data"
        frames = list(
            load_or_download_history(
                symbols,
                interval=args.timeframe,
                limit=args.lookback_bars,
                data_dir=data_dir,
                api_key=args.api_key,
                api_secret=args.api_secret,
                api_base_url=args.api_base_url,
            ).values()
        )
        if not frames:
            raise RuntimeError("Failed to download training history for any symbol")

        smc_features = parse_smc_features(args.smc_features)
        dataset = SignalModel.build_training_dataset(
            frames,
            horizon=args.horizon,
            reward_risk=args.risk_reward,
            include_smc=args.enable_smc,
            smc_features=smc_features,
        )
        logger.info("Built training dataset with %d rows", len(dataset))

        model_path = root / args.model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model = SignalModel(
            model_path=str(model_path),
            model_type=args.model_type,
            include_smc=args.enable_smc,
            smc_features=smc_features,
        )
        metrics = model.fit(dataset.drop(columns=["target"]), dataset["target"], test_size=args.test_size)
        model.save()
        logger.info("Trained model saved to %s", model_path)

        importance_path = root / args.feature_importance
        importance_path.parent.mkdir(parents=True, exist_ok=True)
        feature_importance = model.feature_importance()
        feature_importance.to_frame("importance").to_csv(importance_path)
        logger.info("Saved feature importance to %s", importance_path)

        metrics_path = root / args.metrics_file
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_path": str(model_path),
            "data_rows": len(dataset),
            "symbols": symbols,
            "model_type": args.model_type,
            "include_smc": args.enable_smc,
            "smc_features": smc_features,
            "training_metrics": metrics,
            "feature_columns": model.feature_columns,
        }
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        logger.info("Saved training metrics to %s", metrics_path)

        logger.info("Training run complete")
        return 0
    except Exception as exc:
        logger.exception("Training failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
