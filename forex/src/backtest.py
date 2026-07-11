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
from src.crypto_signals.backtest import backtest_model
from src.crypto_signals.data import load_or_download_history
from src.crypto_signals.model import SignalModel


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("axion_backtest")
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
    parser = argparse.ArgumentParser(description="Run weekly backtesting for AXION_ML")
    parser.add_argument("--model-path", default="models/signal_model.pkl", help="Path to the trained model")
    parser.add_argument("--timeframe", default="15m", help="OHLCV interval for backtest data")
    parser.add_argument("--backtest-window", type=int, default=100, help="Number of latest rows to evaluate")
    parser.add_argument("--training-pairs", type=int, default=80, help="Number of symbols to fetch for backtesting")
    parser.add_argument("--lookback-bars", type=int, default=500, help="History bars to collect for backtesting")
    parser.add_argument("--horizon", type=int, default=6, help="Label horizon in bars")
    parser.add_argument("--risk-reward", type=float, default=2.0, help="Target risk/reward ratio for labels")
    parser.add_argument("--enable-smc", action="store_true", help="Enable SMC feature extraction")
    parser.add_argument("--smc-features", default=None, help="Comma-separated SMC features to include")
    parser.add_argument("--model-type", default="xgboost", choices=["xgboost", "random_forest"], help="Model family to use")
    parser.add_argument("--api-key", default=None, help="Exchange API key")
    parser.add_argument("--api-secret", default=None, help="Exchange API secret")
    parser.add_argument("--api-base-url", default=None, help="Optional exchange API base URL")
    parser.add_argument("--report-file", default="reports/backtest_report.json", help="Path to save backtest report")
    parser.add_argument("--log-file", default="logs/backtest.log", help="Path to save backtest log")
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
        logger.info("Starting backtest run")
        model_path = root / args.model_path
        model = SignalModel(model_path=str(model_path), model_type=args.model_type, include_smc=args.enable_smc, smc_features=parse_smc_features(args.smc_features))
        if not model.load():
            raise RuntimeError(f"Failed to load model from {model_path}")

        if model.metadata is not None:
            enable_smc = bool(model.metadata.get("include_smc", args.enable_smc))
            smc_features = model.metadata.get("smc_features", parse_smc_features(args.smc_features))
        else:
            enable_smc = args.enable_smc
            smc_features = parse_smc_features(args.smc_features)

        client = ExchangeClient(api_key=args.api_key, api_secret=args.api_secret, base_url=args.api_base_url)
        symbols = client.get_top_symbols(limit=args.training_pairs)
        logger.info("Loaded %d candidate symbols for backtest", len(symbols))

        data_dir = root / "data"
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
            raise RuntimeError("Failed to download backtest history for any symbol")

        dataset = SignalModel.build_training_dataset(
            frames,
            horizon=args.horizon,
            reward_risk=args.risk_reward,
            include_smc=enable_smc,
            smc_features=smc_features,
        )
        if len(dataset) < args.backtest_window:
            raise RuntimeError(f"Not enough data for backtest window: {len(dataset)} rows available")

        test_dataset = dataset.iloc[-args.backtest_window :].reset_index(drop=True)
        test_features = test_dataset.drop(columns=["target"])
        test_labels = test_dataset["target"]
        probabilities = None
        try:
            probabilities = model.predict_proba(test_features)
        except Exception:
            probabilities = None

        backtest_result = backtest_model(model, test_features, test_labels, test_features["pct_return"], probabilities=probabilities)
        returns = test_features["pct_return"].astype(float)
        winning_trades = int((returns > 0).sum())
        losing_trades = int((returns < 0).sum())
        total_trades = int(len(returns))

        report_path = root / args.report_file
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_path": str(model_path),
            "symbols": symbols,
            "enable_smc": enable_smc,
            "smc_features": smc_features,
            "backtest_window": args.backtest_window,
            "metrics": backtest_result.__dict__,
            "summary": {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": backtest_result.win_rate,
                "profit_factor": backtest_result.profit_factor,
                "maximum_drawdown": backtest_result.max_drawdown,
                "sharpe_ratio": backtest_result.sharpe_ratio,
            },
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        logger.info("Saved backtest report to %s", report_path)
        logger.info("Backtest run complete")
        return 0
    except Exception as exc:
        logger.exception("Backtest failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
