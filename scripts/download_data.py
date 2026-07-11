from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_DIR = Path(__file__).resolve().parents[1] / "forex"
sys.path.insert(0, str(PROJECT_DIR))

from src.crypto_signals.api import ExchangeClient
from src.crypto_signals.data import download_and_cache_history


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("axion_download")
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
    parser = argparse.ArgumentParser(description="Download and cache cryptocurrency OHLCV history")
    parser.add_argument("--timeframe", default="15m", help="OHLCV interval for caching")
    parser.add_argument("--symbols-file", default=None, help="Optional file with one symbol per line")
    parser.add_argument("--training-pairs", type=int, default=80, help="Number of top symbols to fetch")
    parser.add_argument("--lookback-bars", type=int, default=500, help="Number of bars to download per symbol")
    parser.add_argument("--data-dir", default="forex/data", help="Directory to save cached CSV files")
    parser.add_argument("--api-key", default=None, help="Exchange API key")
    parser.add_argument("--api-secret", default=None, help="Exchange API secret")
    parser.add_argument("--api-base-url", default=None, help="Optional exchange API base URL")
    parser.add_argument("--log-file", default="logs/download_data.log", help="Path to save download log")
    return parser.parse_args()


def read_symbols_file(path: str) -> List[str]:
    symbol_path = Path(path)
    if not symbol_path.exists():
        raise FileNotFoundError(f"Symbols file not found: {path}")
    return [line.strip() for line in symbol_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    log_path = root / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_path)

    try:
        if args.symbols_file:
            symbols = read_symbols_file(args.symbols_file)
            logger.info("Loaded %d symbols from %s", len(symbols), args.symbols_file)
        else:
            client = ExchangeClient(api_key=args.api_key, api_secret=args.api_secret, base_url=args.api_base_url)
            symbols = client.get_top_symbols(limit=args.training_pairs)
            logger.info("Fetched %d top symbols from provider", len(symbols))

        if not symbols:
            raise RuntimeError("No symbols available to download")

        data_dir = root / args.data_dir
        result = download_and_cache_history(
            symbols,
            interval=args.timeframe,
            limit=args.lookback_bars,
            data_dir=data_dir,
            api_key=args.api_key,
            api_secret=args.api_secret,
            api_base_url=args.api_base_url,
            max_workers=8,
        )

        logger.info("Downloaded and cached history for %d symbols into %s", len(result), data_dir)
        output_path = root / "reports" / "download_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump({"symbol_count": len(result), "symbols": sorted(result.keys()), "data_dir": str(data_dir)}, handle, indent=2)
        logger.info("Saved download summary to %s", output_path)
        return 0
    except Exception as exc:
        logger.exception("Cache download failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
