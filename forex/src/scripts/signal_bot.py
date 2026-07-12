from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.crypto_signals.api import MEXCExchangeClient
from src.crypto_signals.model import SignalModel
from src.crypto_signals.scanner import ScannerConfig, SignalScanner
from src.crypto_signals.telegram import TelegramConfig, TelegramNotifier
from src.crypto_signals.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hourly signal scanner")
    parser.add_argument("--timeframe", default="15m", help="OHLCV interval for scanning")
    parser.add_argument("--scan-limit", type=int, default=80, help="Number of symbols to scan")
    parser.add_argument("--lookback-bars", type=int, default=500, help="Number of bars to download per symbol")
    parser.add_argument("--log-file", default="logs/signal_scan.log", help="Path to save scan log")
    parser.add_argument("--api-key", default=None, help="MEXC API key")
    parser.add_argument("--api-secret", default=None, help="MEXC API secret")
    parser.add_argument("--telegram-token", default=None, help="Telegram bot token")
    parser.add_argument("--telegram-channel-id", default=None, help="Telegram channel/chat id")
    parser.add_argument("--telegram-user-id", default=None, help="Telegram user id for DM notifications")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = PROJECT_DIR
    log_path = root / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_path)

    logger.info("Starting hourly market scan...")

    model_path = root / "models" / "signal_model.pkl"
    model = SignalModel(model_path=str(model_path))
    if not model.exists() or not model.load():
        logger.warning("Trained model not found at %s - skipping scan", model_path)
        return 0

    logger.info("Loaded trained model.")

    # Use MEXC client directly to ensure MEXC-only connection
    client = MEXCExchangeClient(api_key=args.api_key, api_secret=args.api_secret)

    try:
        symbols = client.get_top_symbols(limit=args.scan_limit)
    except Exception as exc:
        logger.error("Failed to fetch top symbols from MEXC: %s", exc)
        return 0

    scanner_config = ScannerConfig(
        timeframe=args.timeframe,
        scan_limit=args.scan_limit,
        lookback_bars=args.lookback_bars,
    )
    scanner = SignalScanner(client=client, model=model, config=scanner_config, logger=logger)

    try:
        signals = scanner.scan(symbols)
    except Exception as exc:
        logger.exception("Scan failed: %s", exc)
        return 0

    if not signals:
        logger.info("No actionable signals found.")
    else:
        # Prepare Telegram notifier if configured
        tg_token = args.telegram_token
        tg_channel = args.telegram_channel_id
        tg_user = args.telegram_user_id or ""
        notifier = None
        if tg_token and tg_channel:
            try:
                tg_config = TelegramConfig(bot_token=tg_token, channel_id=str(tg_channel), user_id=str(tg_user))
                notifier = TelegramNotifier(tg_config)
            except Exception as exc:
                logger.error("Failed to initialize Telegram notifier: %s", exc)

        for s in signals:
            logger.info("Signal: %s %s (%.1f%%)", s.pair, s.direction, s.confidence * 100.0)
            try:
                message = s.format_message() if hasattr(s, "format_message") else str(s)
            except Exception:
                message = str(s)

            if notifier and s.confidence >= scanner_config.signal_threshold:
                try:
                    ok = notifier.send_signal_to_channel(message)
                    logger.info("Telegram notification sent: %s", ok)
                except Exception as exc:
                    logger.error("Telegram send failed: %s", exc)

    logger.info("Hourly scan completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
