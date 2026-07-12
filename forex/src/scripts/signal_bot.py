from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.crypto_signals.api import MEXCExchangeClient
from src.crypto_signals.logger import setup_logger
from src.crypto_signals.model import SignalModel
from src.crypto_signals.scanner import ScannerConfig, SignalScanner
from src.crypto_signals.signals import format_telegram_message
from src.crypto_signals.telegram import TelegramConfig, TelegramNotifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hourly signal scanner")
    parser.add_argument("--timeframe", default="15m", help="OHLCV interval for scanning")
    parser.add_argument("--scan-limit", type=int, default=80, help="Number of symbols to scan")
    parser.add_argument("--lookback-bars", type=int, default=500, help="Number of bars to download per symbol")
    parser.add_argument("--log-file", default="logs/signal_scan.log", help="Path to save scan log")
    parser.add_argument("--api-key", default=None, help="MEXC API key")
    parser.add_argument("--api-secret", default=None, help="MEXC API secret")
    parser.add_argument("--telegram-token", default=None, help="Telegram bot token")
    parser.add_argument("--telegram-channel-id", default=None, help="Telegram channel ID")
    parser.add_argument("--telegram-user-id", default=None, help="Telegram user ID")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = PROJECT_DIR
    log_path = root / args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(name="signal_bot", log_file=str(log_path))

    logger.info("Starting hourly market scan...")

    model_path = root / "models" / "signal_model.pkl"
    model = SignalModel(model_path=str(model_path))
    if not model.exists() or not model.load():
        logger.warning("Trained model not found at %s - skipping scan", model_path)
        return 0

    logger.info("Loaded trained model.")

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
        return 0

    for signal in signals:
        logger.info("Signal: %s %s (%.1f%%)", signal.pair, signal.direction, signal.confidence_score)

    telegram_token = args.telegram_token or os.getenv("TELEGRAM_TOKEN") or os.getenv("TG_BOT_TOKEN")
    telegram_channel_id = args.telegram_channel_id or os.getenv("TELEGRAM_CHANNEL_ID") or os.getenv("TELEGRAM_CHANNEL")
    telegram_user_id = args.telegram_user_id or os.getenv("TELEGRAM_USER_ID")
    message = format_telegram_message(signals)

    if telegram_token and telegram_channel_id:
        channel_config = TelegramConfig(
            bot_token=telegram_token,
            channel_id=str(telegram_channel_id),
            user_id=str(telegram_user_id or ""),
        )
        notifier = TelegramNotifier(channel_config)
        channel_sent = notifier.send_signal_to_channel(message)
        logger.info("Telegram channel send status: %s", channel_sent)

    if telegram_token and telegram_user_id:
        user_config = TelegramConfig(
            bot_token=telegram_token,
            channel_id=str(telegram_channel_id or ""),
            user_id=str(telegram_user_id),
        )
        notifier = TelegramNotifier(user_config)
        user_sent = notifier.send_notification_to_user("MARKET_SIGNAL", message)
        logger.info("Telegram user send status: %s", user_sent)

    logger.info("Market scan completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
