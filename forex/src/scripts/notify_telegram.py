"""Send training and pipeline reports to Telegram."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from src.crypto_signals.telegram import TelegramConfig, TelegramNotifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send training reports to Telegram")
    parser.add_argument("--telegram-token", required=True, help="Telegram bot token")
    parser.add_argument("--telegram-channel-id", default=None, help="Telegram channel/chat id")
    parser.add_argument("--telegram-user-id", default=None, help="Telegram user id for DM notifications")
    parser.add_argument("--download-report", default=None, help="Path to download_summary.json")
    parser.add_argument("--backtest-report", default=None, help="Path to backtest report JSON")
    parser.add_argument("--metrics-file", default="reports/training_metrics.json", help="Path to training_metrics.json")
    return parser.parse_args()


def _load_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    report_path = Path(path)
    if not report_path.exists():
        return None
    try:
        with report_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def format_report_message(
    download_report: Optional[str] = None,
    backtest_report: Optional[str] = None,
    metrics_file: Optional[str] = None,
) -> str:
    """Format reports into a readable Telegram message."""
    lines = ["🤖 AXION_ML Training Report\n"]

    metrics = _load_json(metrics_file)
    if metrics:
        lines.append("🧠 Training Summary:")
        lines.append(f"  • Timestamp: {metrics.get('timestamp', 'n/a')}")
        lines.append(f"  • Model: {metrics.get('model_type', 'unknown')}")
        lines.append(f"  • Rows: {metrics.get('data_rows', 0)}")
        lines.append(f"  • Symbols: {len(metrics.get('symbols', []))}")
        training_metrics = metrics.get("training_metrics", {}) or {}
        if training_metrics:
            for key in ("accuracy", "f1", "precision", "recall"):
                value = training_metrics.get(key)
                if value is not None:
                    lines.append(f"  • {key.title()}: {float(value):.3f}")
        lines.append("")

    if download_report and Path(download_report).exists():
        data = _load_json(download_report)
        if data:
            lines.append("📊 Data Download Summary:")
            lines.append(f"  • Symbols downloaded: {data.get('symbols_downloaded', 0)}")
            lines.append(f"  • Symbols failed: {data.get('symbols_failed', 0)}")
            lines.append(f"  • Total records: {data.get('total_records', 0)}")
            lines.append("")

    if backtest_report and Path(backtest_report).exists():
        data = _load_json(backtest_report)
        if data:
            lines.append("📈 Backtest Results:")
            lines.append(f"  • Total trades: {data.get('total_trades', 0)}")
            lines.append(f"  • Win rate: {data.get('win_rate', 0):.1f}%")
            lines.append(f"  • Total return: {data.get('total_return', 0):.2f}%")
            lines.append(f"  • Max drawdown: {data.get('max_drawdown', 0):.2f}%")
            lines.append(f"  • Sharpe ratio: {data.get('sharpe_ratio', 0):.2f}")
            lines.append("")

    lines.append("✅ Model trained and ready for trading")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    if not args.telegram_token:
        print("Skipping Telegram notification: token not configured")
        return 0

    if not args.telegram_user_id and not args.telegram_channel_id:
        print("Skipping Telegram notification: no user/channel configured")
        return 0

    try:
        tg_config = TelegramConfig(
            bot_token=args.telegram_token,
            channel_id=str(args.telegram_channel_id or ""),
            user_id=str(args.telegram_user_id or ""),
        )
        notifier = TelegramNotifier(tg_config)
    except Exception as exc:
        print(f"Failed to initialize Telegram notifier: {exc}")
        return 1

    message = format_report_message(
        download_report=args.download_report,
        backtest_report=args.backtest_report,
        metrics_file=args.metrics_file,
    )

    try:
        if args.telegram_user_id:
            ok = notifier.send_notification_to_user("TRAINING_REPORT", message)
            print(f"Telegram DM notification sent: {ok}")
        else:
            ok = notifier.send_signal_to_channel(message)
            print(f"Telegram channel notification sent: {ok}")
        return 0 if ok else 1
    except Exception as exc:
        print(f"Failed to send Telegram notification: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
