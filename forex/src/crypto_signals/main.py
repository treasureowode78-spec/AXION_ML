from __future__ import annotations

import json
from pathlib import Path

from src.crypto_signals.api import ExchangeClient
from src.crypto_signals.backtest import backtest_model
from src.crypto_signals.config import Config
from src.crypto_signals.data import load_data, load_or_download_history
from src.crypto_signals.logger import setup_logger
from src.crypto_signals.model import SignalModel
from src.crypto_signals.scanner import ScannerConfig, SignalScanner
from src.crypto_signals.signals import format_telegram_message
from src.crypto_signals.telegram import TelegramConfig, TelegramNotifier
from src.crypto_signals.coin_info import CoinInfoManager
from src.crypto_signals.futures_filter import FuturesFilter, FuturesPairFilter


def main() -> None:
    config = Config.from_cli()
    logger = setup_logger(level=getattr(__import__("logging"), config.log_level.upper(), 20))
    client = ExchangeClient(
        api_key=config.api_key,
        api_secret=config.api_secret,
        base_url=config.api_base_url,
    )

    # Coin info manager (MEXC + CoinGecko)
    coin_manager = CoinInfoManager(
        mexc_key=config.api_key or "",
        mexc_secret=config.api_secret or "",
        coingecko_key=getattr(config, "coingecko_api_key", None),
    )

    model = SignalModel(
        model_path=config.model_path,
        model_type=config.model_type,
        include_smc=config.enable_smc,
        smc_features=config.smc_features,
    )
    if config.retrain or not model.exists():
        logger.info("Training new model using %s symbols", config.training_pairs)
        symbols = client.get_top_symbols(limit=config.training_pairs)
        data_dir = Path("data")
        frames = list(
            load_or_download_history(
                symbols,
                interval=config.timeframe,
                limit=config.lookback_bars,
                data_dir=data_dir,
                api_key=config.api_key,
                api_secret=config.api_secret,
                api_base_url=config.api_base_url,
            ).values()
        )
        if not frames:
            raise RuntimeError("Unable to collect any training history from the exchange")

        training_dataset = SignalModel.build_training_dataset(
            frames,
            reward_risk=config.risk_reward,
            include_smc=config.enable_smc,
            smc_features=config.smc_features,
        )
        metrics = model.fit(training_dataset, training_dataset["target"])
        logger.info("Training metrics: %s", metrics)
        backtest = backtest_model(model, training_dataset[model.feature_columns], training_dataset["target"], training_dataset["pct_return"])
        logger.info("Backtest results: %s", backtest)
        model.save()
    else:
        loaded = model.load()
        if not loaded:
            raise RuntimeError("Failed to load an existing model")
        logger.info("Loaded model from %s", config.model_path)

    # Fetch candidate symbols; prefer futures-only list if configured
    if config.futures_only:
        coin_infos = coin_manager.get_all_futures_coins()
        futures_filter = FuturesFilter(min_volume_usdt=int(config.min_volume))
        pair_filter = FuturesPairFilter(futures_filter)
        valid_coin_infos = pair_filter.filter_pairs(coin_infos)
        symbols = [c.symbol for c in valid_coin_infos][: config.scan_limit]
    else:
        symbols = client.get_top_symbols(limit=config.scan_limit)
    scanner_config = ScannerConfig(
        timeframe=config.timeframe,
        scan_limit=config.scan_limit,
        lookback_bars=config.lookback_bars,
        signal_threshold=config.signal_threshold,
        top_signals=config.top_signals,
        min_risk_reward=config.min_risk_reward,
        stop_loss_multiplier=config.stop_loss_multiplier,
        take_profit_1_multiplier=config.take_profit_1_multiplier,
        take_profit_2_multiplier=config.take_profit_2_multiplier,
    )
    scanner = SignalScanner(client=client, model=model, config=scanner_config, logger=logger)
    signals = scanner.scan(symbols)

    if signals:
        # Send signals to public channel via TelegramNotifier
        telegram_bot = None
        channel_id = config.telegram_channel_id or config.telegram_chat_id
        if config.telegram_token and channel_id:
            tg_config = TelegramConfig(bot_token=config.telegram_token, channel_id=str(channel_id), user_id=str(config.telegram_user_id or ""))
            notifier = TelegramNotifier(tg_config)
            message = format_telegram_message(signals)
            success = notifier.send_signal_to_channel(message)
            logger.info("Telegram channel send status: %s", success)
        else:
            logger.info("Telegram token or channel id not configured; skipping channel post")
    elif not signals:
        logger.info("No high-conviction signals to send at this time")

    result = {
        "symbols_scanned": len(symbols),
        "signals_found": len(signals),
        "model_loaded": model.exists(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
