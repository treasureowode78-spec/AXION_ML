from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.crypto_signals.api import ExchangeClient

logger = logging.getLogger(__name__)

CANDLE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def load_data(
    path: Optional[str] = None,
    api_symbol: Optional[str] = None,
    api_interval: str = "1h",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_base_url: Optional[str] = None,
    n_rows: int = 400,
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file, exchange API, or synthetic sample dataset."""
    if path:
        frame = pd.read_csv(path, parse_dates=["Date"])
        frame = frame.sort_values("Date").reset_index(drop=True)
        required = set(CANDLE_COLUMNS)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
        return frame

    if api_symbol:
        return load_data_from_api(
            api_symbol,
            api_interval,
            n_rows,
            api_key=api_key,
            api_secret=api_secret,
            api_base_url=api_base_url,
        )

    return make_synthetic_crypto_data(n_rows=n_rows)


def load_data_from_api(
    symbol: str,
    interval: str = "1h",
    limit: int = 400,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> pd.DataFrame:
    client = ExchangeClient(api_key=api_key, api_secret=api_secret, base_url=api_base_url)
    return client.get_klines(symbol, interval=interval, limit=limit)


def load_history_for_symbols(
    symbols: List[str],
    interval: str = "15m",
    limit: int = 500,
    max_workers: int = 8,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    client = ExchangeClient(api_key=api_key, api_secret=api_secret, base_url=api_base_url)
    results: Dict[str, pd.DataFrame] = {}

    def _fetch(symbol: str) -> Optional[pd.DataFrame]:
        try:
            df = client.get_klines(symbol, interval=interval, limit=limit)
            if df.shape[0] < 100:
                return None
            return df
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            frame = future.result()
            if frame is not None:
                results[symbol] = frame
    return results


def make_synthetic_crypto_data(n_rows: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drift = np.linspace(0.0002, 0.0015, n_rows)
    noise = rng.normal(0, 0.012, size=n_rows)
    returns = drift + noise
    prices = np.exp(np.cumsum(returns))

    open_prices = prices * (1 + rng.normal(0, 0.003, size=n_rows))
    close_prices = prices * (1 + rng.normal(0, 0.002, size=n_rows))
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(rng.normal(0, 0.004, size=n_rows)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(rng.normal(0, 0.004, size=n_rows)))
    volume = rng.integers(1_200, 7_000, size=n_rows)

    frame = pd.DataFrame(
        {
            "Date": pd.date_range(start="2020-01-01", periods=n_rows, freq="D"),
            "Open": open_prices,
            "High": high_prices,
            "Low": low_prices,
            "Close": close_prices,
            "Volume": volume,
        }
    )
    return frame


def ensure_data_directory(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def symbol_cache_path(symbol: str, interval: str, data_dir: Path) -> Path:
    sanitized = symbol.replace("/", "_").replace(" ", "_")
    return data_dir / f"{sanitized}_{interval}.csv"


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["Date"])
    required = set(CANDLE_COLUMNS)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cached CSV {path} is missing required columns: {sorted(missing)}")
    return frame.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)


def save_symbol_history(symbol: str, interval: str, frame: pd.DataFrame, data_dir: Path) -> Path:
    data_dir = ensure_data_directory(data_dir)
    path = symbol_cache_path(symbol, interval, data_dir)
    frame = frame.copy()
    frame = frame.loc[:, CANDLE_COLUMNS]
    frame.to_csv(path, index=False)
    return path


def append_symbol_history(symbol: str, interval: str, new_frame: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    path = symbol_cache_path(symbol, interval, ensure_data_directory(data_dir))
    existing = _load_csv(path)
    if existing is None:
        save_symbol_history(symbol, interval, new_frame, data_dir)
        return new_frame

    combined = pd.concat([existing, new_frame], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    save_symbol_history(symbol, interval, combined, data_dir)
    return combined


def load_cached_symbol_history(symbol: str, interval: str, data_dir: Path) -> Optional[pd.DataFrame]:
    return _load_csv(symbol_cache_path(symbol, interval, Path(data_dir)))


def load_all_cached_history(interval: str, data_dir: Path) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    results: Dict[str, pd.DataFrame] = {}
    if not data_dir.exists():
        return results

    for path in sorted(data_dir.glob(f"*_{interval}.csv")):
        try:
            frame = _load_csv(path)
            if frame is None or frame.empty:
                continue
            symbol = path.stem.rsplit("_", 1)[0]
            results[symbol] = frame
        except Exception:
            continue
    return results


def _merge_candles(existing: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, new_data], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return combined


def load_or_download_history(
    symbols: List[str],
    interval: str,
    limit: int,
    data_dir: Path,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_base_url: Optional[str] = None,
    max_workers: int = 8,
) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    cached = load_all_cached_history(interval, data_dir)
    if cached:
        logger.info("Loaded %d symbol histories from cache at %s", len(cached), data_dir)
        if api_key and symbols:
            missing = [symbol for symbol in symbols if symbol not in cached]
            if missing:
                try:
                    downloaded = download_and_cache_history(
                        missing,
                        interval,
                        limit,
                        data_dir,
                        api_key=api_key,
                        api_secret=api_secret,
                        api_base_url=api_base_url,
                        max_workers=max_workers,
                    )
                    cached.update(downloaded)
                except Exception as exc:
                    logger.warning("Unable to download missing symbols; continuing with cached data: %s", exc)
        return cached

    if not api_key:
        logger.warning("No cached history available at %s and no API credentials provided", data_dir)
        return {}

    logger.info("Downloading market history for %d symbols into %s", len(symbols), data_dir)
    return download_and_cache_history(
        symbols,
        interval,
        limit,
        data_dir,
        api_key=api_key,
        api_secret=api_secret,
        api_base_url=api_base_url,
        max_workers=max_workers,
    )


def download_symbol_history(
    symbol: str,
    interval: str,
    limit: int,
    data_dir: Path,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> pd.DataFrame:
    data_dir = ensure_data_directory(data_dir)
    cache_path = symbol_cache_path(symbol, interval, data_dir)
    existing = _load_csv(cache_path)
    client = ExchangeClient(api_key=api_key, api_secret=api_secret, base_url=api_base_url)

    try:
        latest = client.get_klines(symbol, interval=interval, limit=limit)
        if latest.empty:
            if existing is not None:
                return existing
            raise RuntimeError(f"Provider returned no candles for {symbol}")

        latest = latest.loc[:, CANDLE_COLUMNS].copy()
        if existing is None:
            save_symbol_history(symbol, interval, latest, data_dir)
            return latest

        merged = _merge_candles(existing, latest)
        save_symbol_history(symbol, interval, merged, data_dir)
        return merged
    except Exception as exc:
        if existing is not None:
            logger.warning("Fresh download failed for %s (%s); using cached %s", symbol, exc, cache_path)
            return existing
        logger.error("Failed to download %s and no cache exists: %s", symbol, exc)
        raise


def download_and_cache_history(
    symbols: List[str],
    interval: str,
    limit: int,
    data_dir: Path,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    api_base_url: Optional[str] = None,
    max_workers: int = 8,
) -> Dict[str, pd.DataFrame]:
    data_dir = ensure_data_directory(data_dir)
    results: Dict[str, pd.DataFrame] = {}

    def _fetch(symbol: str) -> Optional[tuple[str, pd.DataFrame]]:
        try:
            frame = download_symbol_history(
                symbol,
                interval,
                limit,
                data_dir,
                api_key=api_key,
                api_secret=api_secret,
                api_base_url=api_base_url,
            )
            if frame.shape[0] < 100:
                logger.warning("Skipping %s: cached/downloaded history has %s rows", symbol, frame.shape[0])
                return None
            return symbol, frame
        except Exception as exc:
            logger.warning("Skipping %s due to download/cache failure: %s", symbol, exc)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                symbol, frame = result
                results[symbol] = frame

    if not results:
        raise RuntimeError("No historical data available after download and cache attempts")
    return results
