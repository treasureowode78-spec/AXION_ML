from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.crypto_signals.smc_features import DEFAULT_SMC_FEATURES, SMCFeatureExtractor, SMC_FEATURE_COLUMNS


FEATURE_COLUMNS = [
    "close",
    "open",
    "high",
    "low",
    "volume",
    "body_size",
    "upper_wick",
    "lower_wick",
    "pct_return",
    "log_return",
    "ema_20",
    "ema_50",
    "ema_100",
    "ema_200",
    "sma_50",
    "sma_200",
    "vwap",
    "supertrend",
    "rsi_14",
    "macd",
    "macd_signal",
    "roc_12",
    "cci_20",
    "sto_k",
    "sto_d",
    "atr_14",
    "bb_width",
    "hist_vol_14",
    "volatility_21",
    "avg_candle_size_7",
    "obv",
    "relative_volume",
    "volume_spike",
    "trend_bias",
    "volatility_regime",
    "breakout_strength",
    "stop_hunt_risk",
]


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def simple_moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def compute_vwap(frame: pd.DataFrame) -> pd.Series:
    typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3
    return (typical * frame["Volume"]).cumsum() / frame["Volume"].cumsum()


def compute_true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["Close"].shift(1)
    ranges = pd.concat(
        [frame["High"] - frame["Low"], (frame["High"] - previous_close).abs(), (frame["Low"] - previous_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def compute_atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    return compute_true_range(frame).rolling(window=window, min_periods=1).mean()


def compute_supertrend(frame: pd.DataFrame, atr_multiplier: float = 3.0, atr_window: int = 14) -> pd.Series:
    atr = compute_atr(frame, window=atr_window)
    hl2 = (frame["High"] + frame["Low"]) / 2
    basic_upper = hl2 + atr_multiplier * atr
    basic_lower = hl2 - atr_multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(index=frame.index, dtype=float)
    direction = 1

    for i in range(len(frame)):
        if i == 0:
            final_upper.iat[0] = basic_upper.iat[0]
            final_lower.iat[0] = basic_lower.iat[0]
            supertrend.iat[0] = final_upper.iat[0]
            continue

        final_upper.iat[i] = (
            basic_upper.iat[i]
            if basic_upper.iat[i] < final_upper.iat[i - 1] or frame["Close"].iat[i - 1] > final_upper.iat[i - 1]
            else final_upper.iat[i - 1]
        )
        final_lower.iat[i] = (
            basic_lower.iat[i]
            if basic_lower.iat[i] > final_lower.iat[i - 1] or frame["Close"].iat[i - 1] < final_lower.iat[i - 1]
            else final_lower.iat[i - 1]
        )

        if frame["Close"].iat[i] > final_upper.iat[i - 1]:
            direction = 1
        elif frame["Close"].iat[i] < final_lower.iat[i - 1]:
            direction = -1

        supertrend.iat[i] = final_lower.iat[i] if direction > 0 else final_upper.iat[i]

    return supertrend.ffill().fillna(frame["Close"])


def compute_macd(frame: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    fast = exponential_moving_average(frame["Close"], span=12)
    slow = exponential_moving_average(frame["Close"], span=26)
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def compute_rsi(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = frame["Close"].diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(window=window, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def compute_roc(frame: pd.DataFrame, window: int = 12) -> pd.Series:
    return frame["Close"].pct_change(window).fillna(0)


def compute_cci(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    typical = (frame["High"] + frame["Low"] + frame["Close"]) / 3
    ma = typical.rolling(window=window, min_periods=1).mean()
    mad = typical.rolling(window=window, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical - ma) / (0.015 * mad.replace(0, np.nan))
    return cci.fillna(0)


def compute_stochastic(frame: pd.DataFrame, window: int = 14, smooth_k: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_min = frame["Low"].rolling(window=window, min_periods=1).min()
    high_max = frame["High"].rolling(window=window, min_periods=1).max()
    sto_k = 100 * (frame["Close"] - low_min) / (high_max - low_min + 1e-9)
    sto_d = sto_k.rolling(window=smooth_k, min_periods=1).mean()
    return sto_k.fillna(50), sto_d.fillna(50)


def compute_bollinger_width(frame: pd.DataFrame, window: int = 20) -> pd.Series:
    sma = frame["Close"].rolling(window=window, min_periods=1).mean()
    std = frame["Close"].rolling(window=window, min_periods=1).std()
    return (std * 2) / (sma.replace(0, np.nan))


def compute_obv(frame: pd.DataFrame) -> pd.Series:
    direction = np.sign(frame["Close"].diff().fillna(0))
    return (direction * frame["Volume"]).cumsum().fillna(0)


def compute_market_regime(frame: pd.DataFrame) -> pd.DataFrame:
    trend = np.sign(frame["ema_50"] - frame["ema_200"])
    strength = np.where(frame["rsi_14"] > 60, 1, np.where(frame["rsi_14"] < 40, -1, 0))
    regime = trend * 2 + strength
    frame["trend_bias"] = regime.clip(-2, 2).fillna(0)
    frame["volatility_regime"] = np.where(frame["atr_14"] > frame["hist_vol_14"], 1, -1)
    return frame


def compute_market_structure(frame: pd.DataFrame) -> pd.DataFrame:
    high_range = frame["High"].rolling(window=5, center=True, min_periods=1).max()
    low_range = frame["Low"].rolling(window=5, center=True, min_periods=1).min()
    frame["breakout_strength"] = (frame["Close"] - high_range.shift(1)) / (frame["atr_14"] + 1e-9)
    frame["stop_hunt_risk"] = (frame["Low"] < low_range.shift(1)).astype(int)
    return frame


def prepare_smc_features(frame: pd.DataFrame, enabled_features: Optional[List[str]] = None) -> pd.DataFrame:
    extractor = SMCFeatureExtractor(enabled_features=enabled_features)
    return extractor.extract(frame)


def prepare_features(frame: pd.DataFrame, include_smc: bool = False, smc_features: Optional[List[str]] = None) -> pd.DataFrame:
    data = frame.copy().sort_values("Date").reset_index(drop=True)
    data["open"] = data["Open"]
    data["high"] = data["High"]
    data["low"] = data["Low"]
    data["close"] = data["Close"]
    data["volume"] = data["Volume"]
    data["body_size"] = (data["Close"] - data["Open"]).abs()
    data["upper_wick"] = data["High"] - data[["Close", "Open"]].max(axis=1)
    data["lower_wick"] = data[["Close", "Open"]].min(axis=1) - data["Low"]
    data["pct_return"] = data["Close"].pct_change().fillna(0)
    data["log_return"] = np.log(data["Close"] / data["Close"].shift(1)).replace(-np.inf, 0).fillna(0)

    data["ema_20"] = exponential_moving_average(data["Close"], span=20)
    data["ema_50"] = exponential_moving_average(data["Close"], span=50)
    data["ema_100"] = exponential_moving_average(data["Close"], span=100)
    data["ema_200"] = exponential_moving_average(data["Close"], span=200)
    data["sma_50"] = simple_moving_average(data["Close"], window=50)
    data["sma_200"] = simple_moving_average(data["Close"], window=200)
    data["vwap"] = compute_vwap(data)
    data["supertrend"] = compute_supertrend(data)
    data["rsi_14"] = compute_rsi(data)
    data["macd"], data["macd_signal"] = compute_macd(data)
    data["roc_12"] = compute_roc(data)
    data["cci_20"] = compute_cci(data)
    data["sto_k"], data["sto_d"] = compute_stochastic(data)
    data["atr_14"] = compute_atr(data, window=14)
    data["bb_width"] = compute_bollinger_width(data)
    data["hist_vol_14"] = data["log_return"].rolling(window=14, min_periods=1).std().fillna(0)
    data["volatility_21"] = data["log_return"].rolling(window=21, min_periods=1).std().fillna(0)
    data["avg_candle_size_7"] = data["body_size"].rolling(window=7, min_periods=1).mean().fillna(0)
    data["obv"] = compute_obv(data)
    data["relative_volume"] = data["Volume"] / (data["Volume"].rolling(window=20, min_periods=1).mean().replace(0, np.nan))
    data["relative_volume"] = data["relative_volume"].fillna(1.0)
    data["volume_spike"] = data["relative_volume"] > 2.0
    data["volume_spike"] = data["volume_spike"].astype(int)

    data = compute_market_regime(data)
    data = compute_market_structure(data)

    if include_smc:
        smc_data = prepare_smc_features(data, enabled_features=smc_features)
        data = pd.concat([data, smc_data], axis=1)

    data = data.fillna(0)
    return data


def label_actions(frame: pd.DataFrame, horizon: int = 6, reward_risk: float = 2.0) -> pd.Series:
    target_high = frame["High"].rolling(window=horizon, min_periods=1).max().shift(-horizon + 1)
    target_low = frame["Low"].rolling(window=horizon, min_periods=1).min().shift(-horizon + 1)
    stop_loss_buy = frame["Close"] - frame["atr_14"]
    target_buy = frame["Close"] + frame["atr_14"] * reward_risk
    stop_loss_sell = frame["Close"] + frame["atr_14"]
    target_sell = frame["Close"] - frame["atr_14"] * reward_risk

    buy_hit = target_high >= target_buy
    sell_hit = target_low <= target_sell

    labels = np.zeros(len(frame), dtype=int)
    labels = np.where(buy_hit & ~sell_hit, 1, labels)
    labels = np.where(sell_hit & ~buy_hit, 2, labels)
    return pd.Series(labels, index=frame.index)


def prepare_training_data(
    frame: pd.DataFrame,
    horizon: int = 6,
    reward_risk: float = 2.0,
    include_smc: bool = False,
    smc_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    features = prepare_features(frame, include_smc=include_smc, smc_features=smc_features)
    labels = label_actions(features, horizon=horizon, reward_risk=reward_risk)
    features = features.iloc[:-horizon].copy()
    labels = labels.iloc[:-horizon]
    features = features.dropna().reset_index(drop=True)
    labels = labels.loc[features.index].reset_index(drop=True)
    return features, labels


def get_feature_columns(include_smc: bool = False, smc_features: Optional[List[str]] = None) -> List[str]:
    columns = FEATURE_COLUMNS.copy()
    if include_smc:
        selected = smc_features if smc_features is not None else DEFAULT_SMC_FEATURES.copy()
        columns.extend([feature for feature in selected if feature in SMC_FEATURE_COLUMNS])
    return columns
