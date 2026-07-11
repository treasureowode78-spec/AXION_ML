from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

logger = logging.getLogger(__name__)

DEFAULT_API_BASE_URL = "https://api.binance.com"
RETRYABLE_STATUS_CODES = {403, 429, 451}
MAX_RETRIES = 3
INITIAL_BACKOFF_SECONDS = 1.0


class ProviderError(Exception):
    """Base class for exchange provider failures."""


class ProviderRequestError(ProviderError):
    """Raised when a provider cannot complete a request."""


class ProviderManagerError(ProviderError):
    """Raised when all providers fail during a failover operation."""


class BaseExchangeClient(ABC):
    name = "base"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: int = 15,
        base_url: Optional[str] = None,
        auth_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.base_url = base_url
        self.session = requests.Session()
        if auth_headers:
            self.session.headers.update(auth_headers)

    def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        try:
            response = self.session.request(
                method,
                url,
                params=params,
                json=json_payload,
                headers=headers,
                timeout=self.timeout,
            )
        except (Timeout, ConnectionError, RequestException) as exc:
            raise ProviderRequestError(str(exc)) from exc

        if response.status_code >= 400:
            raise HTTPError(f"{response.status_code} from {self.name}", response=response)

        try:
            return response.json()
        except ValueError as exc:
            raise ProviderRequestError(f"{self.name} returned invalid JSON") from exc

    @abstractmethod
    def get_top_symbols(self, limit: int = 300) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_exchange_symbols(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_klines(self, symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
        raise NotImplementedError

    def safe_sleep(self, pause_seconds: float = 0.2) -> None:
        time.sleep(pause_seconds)


class BinanceExchangeClient(BaseExchangeClient):
    name = "Binance"
    BASE_URL = DEFAULT_API_BASE_URL

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: int = 15,
        base_url: Optional[str] = None,
        auth_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(api_key=api_key, api_secret=api_secret, timeout=timeout, base_url=base_url or self.BASE_URL, auth_headers=auth_headers)
        if auth_headers:
            self.session.headers.update(auth_headers)
        elif self.api_key:
            self.session.headers["X-MBX-APIKEY"] = self.api_key

    def _normalized_symbol(self, symbol: str) -> str:
        return symbol.replace("_", "") if "_" in symbol and symbol.endswith("USDT") else symbol

    def get_top_symbols(self, limit: int = 300) -> List[str]:
        payload = self._request("GET", f"{self.base_url}/api/v3/ticker/24hr")
        filtered = [
            ticker
            for ticker in payload
            if ticker.get("symbol", "").endswith("USDT") and float(ticker.get("quoteVolume", 0)) > 10_000
        ]
        filtered.sort(key=lambda item: float(item.get("quoteVolume", 0)), reverse=True)
        return [item["symbol"] for item in filtered[:limit]]

    def get_exchange_symbols(self) -> List[str]:
        info = self._request("GET", f"{self.base_url}/api/v3/exchangeInfo")
        return [
            symbol["symbol"]
            for symbol in info.get("symbols", [])
            if symbol.get("status") == "TRADING"
            and symbol.get("quoteAsset") == "USDT"
            and symbol.get("isSpotTradingAllowed", True)
        ]

    def get_klines(self, symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
        symbol = self._normalized_symbol(symbol)
        payload = self._request(
            "GET",
            f"{self.base_url}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        frame = pd.DataFrame(
            payload,
            columns=[
                "open_time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        frame["Date"] = pd.to_datetime(frame["open_time"], unit="ms")
        frame = frame[["Date", "Open", "High", "Low", "Close", "Volume"]]
        return frame.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})


class MEXCExchangeClient(BaseExchangeClient):
    name = "MEXC"
    BASE_URL = "https://contract.mexc.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: int = 15,
        base_url: Optional[str] = None,
        auth_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(api_key=api_key, api_secret=api_secret, timeout=timeout, base_url=self.BASE_URL, auth_headers=auth_headers)
        if self.api_key:
            self.session.headers.update({"X-MEXC-APIKEY": self.api_key})

    def get_top_symbols(self, limit: int = 300) -> List[str]:
        payload = self._request("GET", f"{self.base_url}/open/api/v2/productList")
        if not payload.get("success", False):
            raise ProviderRequestError("MEXC product list did not return success")

        products = payload.get("data", [])
        symbols = []
        for product in products:
            symbol = product.get("symbol", "")
            if not symbol.endswith("_USDT") or product.get("productType") != 1 or product.get("state") != 0:
                continue
            volume = self._parse_volume(product)
            symbols.append((symbol, volume))

        symbols.sort(key=lambda item: item[1], reverse=True)
        return [symbol for symbol, _ in symbols[:limit]]

    def get_exchange_symbols(self) -> List[str]:
        payload = self._request("GET", f"{self.base_url}/open/api/v2/productList")
        if not payload.get("success", False):
            raise ProviderRequestError("MEXC product list did not return success")
        return [
            product.get("symbol")
            for product in payload.get("data", [])
            if product.get("symbol", "").endswith("_USDT") and product.get("productType") == 1 and product.get("state") == 0
        ]

    def get_klines(self, symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
        normalized_symbol = self._normalize_symbol(symbol)
        payload = self._request(
            "GET",
            f"{self.base_url}/api/v1/contract/market/kline",
            params={"symbol": normalized_symbol, "interval": interval, "limit": limit},
        )
        raw_data = payload if isinstance(payload, list) else payload.get("data")
        if not isinstance(raw_data, list):
            raise ProviderRequestError("MEXC returned unexpected kline payload")

        frame = pd.DataFrame(raw_data)
        if frame.shape[1] < 6:
            raise ProviderRequestError("MEXC kline payload has unexpected columns")

        frame = frame.iloc[:, :6]
        frame.columns = ["open_time", "Open", "High", "Low", "Close", "Volume"]
        frame["Date"] = pd.to_datetime(frame["open_time"].astype(float).apply(self._timestamp_to_ms), unit="ms")
        frame = frame[["Date", "Open", "High", "Low", "Close", "Volume"]]
        return frame.astype({"Open": float, "High": float, "Low": float, "Close": float, "Volume": float})

    def _normalize_symbol(self, symbol: str) -> str:
        if symbol.endswith("USDT") and "_" not in symbol:
            return f"{symbol[:-4]}_USDT"
        return symbol

    def _parse_volume(self, product: Dict[str, Any]) -> float:
        volume = product.get("volume24h") or product.get("contractSize") or product.get("turnover24h") or 0
        try:
            return float(volume)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _timestamp_to_ms(value: Any) -> float:
        timestamp = float(value)
        return timestamp if timestamp > 1e12 else timestamp * 1000


class CoinGeckoExchangeClient(BaseExchangeClient):
    name = "CoinGecko"
    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: int = 15,
        base_url: Optional[str] = None,
        auth_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(api_key=api_key, api_secret=api_secret, timeout=timeout, base_url=self.BASE_URL, auth_headers=auth_headers)

    def get_top_symbols(self, limit: int = 300) -> List[str]:
        payload = self._request("GET", f"{self.base_url}/search/trending")
        coin_ids = [item.get("item", {}).get("id") for item in payload.get("coins", []) if item.get("item", {}).get("id")]
        return coin_ids[:limit]

    def get_exchange_symbols(self) -> List[str]:
        return self.get_top_symbols(limit=300)

    def get_klines(self, symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
        raise ProviderRequestError("CoinGecko klines are not supported for AXION_ML timeframes")


class ProviderManager:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: int = 15,
        base_url: Optional[str] = None,
        auth_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.providers = self._build_providers(api_key, api_secret, timeout, base_url, auth_headers)

    def _build_providers(
        self,
        api_key: Optional[str],
        api_secret: Optional[str],
        timeout: int,
        base_url: Optional[str],
        auth_headers: Optional[Dict[str, str]],
    ) -> List[BaseExchangeClient]:
        return [
            MEXCExchangeClient(api_key=api_key, api_secret=api_secret, timeout=timeout, auth_headers=auth_headers),
            BinanceExchangeClient(api_key=api_key, api_secret=api_secret, timeout=timeout, base_url=base_url, auth_headers=auth_headers),
            CoinGeckoExchangeClient(api_key=api_key, timeout=timeout, auth_headers=auth_headers),
        ]

    def _request_with_failover(self, action_name: str, operation: str, *args: Any, **kwargs: Any) -> Any:
        failures: List[str] = []
        for provider in self.providers:
            logger.info("Using %s", provider.name)
            try:
                return self._attempt_request(provider, operation, *args, **kwargs)
            except ProviderError as exc:
                failures.append(f"{provider.name}: {exc}")
                logger.warning("%s unavailable: %s", provider.name, exc)

        raise ProviderManagerError(
            f"{action_name} failed for all providers. Attempts: {', '.join(failures)}"
        )

    def _attempt_request(self, provider: BaseExchangeClient, operation: str, *args: Any, **kwargs: Any) -> Any:
        backoff = INITIAL_BACKOFF_SECONDS
        for attempt in range(MAX_RETRIES):
            try:
                if operation == "top_symbols":
                    return provider.get_top_symbols(*args, **kwargs)
                if operation == "exchange_symbols":
                    return provider.get_exchange_symbols(*args, **kwargs)
                if operation == "klines":
                    return provider.get_klines(*args, **kwargs)
                raise ProviderRequestError(f"Unsupported operation: {operation}")
            except HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                message = self._format_http_error(provider.name, status_code)
                if status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
                    logger.warning("%s returned %s. Retrying in %.1fs", provider.name, status_code, backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise ProviderRequestError(message)
            except (Timeout, ConnectionError, RequestException) as exc:
                if attempt < MAX_RETRIES - 1:
                    logger.warning("%s temporary failure: %s. Retrying in %.1fs", provider.name, exc, backoff)
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise ProviderRequestError(str(exc))

    @staticmethod
    def _format_http_error(provider_name: str, status_code: Optional[int]) -> str:
        if status_code is None:
            return f"{provider_name} returned an unknown HTTP error"
        return f"{provider_name} returned HTTP {status_code}" if status_code not in RETRYABLE_STATUS_CODES else f"{provider_name} returned retryable HTTP {status_code}"

    def get_top_symbols(self, limit: int = 300) -> List[str]:
        return self._request_with_failover("get_top_symbols", "top_symbols", limit)

    def get_exchange_symbols(self) -> List[str]:
        return self._request_with_failover("get_exchange_symbols", "exchange_symbols")

    def get_klines(self, symbol: str, interval: str = "15m", limit: int = 500) -> pd.DataFrame:
        return self._request_with_failover("get_klines", "klines", symbol, interval, limit)

    def safe_sleep(self, pause_seconds: float = 0.2) -> None:
        time.sleep(pause_seconds)


class ExchangeClient(ProviderManager):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        auth_headers: Optional[Dict[str, str]] = None,
        timeout: int = 15,
    ) -> None:
        super().__init__(api_key=api_key, api_secret=api_secret, timeout=timeout, base_url=base_url, auth_headers=auth_headers)
