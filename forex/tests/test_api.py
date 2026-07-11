import json
import pytest
import requests

from src.crypto_signals.api import ExchangeClient, ProviderManagerError


class DummyResponse:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def fake_request_factory(call_sequence):
    def fake_request(self, method, url, params=None, json=None, headers=None, timeout=None):
        key = next(call_sequence)
        if key == "timeout":
            raise requests.exceptions.Timeout("request timed out")
        if key == "connection":
            raise requests.exceptions.ConnectionError("connection failed")
        if isinstance(key, tuple):
            status_code, data = key
            return DummyResponse(status_code=status_code, payload=data)
        raise RuntimeError("Unexpected fake request key: %r" % key)

    return fake_request


def test_get_top_symbols_uses_mexc_primary_provider(monkeypatch):
    sequence = iter([
        (200, {"success": True, "data": [{"symbol": "BTC_USDT", "productType": 1, "state": 0, "volume24h": 20000}]}),
    ])
    monkeypatch.setattr("src.crypto_signals.api.time.sleep", lambda _: None)
    monkeypatch.setattr(requests.Session, "request", fake_request_factory(sequence))

    client = ExchangeClient(api_key="mexc-key", api_secret="mexc-secret")
    symbols = client.get_top_symbols(limit=1)

    assert symbols == ["BTC_USDT"]


def test_provider_failover_switches_to_coingecko_on_mexc_http_451(monkeypatch):
    sequence = iter([
        (451, {}),
        (451, {}),
        (451, {}),
        (200, {"coins": [{"item": {"id": "ethereum"}}]}),
    ])
    monkeypatch.setattr("src.crypto_signals.api.time.sleep", lambda _: None)
    monkeypatch.setattr(requests.Session, "request", fake_request_factory(sequence))

    client = ExchangeClient(api_key="mexc-key", api_secret="mexc-secret")
    symbols = client.get_top_symbols(limit=1)

    assert symbols == ["ethereum"]


def test_timeout_on_mexc_falls_back_to_coingecko(monkeypatch):
    sequence = iter([
        "timeout",
        (200, {"coins": [{"item": {"id": "bitcoin"}}]}),
    ])
    monkeypatch.setattr("src.crypto_signals.api.time.sleep", lambda _: None)
    monkeypatch.setattr(requests.Session, "request", fake_request_factory(sequence))

    client = ExchangeClient(api_key="mexc-key", api_secret="mexc-secret")
    symbols = client.get_top_symbols(limit=1)

    assert symbols == ["bitcoin"]


def test_retry_logic_on_mexc_retries_before_success(monkeypatch):
    sequence = iter([
        (429, {}),
        (429, {}),
        (200, {"success": True, "data": [{"symbol": "XRP_USDT", "productType": 1, "state": 0, "volume24h": 25000}]}),
    ])
    monkeypatch.setattr("src.crypto_signals.api.time.sleep", lambda _: None)
    monkeypatch.setattr(requests.Session, "request", fake_request_factory(sequence))

    client = ExchangeClient(api_key="mexc-key", api_secret="mexc-secret")
    symbols = client.get_top_symbols(limit=1)

    assert symbols == ["XRP_USDT"]


def test_provider_error_contains_all_failed_providers(monkeypatch):
    sequence = iter([
        (451, {}),
        (451, {}),
        (451, {}),
        (451, {}),
        (451, {}),
        (451, {}),
        (451, {}),
        (451, {}),
        (451, {}),
    ])
    monkeypatch.setattr("src.crypto_signals.api.time.sleep", lambda _: None)
    monkeypatch.setattr(requests.Session, "request", fake_request_factory(sequence))

    client = ExchangeClient(api_key="mexc-key", api_secret="mexc-secret")

    with pytest.raises(ProviderManagerError) as excinfo:
        client.get_top_symbols(limit=1)

    message = str(excinfo.value)
    assert "MEXC" in message
    assert "CoinGecko" in message
    assert "Binance" not in message
