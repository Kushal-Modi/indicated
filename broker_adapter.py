"""
broker_adapter.py — Modular Broker Integration Layer
=====================================================
Implements an Abstract Base Class (BaseBrokerAdapter) defining the
contract that every broker must fulfill, plus concrete adapter stubs
for Upstox, Dhan, and Zerodha Kite Connect.

To add a new broker:
  1. Subclass BaseBrokerAdapter
  2. Implement all abstract methods
  3. Register it in BROKER_REGISTRY

The factory function `get_broker_adapter()` returns the correct
adapter based on cfg.BROKER_NAME.

NOTE: In MOCK_MODE, data_engine.py bypasses these adapters entirely
and uses its own simulated data generator.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd

from config import cfg

logger = logging.getLogger(__name__)


# ===========================================================================
# Data Models (plain dicts / typed return shapes for documentation clarity)
# ===========================================================================

# OHLCV candle row keys: timestamp, open, high, low, close, volume
# Option chain row keys: strike, ce_oi, ce_chg_oi, ce_volume, ce_iv,
#                        pe_oi, pe_chg_oi, pe_volume, pe_iv, ce_ltp, pe_ltp


# ===========================================================================
# Abstract Base Adapter
# ===========================================================================

class BaseBrokerAdapter(ABC):
    """
    Every broker adapter must implement these five methods.
    All methods should raise BrokerError on failure (never return None silently).
    """

    @abstractmethod
    def get_spot_price(self, symbol: str) -> float:
        """
        Return the current spot/index price for the given symbol.
        
        Args:
            symbol: "NIFTY" | "BANKNIFTY" | "FINNIFTY"
        Returns:
            float: Latest traded price
        """
        ...

    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        """
        Fetch historical OHLCV candle data.

        Args:
            symbol: Index/futures symbol string
            timeframe: "5m" | "15m" | "1h"
            count: Number of candles to retrieve (most recent N)
        Returns:
            pd.DataFrame with columns: [timestamp, open, high, low, close, volume]
                - timestamp must be timezone-aware (IST preferred)
                - Sorted ascending by timestamp
        """
        ...

    @abstractmethod
    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch option chain data around ATM.

        Args:
            symbol: "NIFTY" | "BANKNIFTY" | "FINNIFTY"
            expiry: "YYYY-MM-DD" or None (uses nearest expiry)
        Returns:
            pd.DataFrame with columns:
                strike, ce_oi, ce_chg_oi, ce_volume, ce_iv, ce_ltp,
                pe_oi, pe_chg_oi, pe_volume, pe_iv, pe_ltp
                - Sorted ascending by strike
        """
        ...

    @abstractmethod
    def get_india_vix(self) -> float:
        """
        Return the current India VIX value.
        Returns:
            float: India VIX (typically 10–40)
        """
        ...

    @abstractmethod
    def get_nearest_expiry(self, symbol: str) -> str:
        """
        Return the nearest weekly/monthly expiry date string.
        Returns:
            str: "YYYY-MM-DD"
        """
        ...


# ===========================================================================
# Custom Exception
# ===========================================================================

class BrokerError(Exception):
    """Raised when a broker API call fails."""
    pass


# ===========================================================================
# Upstox Adapter (v2 API)
# ===========================================================================

class UpstoxAdapter(BaseBrokerAdapter):
    """
    Upstox v2 REST API adapter.
    
    API Docs: https://upstox.com/developer/api-documentation/
    
    Required env vars:
        BROKER_API_KEY    — Your Upstox API key
        BROKER_API_SECRET — Your Upstox API secret
    
    Authentication flow:
        1. Direct user to auth URL to get auth_code
        2. Exchange auth_code for access_token via /login/authorization/token
        3. Use access_token as Bearer token in all subsequent calls
    """

    BASE_URL = "https://api.upstox.com/v2"

    # Upstox instrument keys for indices
    INSTRUMENT_KEYS = {
        "NIFTY": "NSE_INDEX|Nifty 50",
        "BANKNIFTY": "NSE_INDEX|Nifty Bank",
        "FINNIFTY": "NSE_INDEX|Nifty Fin Service",
        "SENSEX": "BSE_INDEX|SENSEX",
        "INDIAVIX": "NSE_INDEX|India VIX",
    }

    # Upstox candle interval mapping
    INTERVAL_MAP = {
        "5m": "5minute",
        "15m": "15minute",
        "1h": "60minute",
    }

    def __init__(self):
        self._access_token: str = cfg.BROKER_API_KEY  # assumes pre-auth token
        if not self._access_token:
            raise BrokerError("BROKER_API_KEY is empty. Cannot initialise UpstoxAdapter.")
        self._headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }
        logger.info("UpstoxAdapter initialised.")

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Internal helper for authenticated GET requests."""
        import requests
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = requests.get(url, headers=self._headers, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise BrokerError(f"Upstox API error [{endpoint}]: {exc}") from exc

    def get_spot_price(self, symbol: str) -> float:
        key = self.INSTRUMENT_KEYS.get(symbol)
        if not key:
            raise BrokerError(f"Unknown symbol: {symbol}")
        data = self._get("/market-quote/ltp", params={"instrument_key": key})
        return float(data["data"][key]["last_price"])

    def get_ohlcv(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        import requests
        from datetime import date, timedelta
        key = self.INSTRUMENT_KEYS.get(symbol, symbol)
        interval = self.INTERVAL_MAP.get(timeframe, "5minute")
        to_date = date.today().isoformat()
        from_date = (date.today() - timedelta(days=5)).isoformat()
        url = f"{self.BASE_URL}/historical-candle/{key}/{interval}/{to_date}/{from_date}"
        resp = self._get(url.replace(self.BASE_URL, ""))
        candles = resp["data"]["candles"]
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").tail(count).reset_index(drop=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        # Upstox option chain endpoint
        expiry = expiry or self.get_nearest_expiry(symbol)
        data = self._get("/option/chain", params={
            "instrument_key": self.INSTRUMENT_KEYS.get(symbol, symbol),
            "expiry_date": expiry,
        })
        rows = []
        for item in data.get("data", []):
            ce = item.get("call_options", {}).get("market_data", {})
            pe = item.get("put_options", {}).get("market_data", {})
            rows.append({
                "strike": item["strike_price"],
                "ce_oi": ce.get("oi", 0),
                "ce_chg_oi": ce.get("oi_day_change", 0),
                "ce_volume": ce.get("volume", 0),
                "ce_iv": ce.get("implied_volatility", 0),
                "ce_ltp": ce.get("ltp", 0),
                "pe_oi": pe.get("oi", 0),
                "pe_chg_oi": pe.get("oi_day_change", 0),
                "pe_volume": pe.get("volume", 0),
                "pe_iv": pe.get("implied_volatility", 0),
                "pe_ltp": pe.get("ltp", 0),
            })
        return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

    def get_india_vix(self) -> float:
        data = self._get("/market-quote/ltp", params={"instrument_key": self.INSTRUMENT_KEYS["INDIAVIX"]})
        return float(data["data"][self.INSTRUMENT_KEYS["INDIAVIX"]]["last_price"])

    def get_nearest_expiry(self, symbol: str) -> str:
        data = self._get("/option/expiry", params={"instrument_key": self.INSTRUMENT_KEYS.get(symbol, symbol)})
        dates = sorted(data["data"])
        return dates[0]


# ===========================================================================
# Dhan Adapter
# ===========================================================================

class DhanAdapter(BaseBrokerAdapter):
    """
    Dhan HQ API adapter.
    
    API Docs: https://dhanhq.co/docs/v2/
    
    Required env vars:
        BROKER_API_KEY — Dhan access token
    
    Dhan uses a single access token (no secret), obtained from the
    Dhan developer portal.
    """

    BASE_URL = "https://api.dhan.co"

    SYMBOL_MAP = {
        "NIFTY": "NIFTY",
        "BANKNIFTY": "BANKNIFTY",
        "FINNIFTY": "FINNIFTY",
        "SENSEX": "SENSEX",
    }

    INTERVAL_MAP = {
        "5m": "5",
        "15m": "15",
        "1h": "60",
    }

    def __init__(self):
        self._token = cfg.BROKER_API_KEY
        if not self._token:
            raise BrokerError("BROKER_API_KEY is empty. Cannot initialise DhanAdapter.")
        self._headers = {
            "access-token": self._token,
            "Content-Type": "application/json",
        }
        logger.info("DhanAdapter initialised.")

    def _post(self, endpoint: str, payload: dict) -> dict:
        import requests
        url = f"{self.BASE_URL}{endpoint}"
        try:
            resp = requests.post(url, headers=self._headers, json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise BrokerError(f"Dhan API error [{endpoint}]: {exc}") from exc

    def get_spot_price(self, symbol: str) -> float:
        # Dhan LTP via market feed
        data = self._post("/v2/marketfeed/ltp", {"NSE_INDEX": [self.SYMBOL_MAP[symbol]]})
        return float(data["data"]["NSE_INDEX"][self.SYMBOL_MAP[symbol]]["last_price"])

    def get_ohlcv(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        from datetime import date, timedelta
        payload = {
            "securityId": self.SYMBOL_MAP[symbol],
            "exchangeSegment": "IDX_I",
            "instrument": "INDEX",
            "interval": self.INTERVAL_MAP.get(timeframe, "5"),
            "fromDate": (date.today() - timedelta(days=5)).isoformat(),
            "toDate": date.today().isoformat(),
        }
        data = self._post("/v2/charts/intraday", payload)
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["timestamp"], unit="s", utc=True),
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
        })
        return df.sort_values("timestamp").tail(count).reset_index(drop=True)

    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        # Dhan option chain — instrument-based approach
        expiry = expiry or self.get_nearest_expiry(symbol)
        # NOTE: Dhan requires security_id lookup for each strike; this is a simplified stub
        logger.warning("DhanAdapter.get_option_chain is a stub; full implementation requires security ID mapping.")
        raise NotImplementedError("DhanAdapter.get_option_chain requires full security ID mapping. Use MOCK_MODE.")

    def get_india_vix(self) -> float:
        data = self._post("/v2/marketfeed/ltp", {"NSE_INDEX": ["INDIAVIX"]})
        return float(data["data"]["NSE_INDEX"]["INDIAVIX"]["last_price"])

    def get_nearest_expiry(self, symbol: str) -> str:
        from datetime import date, timedelta
        # Approximate: Thursday for NIFTY/FINNIFTY, Wednesday for BANKNIFTY, Friday for SENSEX
        today = date.today()
        target_weekday = 3  # Thursday
        if symbol == "BANKNIFTY":
            target_weekday = 2  # Wednesday
        elif symbol == "SENSEX":
            target_weekday = 4  # Friday
        days_ahead = (target_weekday - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (today + timedelta(days=days_ahead)).isoformat()


# ===========================================================================
# Zerodha Kite Connect Adapter
# ===========================================================================

class ZerodhaAdapter(BaseBrokerAdapter):
    """
    Zerodha Kite Connect v3 adapter.
    
    API Docs: https://kite.trade/docs/connect/v3/
    
    Required env vars:
        BROKER_API_KEY    — Kite Connect API key
        BROKER_API_SECRET — Kite Connect API secret
    
    Authentication flow (one-time per session):
        1. Direct user to login URL: kite.login_url()
        2. User logs in; Kite redirects to your redirect_url with request_token
        3. Call kite.generate_session(request_token, api_secret) to get access_token
        4. Set BROKER_API_KEY to the resulting access_token for subsequent runs
        
    Install kiteconnect separately: pip install kiteconnect
    """

    def __init__(self):
        try:
            from kiteconnect import KiteConnect  # type: ignore
        except ImportError:
            raise BrokerError("kiteconnect package not installed. Run: pip install kiteconnect")
        self._kite = KiteConnect(api_key=cfg.BROKER_API_KEY)
        self._kite.set_access_token(cfg.BROKER_API_SECRET)  # SECRET field holds access_token
        logger.info("ZerodhaAdapter initialised.")

    # Zerodha index trading symbols
    SYMBOL_MAP = {
        "NIFTY": "NSE:NIFTY 50",
        "BANKNIFTY": "NSE:NIFTY BANK",
        "FINNIFTY": "NSE:NIFTY FIN SERVICE",
        "SENSEX": "BSE:SENSEX",
    }

    INTERVAL_MAP = {
        "5m": "5minute",
        "15m": "15minute",
        "1h": "60minute",
    }

    def get_spot_price(self, symbol: str) -> float:
        quote = self._kite.ltp(self.SYMBOL_MAP[symbol])
        return float(quote[self.SYMBOL_MAP[symbol]]["last_price"])

    def get_ohlcv(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        from datetime import date, timedelta
        from_date = date.today() - timedelta(days=5)
        records = self._kite.historical_data(
            instrument_token=self._get_token(symbol),
            from_date=from_date,
            to_date=date.today(),
            interval=self.INTERVAL_MAP.get(timeframe, "5minute"),
        )
        df = pd.DataFrame(records)
        df = df.rename(columns={"date": "timestamp"})
        return df.sort_values("timestamp").tail(count).reset_index(drop=True)[
            ["timestamp", "open", "high", "low", "close", "volume"]
        ]

    def _get_token(self, symbol: str) -> int:
        """Lookup instrument token for index — requires instruments() call."""
        instruments = self._kite.instruments("BSE" if symbol == "SENSEX" else "NSE")
        mapping = {
            "NIFTY": "NIFTY 50",
            "BANKNIFTY": "NIFTY BANK",
            "FINNIFTY": "NIFTY FIN SERVICE",
            "SENSEX": "SENSEX",
        }
        name = mapping.get(symbol, symbol)
        for inst in instruments:
            if inst["name"] == name and inst["segment"] == "INDICES":
                return inst["instrument_token"]
        raise BrokerError(f"Could not find instrument token for {symbol}")

    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> pd.DataFrame:
        # Zerodha option chain requires fetching all instruments and querying quote for each
        # This is a simplified implementation — production use should cache instruments list
        import requests
        expiry = expiry or self.get_nearest_expiry(symbol)
        all_instruments = self._kite.instruments("NFO")
        filtered = [
            i for i in all_instruments
            if i["name"] == symbol
            and i["expiry"].isoformat() == expiry
            and i["instrument_type"] in ("CE", "PE")
        ]
        rows: Dict[float, dict] = {}
        for inst in filtered:
            strike = inst["strike"]
            if strike not in rows:
                rows[strike] = {k: 0 for k in ["strike","ce_oi","ce_chg_oi","ce_volume","ce_iv","ce_ltp","pe_oi","pe_chg_oi","pe_volume","pe_iv","pe_ltp"]}
                rows[strike]["strike"] = strike
            prefix = "ce_" if inst["instrument_type"] == "CE" else "pe_"
            q = self._kite.quote(f"NFO:{inst['tradingsymbol']}")
            d = q.get(f"NFO:{inst['tradingsymbol']}", {})
            rows[strike][f"{prefix}oi"] = d.get("oi", 0)
            rows[strike][f"{prefix}chg_oi"] = d.get("oi_day_change", 0)
            rows[strike][f"{prefix}volume"] = d.get("volume", 0)
            rows[strike][f"{prefix}iv"] = d.get("implied_volatility", 0)
            rows[strike][f"{prefix}ltp"] = d.get("last_price", 0)
        return pd.DataFrame(list(rows.values())).sort_values("strike").reset_index(drop=True)

    def get_india_vix(self) -> float:
        quote = self._kite.ltp("NSE:INDIA VIX")
        return float(quote["NSE:INDIA VIX"]["last_price"])

    def get_nearest_expiry(self, symbol: str) -> str:
        from datetime import date, timedelta
        today = date.today()
        target_weekday = 3
        if symbol == "BANKNIFTY":
            target_weekday = 2
        elif symbol == "SENSEX":
            target_weekday = 4
        days_ahead = (target_weekday - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return (today + timedelta(days=days_ahead)).isoformat()


# ===========================================================================
# Adapter Registry & Factory
# ===========================================================================

BROKER_REGISTRY: Dict[str, type] = {
    "upstox": UpstoxAdapter,
    "dhan": DhanAdapter,
    "zerodha": ZerodhaAdapter,
}


def get_broker_adapter() -> BaseBrokerAdapter:
    """
    Factory function — returns the configured broker adapter instance.

    Usage:
        from broker_adapter import get_broker_adapter
        adapter = get_broker_adapter()
        spot = adapter.get_spot_price("NIFTY")

    Raises:
        BrokerError: If the broker name is unknown or credentials are missing.
    """
    broker = cfg.BROKER_NAME
    cls = BROKER_REGISTRY.get(broker)
    if cls is None:
        available = list(BROKER_REGISTRY.keys())
        raise BrokerError(
            f"Unknown broker '{broker}'. "
            f"Available brokers: {available}. "
            f"Set BROKER_NAME in .env or enable MOCK_MODE=True."
        )
    return cls()
