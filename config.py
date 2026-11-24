"""
config.py — Centralised Configuration Loader
=============================================
Reads all environment variables from .env file and exposes typed
constants used across every module. Import this everywhere instead
of calling os.getenv() directly.

Usage:
    from config import cfg
    if cfg.MOCK_MODE:
        ...
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root (the directory containing this script)
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH, override=True)


def _bool(val: str) -> bool:
    """Convert env-string to bool (handles True/False/1/0/yes/no)."""
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _int(val: str, default: int) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _list(val: str, sep: str = ",") -> List[str]:
    """Split a comma-separated env string into a list of stripped strings."""
    if not val:
        return []
    return [v.strip() for v in val.split(sep) if v.strip()]


# ---------------------------------------------------------------------------
# Configuration dataclass — single source of truth
# ---------------------------------------------------------------------------
@dataclass
class AppConfig:
    # ---- Core ----
    MOCK_MODE: bool = True
    SYMBOL: str = "NIFTY"
    TIMEFRAME: str = "5m"
    AUTO_REFRESH_SECONDS: int = 60

    # ---- Broker ----
    BROKER_NAME: str = "upstox"
    BROKER_API_KEY: str = ""
    BROKER_API_SECRET: str = ""

    # ---- LLM ----
    LLM_PROVIDER: str = "gemini"          # "gemini" | "openai" | ""
    LLM_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ---- News ----
    RSS_FEEDS: List[str] = field(default_factory=list)

    # ---- Analytics constants (not in .env, hardcoded sensible defaults) ----
    ADX_PERIOD: int = 14
    ATR_PERIOD: int = 14
    EMA_PERIOD: int = 20
    OI_STRIKES_RANGE: int = 10          # ATM ± 10 strikes
    CANDLE_COUNT: int = 100             # Number of historical candles to fetch/generate

    # ---- Index-specific lot sizes (NSE / BSE standard) ----
    LOT_SIZES: dict = field(default_factory=lambda: {
        "NIFTY": 25,
        "BANKNIFTY": 15,
        "FINNIFTY": 40,
        "SENSEX": 20,
    })

    # ---- Index-specific strike gaps ----
    STRIKE_GAPS: dict = field(default_factory=lambda: {
        "NIFTY": 50,
        "BANKNIFTY": 100,
        "FINNIFTY": 50,
        "SENSEX": 100,
    })

    # ---- Approximate spot reference (used for mock seed) ----
    SPOT_SEEDS: dict = field(default_factory=lambda: {
        "NIFTY": 25349.65,
        "BANKNIFTY": 62585.35,
        "FINNIFTY": 28153.50,
        "SENSEX": 82435.99,
    })


def _load_config() -> AppConfig:
    """Read environment variables and build the AppConfig instance."""
    default_feeds = [
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.livemint.com/rss/markets",
    ]
    return AppConfig(
        MOCK_MODE=_bool(os.getenv("MOCK_MODE", "True")),
        SYMBOL=os.getenv("SYMBOL", "NIFTY").upper(),
        TIMEFRAME=os.getenv("TIMEFRAME", "5m"),
        AUTO_REFRESH_SECONDS=_int(os.getenv("AUTO_REFRESH_SECONDS", "60"), 60),

        BROKER_NAME=os.getenv("BROKER_NAME", "upstox").lower(),
        BROKER_API_KEY=os.getenv("BROKER_API_KEY", ""),
        BROKER_API_SECRET=os.getenv("BROKER_API_SECRET", ""),

        LLM_PROVIDER=os.getenv("LLM_PROVIDER", "gemini").lower(),
        LLM_API_KEY=os.getenv("LLM_API_KEY", ""),
        GEMINI_MODEL=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        OPENAI_MODEL=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),

        RSS_FEEDS=_list(os.getenv("RSS_FEEDS", "")) or default_feeds,
    )


# ---------------------------------------------------------------------------
# Singleton — import `cfg` everywhere
# ---------------------------------------------------------------------------
cfg: AppConfig = _load_config()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
VALID_SYMBOLS = ["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"]
VALID_TIMEFRAMES = ["5m", "15m", "1h"]
VALID_BROKERS = ["upstox", "dhan", "zerodha"]
VALID_LLM_PROVIDERS = ["gemini", "openai", ""]


def validate_config() -> List[str]:
    """Return a list of validation warnings (empty = all good)."""
    warnings: List[str] = []
    if cfg.SYMBOL not in VALID_SYMBOLS:
        warnings.append(f"SYMBOL '{cfg.SYMBOL}' unknown; defaulting to NIFTY.")
    if cfg.TIMEFRAME not in VALID_TIMEFRAMES:
        warnings.append(f"TIMEFRAME '{cfg.TIMEFRAME}' not in {VALID_TIMEFRAMES}.")
    if not cfg.MOCK_MODE and not cfg.BROKER_API_KEY:
        warnings.append("MOCK_MODE=False but BROKER_API_KEY is empty. Live data will fail.")
    if cfg.LLM_PROVIDER and cfg.LLM_PROVIDER not in VALID_LLM_PROVIDERS:
        warnings.append(f"LLM_PROVIDER '{cfg.LLM_PROVIDER}' unknown. Falling back to rule-based.")
    return warnings
