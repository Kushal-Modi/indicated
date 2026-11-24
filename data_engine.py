"""
data_engine.py — Market Data Fetcher & Mock Generator
======================================================
Provides a unified interface for fetching:
  - Spot / index price
  - OHLCV candlestick data
  - Option Chain (CE/PE OI, Change in OI, Volume, IV)
  - India VIX

Two operational modes controlled by cfg.MOCK_MODE:
  - MOCK_MODE=True  → Generates realistic synthetic data using
                      Geometric Brownian Motion + intraday vol patterns.
                      Zero API keys required.
  - MOCK_MODE=False → Routes calls through the configured broker adapter.

All public functions return pandas DataFrames or simple scalars with
consistent column names regardless of mode.
"""

from __future__ import annotations

import logging
import math
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# India Standard Time offset
# ---------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------------------
# Mock market session constants
# ---------------------------------------------------------------------------
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30

# Annualised volatility assumptions per index (used in GBM)
_ANNUAL_VOL: Dict[str, float] = {
    "NIFTY": 0.16,
    "BANKNIFTY": 0.20,
    "FINNIFTY": 0.18,
    "SENSEX": 0.15,
}

# Daily drift (slight bullish bias)
_ANNUAL_DRIFT: Dict[str, float] = {
    "NIFTY": 0.12,
    "BANKNIFTY": 0.10,
    "FINNIFTY": 0.11,
    "SENSEX": 0.11,
}


# ===========================================================================
# MOCK DATA GENERATORS
# ===========================================================================

def _timeframe_minutes(tf: str) -> int:
    """Convert timeframe string to minutes."""
    return {"5m": 5, "15m": 15, "1h": 60}.get(tf, 5)


def _generate_mock_candles(
    symbol: str,
    timeframe: str,
    count: int,
    seed_price: float,
) -> pd.DataFrame:
    """
    Generate realistic OHLCV candles using Geometric Brownian Motion.

    The simulation applies an intraday volatility smile — higher volatility
    at open and close, lower in mid-session — to mimic real market behaviour.

    Args:
        symbol: Index name for vol/drift params
        timeframe: "5m" | "15m" | "1h"
        count: Number of candles to generate
        seed_price: Starting price for the simulation

    Returns:
        pd.DataFrame[timestamp, open, high, low, close, volume]
    """
    tf_minutes = _timeframe_minutes(timeframe)
    annual_vol = _ANNUAL_VOL.get(symbol, 0.16)
    annual_drift = _ANNUAL_DRIFT.get(symbol, 0.12)

    # Scale to per-candle parameters
    minutes_per_year = 252 * 6.25 * 60  # ~252 trading days, 6h15m sessions
    dt = tf_minutes / minutes_per_year
    per_candle_vol = annual_vol * math.sqrt(dt)
    per_candle_drift = annual_drift * dt

    # Build timestamps going backwards from "now" (capped to market hours)
    now_ist = datetime.now(IST)
    # Snap to previous candle boundary
    minute_snap = (now_ist.minute // tf_minutes) * tf_minutes
    latest = now_ist.replace(minute=minute_snap, second=0, microsecond=0)

    timestamps = []
    t = latest
    while len(timestamps) < count:
        # Only include bars within market hours (09:15 – 15:30 IST)
        bar_ok = (
            (t.hour == MARKET_OPEN_HOUR and t.minute >= MARKET_OPEN_MINUTE) or
            (MARKET_OPEN_HOUR < t.hour < MARKET_CLOSE_HOUR) or
            (t.hour == MARKET_CLOSE_HOUR and t.minute <= MARKET_CLOSE_MINUTE)
        )
        if bar_ok and t.weekday() < 5:  # Mon–Fri
            timestamps.append(t)
        t -= timedelta(minutes=tf_minutes)

    timestamps = list(reversed(timestamps))

    # Simulate GBM price path
    np.random.seed(int(seed_price) % 1000)  # reproducible per symbol
    prices = [seed_price]
    for i in range(1, len(timestamps)):
        # Intraday vol smile: higher vol at open and close
        minute_of_day = timestamps[i].hour * 60 + timestamps[i].minute
        market_open_min = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MINUTE
        market_close_min = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MINUTE
        session_progress = (minute_of_day - market_open_min) / (market_close_min - market_open_min)
        session_progress = max(0.0, min(1.0, session_progress))
        # U-shaped intraday vol multiplier
        intraday_mult = 1.0 + 1.5 * (1 - 4 * session_progress * (1 - session_progress))

        z = np.random.standard_normal()
        log_return = (per_candle_drift - 0.5 * per_candle_vol ** 2) + per_candle_vol * intraday_mult * z
        prices.append(prices[-1] * math.exp(log_return))

    # Scale series so the latest candle close matches the real-world current spot price
    if prices[-1] != 0:
        scale = seed_price / prices[-1]
        prices = [p * scale for p in prices]

    # Build OHLCV candles from close prices
    rows = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        prev_close = prices[i - 1] if i > 0 else close
        open_price = prev_close

        # Realistic high/low using intrabar volatility
        intrabar_range = abs(close - open_price) + abs(np.random.normal(0, per_candle_vol * prev_close * 0.3))
        high = max(open_price, close) + abs(np.random.normal(0, intrabar_range * 0.5))
        low = min(open_price, close) - abs(np.random.normal(0, intrabar_range * 0.5))

        # Volume: higher at open/close, random with positive noise
        base_vol = random.randint(50_000, 500_000)
        rows.append({
            "timestamp": ts,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": int(base_vol * random.uniform(0.7, 1.4)),
        })

    return pd.DataFrame(rows)


def _generate_mock_option_chain(
    symbol: str,
    spot: float,
    vix: float,
) -> pd.DataFrame:
    """
    Generate a realistic option chain around ATM ± cfg.OI_STRIKES_RANGE strikes.

    OI distribution follows a bell curve centred near ATM, with:
      - CE OI peaking slightly OTM (resistance above spot)
      - PE OI peaking slightly OTM below spot (support)
      - Change in OI adds directional bias noise
      - IV uses a realistic vol surface smile (higher at wings)

    Args:
        symbol: Index name
        spot: Current spot price
        vix: India VIX (used to scale IV)
    Returns:
        pd.DataFrame with full option chain columns
    """
    gap = cfg.STRIKE_GAPS[symbol]
    atm = round(spot / gap) * gap  # Round to nearest valid strike
    n = cfg.OI_STRIKES_RANGE
    strikes = [atm + (i - n) * gap for i in range(2 * n + 1)]

    # Base OI quantities (realistic NSE scale)
    ce_base_oi = random.randint(1_000_000, 8_000_000)
    pe_base_oi = random.randint(1_000_000, 8_000_000)

    rows = []
    for strike in strikes:
        distance = (strike - atm) / gap  # +ve = OTM for CE, ITM for PE

        # CE OI: peaks at +1 to +3 strikes OTM
        ce_peak_offset = random.uniform(1.0, 3.0)
        ce_oi_weight = math.exp(-0.15 * (distance - ce_peak_offset) ** 2)
        ce_oi = int(ce_base_oi * ce_oi_weight * random.uniform(0.8, 1.2))

        # PE OI: peaks at -1 to -3 strikes OTM
        pe_peak_offset = random.uniform(-3.0, -1.0)
        pe_oi_weight = math.exp(-0.15 * (distance - pe_peak_offset) ** 2)
        pe_oi = int(pe_base_oi * pe_oi_weight * random.uniform(0.8, 1.2))

        # Change in OI (directional noise — slight long bias)
        ce_chg_oi = int(ce_oi * random.uniform(-0.08, 0.15))
        pe_chg_oi = int(pe_oi * random.uniform(-0.10, 0.12))

        # Volume proportional to OI
        ce_vol = int(ce_oi * random.uniform(0.03, 0.12))
        pe_vol = int(pe_oi * random.uniform(0.03, 0.12))

        # IV smile: higher at wings (ATM IV ~ vix/sqrt(252)*sqrt(dte*252))
        atm_iv = vix / 100 * math.sqrt(7 / 365) * 100  # 7 DTE approximation
        smile_factor = 1.0 + 0.02 * distance ** 2        # quadratic smile
        ce_iv = round(atm_iv * smile_factor * random.uniform(0.95, 1.05), 2)
        pe_iv = round(atm_iv * smile_factor * random.uniform(0.95, 1.05), 2)

        # LTP: intrinsic + time value approximation
        ce_intrinsic = max(0.0, spot - strike)
        pe_intrinsic = max(0.0, strike - spot)
        time_val = spot * atm_iv / 100 * math.sqrt(7 / 365) * random.uniform(0.8, 1.2)
        ce_ltp = round(ce_intrinsic + time_val, 2)
        pe_ltp = round(pe_intrinsic + time_val, 2)

        rows.append({
            "strike": strike,
            "ce_oi": ce_oi,
            "ce_chg_oi": ce_chg_oi,
            "ce_volume": ce_vol,
            "ce_iv": ce_iv,
            "ce_ltp": ce_ltp,
            "pe_oi": pe_oi,
            "pe_chg_oi": pe_chg_oi,
            "pe_volume": pe_vol,
            "pe_iv": pe_iv,
            "pe_ltp": pe_ltp,
        })

    return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)


def _generate_mock_vix(symbol: str) -> float:
    """
    Generate a realistic India VIX reading.
    VIX typically oscillates 10–28, mean-reverting around 15.
    """
    # Use deterministic seed so VIX doesn't jump every refresh
    base_vix = {"NIFTY": 14.5, "BANKNIFTY": 16.0, "FINNIFTY": 15.5, "SENSEX": 14.0}.get(symbol, 14.5)
    noise = np.random.normal(0, 1.5)
    vix = base_vix + noise
    # Mean-revert bounds: clamp to [9, 35]
    return round(max(9.0, min(35.0, vix)), 2)


# ===========================================================================
# PUBLIC INTERFACE
# ===========================================================================

def get_spot_price(symbol: str) -> float:
    """
    Return the current spot price for the given index.
    """
    if cfg.MOCK_MODE:
        return round(float(cfg.SPOT_SEEDS.get(symbol, 25514.30)), 2)
    else:
        from broker_adapter import get_broker_adapter
        adapter = get_broker_adapter()
        return adapter.get_spot_price(symbol)


def get_ohlcv(symbol: str, timeframe: str = None, count: int = None) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data.

    Args:
        symbol: "NIFTY" | "BANKNIFTY" | "FINNIFTY"
        timeframe: "5m" | "15m" | "1h" (defaults to cfg.TIMEFRAME)
        count: Number of candles (defaults to cfg.CANDLE_COUNT)
    Returns:
        pd.DataFrame[timestamp, open, high, low, close, volume]
    """
    timeframe = timeframe or cfg.TIMEFRAME
    count = count or cfg.CANDLE_COUNT
    if cfg.MOCK_MODE:
        seed_price = cfg.SPOT_SEEDS.get(symbol, 24500.0)
        return _generate_mock_candles(symbol, timeframe, count, seed_price)
    else:
        from broker_adapter import get_broker_adapter
        adapter = get_broker_adapter()
        return adapter.get_ohlcv(symbol, timeframe, count)


def get_option_chain(symbol: str, spot: float = None, vix: float = None) -> pd.DataFrame:
    """
    Fetch the option chain (CE & PE) around ATM ± cfg.OI_STRIKES_RANGE strikes.

    Args:
        symbol: Index name
        spot: Current spot price (used for ATM calculation in mock mode)
        vix: India VIX (used for IV simulation in mock mode)
    Returns:
        pd.DataFrame with columns:
            strike, ce_oi, ce_chg_oi, ce_volume, ce_iv, ce_ltp,
            pe_oi, pe_chg_oi, pe_volume, pe_iv, pe_ltp
    """
    if cfg.MOCK_MODE:
        spot = spot or get_spot_price(symbol)
        vix = vix or get_india_vix(symbol)
        return _generate_mock_option_chain(symbol, spot, vix)
    else:
        from broker_adapter import get_broker_adapter
        adapter = get_broker_adapter()
        return adapter.get_option_chain(symbol)


def get_india_vix(symbol: str = "NIFTY") -> float:
    """
    Return the current India VIX reading.
    
    Args:
        symbol: Passed for mock mode seeding only (ignored in live mode)
    Returns:
        float: VIX value
    """
    if cfg.MOCK_MODE:
        return _generate_mock_vix(symbol)
    else:
        from broker_adapter import get_broker_adapter
        adapter = get_broker_adapter()
        return adapter.get_india_vix()


def get_all_market_data(symbol: str, timeframe: str = None) -> Dict:
    """
    Convenience function: fetches all data in one call and returns a dict.

    Returns:
        {
            "spot": float,
            "vix": float,
            "candles": pd.DataFrame,
            "option_chain": pd.DataFrame,
        }
    """
    timeframe = timeframe or cfg.TIMEFRAME
    logger.info(f"Fetching market data for {symbol} [{timeframe}] | mock={cfg.MOCK_MODE}")

    spot = get_spot_price(symbol)
    vix = get_india_vix(symbol)
    candles = get_ohlcv(symbol, timeframe)
    option_chain = get_option_chain(symbol, spot=spot, vix=vix)

    return {
        "spot": spot,
        "vix": vix,
        "candles": candles,
        "option_chain": option_chain,
    }
