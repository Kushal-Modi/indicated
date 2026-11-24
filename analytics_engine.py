"""
analytics_engine.py — Technical & Options Analytics Core
=========================================================
Three analytical subsystems:

1. Market Regime Filter
   ─────────────────────
   Uses ADX(14), ATR(14), EMA(20), and VWAP computed via pandas-ta.
   Classifies market state into one of:
     • TRENDING BULLISH
     • TRENDING BEARISH
     • SIDEWAYS / CHOP (AVOID BUYING)

2. Option Chain & OI Analyzer
   ───────────────────────────
   Computes:
     • Put-Call Ratio (PCR) by OI
     • Max Pain strike
     • Highest CE OI strike (resistance wall)
     • Highest PE OI strike (support wall)
     • Per-strike buildup classification:
       Long Buildup | Short Buildup | Short Covering | Long Unwinding

3. Confluence Signal Generator
   ────────────────────────────
   Fires BUY CE / BUY PE / NO TRADE only when ≥2 of 3 conditions align:
     • Technical direction (from EMA + VWAP slope)
     • Market regime (from ADX)
     • OI buildup direction
   Outputs Entry (spot), ATR-based Stop-Loss, and 2:1 RR Target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from config import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import pandas-ta; graceful fallback to manual calculations
# ---------------------------------------------------------------------------
try:
    import pandas_ta as ta  # type: ignore
    _PANDAS_TA_AVAILABLE = True
except ImportError:
    _PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta not available; falling back to manual indicator calculations.")


# ===========================================================================
# Data Containers
# ===========================================================================

@dataclass
class MarketRegime:
    """Output of the Market Regime Filter."""
    state: str                          # "TRENDING BULLISH" | "TRENDING BEARISH" | "SIDEWAYS / CHOP"
    adx: float                          # ADX value (0–100)
    atr: float                          # ATR value in price units
    ema20: float                        # Latest 20-period EMA
    vwap: float                         # Session VWAP
    ema_slope: float                    # EMA slope (last bar change)
    emoji: str = field(init=False)      # Visual indicator

    def __post_init__(self):
        emoji_map = {
            "TRENDING BULLISH": "🟢",
            "TRENDING BEARISH": "🔴",
            "SIDEWAYS / CHOP": "🟡",
        }
        self.emoji = emoji_map.get(self.state, "⚪")

    @property
    def color(self) -> str:
        color_map = {
            "TRENDING BULLISH": "#00d26a",
            "TRENDING BEARISH": "#ff4b4b",
            "SIDEWAYS / CHOP": "#ffd700",
        }
        return color_map.get(self.state, "#aaaaaa")

    @property
    def short_label(self) -> str:
        label_map = {
            "TRENDING BULLISH": "BULL TREND",
            "TRENDING BEARISH": "BEAR TREND",
            "SIDEWAYS / CHOP": "SIDEWAYS",
        }
        return label_map.get(self.state, self.state)


@dataclass
class OIAnalysis:
    """Output of the OI Analyzer."""
    pcr: float                          # Put-Call Ratio by OI
    max_pain: float                     # Max pain strike
    ce_resistance: float                # Strike with highest CE OI (resistance)
    pe_support: float                   # Strike with highest PE OI (support)
    total_ce_oi: int
    total_pe_oi: int
    # Per-strike buildup classification
    buildup_df: pd.DataFrame            # Columns: strike, buildup, ce_oi, pe_oi, ...
    # Overall dominant buildup
    dominant_buildup: str               # "Long Buildup" | "Short Buildup" | etc.
    oi_bias: str                        # "BULLISH" | "BEARISH" | "NEUTRAL"
    # Volume statistics (NSE Traded Contracts)
    total_ce_volume: int = 0
    total_pe_volume: int = 0
    volume_pcr: float = 1.0             # PE Volume / CE Volume
    volume_bias: str = "BALANCED"       # "CE (CALL) HEAVY" | "PE (PUT) HEAVY" | "BALANCED"


@dataclass
class TradeSignal:
    """Output of the Confluence Signal Generator."""
    signal: str                         # "BUY CE" | "BUY PE" | "NO TRADE"
    entry: float
    stop_loss: float
    target: float
    risk_reward: float
    confidence: int                     # 1–3 (number of aligned conditions)
    reasons: List[str]                  # Human-readable alignment reasons
    atr: float
    # Practical Option Execution Plan (Actionable for Option Buyers)
    recommended_strike: float = 0.0
    strike_label: str = ""              # e.g. "NIFTY 24500 CE (ATM)"
    option_entry: float = 0.0           # Current estimated option premium LTP
    option_sl: float = 0.0              # Option Stop-Loss premium
    option_target_1: float = 0.0        # Target 1 (book 50% & trail SL to cost)
    option_target_2: float = 0.0        # Target 2 (runner exit / 2:1 RR)
    entry_rule: str = ""                # When to enter
    exit_target_1_rule: str = ""        # When to take partial profit
    exit_target_2_rule: str = ""        # When to exit full
    emergency_exit_rule: str = ""       # Invalidation rule (sell immediately)

    @property
    def emoji(self) -> str:
        return {"BUY CE": "🔼", "BUY PE": "🔽", "NO TRADE": "⏸️"}.get(self.signal, "❓")

    @property
    def color(self) -> str:
        return {"BUY CE": "#00d26a", "BUY PE": "#ff4b4b", "NO TRADE": "#ffd700"}.get(self.signal, "#aaa")


# ===========================================================================
# 1. MARKET REGIME FILTER
# ===========================================================================

def _manual_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute EMA manually (fallback if pandas-ta unavailable)."""
    return series.ewm(span=period, adjust=False).mean()


def _manual_adx(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Manual ADX calculation using Wilder's smoothing.
    Returns pd.Series of ADX values aligned to df index.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder smoothing (EWM approximation)
    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    adx.index = df.index
    return adx


def _manual_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Compute ATR manually."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute VWAP (Volume Weighted Average Price).
    Uses (H+L+C)/3 as typical price, resets each session.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (typical * df["volume"]).cumsum()
    return (cum_tp_vol / cum_vol.replace(0, np.nan)).ffill()


def compute_market_regime(df: pd.DataFrame, adx_period: int = 14, atr_period: int = 14, ema_period: int = 20) -> Tuple[MarketRegime, pd.DataFrame]:
    """
    Compute technical indicators and classify market regime.

    Args:
        df: OHLCV DataFrame (columns: timestamp, open, high, low, close, volume)
        adx_period: ADX lookback period
        atr_period: ATR lookback period
        ema_period: EMA lookback period

    Returns:
        Tuple of (MarketRegime, enriched_df)
        enriched_df adds columns: ema, vwap, atr, adx
    """
    df = df.copy().reset_index(drop=True)

    if _PANDAS_TA_AVAILABLE:
        # Use pandas-ta for all indicators
        df.ta.ema(length=ema_period, append=True)
        df.ta.atr(length=atr_period, append=True)
        df.ta.adx(length=adx_period, append=True)

        ema_col = f"EMA_{ema_period}"
        atr_col = f"ATRr_{atr_period}"
        adx_col = f"ADX_{adx_period}"

        # Rename to standard names
        df["ema"] = df.get(ema_col, _manual_ema(df["close"], ema_period))
        df["atr"] = df.get(atr_col, _manual_atr(df, atr_period))
        df["adx"] = df.get(adx_col, _manual_adx(df, adx_period))
    else:
        # Manual fallback
        df["ema"] = _manual_ema(df["close"], ema_period)
        df["atr"] = _manual_atr(df, atr_period)
        df["adx"] = _manual_adx(df, adx_period)

    # Always compute VWAP manually (pandas-ta's VWAP needs datetime index)
    df["vwap"] = _compute_vwap(df)

    # Extract last-row indicators
    last = df.iloc[-1]
    adx_val = float(last.get("adx", 20.0))
    atr_val = float(last.get("atr", 50.0))
    ema_val = float(last.get("ema", last["close"]))
    vwap_val = float(last.get("vwap", last["close"]))
    close_val = float(last["close"])

    # EMA slope: last EMA minus EMA 3 bars ago
    ema_slope = float(df["ema"].iloc[-1] - df["ema"].iloc[-4]) if len(df) >= 4 else 0.0

    # ── Regime Classification Logic ──────────────────────────────────────────
    # ADX > 25 → trending; ADX ≤ 25 → sideways/chop
    # Bullish conditions: price above VWAP, above EMA, EMA slope positive
    # Bearish conditions: price below VWAP, below EMA, EMA slope negative
    is_trending = adx_val >= 25.0
    above_vwap = close_val > vwap_val
    above_ema = close_val > ema_val
    positive_slope = ema_slope > 0

    bullish_score = sum([above_vwap, above_ema, positive_slope])  # 0-3
    bearish_score = sum([not above_vwap, not above_ema, not positive_slope])  # 0-3

    if is_trending and bullish_score >= 2:
        state = "TRENDING BULLISH"
    elif is_trending and bearish_score >= 2:
        state = "TRENDING BEARISH"
    else:
        state = "SIDEWAYS / CHOP"

    regime = MarketRegime(
        state=state,
        adx=round(adx_val, 2),
        atr=round(atr_val, 2),
        ema20=round(ema_val, 2),
        vwap=round(vwap_val, 2),
        ema_slope=round(ema_slope, 4),
    )

    logger.info(f"Market Regime: {state} | ADX={adx_val:.1f} | ATR={atr_val:.1f} | EMA={ema_val:.1f}")
    return regime, df


# ===========================================================================
# 2. OPTION CHAIN & OI ANALYZER
# ===========================================================================

_BUILDUP_RULES = {
    # (price_up, oi_up) → buildup type
    (True, True): "Long Buildup",
    (False, True): "Short Buildup",
    (False, False): "Short Covering",
    (True, False): "Long Unwinding",
}


def _classify_buildup(price_change: float, oi_change: float) -> str:
    """Classify buildup based on price and OI change direction."""
    price_up = price_change >= 0
    oi_up = oi_change >= 0
    return _BUILDUP_RULES[(price_up, oi_up)]


def compute_max_pain(option_chain: pd.DataFrame) -> float:
    """
    Calculate the Max Pain strike — the price at which maximum options expire worthless.

    For each strike S, compute total loss = sum over all strikes k of:
      CE loss: max(S - k, 0) * CE_OI(k)
      PE loss: max(k - S, 0) * PE_OI(k)

    Max Pain = S that minimises total loss.

    Args:
        option_chain: DataFrame with columns strike, ce_oi, pe_oi
    Returns:
        float: Max Pain strike price
    """
    strikes = option_chain["strike"].values
    ce_oi = option_chain["ce_oi"].values
    pe_oi = option_chain["pe_oi"].values

    min_loss = float("inf")
    max_pain_strike = strikes[len(strikes) // 2]  # default to ATM

    for s in strikes:
        ce_loss = np.sum(np.maximum(s - strikes, 0) * ce_oi)
        pe_loss = np.sum(np.maximum(strikes - s, 0) * pe_oi)
        total = ce_loss + pe_loss
        if total < min_loss:
            min_loss = total
            max_pain_strike = s

    return float(max_pain_strike)


def analyze_option_chain(option_chain: pd.DataFrame, spot: float) -> OIAnalysis:
    """
    Perform complete OI analysis on the option chain.

    Args:
        option_chain: DataFrame from data_engine.get_option_chain()
        spot: Current spot price
    Returns:
        OIAnalysis dataclass
    """
    df = option_chain.copy()

    # ── PCR (Open Interest) ───────────────────────────────────────────────────
    total_ce_oi = int(df["ce_oi"].sum())
    total_pe_oi = int(df["pe_oi"].sum())
    pcr = round(total_pe_oi / max(total_ce_oi, 1), 3)

    # ── Volume Analysis (NSE Traded Contracts) ────────────────────────────────
    total_ce_volume = int(df["ce_volume"].sum()) if "ce_volume" in df.columns else 0
    total_pe_volume = int(df["pe_volume"].sum()) if "pe_volume" in df.columns else 0
    volume_pcr = round(total_pe_volume / max(total_ce_volume, 1), 3)

    if total_pe_volume > total_ce_volume * 1.15:
        volume_bias = "PE (PUT) HEAVY"
    elif total_ce_volume > total_pe_volume * 1.15:
        volume_bias = "CE (CALL) HEAVY"
    else:
        volume_bias = "BALANCED"

    # ── Max Pain ──────────────────────────────────────────────────────────────
    max_pain = compute_max_pain(df)

    # ── Support & Resistance Walls ────────────────────────────────────────────
    # Resistance: highest CE OI (call writers have written most calls here)
    ce_resistance_idx = df["ce_oi"].idxmax()
    ce_resistance = float(df.loc[ce_resistance_idx, "strike"])

    # Support: highest PE OI (put writers have written most puts here)
    pe_support_idx = df["pe_oi"].idxmax()
    pe_support = float(df.loc[pe_support_idx, "strike"])

    # ── Per-Strike Buildup Classification ────────────────────────────────────
    buildup_rows = []
    for _, row in df.iterrows():
        # Classify CE buildup (use CE price change proxy: ltp vs "yesterday" = assume chg_oi direction)
        ce_price_up = row["ce_ltp"] > 0  # simplified: positive LTP means CE going up
        ce_buildup = _classify_buildup(
            price_change=row["ce_chg_oi"],   # use chg_oi as proxy for activity direction
            oi_change=row["ce_chg_oi"]
        )
        pe_buildup = _classify_buildup(
            price_change=row["pe_chg_oi"],
            oi_change=row["pe_chg_oi"]
        )
        buildup_rows.append({
            "strike": row["strike"],
            "ce_oi": row["ce_oi"],
            "ce_chg_oi": row["ce_chg_oi"],
            "ce_iv": row["ce_iv"],
            "ce_ltp": row["ce_ltp"],
            "pe_oi": row["pe_oi"],
            "pe_chg_oi": row["pe_chg_oi"],
            "pe_iv": row["pe_iv"],
            "pe_ltp": row["pe_ltp"],
            "ce_buildup": ce_buildup,
            "pe_buildup": pe_buildup,
            "is_atm": abs(row["strike"] - spot) < 50,
            "is_ce_resistance": row["strike"] == ce_resistance,
            "is_pe_support": row["strike"] == pe_support,
        })

    buildup_df = pd.DataFrame(buildup_rows)

    # ── Dominant Buildup & OI Bias ────────────────────────────────────────────
    # Aggregate: compare total CE chg_oi vs PE chg_oi
    ce_chg_sum = df["ce_chg_oi"].sum()
    pe_chg_sum = df["pe_chg_oi"].sum()

    if ce_chg_sum > 0 and pe_chg_sum <= 0:
        # CE building, PE not → bears writing calls → bearish
        dominant_buildup = "Short Buildup (CE)"
        oi_bias = "BEARISH"
    elif pe_chg_sum > 0 and ce_chg_sum <= 0:
        # PE building, CE not → bulls writing puts → bullish
        dominant_buildup = "Long Buildup (PE)"
        oi_bias = "BULLISH"
    elif pe_chg_sum > 0 and ce_chg_sum > 0:
        # Both building; whichever is larger dominates
        if pe_chg_sum > ce_chg_sum:
            dominant_buildup = "Long Buildup"
            oi_bias = "BULLISH"
        else:
            dominant_buildup = "Short Buildup"
            oi_bias = "BEARISH"
    elif pe_chg_sum <= 0 and ce_chg_sum <= 0:
        # Both unwinding
        dominant_buildup = "Long Unwinding"
        oi_bias = "NEUTRAL"
    else:
        dominant_buildup = "Mixed"
        oi_bias = "NEUTRAL"

    # PCR override: PCR > 1.2 → bullish, PCR < 0.8 → bearish
    if pcr > 1.2:
        oi_bias = "BULLISH"
    elif pcr < 0.8:
        oi_bias = "BEARISH"

    logger.info(
        f"OI Analysis: PCR={pcr:.3f} | MaxPain={max_pain} | "
        f"CE Resist={ce_resistance} | PE Support={pe_support} | "
        f"Bias={oi_bias}"
    )

    return OIAnalysis(
        pcr=pcr,
        max_pain=max_pain,
        ce_resistance=ce_resistance,
        pe_support=pe_support,
        total_ce_oi=total_ce_oi,
        total_pe_oi=total_pe_oi,
        buildup_df=buildup_df,
        dominant_buildup=dominant_buildup,
        oi_bias=oi_bias,
        total_ce_volume=total_ce_volume,
        total_pe_volume=total_pe_volume,
        volume_pcr=volume_pcr,
        volume_bias=volume_bias,
    )


# ===========================================================================
# 3. CONFLUENCE SIGNAL GENERATOR
# ===========================================================================

def generate_signal(
    regime: MarketRegime,
    oi_analysis: OIAnalysis,
    spot: float,
    vix: float,
    symbol: str = "NIFTY",
) -> TradeSignal:
    """
    Generate a trading signal using confluence of:
      1. Technical trend direction (EMA slope + VWAP position)
      2. Market regime (ADX-based: trending vs sideways)
      3. OI buildup direction (from PCR + chg OI analysis)

    Signal fires ONLY when ≥2 of 3 conditions align.
    Calculates exact Spot and Option Premium Entry, Stop-Loss, and Targets.
    """
    reasons: List[str] = []
    bull_score = 0
    bear_score = 0

    # ── Condition 1: Technical Trend ─────────────────────────────────────────
    tech_bull = regime.state == "TRENDING BULLISH"
    tech_bear = regime.state == "TRENDING BEARISH"
    tech_side = regime.state == "SIDEWAYS / CHOP"

    if tech_bull:
        bull_score += 1
        reasons.append(f"✅ Trend: BULLISH (ADX={regime.adx:.1f}, price above EMA & VWAP)")
    elif tech_bear:
        bear_score += 1
        reasons.append(f"✅ Trend: BEARISH (ADX={regime.adx:.1f}, price below EMA & VWAP)")
    else:
        reasons.append(f"⚠️ Trend: SIDEWAYS (ADX={regime.adx:.1f} — below threshold)")

    # ── Condition 2: Market Regime Strength ──────────────────────────────────
    if regime.adx >= 25:
        if regime.ema_slope > 0:
            bull_score += 1
            reasons.append(f"✅ Regime: Strong trend (ADX={regime.adx:.1f}, EMA slope +ve)")
        else:
            bear_score += 1
            reasons.append(f"✅ Regime: Strong trend (ADX={regime.adx:.1f}, EMA slope -ve)")
    else:
        reasons.append(f"⚠️ Regime: Weak/No trend (ADX={regime.adx:.1f} < 25)")

    # ── Condition 3: OI Buildup ───────────────────────────────────────────────
    if oi_analysis.oi_bias == "BULLISH":
        bull_score += 1
        reasons.append(f"✅ OI: Bullish buildup (PCR={oi_analysis.pcr:.2f} > 1.0)")
    elif oi_analysis.oi_bias == "BEARISH":
        bear_score += 1
        reasons.append(f"✅ OI: Bearish buildup (PCR={oi_analysis.pcr:.2f} < 1.0)")
    else:
        reasons.append(f"⚠️ OI: Neutral (PCR={oi_analysis.pcr:.2f})")

    # ── Volume Confirmation (NSE Traded Contracts) ───────────────────────────
    if oi_analysis.volume_bias == "CE (CALL) HEAVY":
        reasons.append(f"📊 Volume: Call side dominant ({oi_analysis.total_ce_volume:,} CE vs {oi_analysis.total_pe_volume:,} PE | Vol PCR: {oi_analysis.volume_pcr:.2f})")
    elif oi_analysis.volume_bias == "PE (PUT) HEAVY":
        reasons.append(f"📊 Volume: Put side dominant ({oi_analysis.total_pe_volume:,} PE vs {oi_analysis.total_ce_volume:,} CE | Vol PCR: {oi_analysis.volume_pcr:.2f})")
    else:
        reasons.append(f"📊 Volume: Balanced activity (Vol PCR: {oi_analysis.volume_pcr:.2f})")

    # ── VIX Risk Check ────────────────────────────────────────────────────────
    high_vix = vix > 25.0
    if high_vix:
        reasons.append(f"⚠️ HIGH VIX ({vix:.1f}) — elevated event risk; reduce position size")

    # ── Signal Decision ───────────────────────────────────────────────────────
    atr = regime.atr
    sl_mult = 1.5
    tgt_mult = 3.0  # 2:1 R:R

    if bull_score >= 2 and bull_score > bear_score:
        signal = "BUY CE"
        entry = spot
        stop_loss = round(entry - sl_mult * atr, 2)
        target = round(entry + tgt_mult * atr, 2)
        confidence = bull_score
        reasons.append(f"🎯 Signal: BUY CE | {bull_score}/3 bullish conditions met")
    elif bear_score >= 2 and bear_score > bull_score:
        signal = "BUY PE"
        entry = spot
        stop_loss = round(entry + sl_mult * atr, 2)
        target = round(entry - tgt_mult * atr, 2)
        confidence = bear_score
        reasons.append(f"🎯 Signal: BUY PE | {bear_score}/3 bearish conditions met")
    else:
        signal = "NO TRADE"
        entry = spot
        stop_loss = round(entry - sl_mult * atr, 2)
        target = round(entry + tgt_mult * atr, 2)
        confidence = max(bull_score, bear_score)
        reasons.append(f"⏸️ No confluence — wait for clearer alignment (Bull:{bull_score}/3, Bear:{bear_score}/3)")

    rr = round(abs(target - entry) / max(abs(stop_loss - entry), 0.01), 2)

    # ── Actionable Option Strike & Premium Execution Plan ─────────────────────
    gap = cfg.STRIKE_GAPS.get(symbol, 50)
    atm_strike = round(spot / gap) * gap

    ce_ltp = round(spot * 0.007, 2)
    pe_ltp = round(spot * 0.007, 2)
    if oi_analysis.buildup_df is not None and not oi_analysis.buildup_df.empty:
        atm_row = oi_analysis.buildup_df[oi_analysis.buildup_df["strike"] == atm_strike]
        if not atm_row.empty:
            ce_ltp = float(atm_row.iloc[0].get("ce_ltp", ce_ltp))
            pe_ltp = float(atm_row.iloc[0].get("pe_ltp", pe_ltp))

    delta = 0.50  # ATM Delta estimate

    if signal == "BUY CE":
        recommended_strike = float(atm_strike)
        strike_label = f"{symbol} {int(atm_strike)} CE (ATM)"
        option_entry = round(ce_ltp, 2)
        # Option Risk Management (25-30% premium risk for 1:2 R:R)
        opt_risk = round(min(option_entry * 0.28, max(option_entry * 0.20, sl_mult * atr * delta * 0.5)), 2)
        option_sl = round(max(5.0, option_entry - opt_risk), 2)
        option_t1 = round(option_entry + (opt_risk * 1.25), 2)
        option_t2 = round(option_entry + (opt_risk * 2.50), 2)
        entry_rule = "BUY when 5m candle closes ABOVE VWAP with high Call Volume."
        t1_rule = f"Book 50% profit at ₹{option_t1:,.1f} (+{round(((option_t1-option_entry)/max(option_entry,1))*100)}%). Immediately trail SL to Cost (₹{option_entry:,.1f}) to guarantee ZERO RISK."
        t2_rule = f"Sell remaining 50% at Target 2: ₹{option_t2:,.1f} (+{round(((option_t2-option_entry)/max(option_entry,1))*100)}%) or exit when 20 EMA is breached."
        emergency_rule = f"SELL/EXIT IMMEDIATELY if spot drops below ₹{stop_loss:,.0f} or Option LTP drops to ₹{option_sl:,.1f}."
    elif signal == "BUY PE":
        recommended_strike = float(atm_strike)
        strike_label = f"{symbol} {int(atm_strike)} PE (ATM)"
        option_entry = round(pe_ltp, 2)
        opt_risk = round(min(option_entry * 0.28, max(option_entry * 0.20, sl_mult * atr * delta * 0.5)), 2)
        option_sl = round(max(5.0, option_entry - opt_risk), 2)
        option_t1 = round(option_entry + (opt_risk * 1.25), 2)
        option_t2 = round(option_entry + (opt_risk * 2.50), 2)
        entry_rule = "BUY when 5m candle closes BELOW VWAP with Put Volume dominant."
        t1_rule = f"Book 50% profit at ₹{option_t1:,.1f} (+{round(((option_t1-option_entry)/max(option_entry,1))*100)}%). Immediately trail SL to Cost (₹{option_entry:,.1f}) to guarantee ZERO RISK."
        t2_rule = f"Sell remaining 50% at Target 2: ₹{option_t2:,.1f} (+{round(((option_t2-option_entry)/max(option_entry,1))*100)}%) or exit when 20 EMA is crossed up."
        emergency_rule = f"SELL/EXIT IMMEDIATELY if spot rises above ₹{stop_loss:,.0f} or Option LTP drops to ₹{option_sl:,.1f}."
    else:
        recommended_strike = float(atm_strike)
        strike_label = f"{symbol} {int(atm_strike)} — NO POSITION"
        option_entry = 0.0
        option_sl = 0.0
        option_t1 = 0.0
        option_t2 = 0.0
        entry_rule = "WAIT — Market in sideways chop. Do NOT buy options (Theta decay eats premium)."
        t1_rule = "N/A"
        t2_rule = "N/A"
        emergency_rule = "Keep capital in cash until trend aligns."

    logger.info(f"Signal: {signal} | Strike: {strike_label} | Entry={entry} SL={stop_loss} Tgt={target}")

    return TradeSignal(
        signal=signal,
        entry=entry,
        stop_loss=stop_loss,
        target=target,
        risk_reward=rr,
        confidence=confidence,
        reasons=reasons,
        atr=round(atr, 2),
        recommended_strike=recommended_strike,
        strike_label=strike_label,
        option_entry=option_entry,
        option_sl=option_sl,
        option_target_1=option_t1,
        option_target_2=option_t2,
        entry_rule=entry_rule,
        exit_target_1_rule=t1_rule,
        exit_target_2_rule=t2_rule,
        emergency_exit_rule=emergency_rule,
    )


# ===========================================================================
# CONVENIENCE: Run full analytics pipeline
# ===========================================================================

def run_full_analytics(
    candles: pd.DataFrame,
    option_chain: pd.DataFrame,
    spot: float,
    vix: float,
    symbol: str = "NIFTY",
) -> Dict:
    """
    Run all three analytical subsystems and return a combined results dict.
    """
    regime, enriched = compute_market_regime(candles)
    oi_analysis = analyze_option_chain(option_chain, spot)
    signal = generate_signal(regime, oi_analysis, spot, vix, symbol=symbol)

    return {
        "regime": regime,
        "enriched_candles": enriched,
        "oi_analysis": oi_analysis,
        "signal": signal,
    }
