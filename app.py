"""
app.py — AI-Powered Indian F&O Market Analysis & Trading Dashboard
==================================================================
Main Streamlit application entry point.

Layout:
  ┌─────────────────────────────────────────────────────┐
  │  SIDEBAR: Controls (Symbol, TF, MockMode, Refresh)  │
  ├─────────────────────────────────────────────────────┤
  │  TOP KPI BANNER: Spot | VIX | PCR | Regime | Signal │
  ├───────────────────────┬─────────────────────────────┤
  │  CANDLESTICK CHART    │  OI BAR CHART               │
  │  EMA + VWAP overlays  │  CE vs PE per strike        │
  ├───────────────────────┴─────────────────────────────┤
  │  AI SENTIMENT CARD: Headlines + Bias + Events       │
  └─────────────────────────────────────────────────────┘

Run:
    streamlit run app.py

Auto-refresh:
    Uses streamlit-autorefresh. Set AUTO_REFRESH_SECONDS in .env.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page config MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Indicated — F&O Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Local module imports
# ---------------------------------------------------------------------------
from config import cfg, validate_config, VALID_SYMBOLS, VALID_TIMEFRAMES
import data_engine as de
import analytics_engine as ae
import ai_news_engine as ne

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# ===========================================================================
# CUSTOM CSS — Dark Pro Theme
# ===========================================================================

DARK_CSS = """
<style>
:root {
    --bg-primary:   #0e1117;
    --bg-card:      #1a1d2e;
    --bg-card2:     #12151f;
    --accent-green: #00d26a;
    --accent-red:   #ff4b4b;
    --accent-gold:  #ffd700;
    --accent-blue:  #4f8ef7;
    --text-primary: #e8ecf4;
    --text-muted:   #8892a4;
    --border:       #2a2f42;
}
.stApp { background-color: var(--bg-primary); }
.block-container { padding-top: 1rem !important; }
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    min-height: 90px;
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
}
.kpi-sub {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
}
.sentiment-card {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
}
.trade-plan-card {
    border-radius: 12px;
    padding: 18px 22px;
    margin: 10px 0 16px 0;
}
.trade-plan-bullish {
    background: linear-gradient(135deg, #0b2216 0%, #10291e 100%);
    border: 1.5px solid #10b981;
}
.trade-plan-bearish {
    background: linear-gradient(135deg, #290e14 0%, #221217 100%);
    border: 1.5px solid #ef4444;
}
.trade-plan-neutral {
    background: linear-gradient(135deg, #221d0a 0%, #1b1708 100%);
    border: 1.5px solid #f59e0b;
}
.trade-pill {
    display: inline-block;
    padding: 8px 14px;
    border-radius: 8px;
    background: #171d2d;
    border: 1px solid #2d3748;
    color: #f8fafc !important;
    font-size: 13.5px;
    margin: 4px 6px 4px 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.25);
}
.trade-pill b {
    color: #cbd5e1;
}
.headline-item {
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
    color: var(--text-primary);
}
.headline-source {
    font-size: 10px;
    color: var(--text-muted);
    font-weight: 600;
    text-transform: uppercase;
}
.section-header {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--text-muted);
    margin: 16px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
}
[data-testid="stSidebar"] { background: var(--bg-card2) !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)


# ===========================================================================
# AUTO-REFRESH
# ===========================================================================

def _setup_autorefresh(interval_sec: int):
    if interval_sec <= 0:
        return
    try:
        from streamlit_autorefresh import st_autorefresh  # type: ignore
        st_autorefresh(interval=interval_sec * 1000, key="auto_refresh_timer")
    except ImportError:
        pass


# ===========================================================================
# SESSION STATE
# ===========================================================================

def _init_session_state():
    defaults = {
        "symbol": cfg.SYMBOL,
        "timeframe": cfg.TIMEFRAME,
        "mock_mode": cfg.MOCK_MODE,
        "last_refresh": None,
        "refresh_trigger": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ===========================================================================
# CACHED DATA LOADERS
# ===========================================================================

@st.cache_data(ttl=30, show_spinner=False)
def _load_market_data(symbol: str, timeframe: str, mock_mode: bool, _trigger: int):
    cfg.MOCK_MODE = mock_mode
    return de.get_all_market_data(symbol, timeframe)


@st.cache_data(ttl=30, show_spinner=False)
def _run_analytics(symbol: str, timeframe: str, mock_mode: bool, _trigger: int):
    data = _load_market_data(symbol, timeframe, mock_mode, _trigger)
    return ae.run_full_analytics(
        candles=data["candles"],
        option_chain=data["option_chain"],
        spot=data["spot"],
        vix=data["vix"],
        symbol=symbol,
    )


@st.cache_data(ttl=120, show_spinner=False)
def _load_sentiment(pcr: float, oi_bias: str, vix: float, _trigger: int):
    return ne.get_market_sentiment(pcr=pcr, oi_bias=oi_bias, vix=vix)


# ===========================================================================
# PLOTLY THEME
# ===========================================================================

PLOTLY_DARK = dict(
    paper_bgcolor="#0e1117",
    plot_bgcolor="#12151f",
    font=dict(color="#e8ecf4", size=11),
    xaxis=dict(gridcolor="#1e2235", showgrid=True, rangeslider=dict(visible=False), type="category"),
    yaxis=dict(gridcolor="#1e2235", showgrid=True),
    margin=dict(l=10, r=10, t=35, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2f42", font=dict(size=10)),
    hovermode="x unified",
)


# ===========================================================================
# CANDLESTICK CHART
# ===========================================================================

def _build_candlestick_chart(df, symbol, signal_type, entry, sl, target):
    df = df.tail(60).copy()
    df["ts_label"] = df["timestamp"].dt.strftime("%H:%M")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.78, 0.22],
    )

    fig.add_trace(go.Candlestick(
        x=df["ts_label"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color="#00d26a", decreasing_line_color="#ff4b4b",
        increasing_fillcolor="#00d26a", decreasing_fillcolor="#ff4b4b",
        line=dict(width=1),
    ), row=1, col=1)

    if "ema" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["ts_label"], y=df["ema"], name="EMA 20",
            line=dict(color="#4f8ef7", width=1.5, dash="dot"),
        ), row=1, col=1)

    if "vwap" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["ts_label"], y=df["vwap"], name="VWAP",
            line=dict(color="#ffd700", width=1.5, dash="dash"),
        ), row=1, col=1)

    vol_colors = ["#00d26a" if c >= o else "#ff4b4b" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["ts_label"], y=df["volume"], name="Volume",
        marker_color=vol_colors, opacity=0.7, showlegend=False,
    ), row=2, col=1)

    if signal_type in ("BUY CE", "BUY PE"):
        last_ts = df["ts_label"].iloc[-1]
        is_ce = signal_type == "BUY CE"
        marker_y = df["low"].iloc[-1] * 0.9995 if is_ce else df["high"].iloc[-1] * 1.0005
        fig.add_trace(go.Scatter(
            x=[last_ts], y=[marker_y],
            mode="markers+text",
            marker=dict(symbol="triangle-up" if is_ce else "triangle-down", size=18,
                        color="#00d26a" if is_ce else "#ff4b4b"),
            text=["▲ BUY CE" if is_ce else "▼ BUY PE"],
            textposition="bottom center" if is_ce else "top center",
            textfont=dict(size=11, color="#00d26a" if is_ce else "#ff4b4b"),
            name=signal_type, showlegend=True,
        ), row=1, col=1)
        fig.add_hline(y=sl, line_dash="dot", line_color="#ff6b6b", line_width=1,
                      annotation_text=f"SL {sl:.0f}", annotation_font_color="#ff6b6b",
                      annotation_font_size=10, row=1, col=1)
        fig.add_hline(y=target, line_dash="dot", line_color="#00d26a", line_width=1,
                      annotation_text=f"TGT {target:.0f}", annotation_font_color="#00d26a",
                      annotation_font_size=10, row=1, col=1)

    fig.update_layout(**PLOTLY_DARK,
                      title=dict(text=f"<b>{symbol}</b> — Candlestick",
                                 font=dict(size=13, color="#e8ecf4"), x=0.01),
                      height=480)
    fig.update_yaxes(title_text="Price", row=1, col=1, tickformat=".0f")
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=9), row=2, col=1)
    return fig


# ===========================================================================
# OI BAR CHART
# ===========================================================================

def _build_oi_chart(buildup_df, ce_resistance, pe_support, spot, symbol="NIFTY"):
    df = buildup_df.copy()
    df["ce_oi_L"] = df["ce_oi"] / 1e5
    df["pe_oi_L"] = df["pe_oi"] / 1e5
    gap = cfg.STRIKE_GAPS.get(symbol, 50)
    atm_strike = round(spot / gap) * gap

    def _label(s):
        if s == atm_strike:
            return f"🎯 {s:.0f} ATM"
        elif s == ce_resistance:
            return f"🔴 {s:.0f} RESIST"
        elif s == pe_support:
            return f"🟢 {s:.0f} SUPPORT"
        return f"{s:.0f}"

    labels = [_label(s) for s in df["strike"]]
    ce_colors = ["#ff2244" if s == ce_resistance else "#ff6680" for s in df["strike"]]
    pe_colors = ["#00aa44" if s == pe_support else "#00cc66" for s in df["strike"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=df["ce_oi_L"], name="Call OI (CE)", orientation="h",
        marker_color=ce_colors,
        hovertemplate="CE OI: %{x:.1f}L<extra></extra>",
        text=df["ce_oi_L"].round(1).astype(str) + "L",
        textposition="outside", textfont=dict(size=9, color="#ff6680"),
    ))
    fig.add_trace(go.Bar(
        y=labels, x=-df["pe_oi_L"], name="Put OI (PE)", orientation="h",
        marker_color=pe_colors,
        hovertemplate="PE OI: %{customdata:.1f}L<extra></extra>",
        customdata=df["pe_oi_L"],
        text=df["pe_oi_L"].round(1).astype(str) + "L",
        textposition="outside", textfont=dict(size=9, color="#00cc66"),
    ))
    fig.add_vline(x=0, line_color="#ffd700", line_width=1.5, opacity=0.5)
    max_oi = max(df["ce_oi_L"].max(), df["pe_oi_L"].max()) * 1.3

    layout = dict(PLOTLY_DARK)
    layout.pop("xaxis", None)
    layout.pop("yaxis", None)
    layout.pop("legend", None)
    fig.update_layout(
        **layout,
        barmode="overlay",
        title=dict(text="<b>Call vs Put OI</b> by Strike",
                   font=dict(size=13, color="#e8ecf4"), x=0.01),
        xaxis=dict(title="OI (Lakhs)", range=[-max_oi, max_oi],
                   tickformat=".0f", gridcolor="#1e2235",
                   zeroline=True, zerolinecolor="#ffd700", zerolinewidth=1),
        yaxis=dict(gridcolor="#1e2235", tickfont=dict(size=10), automargin=True),
        height=480,
        legend=dict(orientation="h", x=0.25, y=1.04, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ===========================================================================
# SENTIMENT GAUGE
# ===========================================================================

def _build_sentiment_gauge(score: float, bias: str) -> go.Figure:
    color_map = {"Bullish": "#00d26a", "Bearish": "#ff4b4b", "Neutral": "#ffd700"}
    bar_color = color_map.get(bias, "#aaaaaa")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="", font=dict(size=28, color=bar_color),
                    prefix="+" if score >= 0 else ""),
        gauge=dict(
            axis=dict(range=[-10, 10], tickwidth=1, tickcolor="#2a2f42",
                      tickvals=[-10, -5, 0, 5, 10],
                      ticktext=["-10", "-5", "0", "+5", "+10"],
                      tickfont=dict(size=10, color="#8892a4")),
            bar=dict(color=bar_color, thickness=0.25),
            bgcolor="#12151f", borderwidth=1, bordercolor="#2a2f42",
            steps=[
                dict(range=[-10, -5], color="#2a0a0a"),
                dict(range=[-5, 0], color="#1a1015"),
                dict(range=[0, 5], color="#0a1a10"),
                dict(range=[5, 10], color="#0a2a15"),
            ],
            threshold=dict(line=dict(color="#ffffff", width=2), thickness=0.75, value=score),
        ),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(paper_bgcolor="#12151f", font=dict(color="#e8ecf4"),
                      height=180, margin=dict(l=20, r=20, t=30, b=10))
    return fig


# ===========================================================================
# KPI CARD
# ===========================================================================

def _kpi_card(label, value, sub="", color="#e8ecf4"):
    return f"""
<div class="kpi-card">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value" style="color:{color}">{value}</div>
    <div class="kpi-sub">{sub}</div>
</div>"""


# ===========================================================================
# SIDEBAR
# ===========================================================================

def _render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style='text-align:center; padding:10px 0 20px 0;'>
    <span style='font-size:28px'>📊</span><br>
    <span style='font-size:18px; font-weight:800; color:#e8ecf4'>Indicated</span><br>
    <span style='font-size:11px; color:#8892a4; letter-spacing:2px'>F&O DASHBOARD</span>
</div>""", unsafe_allow_html=True)
        st.markdown("---")

        symbol = st.selectbox(
            "📈 Index", options=VALID_SYMBOLS,
            index=VALID_SYMBOLS.index(st.session_state.symbol)
                  if st.session_state.symbol in VALID_SYMBOLS else 0,
        )
        timeframe = st.selectbox(
            "⏱️ Timeframe", options=VALID_TIMEFRAMES,
            index=VALID_TIMEFRAMES.index(st.session_state.timeframe)
                  if st.session_state.timeframe in VALID_TIMEFRAMES else 0,
        )
        mock_mode = st.toggle("🧪 Mock Mode", value=st.session_state.mock_mode,
                              help="Simulated data — no API key required")

        if mock_mode:
            st.info("🧪 Running in **Mock Mode**", icon="ℹ️")
        else:
            st.warning("⚡ Live Mode — broker key required", icon="⚠️")

        st.markdown("---")
        refresh_clicked = st.button("🔄 Refresh AI & Market Scan",
                                    use_container_width=True, type="primary")

        if cfg.AUTO_REFRESH_SECONDS > 0:
            st.caption(f"⏰ Auto-refresh every **{cfg.AUTO_REFRESH_SECONDS}s**")

        st.markdown("---")
        for w in validate_config():
            st.warning(w, icon="⚠️")

        now_ist = datetime.now(IST)
        st.markdown(f"""
<div style='font-size:10px; color:#8892a4; padding-top:8px;'>
    🕐 IST: {now_ist.strftime('%d %b %Y  %H:%M:%S')}<br>
    {'📡 Mock Data' if mock_mode else '📡 Live Data'}
</div>""", unsafe_allow_html=True)

    return symbol, timeframe, mock_mode, refresh_clicked


# ===========================================================================
# KPI BANNER
# ===========================================================================

def _render_kpi_banner(spot, vix, oi_analysis, regime, signal):
    pcr_color = "#00d26a" if oi_analysis.pcr > 1.0 else "#ff4b4b"
    vix_color = "#00d26a" if vix < 15 else ("#ffd700" if vix < 22 else "#ff4b4b")

    cols = st.columns([1.2, 0.9, 0.9, 1.4, 1.5, 1.5, 1.5])

    with cols[0]:
        st.markdown(_kpi_card("Spot Price", f"₹{spot:,.2f}", "Index LTP"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(_kpi_card("India VIX", f"{vix:.2f}", "Impl. Volatility", vix_color), unsafe_allow_html=True)
    with cols[2]:
        vol_sub = f"Vol PCR {oi_analysis.volume_pcr:.2f} | {oi_analysis.volume_bias}"
        st.markdown(_kpi_card("PCR (OI)", f"{oi_analysis.pcr:.3f}", vol_sub, pcr_color), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(_kpi_card("Market Regime", regime.short_label, f"ADX {regime.adx:.1f}", regime.color), unsafe_allow_html=True)
    with cols[4]:
        sig_label = f"{signal.emoji} {signal.signal}"
        st.markdown(_kpi_card("Master Signal", sig_label, f"Conf: {signal.confidence}/3", signal.color), unsafe_allow_html=True)
    with cols[5]:
        st.markdown(_kpi_card("Entry / SL", f"₹{signal.entry:,.0f}", f"SL ₹{signal.stop_loss:,.0f}", "#e8ecf4"), unsafe_allow_html=True)
    with cols[6]:
        st.markdown(_kpi_card("Target", f"₹{signal.target:,.0f}", f"R:R {signal.risk_reward:.1f}x", "#00d26a"), unsafe_allow_html=True)

    st.markdown("<div style='margin:8px 0'></div>", unsafe_allow_html=True)


# ===========================================================================
# TRADE EXECUTION PLAN (ACTIONABLE BUY/SELL/TARGET GUIDELINES)
# ===========================================================================

def _render_trade_execution_plan(signal, symbol):
    """Render clear, step-by-step option execution rules for profitability."""
    if signal.signal == "BUY CE":
        theme_class = "trade-plan-bullish"
        action_text = f"🟢 ACTION: BUY CALL OPTION (CE) — {signal.strike_label}"
    elif signal.signal == "BUY PE":
        theme_class = "trade-plan-bearish"
        action_text = f"🔴 ACTION: BUY PUT OPTION (PE) — {signal.strike_label}"
    else:
        theme_class = "trade-plan-neutral"
        action_text = f"🟡 ACTION: AVOID BUYING OPTIONS — CAPITAL PRESERVATION MODE"

    st.markdown(f"""
<div class="trade-plan-card {theme_class}">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; margin-bottom:10px;">
        <span style="font-size:16px; font-weight:800; color:#fff;">{action_text}</span>
        <span style="font-size:12px; color:#cbd5e1; background:rgba(0,0,0,0.3); padding:4px 10px; border-radius:6px;">
            Target R:R 2.0x | ATR {signal.atr:.1f} pts
        </span>
    </div>
""", unsafe_allow_html=True)

    if signal.signal in ("BUY CE", "BUY PE"):
        risk_per_share = round(signal.option_entry - signal.option_sl, 1)
        reward_t1 = round(signal.option_target_1 - signal.option_entry, 1)
        reward_t2 = round(signal.option_target_2 - signal.option_entry, 1)

        st.markdown(f"""
    <div style="margin-bottom:12px;">
        <span class="trade-pill" style="border-color:#38bdf8;">🎯 <b style="color:#94a3b8;">Contract:</b> <span style="color:#38bdf8; font-weight:800; font-size:14.5px;">{signal.strike_label}</span></span>
        <span class="trade-pill" style="border-color:#fbbf24; background:#241d08;">💵 <b style="color:#fbbf24;">Est. Entry:</b> <span style="color:#fde047; font-weight:800; font-size:15px; padding-left:4px;">₹{signal.option_entry:,.1f}</span></span>
        <span class="trade-pill" style="border-color:#f87171;">🛑 <b style="color:#fca5a5;">Stop-Loss:</b> <span style="color:#ff6b6b; font-weight:800; font-size:15px; padding-left:4px;">₹{signal.option_sl:,.1f}</span> <span style="color:#fca5a5; font-size:11px;">(-₹{risk_per_share}/sh)</span></span>
        <span class="trade-pill" style="border-color:#34d399;">🎯 <b style="color:#86efac;">Target 1:</b> <span style="color:#4ade80; font-weight:800; font-size:15px; padding-left:4px;">₹{signal.option_target_1:,.1f}</span> <span style="color:#86efac; font-size:11px;">(+₹{reward_t1}/sh)</span></span>
        <span class="trade-pill" style="border-color:#2dd4bf;">🚀 <b style="color:#99f6e4;">Target 2:</b> <span style="color:#2dd4bf; font-weight:800; font-size:15px; padding-left:4px;">₹{signal.option_target_2:,.1f}</span> <span style="color:#99f6e4; font-size:11px;">(+₹{reward_t2}/sh)</span></span>
    </div>
    <div style="font-size:13.5px; line-height:1.8; color:#f8fafc; background:#0f172a; border:1px solid #334155; padding:14px 18px; border-radius:8px;">
        <div><b style="color:#38bdf8;">1. ⏱️ When to Enter:</b> {signal.entry_rule}</div>
        <div><b style="color:#4ade80;">2. 💰 When to Sell Part 1 (Target 1):</b> {signal.exit_target_1_rule}</div>
        <div><b style="color:#2dd4bf;">3. 🏁 When to Sell Part 2 (Target 2):</b> {signal.exit_target_2_rule}</div>
        <div><b style="color:#f87171;">4. 🚨 Invalidation Rule (Cut Losses):</b> {signal.emergency_exit_rule}</div>
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
    <div style="font-size:13.5px; line-height:1.8; color:#f8fafc; background:#0f172a; border:1px solid #334155; padding:14px 18px; border-radius:8px;">
        <div><b style="color:#fbbf24;">⚠️ Why No Trade:</b> Market regime is <b>{signal.reasons[-1]}</b>.</div>
        <div><b style="color:#38bdf8;">💡 Golden Rule for Option Buyers:</b> Do NOT buy options in sideways chop. Option sellers make money from <b>Theta (time decay)</b> while buyers bleed. Wait for trend confirmation (ADX &gt; 25 + breakout above/below VWAP).</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ===========================================================================
# SIGNAL DETAILS
# ===========================================================================

def _render_signal_details(signal, regime, oi_analysis):
    with st.expander(f"{signal.emoji} Signal Analysis — {signal.signal}", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📋 Confluence Reasons:**")
            for r in signal.reasons:
                st.markdown(f"- {r}")
        with c2:
            st.markdown("**📐 Risk Management:**")
            st.markdown(f"""
| Level | Price |
|-------|-------|
| Entry | ₹{signal.entry:,.2f} |
| Stop-Loss | ₹{signal.stop_loss:,.2f} |
| Target | ₹{signal.target:,.2f} |
| ATR | {signal.atr:.2f} pts |
| Risk-Reward | {signal.risk_reward:.2f}x |
""")
            st.markdown(f"""
**🔍 OI & Traded Volume (NSE Standard):**
- **Max Pain:** ₹{oi_analysis.max_pain:,.0f}
- **CE Resistance (Max Call OI):** ₹{oi_analysis.ce_resistance:,.0f}
- **PE Support (Max Put OI):** ₹{oi_analysis.pe_support:,.0f}
- **Dominant Buildup:** {oi_analysis.dominant_buildup}
- **Total Call Volume:** {oi_analysis.total_ce_volume:,} contracts
- **Total Put Volume:** {oi_analysis.total_pe_volume:,} contracts
- **Volume Ratio (Vol PCR):** {oi_analysis.volume_pcr:.2f} ({oi_analysis.volume_bias})
""")


# ===========================================================================
# CHARTS
# ===========================================================================

def _render_charts(enriched_candles, signal, spot, buildup_df, ce_resistance, pe_support, symbol):
    col_left, col_right = st.columns([1, 1], gap="medium")
    with col_left:
        st.markdown('<div class="section-header">📊 Candlestick Chart</div>', unsafe_allow_html=True)
        fig = _build_candlestick_chart(enriched_candles, symbol, signal.signal,
                                       signal.entry, signal.stop_loss, signal.target)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with col_right:
        st.markdown('<div class="section-header">📉 Open Interest — Call vs Put</div>', unsafe_allow_html=True)
        fig2 = _build_oi_chart(buildup_df, ce_resistance, pe_support, spot, symbol=symbol)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})


# ===========================================================================
# SENTIMENT SECTION
# ===========================================================================

def _render_sentiment_section(sentiment):
    st.markdown('<div class="section-header">🤖 AI Market Sentiment & Live News</div>', unsafe_allow_html=True)
    col_gauge, col_summary, col_news = st.columns([1, 1.5, 2], gap="medium")

    with col_gauge:
        st.markdown(f"""
<div class="sentiment-card" style="text-align:center; padding-bottom:8px">
    <div class="kpi-label">Sentiment Score</div>
    <div style="font-size:14px; color:{sentiment.color}; font-weight:800; margin:4px 0">
        {sentiment.emoji} {sentiment.bias.upper()}
    </div>""", unsafe_allow_html=True)
        gauge_fig = _build_sentiment_gauge(sentiment.sentiment_score, sentiment.bias)
        st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Source: {sentiment.source}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_summary:
        with st.container():
            st.markdown("**📝 AI Summary**")
            st.markdown(f'<div style="font-size:13px; color:#c8d0e0; line-height:1.6">{sentiment.summary}</div>',
                        unsafe_allow_html=True)
            if sentiment.event_risk_warning and sentiment.event_risk_warning.lower() != "none":
                st.markdown("<br>**⚠️ Event Risk**", unsafe_allow_html=True)
                st.error(sentiment.event_risk_warning, icon="🚨")

    with col_news:
        st.markdown("**📰 Live Headlines**")
        for h in sentiment.headlines[:8]:
            st.markdown(
                f'<div class="headline-item">'
                f'<span class="headline-source">{h.source}</span> &nbsp; {h.title}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ===========================================================================
# FOOTER
# ===========================================================================

def _render_footer(mock_mode, last_refresh):
    refresh_str = last_refresh.strftime("%H:%M:%S IST") if last_refresh else "Never"
    st.markdown("---")
    st.markdown(
        f'<div style="text-align:center; font-size:11px; color:#4a5568">'
        f'⚠️ <b>Disclaimer:</b> Educational purposes only. Not investment advice. '
        f'F&O trading carries substantial risk. &nbsp;|&nbsp; '
        f'{"🧪 Mock Mode" if mock_mode else "📡 Live Mode"} &nbsp;|&nbsp; '
        f'Last Refresh: {refresh_str}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    _init_session_state()
    _setup_autorefresh(cfg.AUTO_REFRESH_SECONDS)

    symbol, timeframe, mock_mode, refresh_clicked = _render_sidebar()

    st.session_state.symbol = symbol
    st.session_state.timeframe = timeframe
    st.session_state.mock_mode = mock_mode
    cfg.MOCK_MODE = mock_mode

    if refresh_clicked:
        st.session_state.refresh_trigger += 1
        st.cache_data.clear()

    trigger = st.session_state.refresh_trigger

    # Page title
    st.markdown(f"""
<div style='margin-bottom:12px'>
    <span style='font-size:22px; font-weight:800; color:#e8ecf4'>📊 Indicated</span>
    <span style='font-size:14px; color:#8892a4; margin-left:12px'>
        F&O Market Analysis — <b>{symbol}</b> ({timeframe})
    </span>
</div>""", unsafe_allow_html=True)

    # Load data
    with st.spinner(f"{'🧪 Generating mock' if mock_mode else '📡 Fetching live'} market data for {symbol}..."):
        try:
            market_data = _load_market_data(symbol, timeframe, mock_mode, trigger)
            analytics = _run_analytics(symbol, timeframe, mock_mode, trigger)
        except Exception as exc:
            st.error(f"❌ Failed to load market data: {exc}", icon="🚨")
            st.info("💡 Enable **Mock Mode** in the sidebar to run without API keys.", icon="ℹ️")
            st.stop()

    spot = market_data["spot"]
    vix = market_data["vix"]
    regime = analytics["regime"]
    enriched_candles = analytics["enriched_candles"]
    oi_analysis = analytics["oi_analysis"]
    signal = analytics["signal"]

    _render_kpi_banner(spot, vix, oi_analysis, regime, signal)
    _render_trade_execution_plan(signal, symbol)
    _render_signal_details(signal, regime, oi_analysis)
    _render_charts(
        enriched_candles=enriched_candles,
        signal=signal,
        spot=spot,
        buildup_df=oi_analysis.buildup_df,
        ce_resistance=oi_analysis.ce_resistance,
        pe_support=oi_analysis.pe_support,
        symbol=symbol,
    )

    with st.spinner("🤖 Analysing news sentiment..."):
        try:
            sentiment = _load_sentiment(oi_analysis.pcr, oi_analysis.oi_bias, vix, trigger)
        except Exception as exc:
            logger.warning(f"Sentiment fetch failed: {exc}")
            sentiment = ne.SentimentResult(
                sentiment_score=0.0, bias="Neutral",
                event_risk_warning="Sentiment unavailable",
                summary="Unable to fetch sentiment. Please check network connection.",
                headlines=ne._mock_headlines(), source="Fallback",
            )

    _render_sentiment_section(sentiment)

    st.session_state.last_refresh = datetime.now(IST)
    _render_footer(mock_mode, st.session_state.last_refresh)


if __name__ == "__main__":
    main()
