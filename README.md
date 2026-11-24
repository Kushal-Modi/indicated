# Indicated — AI-Powered Indian F&O Market Analysis Dashboard

A complete, local-first, modular trading dashboard for Indian NIFTY / BANKNIFTY / FINNIFTY / SENSEX Futures & Options analysis. Built with Python + Streamlit, it works **out of the box in Mock Mode** with zero API keys.

---

## Features

| Feature | Details |
|---------|---------|
| 📊 Live candlestick charts | Plotly interactive OHLCV with EMA-20, VWAP, signal arrows |
| 📉 Option Chain OI chart | CE vs PE OI across ATM ±10 strikes, support/resistance walls |
| 🧠 Market Regime Filter | ADX(14) + EMA(20) + VWAP → TRENDING / SIDEWAYS classification |
| 📐 Confluence Signals | BUY CE / BUY PE / NO TRADE with ATR-based Entry, SL, Target |
| 🤖 AI News Sentiment | Live RSS → Gemini/OpenAI → structured JSON (rule-based fallback) |
| 🧪 Mock Mode | Full GBM-simulated NIFTY/BANKNIFTY/FINNIFTY — no API keys needed |
| 🔌 Broker Adapters | Modular stubs for Upstox, Dhan, Zerodha Kite Connect |

---

## Quick Start (Mock Mode — Zero API Keys)

```bash
# 1. Clone / navigate to project
cd indicated

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Copy environment template
copy .env.example .env       # Windows
# cp .env.example .env       # macOS/Linux

# 6. Run the dashboard (Mock Mode is ON by default)
streamlit run app.py
```

The dashboard opens at **http://localhost:8501** in your browser.

---

## Project Structure

```
indicated/
├── app.py                  # Main Streamlit dashboard (UI + charts)
├── config.py               # Typed .env loader (all settings in one place)
├── data_engine.py          # Mock GBM data generator + live broker routing
├── broker_adapter.py       # ABC + Upstox / Dhan / Zerodha adapter stubs
├── analytics_engine.py     # Market Regime + OI Analyzer + Signal Generator
├── ai_news_engine.py       # RSS fetch → LLM sentiment (with rule fallback)
├── requirements.txt        # All pip dependencies
├── .env.example            # Environment variable template
└── README.md               # This file
```

---

## Configuration (`.env`)

Copy `.env.example` to `.env` and edit as needed:

```env
# Core
MOCK_MODE=True           # True = simulated data, no API keys needed
SYMBOL=NIFTY             # NIFTY | BANKNIFTY | FINNIFTY
TIMEFRAME=5m             # 5m | 15m | 1h
AUTO_REFRESH_SECONDS=60  # 0 to disable

# Broker (only needed when MOCK_MODE=False)
BROKER_NAME=upstox       # upstox | dhan | zerodha
BROKER_API_KEY=...
BROKER_API_SECRET=...

# AI Sentiment (optional — rule-based fallback if blank)
LLM_PROVIDER=gemini      # gemini | openai
LLM_API_KEY=...
```

---

## Plugging in Your API Keys

### Gemini API (Recommended)
1. Get a key at [Google AI Studio](https://aistudio.google.com)
2. Set in `.env`:
   ```
   LLM_PROVIDER=gemini
   LLM_API_KEY=your_gemini_key
   ```

### OpenAI API
1. Get a key at [OpenAI Platform](https://platform.openai.com)
2. Set in `.env`:
   ```
   LLM_PROVIDER=openai
   LLM_API_KEY=your_openai_key
   ```

### Broker Integration (Live Trading Data)
Set `MOCK_MODE=False` and configure your broker:

**Upstox:**
```env
BROKER_NAME=upstox
BROKER_API_KEY=your_upstox_access_token
```

**Dhan:**
```env
BROKER_NAME=dhan
BROKER_API_KEY=your_dhan_access_token
```

**Zerodha Kite:**
```env
BROKER_NAME=zerodha
BROKER_API_KEY=your_kite_api_key
BROKER_API_SECRET=your_kite_access_token
```
> Also install: `pip install kiteconnect`

---

## Analytics Engine — How Signals Work

### Market Regime (3 states)

| Condition | Regime |
|-----------|--------|
| ADX ≥ 25 + price above EMA & VWAP | 🟢 TRENDING BULLISH |
| ADX ≥ 25 + price below EMA & VWAP | 🔴 TRENDING BEARISH |
| ADX < 25 or mixed signals | 🟡 SIDEWAYS / CHOP |

### Confluence Signal Logic

Signal fires **only when ≥ 2 of 3 conditions align**:

1. **Technical Trend** — EMA slope + VWAP position
2. **Regime Strength** — ADX threshold + EMA slope direction  
3. **OI Buildup Bias** — PCR + Change in OI direction

### Risk Management (ATR-based)

```
Entry     = Current Spot
Stop-Loss = Entry ± 1.5 × ATR(14)
Target    = Entry ± 3.0 × ATR(14)   [2:1 Risk-Reward]
```

---

## AI Sentiment Engine

```
RSS Feeds → feedparser → Top 10 Headlines
                              ↓
              LLM Prompt (Headlines + PCR + VIX + OI bias)
                              ↓
              Gemini / OpenAI → JSON response
                              ↓ (fallback if no key)
              Rule-based keyword scorer
                              ↓
              SentimentResult {score, bias, event_risk, summary}
```

**Fallback scoring** uses a 60+ keyword dictionary with:
- Bullish/bearish keyword weights
- PCR tilt (± 1.0 point)
- OI bias tilt (± 0.5 point)
- VIX dampening (> 25 → 30% pull toward neutral)

---

## Disclaimer

> This dashboard is for **educational and informational purposes only**. It does not constitute investment advice. F&O trading involves substantial risk of loss. Always do your own research and consult a SEBI-registered advisor before trading.
