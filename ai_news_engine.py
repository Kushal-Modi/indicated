"""
ai_news_engine.py — AI-Powered News Sentiment Analyzer
=======================================================
Pipeline:
  1. Fetch top 10 live Indian financial headlines from RSS feeds
     (Moneycontrol / Economic Times / Livemint)
  2. Build a structured LLM prompt with:
     - Headlines list
     - Current PCR, OI bias, India VIX
  3. Call configured LLM (Gemini or OpenAI) for structured JSON sentiment
  4. Parse and return structured SentimentResult

If LLM_API_KEY is blank OR LLM_PROVIDER is unset, the engine automatically
falls back to a smart rule-based keyword scorer:
  - Weighted bullish/bearish keyword dictionary
  - VIX-based event risk scaling
  - PCR-based directional tilt
  - Returns the same SentimentResult shape — UI code never needs to branch

Output JSON schema:
  {
    "sentiment_score": float (-10 to +10),
    "bias": "Bullish" | "Bearish" | "Neutral",
    "event_risk_warning": str,
    "summary": str
  }
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try optional dependencies
# ---------------------------------------------------------------------------
try:
    import feedparser  # type: ignore
    _FEEDPARSER_AVAILABLE = True
except ImportError:
    _FEEDPARSER_AVAILABLE = False
    logger.warning("feedparser not installed; news fetching disabled.")

from config import cfg


# ===========================================================================
# Data Models
# ===========================================================================

@dataclass
class Headline:
    title: str
    source: str
    link: str
    published: str = ""


@dataclass
class SentimentResult:
    sentiment_score: float              # -10 (very bearish) to +10 (very bullish)
    bias: str                           # "Bullish" | "Bearish" | "Neutral"
    event_risk_warning: str             # e.g. "RBI Policy tomorrow — elevated volatility expected"
    summary: str                        # 2-3 sentence market commentary
    headlines: List[Headline] = field(default_factory=list)
    source: str = "LLM"                 # "LLM" | "Rule-Based"

    @property
    def emoji(self) -> str:
        return {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(self.bias, "⚪")

    @property
    def color(self) -> str:
        return {"Bullish": "#00d26a", "Bearish": "#ff4b4b", "Neutral": "#ffd700"}.get(self.bias, "#aaa")

    @property
    def score_display(self) -> str:
        """Signed string like '+7.2' or '-3.4'."""
        return f"+{self.sentiment_score:.1f}" if self.sentiment_score >= 0 else f"{self.sentiment_score:.1f}"


# ===========================================================================
# STEP 1: RSS NEWS FETCHER
# ===========================================================================

_RSS_TIMEOUT = 8  # seconds per feed

def fetch_headlines(max_per_feed: int = 5) -> List[Headline]:
    """
    Fetch headlines from configured RSS feeds.

    Args:
        max_per_feed: Maximum headlines to take per RSS feed

    Returns:
        List of Headline objects (deduplicated, up to 10 total)
    """
    if not _FEEDPARSER_AVAILABLE:
        logger.warning("feedparser unavailable; returning mock headlines.")
        return _mock_headlines()

    headlines: List[Headline] = []
    seen_titles = set()

    for feed_url in cfg.RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            source = feed.feed.get("title", feed_url.split("/")[2])
            count = 0
            for entry in feed.entries:
                if count >= max_per_feed:
                    break
                title = entry.get("title", "").strip()
                if not title or title.lower() in seen_titles:
                    continue
                seen_titles.add(title.lower())
                headlines.append(Headline(
                    title=title,
                    source=source,
                    link=entry.get("link", ""),
                    published=entry.get("published", ""),
                ))
                count += 1
        except Exception as exc:
            logger.warning(f"Failed to fetch feed {feed_url}: {exc}")

    # Limit to 10 headlines total
    result = headlines[:10]
    if not result:
        logger.warning("No headlines fetched from RSS; using mock data.")
        return _mock_headlines()

    logger.info(f"Fetched {len(result)} headlines from {len(cfg.RSS_FEEDS)} RSS feeds.")
    return result


def _mock_headlines() -> List[Headline]:
    """Fallback mock headlines for offline / test mode."""
    return [
        Headline("NIFTY eyes 25,000 mark amid broad-based buying", "Moneycontrol", "#"),
        Headline("FIIs turn net buyers; pour ₹3,200 cr into Indian equities", "Economic Times", "#"),
        Headline("RBI keeps repo rate unchanged at 6.5%", "Livemint", "#"),
        Headline("IT stocks rally on strong US jobs data", "Economic Times", "#"),
        Headline("India VIX drops to 14.2, signals reduced uncertainty", "Moneycontrol", "#"),
        Headline("Auto sector sees demand slowdown, analysts cautious", "Livemint", "#"),
        Headline("GST collections hit record ₹2.1 lakh cr in October", "Economic Times", "#"),
        Headline("Bank Nifty resistance at 52,500 — analysts say", "Moneycontrol", "#"),
        Headline("US Fed signals one more rate cut in 2024", "Economic Times", "#"),
        Headline("India GDP growth revised to 7.2% for FY25: IMF", "Livemint", "#"),
    ]


# ===========================================================================
# STEP 2: LLM PROMPT BUILDER
# ===========================================================================

def _build_llm_prompt(
    headlines: List[Headline],
    pcr: float,
    oi_bias: str,
    vix: float,
) -> str:
    """
    Construct a structured prompt for the LLM to analyse market sentiment.
    
    The prompt asks for a strict JSON response to make parsing reliable.
    """
    headlines_text = "\n".join(
        f"{i+1}. [{h.source}] {h.title}" for i, h in enumerate(headlines)
    )

    return f"""You are an expert Indian stock market analyst specialising in F&O (Futures & Options) trading.

Analyse the following market data and return ONLY a valid JSON object — no markdown, no extra text.

## Current Market Conditions
- India VIX: {vix:.2f}
- Put-Call Ratio (PCR): {pcr:.3f}
- OI Buildup Bias: {oi_bias}

## Top {len(headlines)} Indian Market Headlines
{headlines_text}

## Instructions
Based on the headlines and market conditions above, return this exact JSON structure:
{{
  "sentiment_score": <float from -10 (very bearish) to +10 (very bullish)>,
  "bias": "<Bullish | Bearish | Neutral>",
  "event_risk_warning": "<string: mention any specific high-impact events like RBI, Fed, elections, earnings, budget — or 'None' if none>",
  "summary": "<2-3 sentence actionable market summary for an F&O trader>"
}}

Rules:
- VIX above 20 = elevated uncertainty, reduce scores toward neutral
- PCR > 1.2 = bullish underpinning from put writing
- PCR < 0.8 = bearish call buildup
- Be concise and specific to Indian markets
- Do NOT wrap the JSON in markdown code blocks"""


# ===========================================================================
# STEP 3: LLM CALL (Gemini / OpenAI)
# ===========================================================================

def _call_gemini(prompt: str) -> Optional[str]:
    """Call Google Gemini API and return raw text response."""
    try:
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=cfg.LLM_API_KEY)
        model = genai.GenerativeModel(cfg.GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=512,
            ),
        )
        return response.text.strip()
    except Exception as exc:
        logger.error(f"Gemini API error: {exc}")
        return None


def _call_openai(prompt: str) -> Optional[str]:
    """Call OpenAI API and return raw text response."""
    try:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=cfg.LLM_API_KEY)
        response = client.chat.completions.create(
            model=cfg.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert Indian stock market F&O analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error(f"OpenAI API error: {exc}")
        return None


def _parse_llm_response(raw: str, headlines: List[Headline]) -> Optional[SentimentResult]:
    """
    Parse the raw LLM response string into a SentimentResult.
    Handles potential markdown code block wrapping.
    """
    if not raw:
        return None

    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        data = json.loads(raw)
        score = float(data.get("sentiment_score", 0.0))
        score = max(-10.0, min(10.0, score))  # clamp
        bias = data.get("bias", "Neutral")
        if bias not in ("Bullish", "Bearish", "Neutral"):
            bias = "Neutral"
        return SentimentResult(
            sentiment_score=score,
            bias=bias,
            event_risk_warning=str(data.get("event_risk_warning", "None")),
            summary=str(data.get("summary", "")),
            headlines=headlines,
            source="LLM",
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning(f"Failed to parse LLM response: {exc}\nRaw: {raw[:200]}")
        return None


# ===========================================================================
# STEP 4: RULE-BASED FALLBACK SENTIMENT ENGINE
# ===========================================================================

# Weighted keyword dictionaries (Indian market context)
_BULLISH_KEYWORDS: dict = {
    # Strong bullish signals
    "rally": 2.0, "surge": 2.0, "soar": 2.0, "record high": 2.5, "breakout": 2.0,
    "bull run": 2.5, "buy": 1.0, "strong": 1.0, "positive": 1.0, "gains": 1.5,
    "rise": 1.0, "climb": 1.0, "upside": 1.5, "outperform": 1.5, "upgrade": 2.0,
    "recovery": 1.5, "rebound": 1.5, "buying": 1.5, "FII buying": 2.0, "FIIs buy": 2.0,
    "inflow": 1.5, "rate cut": 2.0, "stimulus": 1.5, "GDP growth": 1.0, "profit": 1.0,
    "beat estimates": 2.0, "above expectations": 1.5, "good monsoon": 1.0,
    "market up": 1.5, "nifty up": 1.5, "sensex up": 1.5, "52-week high": 2.0,
    "all-time high": 2.5, "lower vix": 1.5, "pcr bullish": 1.5,
}

_BEARISH_KEYWORDS: dict = {
    # Strong bearish signals
    "crash": -2.5, "fall": -1.5, "drop": -1.5, "plunge": -2.5, "slump": -2.0,
    "sell-off": -2.0, "selloff": -2.0, "decline": -1.5, "weak": -1.5, "loss": -1.5,
    "downside": -1.5, "underperform": -1.5, "downgrade": -2.0, "recession": -2.0,
    "inflation": -1.0, "rate hike": -2.0, "FII selling": -2.0, "FIIs sell": -2.0,
    "outflow": -1.5, "default": -2.5, "miss estimates": -2.0, "below expectations": -1.5,
    "uncertainty": -1.0, "risk": -0.8, "concern": -0.8, "worry": -1.0, "caution": -0.8,
    "slowdown": -1.5, "market down": -1.5, "nifty down": -1.5, "sensex down": -1.5,
    "52-week low": -2.0, "all-time low": -2.5, "high vix": -1.5,
    "geopolitical": -1.0, "tension": -1.0, "sanctions": -1.5, "war": -2.0,
}

# High-impact event keywords (trigger event_risk_warning)
_EVENT_KEYWORDS = {
    "rbi": "RBI policy decision",
    "mpc": "RBI MPC meeting",
    "fed": "US Federal Reserve decision",
    "fomc": "FOMC meeting / US Fed policy",
    "cpi": "CPI inflation data release",
    "gdp": "GDP data release",
    "result": "Quarterly earnings season",
    "earnings": "Quarterly earnings",
    "budget": "Union Budget",
    "election": "Election-related uncertainty",
    "expiry": "Options expiry week",
    "nse expiry": "NSE weekly expiry",
    "quarterly results": "Quarterly results season",
}


def _rule_based_sentiment(
    headlines: List[Headline],
    pcr: float,
    oi_bias: str,
    vix: float,
) -> SentimentResult:
    """
    Rule-based sentiment scoring using keyword matching.

    Scoring logic:
      1. Keyword scan all headlines; sum weighted scores
      2. Normalise to [-10, +10]
      3. Apply PCR tilt: PCR > 1.2 → +1.0, PCR < 0.8 → -1.0
      4. Apply VIX dampening: VIX > 25 → scale toward neutral by 30%
      5. Classify bias
      6. Detect event risk keywords
    """
    all_text = " ".join(h.title.lower() for h in headlines)

    # Keyword scoring
    raw_score = 0.0
    for kw, weight in _BULLISH_KEYWORDS.items():
        if kw.lower() in all_text:
            raw_score += weight
    for kw, weight in _BEARISH_KEYWORDS.items():
        if kw.lower() in all_text:
            raw_score += weight  # weight is already negative

    # Normalise to [-10, +10] (assume max possible raw ~20)
    score = max(-10.0, min(10.0, raw_score * (10.0 / 20.0)))

    # PCR tilt
    if pcr > 1.2:
        score = min(10.0, score + 1.0)
    elif pcr < 0.8:
        score = max(-10.0, score - 1.0)

    # OI bias tilt
    if oi_bias == "BULLISH":
        score = min(10.0, score + 0.5)
    elif oi_bias == "BEARISH":
        score = max(-10.0, score - 0.5)

    # VIX dampening: high VIX → pull score toward 0
    if vix > 25.0:
        dampen = 0.7  # 30% reduction toward neutral
        score = score * dampen

    score = round(score, 2)

    # Bias classification
    if score >= 2.0:
        bias = "Bullish"
    elif score <= -2.0:
        bias = "Bearish"
    else:
        bias = "Neutral"

    # Event risk detection
    event_warnings = []
    for kw, label in _EVENT_KEYWORDS.items():
        if kw.lower() in all_text:
            event_warnings.append(label)

    # VIX-based event risk
    if vix > 22.0:
        event_warnings.append(f"India VIX at {vix:.1f} — elevated implied volatility")

    event_risk = "; ".join(event_warnings) if event_warnings else "No major events detected"

    # Summary
    pcr_desc = "bullish (put writers active)" if pcr > 1.2 else ("bearish (call writers active)" if pcr < 0.8 else "neutral")
    summary = (
        f"News flow is {bias.lower()} with a sentiment score of {score:+.1f}/10. "
        f"PCR at {pcr:.2f} indicates {pcr_desc} positioning. "
        f"India VIX at {vix:.1f} suggests {'elevated' if vix > 20 else 'moderate'} market uncertainty — "
        f"{'reduce position size and avoid directional bets' if vix > 20 else 'normal trading conditions prevail'}."
    )

    return SentimentResult(
        sentiment_score=score,
        bias=bias,
        event_risk_warning=event_risk,
        summary=summary,
        headlines=headlines,
        source="Rule-Based",
    )


# ===========================================================================
# PUBLIC INTERFACE
# ===========================================================================

def get_market_sentiment(
    pcr: float,
    oi_bias: str,
    vix: float,
    force_rule_based: bool = False,
) -> SentimentResult:
    """
    Main entry point: fetch news + analyse sentiment using LLM or fallback.

    Args:
        pcr: Put-Call Ratio from OI analysis
        oi_bias: "BULLISH" | "BEARISH" | "NEUTRAL"
        vix: India VIX
        force_rule_based: Set True to skip LLM even if key is available

    Returns:
        SentimentResult with all fields populated
    """
    # Fetch headlines
    headlines = fetch_headlines(max_per_feed=4)

    # Determine if LLM is available
    has_api_key = bool(cfg.LLM_API_KEY and cfg.LLM_API_KEY != "your_llm_api_key_here")
    has_provider = cfg.LLM_PROVIDER in ("gemini", "openai")
    use_llm = has_api_key and has_provider and not force_rule_based

    if use_llm:
        logger.info(f"Using {cfg.LLM_PROVIDER.upper()} for sentiment analysis.")
        prompt = _build_llm_prompt(headlines, pcr, oi_bias, vix)

        raw_response = None
        if cfg.LLM_PROVIDER == "gemini":
            raw_response = _call_gemini(prompt)
        elif cfg.LLM_PROVIDER == "openai":
            raw_response = _call_openai(prompt)

        result = _parse_llm_response(raw_response, headlines) if raw_response else None

        if result:
            return result
        else:
            logger.warning("LLM call failed or returned unparseable response; falling back to rule-based.")

    # Rule-based fallback
    logger.info("Using rule-based sentiment engine.")
    return _rule_based_sentiment(headlines, pcr, oi_bias, vix)
