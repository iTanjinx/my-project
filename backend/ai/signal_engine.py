"""
Scalping signal engine — the AI's decision brain.

Two core strategies:
  1. TREND CONTINUATION — ride momentum pullbacks in trending markets
  2. MEAN REVERSION — fade extremes at Bollinger/StochRSI levels in ranging markets

Each signal is weighted by reliability. A trade fires when weighted score ≥ threshold.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SignalResult:
    direction: int          # +1 = LONG, -1 = SHORT, 0 = HOLD
    confidence: float       # 0.0 – 1.0
    signal_count: int       # How many independent signals agreed
    reasoning: str          # Human-readable explanation
    raw_signals: dict       # Individual signal values
    strategy: str = ""      # "TREND", "REVERSAL", "RL_OVERRIDE", ""
    # ── Enriched AI data (exposed to frontend) ──
    bull_score: float = 0.0
    bear_score: float = 0.0
    bull_count: int = 0
    bear_count: int = 0
    weighted_signals: dict = field(default_factory=dict)
    trend_confidence: float = 0.0
    reversion_confidence: float = 0.0
    strategy_winner: str = ""


# ── Signal weights (tuned for 1m scalping) ───────────────────────────────────
# Higher weight = more influence on the final decision
# These are the DEFAULT weights — overridden per-regime below
SIGNAL_WEIGHTS = {
    # Trend signals (most reliable on 1m)
    "ema_1m":        1.5,   # EMA ribbon alignment — the backbone
    "supertrend":    1.4,   # Supertrend direction — strong trend filter
    "structure":     1.3,   # Market structure (HH/HL or LH/LL)
    "macd_1m":       1.0,   # MACD histogram direction

    # Entry timing signals
    "stochrsi":      1.2,   # StochRSI crosses — precise entry timing
    "candle":        1.1,   # Candle patterns — reversal/continuation confirmation
    "rsi_1m":        0.8,   # RSI — overbought/oversold
    "bb_1m":         0.9,   # Bollinger band position
    "vwap":          1.0,   # VWAP — institutional level

    # Momentum signals
    "momentum":      0.9,   # ROC — rate of change
    "accel":         0.7,   # Momentum acceleration

    # Multi-timeframe confirmation
    "ema_5m":        1.3,   # 5m trend — very reliable
    "macd_5m":       0.8,
    "ema_15m":       1.1,   # 15m macro trend
}

# ── Regime-specific weight multipliers ────────────────────────────────────────
# Applied on top of base SIGNAL_WEIGHTS per market regime.
# >1.0 = boost that signal, <1.0 = dampen it for this regime.
REGIME_WEIGHT_MULTS = {
    "TRENDING_UP": {
        "ema_1m": 1.3, "supertrend": 1.3, "structure": 1.2, "ema_5m": 1.3,
        "ema_15m": 1.2, "momentum": 1.2, "macd_1m": 1.1,
        "stochrsi": 0.7, "bb_1m": 0.6, "rsi_1m": 0.6,  # dampened: overbought traps in trends
    },
    "TRENDING_DOWN": {
        "ema_1m": 1.3, "supertrend": 1.3, "structure": 1.2, "ema_5m": 1.3,
        "ema_15m": 1.2, "momentum": 1.2, "macd_1m": 1.1,
        "stochrsi": 0.7, "bb_1m": 0.6, "rsi_1m": 0.6,
    },
    "RANGING": {
        "stochrsi": 1.4, "bb_1m": 1.4, "rsi_1m": 1.3, "vwap": 1.3,
        "candle": 1.2,  # reversal candles matter in ranges
        "ema_1m": 0.6, "supertrend": 0.5, "structure": 0.7,  # trend signals noisy in ranges
        "momentum": 0.6, "ema_5m": 0.7,
    },
    "BREAKOUT": {
        "supertrend": 1.4, "momentum": 1.4, "accel": 1.3, "structure": 1.3,
        "bb_1m": 1.2,  # BB squeeze breakout
        "ema_1m": 1.1, "vwap": 1.1,
        "stochrsi": 0.6, "rsi_1m": 0.5,  # oscillators lag on breakouts
    },
    "TRANSITION": {
        # Conservative: slightly dampen everything, boost confirmation signals
        "ema_5m": 1.3, "ema_15m": 1.3, "vwap": 1.2,
        "ema_1m": 0.8, "supertrend": 0.8, "momentum": 0.7,
        "stochrsi": 0.8, "bb_1m": 0.8,
    },
}


def get_regime_weights(regime: str) -> dict:
    """Get signal weights adjusted for the current market regime."""
    mults = REGIME_WEIGHT_MULTS.get(regime, {})
    adjusted = dict(SIGNAL_WEIGHTS)
    for name, mult in mults.items():
        if name in adjusted:
            adjusted[name] = round(adjusted[name] * mult, 2)
    return adjusted


def evaluate_signals(
    ind_1m: dict,
    ind_5m: dict,
    ind_15m: dict,
    min_signals: int = 3,
    min_confidence: float = 0.55,
    preferred_direction: int = 0,
    weight_overrides: dict = None,
    regime: str = "",
) -> SignalResult:
    """
    Evaluate all signals and produce a trade decision.
    Uses weighted scoring — signals with higher weights count more.
    weight_overrides: optional dict of {signal_name: multiplier} from adaptive learning.
    regime: current market regime for regime-specific weight adjustments.
    """
    if not ind_1m:
        return SignalResult(0, 0.0, 0, "No 1m data", {})

    adx = ind_1m.get("adx", 20)
    price = ind_1m.get("price", 0)

    # Start with regime-specific base weights (or defaults if no regime)
    active_weights = get_regime_weights(regime) if regime else dict(SIGNAL_WEIGHTS)

    # Then apply adaptive weight overrides from signal learner on top
    if weight_overrides:
        for name, mult in weight_overrides.items():
            if name in active_weights:
                active_weights[name] = round(active_weights[name] * mult, 2)

    # ── Collect all raw signals ──────────────────────────────────────────
    raw = {}

    # 1m signals
    raw["ema_1m"]     = ind_1m.get("ema_signal", 0.0)
    raw["macd_1m"]    = ind_1m.get("macd_signal", 0.0)
    raw["rsi_1m"]     = ind_1m.get("rsi_signal", 0.0)
    raw["bb_1m"]      = ind_1m.get("bb_signal", 0.0)
    raw["stochrsi"]   = ind_1m.get("stochrsi_signal", 0.0)
    raw["vwap"]       = ind_1m.get("vwap_signal", 0.0)
    raw["supertrend"] = ind_1m.get("supertrend_signal", 0.0)
    raw["momentum"]   = ind_1m.get("roc_signal", 0.0)
    raw["candle"]     = ind_1m.get("candle_signal", 0.0)
    raw["structure"]  = ind_1m.get("structure_signal", 0.0)
    raw["accel"]      = ind_1m.get("accel_signal", 0.0)

    # 5m confirmation
    raw["ema_5m"]  = ind_5m.get("ema_signal", 0.0) if ind_5m else 0.0
    raw["macd_5m"] = ind_5m.get("macd_signal", 0.0) if ind_5m else 0.0

    # 15m macro
    raw["ema_15m"] = ind_15m.get("ema_signal", 0.0) if ind_15m else 0.0

    # ── Evaluate trend continuation strategy ─────────────────────────────
    trend_result = _evaluate_trend_strategy(raw, ind_1m, adx, preferred_direction, active_weights)

    # ── Evaluate mean-reversion strategy ─────────────────────────────────
    reversion_result = _evaluate_reversion_strategy(raw, ind_1m, adx, preferred_direction)

    # Pick the stronger signal
    if trend_result.confidence >= reversion_result.confidence and trend_result.direction != 0:
        best = trend_result
    elif reversion_result.direction != 0:
        best = reversion_result
    elif trend_result.direction != 0:
        best = trend_result
    else:
        # Neither strategy fired — return the one closer to firing for display
        best = trend_result if trend_result.confidence >= reversion_result.confidence else reversion_result

    # Enrich with raw signals for the dashboard
    best.raw_signals = raw

    # ── Compute weighted signal contributions for frontend heatmap ──
    weighted_sigs = {}
    for name, value in raw.items():
        weight = active_weights.get(name, 0.5)
        weighted_sigs[name] = round(value * weight, 3)

    best.weighted_signals = weighted_sigs
    best.trend_confidence = round(trend_result.confidence, 3)
    best.reversion_confidence = round(reversion_result.confidence, 3)
    best.strategy_winner = best.strategy

    # Carry forward bull/bear scores from the winning strategy
    # (trend strategy populates these; reversion uses condition counts)
    if best.strategy == "TREND":
        best.bull_score = round(trend_result.bull_score, 3)
        best.bear_score = round(trend_result.bear_score, 3)
        best.bull_count = trend_result.bull_count
        best.bear_count = trend_result.bear_count
    else:
        best.bull_score = round(reversion_result.bull_score, 3)
        best.bear_score = round(reversion_result.bear_score, 3)
        best.bull_count = reversion_result.bull_count
        best.bear_count = reversion_result.bear_count

    return best


def _evaluate_trend_strategy(
    raw: dict, ind_1m: dict, adx: float, preferred_direction: int,
    active_weights: dict = None,
) -> SignalResult:
    if active_weights is None:
        active_weights = SIGNAL_WEIGHTS
    """
    TREND CONTINUATION: enter on pullback in the direction of the trend.
    Best in trending markets (ADX > 20).
    """
    # Trend requires some directional movement
    if adx < 15:
        return SignalResult(0, 0.0, 0, "ADX too low for trend trade", {}, "TREND")

    # Calculate weighted bull/bear scores
    bull_score = 0.0
    bear_score = 0.0
    bull_count = 0
    bear_count = 0
    total_weight = sum(active_weights.values())

    threshold = 0.15  # lower than before — we want to catch more moves

    for name, value in raw.items():
        weight = active_weights.get(name, 0.5)
        if value > threshold:
            bull_score += value * weight
            bull_count += 1
        elif value < -threshold:
            bear_score += abs(value) * weight
            bear_count += 1

    # Normalize to 0-1 range
    bull_conf = bull_score / total_weight if total_weight > 0 else 0
    bear_conf = bear_score / total_weight if total_weight > 0 else 0

    # ADX boost: stronger trends get confidence boost
    adx_boost = min((adx - 15) / 30, 0.3) if adx > 15 else 0
    bull_conf += adx_boost if bull_count > bear_count else 0
    bear_conf += adx_boost if bear_count > bull_count else 0

    # Supertrend alignment bonus (very reliable)
    st = raw.get("supertrend", 0)
    if st > 0:
        bull_conf *= 1.15
    elif st < 0:
        bear_conf *= 1.15

    # 5m trend alignment bonus
    ema5m = raw.get("ema_5m", 0)
    if ema5m > 0.2:
        bull_conf *= 1.1
    elif ema5m < -0.2:
        bear_conf *= 1.1

    # Determine direction
    direction = 0
    confidence = 0.0
    signal_count = 0
    reasoning_parts = []

    min_count = max(2, min_signals_for_adx(adx))  # adaptive min signals

    if bull_count >= min_count and bull_conf >= 0.35 and preferred_direction >= 0:
        direction = 1
        confidence = min(bull_conf, 1.0)
        signal_count = bull_count
        reasoning_parts.append(f"TREND LONG: {bull_count} signals (wt={bull_conf:.2f})")
    elif bear_count >= min_count and bear_conf >= 0.35 and preferred_direction <= 0:
        direction = -1
        confidence = min(bear_conf, 1.0)
        signal_count = bear_count
        reasoning_parts.append(f"TREND SHORT: {bear_count} signals (wt={bear_conf:.2f})")

    # Add key signal breakdown
    for name, val in sorted(raw.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        if abs(val) > threshold:
            arrow = "↑" if val > 0 else "↓"
            reasoning_parts.append(f"{name}={arrow}{abs(val):.2f}")

    reasoning = " | ".join(reasoning_parts) if reasoning_parts else "HOLD: weak trend signals"
    result = SignalResult(direction, confidence, signal_count, reasoning, {}, "TREND")
    result.bull_score = bull_score
    result.bear_score = bear_score
    result.bull_count = bull_count
    result.bear_count = bear_count
    return result


def _evaluate_reversion_strategy(
    raw: dict, ind_1m: dict, adx: float, preferred_direction: int
) -> SignalResult:
    """
    MEAN REVERSION: fade extreme moves at BB/StochRSI extremes.
    Best in ranging markets (ADX < 25).
    """
    # Mean reversion works best in calm markets
    if adx > 35:
        return SignalResult(0, 0.0, 0, "ADX too high for reversion", {}, "REVERSAL")

    rsi = ind_1m.get("rsi", 50)
    stochrsi_k = ind_1m.get("stochrsi_k", 50)
    bb_pct = ind_1m.get("bb_pct", 0.5)
    candle = raw.get("candle", 0)
    bb_signal = raw.get("bb_1m", 0)
    stochrsi_signal = raw.get("stochrsi", 0)

    direction = 0
    confidence = 0.0
    signal_count = 0
    reasoning_parts = []

    # ── Bullish reversal setup ───────────────────────────────────────────
    # StochRSI oversold + BB lower band + bullish candle = strong buy
    bull_conditions = 0
    bull_strength = 0.0

    if stochrsi_k < 25:
        bull_conditions += 1
        bull_strength += 0.3
    if bb_pct < 0.15:
        bull_conditions += 1
        bull_strength += 0.25
    if rsi < 35:
        bull_conditions += 1
        bull_strength += 0.2
    if candle > 0.3:  # bullish candle pattern
        bull_conditions += 1
        bull_strength += 0.25
    if raw.get("vwap", 0) > 0.2:  # below VWAP
        bull_conditions += 1
        bull_strength += 0.15

    # ── Bearish reversal setup ───────────────────────────────────────────
    bear_conditions = 0
    bear_strength = 0.0

    if stochrsi_k > 75:
        bear_conditions += 1
        bear_strength += 0.3
    if bb_pct > 0.85:
        bear_conditions += 1
        bear_strength += 0.25
    if rsi > 65:
        bear_conditions += 1
        bear_strength += 0.2
    if candle < -0.3:  # bearish candle pattern
        bear_conditions += 1
        bear_strength += 0.25
    if raw.get("vwap", 0) < -0.2:  # above VWAP
        bear_conditions += 1
        bear_strength += 0.15

    if bull_conditions >= 3 and bull_strength >= 0.55 and preferred_direction >= 0:
        direction = 1
        confidence = min(bull_strength, 1.0)
        signal_count = bull_conditions
        reasoning_parts.append(
            f"REVERSAL LONG: StochRSI={stochrsi_k:.0f} BB%={bb_pct:.2f} "
            f"RSI={rsi:.0f} ({bull_conditions} conditions)"
        )
    elif bear_conditions >= 3 and bear_strength >= 0.55 and preferred_direction <= 0:
        direction = -1
        confidence = min(bear_strength, 1.0)
        signal_count = bear_conditions
        reasoning_parts.append(
            f"REVERSAL SHORT: StochRSI={stochrsi_k:.0f} BB%={bb_pct:.2f} "
            f"RSI={rsi:.0f} ({bear_conditions} conditions)"
        )

    reasoning = " | ".join(reasoning_parts) if reasoning_parts else "HOLD: no reversal setup"
    result = SignalResult(direction, confidence, signal_count, reasoning, {}, "REVERSAL")
    result.bull_score = bull_strength
    result.bear_score = bear_strength
    result.bull_count = bull_conditions
    result.bear_count = bear_conditions
    return result


def min_signals_for_adx(adx: float) -> int:
    """Adaptive minimum signal count based on trend strength."""
    if adx > 35:
        return 2  # strong trend — fewer signals needed
    elif adx > 25:
        return 3
    elif adx > 20:
        return 3
    else:
        return 4  # ranging — need more confirmation
