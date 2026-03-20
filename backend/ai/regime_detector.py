"""
Market regime detector — classifies current market state.
Uses ADX, BB width, EMA alignment, volatility, and structure.
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Regime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    BREAKOUT = "BREAKOUT"
    TRANSITION = "TRANSITION"


@dataclass
class RegimeState:
    regime: Regime
    confidence: float
    adx: float
    bb_width: float
    volatility: str    # "LOW", "MEDIUM", "HIGH"
    scores: dict = None  # Per-regime probability scores

    def __post_init__(self):
        if self.scores is None:
            self.scores = {}


def _compute_regime_scores(adx: float, ema_signal: float, bb_width: float,
                           supertrend: float, structure: float, ema_5m: float) -> dict:
    """Compute a soft probability score (0-1) for each regime state."""
    scores = {}

    # Breakout: BB squeeze + rising ADX
    breakout_score = 0.0
    if bb_width < 0.02:
        breakout_score += (0.02 - bb_width) / 0.02 * 0.5
    if adx > 18:
        breakout_score += min((adx - 18) / 20, 0.5)
    scores[Regime.BREAKOUT.value] = round(min(breakout_score, 1.0), 3)

    # Trending UP
    trend_up_score = 0.0
    if adx > 20:
        trend_up_score += min((adx - 20) / 30, 0.4)
    if ema_signal > 0:
        trend_up_score += min(ema_signal * 0.4, 0.3)
    if supertrend > 0:
        trend_up_score += 0.15
    if structure > 0:
        trend_up_score += min(abs(structure) * 0.15, 0.15)
    scores[Regime.TRENDING_UP.value] = round(min(trend_up_score, 1.0), 3)

    # Trending DOWN
    trend_down_score = 0.0
    if adx > 20:
        trend_down_score += min((adx - 20) / 30, 0.4)
    if ema_signal < 0:
        trend_down_score += min(abs(ema_signal) * 0.4, 0.3)
    if supertrend < 0:
        trend_down_score += 0.15
    if structure < 0:
        trend_down_score += min(abs(structure) * 0.15, 0.15)
    scores[Regime.TRENDING_DOWN.value] = round(min(trend_down_score, 1.0), 3)

    # Ranging: low ADX, wide BB
    ranging_score = 0.0
    if adx < 25:
        ranging_score += min((25 - adx) / 20, 0.5)
    if bb_width > 0.015:
        ranging_score += min((bb_width - 0.015) / 0.03, 0.3)
    if abs(ema_signal) < 0.2:
        ranging_score += 0.2
    scores[Regime.RANGING.value] = round(min(ranging_score, 1.0), 3)

    # Transition: moderate ADX, mixed signals
    transition_score = 0.0
    if 18 < adx < 30:
        transition_score += 0.3
    if abs(ema_signal) < 0.3 and adx > 20:
        transition_score += 0.3
    scores[Regime.TRANSITION.value] = round(min(transition_score, 1.0), 3)

    # Normalize to sum to 1.0
    total = sum(scores.values())
    if total > 0:
        scores = {k: round(v / total, 3) for k, v in scores.items()}

    return scores


def detect_regime(ind_1m: dict, ind_5m: dict = None) -> RegimeState:
    """
    Classify current market regime using multiple features.
    """
    adx = ind_1m.get("adx", 20)
    ema_signal = ind_1m.get("ema_signal", 0)
    price = ind_1m.get("price", 1)
    atr = ind_1m.get("atr", price * 0.01)
    bb_width = ind_1m.get("bb_width", atr / price if price > 0 else 0.02)
    supertrend = ind_1m.get("supertrend_dir", 0)
    structure = ind_1m.get("structure_signal", 0)

    # Volatility classification
    atr_pct = (atr / price * 100) if price > 0 else 1
    if atr_pct < 0.15:
        volatility = "LOW"
    elif atr_pct < 0.5:
        volatility = "MEDIUM"
    else:
        volatility = "HIGH"

    # 5m trend for confirmation
    ema_5m = ind_5m.get("ema_signal", 0) if ind_5m else 0

    # Compute soft scores for all regimes (for frontend radar chart)
    regime_scores = _compute_regime_scores(adx, ema_signal, bb_width, supertrend, structure, ema_5m)

    # ── Breakout detection: BB squeeze + sudden volume/momentum ──────────
    if bb_width < 0.015 and adx > 20:
        confidence = min((adx - 15) / 20, 1.0)
        return RegimeState(Regime.BREAKOUT, confidence, adx, bb_width, volatility, regime_scores)

    # ── Trending ─────────────────────────────────────────────────────────
    if adx > 25:
        confidence = min((adx - 20) / 25, 1.0)

        # Boost confidence if supertrend and structure agree
        if supertrend != 0 and structure != 0:
            if (supertrend > 0 and structure > 0) or (supertrend < 0 and structure < 0):
                confidence = min(confidence + 0.15, 1.0)

        # Boost if 5m trend aligns
        if abs(ema_5m) > 0.3:
            confidence = min(confidence + 0.1, 1.0)

        if ema_signal >= 0.2 or (supertrend > 0 and structure > 0.2):
            return RegimeState(Regime.TRENDING_UP, confidence, adx, bb_width, volatility, regime_scores)
        elif ema_signal <= -0.2 or (supertrend < 0 and structure < -0.2):
            return RegimeState(Regime.TRENDING_DOWN, confidence, adx, bb_width, volatility, regime_scores)
        else:
            return RegimeState(Regime.TRANSITION, confidence * 0.6, adx, bb_width, volatility, regime_scores)

    # ── Ranging ──────────────────────────────────────────────────────────
    elif adx < 20:
        confidence = min((20 - adx) / 15, 1.0)
        return RegimeState(Regime.RANGING, confidence, adx, bb_width, volatility, regime_scores)

    # ── Transition ───────────────────────────────────────────────────────
    else:
        return RegimeState(Regime.TRANSITION, 0.5, adx, bb_width, volatility, regime_scores)


def get_regime_params(regime: Regime) -> dict:
    """Return risk params appropriate for this regime."""
    # SCALPING CONFIG — tight SL/TP for 1-minute timeframe
    # Designed for quick in-and-out trades (30s-2min hold)
    params = {
        Regime.TRENDING_UP: {
            "stop_atr_mult": 0.3,
            "tp_atr_mult": 0.8,
            "min_signals": 3,
            "size_mult": 1.2,
            "preferred_direction": 1,
            "trail_after_atr": 0.4,
            "cooldown_s": 90,       # wait for quality setups
        },
        Regime.TRENDING_DOWN: {
            "stop_atr_mult": 0.3,
            "tp_atr_mult": 0.8,
            "min_signals": 3,
            "size_mult": 1.2,
            "preferred_direction": -1,
            "trail_after_atr": 0.4,
            "cooldown_s": 90,
        },
        Regime.RANGING: {
            "stop_atr_mult": 0.25,
            "tp_atr_mult": 0.5,
            "min_signals": 4,
            "size_mult": 0.7,
            "preferred_direction": 0,
            "trail_after_atr": 0.3,
            "cooldown_s": 180,      # very patient in choppy markets
        },
        Regime.BREAKOUT: {
            "stop_atr_mult": 0.35,
            "tp_atr_mult": 1.0,
            "min_signals": 3,
            "size_mult": 1.0,
            "preferred_direction": 0,
            "trail_after_atr": 0.5,
            "cooldown_s": 60,       # faster on breakouts
        },
        Regime.TRANSITION: {
            "stop_atr_mult": 0.25,
            "tp_atr_mult": 0.5,
            "min_signals": 4,
            "size_mult": 0.5,
            "preferred_direction": 0,
            "trail_after_atr": 0.3,
            "cooldown_s": 120,
        },
    }
    return params[regime]
