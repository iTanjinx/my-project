"""
Smart trading filters that make the AI genuinely smarter:
1. Multi-timeframe confluence — require alignment across 1m + 5m + 15m
2. Adaptive signal weights — learn which signals work best for each symbol
3. Trade outcome learning — track what conditions produce winners
4. Anti-martingale sizing — reduce size after losses, increase after wins
"""
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Multi-Timeframe Confluence ────────────────────────────────────────────────

def check_mtf_confluence(ind_1m: dict, ind_5m: dict, ind_15m: dict,
                          direction: int) -> dict:
    """
    Check if 1m, 5m, and 15m timeframes agree on direction.
    Returns {aligned: bool, score: 0-1, details: str}
    """
    if not ind_1m:
        return {"aligned": False, "score": 0.0, "details": "No 1m data"}

    def get_bias(ind: dict) -> float:
        """Extract directional bias from indicators: positive = bullish, negative = bearish."""
        if not ind:
            return 0.0
        ema = ind.get("ema_signal", 0) or 0
        supertrend = ind.get("supertrend_signal", 0) or 0
        structure = ind.get("structure_signal", 0) or 0
        macd = ind.get("macd_signal", 0) or 0
        return (ema * 0.35 + supertrend * 0.25 + structure * 0.25 + macd * 0.15)

    bias_1m = get_bias(ind_1m)
    bias_5m = get_bias(ind_5m)
    bias_15m = get_bias(ind_15m)

    # Check alignment with desired direction
    # Threshold raised from 0.05 → 0.12 to stop noise-level "alignment"
    aligned_1m = (direction * bias_1m) > 0.12
    aligned_5m = (direction * bias_5m) > 0.12 if ind_5m else False  # no data = NOT aligned (was True)
    aligned_15m = (direction * bias_15m) > 0.12 if ind_15m else False

    # Check for strong COUNTER-trend on higher TFs (hard block)
    counter_5m = (direction * bias_5m) < -0.15 if ind_5m else False
    counter_15m = (direction * bias_15m) < -0.15 if ind_15m else False

    aligned_count = sum([aligned_1m, aligned_5m, aligned_15m])
    score = aligned_count / 3.0

    # HARD BLOCK: if BOTH higher TFs are counter-trend, never align
    if counter_5m and counter_15m:
        aligned = False
    else:
        # At least 2 of 3 timeframes must agree
        aligned = aligned_count >= 2

    details_parts = []
    for tf, bias, ok in [("1m", bias_1m, aligned_1m), ("5m", bias_5m, aligned_5m), ("15m", bias_15m, aligned_15m)]:
        emoji = "+" if ok else "-"
        details_parts.append(f"{tf}:{emoji}{abs(bias):.2f}")

    return {
        "aligned": aligned,
        "score": round(score, 2),
        "details": " | ".join(details_parts),
        "bias_1m": round(bias_1m, 3),
        "bias_5m": round(bias_5m, 3),
        "bias_15m": round(bias_15m, 3),
    }


# ── Adaptive Signal Weight Learner ────────────────────────────────────────────

@dataclass
class SignalWeightLearner:
    """
    Tracks which signals predict profitable trades and adjusts weights.
    Uses simple exponential moving average of signal accuracy.
    """
    # signal_name → {accuracy_ema, total_trades, correct_predictions}
    signal_scores: dict = field(default_factory=lambda: defaultdict(lambda: {
        "accuracy": 0.5, "total": 0, "correct": 0, "weight_mult": 1.0,
    }))
    learning_rate: float = 0.25  # EMA smoothing — faster adaptation (was 0.1)

    def record_trade_outcome(self, entry_signals: dict, pnl: float, direction: int):
        """
        After a trade closes, update signal accuracy scores.
        entry_signals: {signal_name: signal_value} at time of entry
        """
        is_win = pnl > 0

        for signal_name, signal_value in entry_signals.items():
            if abs(signal_value) < 0.05:
                continue  # signal was near zero, skip

            # Was this signal aligned with the trade direction?
            signal_predicted_direction = 1 if signal_value > 0 else -1
            signal_agreed = (signal_predicted_direction == direction)

            # Was the signal correct? (agreed AND won) or (disagreed AND lost)
            correct = (signal_agreed and is_win) or (not signal_agreed and not is_win)

            score = self.signal_scores[signal_name]
            score["total"] += 1
            if correct:
                score["correct"] += 1

            # Update EMA accuracy
            new_accuracy = 1.0 if correct else 0.0
            score["accuracy"] = score["accuracy"] * (1 - self.learning_rate) + new_accuracy * self.learning_rate

            # Compute weight multiplier: signals with >60% accuracy get boosted
            if score["total"] >= 3:  # faster cold-start (was 5)
                acc = score["accuracy"]
                if acc > 0.65:
                    score["weight_mult"] = 1.0 + (acc - 0.5) * 2  # 0.65 → 1.3x, 0.75 → 1.5x
                elif acc < 0.40:
                    score["weight_mult"] = max(0.3, acc * 1.5)     # reduce unreliable signals
                else:
                    score["weight_mult"] = 1.0

        logger.debug(f"Signal weights updated: {dict(self.signal_scores)}")

    def get_weight_multipliers(self) -> dict:
        """Return current weight multipliers for each signal."""
        return {name: score["weight_mult"] for name, score in self.signal_scores.items()}

    def get_report(self) -> list[dict]:
        """Return signal accuracy report for frontend."""
        result = []
        for name, score in sorted(self.signal_scores.items(), key=lambda x: x[1]["accuracy"], reverse=True):
            if score["total"] < 3:
                continue
            result.append({
                "signal": name,
                "accuracy": round(score["accuracy"] * 100, 1),
                "total": score["total"],
                "correct": score["correct"],
                "weight_mult": round(score["weight_mult"], 2),
            })
        return result


# ── Trade Outcome Tracker ─────────────────────────────────────────────────────

@dataclass
class TradeOutcomeTracker:
    """
    Tracks conditions that produce winning vs losing trades.
    Learns which regimes, sessions, and signal combinations work best.
    """
    # Condition → {wins, losses, total_pnl}
    regime_outcomes: dict = field(default_factory=lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0}))
    session_outcomes: dict = field(default_factory=lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0}))
    strategy_outcomes: dict = field(default_factory=lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0}))

    def record(self, pnl: float, regime: str = "", session: str = "", strategy: str = ""):
        """Record a trade outcome."""
        is_win = pnl > 0
        for tracker, key in [
            (self.regime_outcomes, regime),
            (self.session_outcomes, session),
            (self.strategy_outcomes, strategy),
        ]:
            if key:
                d = tracker[key]
                d["pnl"] += pnl
                if is_win:
                    d["wins"] += 1
                else:
                    d["losses"] += 1

    def should_trade(self, regime: str, session: str) -> dict:
        """
        Check if we should trade in the current regime + session based on history.
        Returns {ok: bool, reason: str, confidence_mult: 0.5-1.5}
        """
        # Check regime performance
        regime_data = self.regime_outcomes.get(regime)
        if regime_data and regime_data["wins"] + regime_data["losses"] >= 5:
            total = regime_data["wins"] + regime_data["losses"]
            wr = regime_data["wins"] / total
            if wr < 0.30:
                return {"ok": False, "reason": f"{regime} win rate only {wr:.0%} ({total} trades)", "confidence_mult": 0.5}

        # Check session performance
        session_data = self.session_outcomes.get(session)
        if session_data and session_data["wins"] + session_data["losses"] >= 5:
            total = session_data["wins"] + session_data["losses"]
            wr = session_data["wins"] / total
            if wr < 0.25:
                return {"ok": False, "reason": f"{session} session win rate only {wr:.0%}", "confidence_mult": 0.5}

        return {"ok": True, "reason": "", "confidence_mult": 1.0}

    def get_report(self) -> dict:
        """Return analysis for frontend."""
        def format_outcomes(data: dict) -> list:
            result = []
            for key, d in sorted(data.items(), key=lambda x: x[1]["wins"] + x[1]["losses"], reverse=True):
                total = d["wins"] + d["losses"]
                if total == 0:
                    continue
                result.append({
                    "name": key,
                    "wins": d["wins"],
                    "losses": d["losses"],
                    "win_rate": round(d["wins"] / total * 100, 1),
                    "pnl": round(d["pnl"], 2),
                })
            return result

        return {
            "by_regime": format_outcomes(dict(self.regime_outcomes)),
            "by_session": format_outcomes(dict(self.session_outcomes)),
            "by_strategy": format_outcomes(dict(self.strategy_outcomes)),
        }


# ── Anti-Martingale Position Sizing ───────────────────────────────────────────

def anti_martingale_size(base_mult: float, consecutive_wins: int,
                          consecutive_losses: int, max_mult: float = 2.0) -> float:
    """
    Scale position size based on recent performance.
    After wins: gradually increase (riding the hot streak)
    After losses: reduce (protect capital)
    """
    if consecutive_losses >= 3:
        # 3+ losses: cut to 50% size
        return base_mult * 0.5
    elif consecutive_losses >= 2:
        # 2 losses: cut to 70%
        return base_mult * 0.7
    elif consecutive_losses >= 1:
        # 1 loss: cut to 85%
        return base_mult * 0.85
    elif consecutive_wins >= 5:
        # 5+ wins: increase to 150%
        return min(base_mult * 1.5, max_mult)
    elif consecutive_wins >= 3:
        # 3+ wins: increase to 130%
        return min(base_mult * 1.3, max_mult)
    elif consecutive_wins >= 2:
        # 2 wins: increase to 115%
        return min(base_mult * 1.15, max_mult)
    return base_mult
