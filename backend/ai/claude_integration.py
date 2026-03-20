"""
Claude Integration Layer — glues the advisory brain, scraper, sentiment,
and notifications into the existing trading pipeline.

This module is imported by main.py and provides:
  • startup/shutdown hooks for the Claude subsystem
  • on_candle_hook() — called after each candle to run Claude analysis
  • on_trade_hook() — called after trade open/close for explanations
  • FastAPI router with chat, alerts, and intel endpoints
"""
import asyncio
import json
import logging
import os
import time
from typing import Optional

from fastapi import APIRouter, WebSocket
from pydantic import BaseModel

from ai.claude_advisor import get_advisor, AdvisorContext, get_cost_tracker
from ai.market_intel import get_market_intel
from ai.sentiment import get_sentiment_analyzer
from ai.notifications import get_notifier, Alert
from ai.news_blackout import get_blackout_manager
from ai.trade_journal import get_journal
from ai.ultra_memory import get_ultra_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/claude", tags=["claude"])

# ── Analysis interval (don't call Claude every single candle) ─────────────────
ANALYSIS_INTERVAL_S = 300     # Full analysis every 5 min (was 2 min — saves 60%)
EVENT_CHECK_INTERVAL_S = 600  # Event risk check every 10 min (was 5 min)
_last_analysis: float = 0.0
_last_event_check: float = 0.0
_latest_analysis: dict = {}
_latest_event_risk: dict = {}


# ── Startup / Shutdown ────────────────────────────────────────────────────────

async def startup(broadcast_fn):
    """Start Claude subsystems — LEAN MODE: only adversarial filter + trade journal.
    Scraper and sentiment are paused to save API costs while the bot learns."""
    notifier = get_notifier()
    notifier.set_broadcast(broadcast_fn)

    # NOTE: Market intel scraper and sentiment analyzer are PAUSED to save API costs.
    # Uncomment these when the PPO models are trained and you want full intelligence:
    # get_market_intel().start()
    # get_sentiment_analyzer().start()

    logger.info("Claude advisory layer started (LEAN MODE: adversarial filter + trade journal only)")


async def shutdown():
    """Gracefully stop all Claude subsystems."""
    await get_market_intel().stop()
    await get_sentiment_analyzer().stop()
    await get_advisor().close()
    await get_notifier().close()
    logger.info("Claude advisory layer stopped")


# ── Pipeline hooks ────────────────────────────────────────────────────────────

def build_advisor_context(symbol: str, eng, ind_1m: dict, regime_state,
                           signal, rl_action: int, rl_confidence: float) -> AdvisorContext:
    """Build an AdvisorContext from the existing pipeline state."""
    intel = get_market_intel()
    sentiment = get_sentiment_analyzer()

    position_str = "FLAT"
    if eng.portfolio.position:
        position_str = "LONG" if eng.portfolio.position.direction == 1 else "SHORT"

    # Get actual price from store (last candle close)
    try:
        last_candle = list(eng.store._candles_1m)[-1] if eng.store._candles_1m else {}
        current_price = last_candle.get("close", 0)
    except Exception:
        current_price = eng.portfolio.position.entry_price if eng.portfolio.position else 0

    return AdvisorContext(
        symbol=symbol,
        price=current_price,
        regime=regime_state.regime.value if hasattr(regime_state, 'regime') else str(regime_state),
        regime_confidence=regime_state.confidence if hasattr(regime_state, 'confidence') else 0,
        signal_direction=signal.direction,
        signal_confidence=signal.confidence,
        signal_count=signal.signal_count,
        rl_action=rl_action,
        rl_confidence=rl_confidence,
        position=position_str,
        pnl_usd=eng.portfolio.metrics().get("total_pnl", 0),
        win_rate=eng.portfolio.metrics().get("win_rate", 0) / 100,
        total_trades=eng.portfolio.metrics().get("total_trades", 0),
        recent_news=intel.get_recent_news(10),
        sentiment_score=sentiment.rolling_sentiment,
        indicators={
            "atr": round(ind_1m.get("atr", 0), 2),
            "rsi": round(ind_1m.get("rsi", 50), 1),
            "vwap": round(ind_1m.get("vwap", 0), 2),
            "ema_trend": "UP" if ind_1m.get("ema_signal", 0) > 0 else
                         "DOWN" if ind_1m.get("ema_signal", 0) < 0 else "FLAT",
            "adx": round(ind_1m.get("adx", 20), 1),
            "bb_pct": round(ind_1m.get("bb_pct", 0.5), 3),
        },
        kb_insights=[],
    )


async def on_candle_hook(symbol: str, eng, ind_1m: dict, regime_state,
                          signal, rl_action: int, rl_confidence: float,
                          broadcast_fn) -> Optional[dict]:
    """LEAN MODE: No per-candle Claude calls. Only blackout check (free, no API).
    Market analysis and sentiment are paused to save costs while the bot trains."""
    result = {}

    # Blackout check is FREE (no API call — just checks economic calendar data)
    blackout_mgr = get_blackout_manager()
    result["blackout"] = blackout_mgr.get_status()

    # Fetch macro data periodically (honor throttle interval)
    global _last_event_check
    now = time.time()
    if now - _last_event_check >= EVENT_CHECK_INTERVAL_S:
        _last_event_check = now
        try:
            await blackout_mgr.fetch_macro_data()
        except Exception:
            pass

    return result


async def on_trade_hook(symbol: str, eng, ind_1m: dict, regime_state,
                         signal, rl_action: int, rl_confidence: float,
                         trade_action: str, entry_price: float = 0,
                         sl: float = 0, tp: float = 0,
                         broadcast_fn=None):
    """LEAN MODE: No Claude call on trade open — adversarial filter already ran.
    Just send a simple Telegram alert (free, no API cost)."""
    try:
        is_entry = trade_action in ("LONG", "SHORT")
        notifier = get_notifier()
        await notifier.send_alert(Alert(
            type="trade_entry" if is_entry else "trade_exit",
            severity="info",
            title=f"{'Entered' if is_entry else 'Exited'} {trade_action} {symbol} @ ${entry_price:,.2f}",
            message=f"SL: ${sl:,.2f} | TP: ${tp:,.2f}" if is_entry else "",
        ))
    except Exception as e:
        logger.warning(f"Trade alert failed: {e}")


async def on_trade_close_hook(symbol: str, eng, ind_1m: dict, regime_state,
                               signal, rl_action: int, rl_confidence: float,
                               trade_data: dict, broadcast_fn=None,
                               signal_learner=None):
    """Called after a trade closes — journal + UltraThink teaching."""
    # 1. Trade journal (quick, non-blocking)
    try:
        ctx = build_advisor_context(symbol, eng, ind_1m, regime_state, signal, rl_action, rl_confidence)
        journal = get_journal()
        await journal.record_trade(trade_data, ctx, broadcast_fn)
    except Exception as e:
        logger.warning(f"Trade journal failed: {e}")

    # 2. UltraThink teaching — deep post-trade analysis (non-blocking)
    try:
        ctx = build_advisor_context(symbol, eng, ind_1m, regime_state, signal, rl_action, rl_confidence)
        advisor = get_advisor()
        result = await advisor.ultra_teach(trade_data, ctx)
        teaching = result.get("teaching", {})

        # Apply signal weight adjustments from UltraThink
        signal_adjustments = teaching.get("signal_adjustments", {})
        if signal_adjustments and signal_learner is not None:
            _apply_ultra_signal_adjustments(signal_learner, signal_adjustments)

        # Store the lesson in unlimited memory
        get_ultra_memory().store_lesson(teaching, trade_data)

        # Broadcast to dashboard
        if broadcast_fn:
            await broadcast_fn("ultra_lesson", {
                "type": "trade_review",
                "trade_grade": teaching.get("trade_grade", "?"),
                "pattern": teaching.get("pattern_name", ""),
                "lessons": teaching.get("lessons", []),
                "strategy_tips": teaching.get("strategy_tips", []),
                "best_practices": teaching.get("best_practices", []),
                "focus_area": teaching.get("focus_area", ""),
                "signal_adjustments": signal_adjustments,
                "optimal_entry": teaching.get("optimal_entry", ""),
                "optimal_exit": teaching.get("optimal_exit", ""),
                "avoid_rule": teaching.get("avoid_rule"),
                "repeat_rule": teaching.get("repeat_rule"),
                "reasoning": teaching.get("reasoning_summary", ""),
                "thinking_preview": result.get("thinking", "")[:500],
                "pnl": trade_data.get("pnl_usd", 0),
                "symbol": symbol,
                "timestamp": time.time(),
            })

        grade = teaching.get("trade_grade", "?")
        logger.info(f"UltraThink teaching: grade={grade} | {teaching.get('pattern_name', '')} | focus: {teaching.get('focus_area', '')}")

    except Exception as e:
        logger.warning(f"UltraThink teaching failed: {e}")


def _apply_ultra_signal_adjustments(signal_learner, adjustments: dict):
    """Apply UltraThink's signal weight recommendations to the SignalWeightLearner."""
    for signal_name, adjustment in adjustments.items():
        if not isinstance(adjustment, (int, float)):
            continue
        adjustment = max(-0.3, min(0.3, float(adjustment)))  # Clamp
        if signal_name in signal_learner.signal_scores:
            score = signal_learner.signal_scores[signal_name]
            # Nudge the weight multiplier by the adjustment
            new_mult = score["weight_mult"] + adjustment
            score["weight_mult"] = max(0.2, min(2.0, new_mult))  # Keep within sane bounds
            logger.debug(f"UltraThink adjusted {signal_name} weight: {adjustment:+.2f} → {score['weight_mult']:.2f}")


# ── Lesson access (delegated to UltraMemory) ────────────────────────────────

def get_lessons(limit: int = 50) -> list[dict]:
    """Get recent UltraThink lessons from persistent memory."""
    return get_ultra_memory().get_recent_lessons(limit)


async def trigger_coaching_session(symbol: str, eng, ind_1m: dict, regime_state,
                                    signal, rl_action: int, rl_confidence: float,
                                    broadcast_fn=None, signal_learner=None,
                                    trade_outcomes=None) -> dict:
    """Trigger a full UltraThink coaching session — call every N trades or on demand."""
    ctx = build_advisor_context(symbol, eng, ind_1m, regime_state, signal, rl_action, rl_confidence)
    advisor = get_advisor()
    journal = get_journal()

    # Gather data for coaching
    recent_entries = journal.get_recent(30)
    signal_report = signal_learner.get_report() if signal_learner else []
    outcome_report = trade_outcomes.get_report() if trade_outcomes else {}

    result = await advisor.ultra_coach(recent_entries, signal_report, outcome_report, ctx)
    coaching = result.get("coaching", {})

    # Store coaching in unlimited memory
    get_ultra_memory().store_coaching(coaching)

    # Apply signal adjustments from coaching
    signal_adjustments = coaching.get("signal_adjustments", {})
    if signal_adjustments and signal_learner is not None:
        _apply_ultra_signal_adjustments(signal_learner, signal_adjustments)

    # Broadcast to dashboard
    if broadcast_fn:
        await broadcast_fn("ultra_coaching", {
            "type": "coaching_session",
            "overall_grade": coaching.get("overall_grade", "?"),
            "strongest_edge": coaching.get("strongest_edge", ""),
            "biggest_weakness": coaching.get("biggest_weakness", ""),
            "regime_strategies": coaching.get("regime_strategies", {}),
            "risk_rules": coaching.get("risk_rules", []),
            "timing_rules": coaching.get("timing_rules", []),
            "execution_tips": coaching.get("execution_tips", []),
            "advanced_techniques": coaching.get("advanced_techniques", []),
            "top_3_improvements": coaching.get("top_3_improvements", []),
            "focus_this_week": coaching.get("focus_this_week", ""),
            "signal_adjustments": signal_adjustments,
            "coaching_summary": coaching.get("coaching_summary", ""),
            "thinking_preview": result.get("thinking", "")[:500],
            "timestamp": time.time(),
        })

    logger.info(f"UltraThink coaching: grade={coaching.get('overall_grade', '?')} | focus: {coaching.get('focus_this_week', '')}")
    return result


# ── FastAPI Endpoints ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    symbol: str = "XAUUSD"


class ChatResponse(BaseModel):
    response: str
    timestamp: float
    sentiment: dict = {}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """Chat with the AI advisor about your trading bot."""
    # Build a minimal context (full context would need engine access)
    intel = get_market_intel()
    sentiment = get_sentiment_analyzer()

    ctx = AdvisorContext(
        symbol=req.symbol,
        recent_news=intel.get_recent_news(10),
        sentiment_score=sentiment.rolling_sentiment,
    )

    response = await get_advisor().chat(req.message, ctx)
    return ChatResponse(
        response=response,
        timestamp=time.time(),
        sentiment=sentiment.get_signal(),
    )


@router.get("/analysis")
async def get_analysis():
    """Get the latest Claude market analysis."""
    return {
        "analysis": _latest_analysis,
        "event_risk": _latest_event_risk,
        "sentiment": get_sentiment_analyzer().get_signal(),
        "news": get_market_intel().get_recent_news(20),
    }


@router.get("/costs")
async def get_costs():
    """Get API cost tracking stats."""
    return get_cost_tracker().get_stats()


@router.get("/news")
async def get_news(limit: int = 30):
    """Get recent market headlines."""
    return {
        "headlines": get_market_intel().get_recent_news(limit),
        "aggregate_sentiment": get_market_intel().aggregate_sentiment,
    }


@router.get("/events")
async def get_events():
    """Get upcoming economic events."""
    intel = get_market_intel()
    return {
        "upcoming": intel.get_upcoming_events(hours=8),
        "high_impact": intel.get_high_impact_events(),
    }


@router.get("/sentiment")
async def get_sentiment():
    """Get current sentiment analysis."""
    analyzer = get_sentiment_analyzer()
    return {
        "signal": analyzer.get_signal(),
        "high_impact_alerts": analyzer.high_impact_alerts[-10:],
    }


@router.get("/alerts")
async def get_alerts(limit: int = 50):
    """Get recent alerts."""
    return {"alerts": get_notifier().get_recent_alerts(limit)}


@router.get("/journal")
async def get_trade_journal(limit: int = 20):
    """Get recent trade post-mortem journal entries."""
    journal = get_journal()
    return {
        "entries": journal.get_recent(limit),
        "patterns": journal.get_patterns(),
    }


@router.get("/blackout")
async def get_blackout():
    """Get current news blackout status and macro data."""
    mgr = get_blackout_manager()
    await mgr.fetch_macro_data()
    return mgr.get_status()


@router.get("/lessons")
async def get_ultra_lessons(limit: int = 50):
    """Get recent UltraThink teaching lessons."""
    return {"lessons": get_lessons(limit)}


@router.get("/memory/stats")
async def get_memory_stats():
    """Get full UltraMemory statistics for the dashboard."""
    return get_ultra_memory().get_full_stats()


@router.get("/memory/patterns")
async def get_memory_patterns():
    """Get all known trade patterns with win/loss stats."""
    return {"patterns": get_ultra_memory().get_patterns()}


@router.get("/memory/rules")
async def get_memory_rules():
    """Get all learned rules (avoid, repeat, risk, timing)."""
    return {"rules": get_ultra_memory().get_rules()}


@router.get("/memory/wisdom")
async def get_memory_wisdom():
    """Get distilled wisdom entries."""
    return {"wisdom": get_ultra_memory().get_wisdom()}


@router.get("/memory/signals")
async def get_memory_signals(signal: str = None):
    """Get signal adjustment history."""
    return {"signals": get_ultra_memory().get_signal_history(signal)}


@router.get("/memory/coaching")
async def get_memory_coaching(limit: int = 10):
    """Get coaching session history."""
    return {"coaching": get_ultra_memory().get_coaching_sessions(limit)}


@router.post("/coaching")
async def trigger_coaching():
    """Trigger a full UltraThink coaching session on demand."""
    # This needs engine access — will be wired via main.py
    return {"status": "Coaching endpoint ready — trigger via WebSocket or autopilot"}


@router.post("/review")
async def trigger_review():
    """Trigger a deep performance review and send it to Telegram."""
    journal = get_journal()
    patterns = journal.get_patterns()
    entries = journal.get_recent(50)

    if not entries:
        return {"status": "No trades to review yet"}

    # Build review prompt from journal data
    advisor = get_advisor()
    entries_summary = "\n".join(
        f"  {'WIN' if e.get('win') else 'LOSS'} {e.get('direction','')} {e.get('symbol','')} "
        f"${e.get('pnl_usd',0):+.2f} | {e.get('exit_reason','')} | {e.get('regime','')}"
        for e in entries[-30:]
    )

    prompt = f"""TRADING BOT PERFORMANCE REVIEW

Stats: {json.dumps(patterns, indent=2, default=str)}

Recent trades:
{entries_summary}

Write a comprehensive performance review (200 words max):
1. Overall assessment — is the bot profitable? Getting better or worse?
2. Strongest patterns — when does it win most?
3. Weakest patterns — when does it lose most?
4. Top 3 specific, actionable improvements ranked by expected impact
5. Risk assessment — any concerning trends?"""

    review = await advisor._call_claude(prompt, tier="smart", use_cache=False)

    # Send to Telegram
    from ai.notifications import get_notifier, Alert
    await get_notifier().send_alert(Alert(
        type="performance_review",
        severity="info",
        title="Weekly Performance Review",
        message=review,
    ))

    return {"status": "Review generated and sent to Telegram", "review": review}
