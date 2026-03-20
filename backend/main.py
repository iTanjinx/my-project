"""
FastAPI backend — dual-symbol AI trading engine.
  • BTCUSDT via Bitstamp WebSocket + REST  (matches TradingView default price, free)
  • XAUUSD  via Twelve Data REST API (history + 110s polling) [preferred]
            OR Yahoo Finance XAUUSD=X spot (fallback, no key needed)
"""
import asyncio
import json
import logging
import os
import sys
import time

# Load .env before anything reads env vars
from dotenv import load_dotenv
load_dotenv()
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

sys.path.insert(0, ".")
from data.bitstamp_feed    import BitstampFeed
from data.twelvedata_feed  import TwelveDataFeed
from data.yfinance_feed    import YFinanceFeed
from data.candle_store     import CandleStore
from indicators.technical import compute_indicators
from ai.regime_detector   import detect_regime, get_regime_params
from ai.signal_engine     import evaluate_signals
from ai.trading_env       import TradingEnv
from ai.rl_agent          import RLAgent
from ai.ha_strategy       import evaluate_ha_strategy
from ai.session_manager   import detect_session
from ai.performance_tracker import PerformanceTracker
from ai.smart_filters import check_mtf_confluence, SignalWeightLearner, TradeOutcomeTracker, anti_martingale_size
from portfolio.paper_portfolio import PaperPortfolio
from portfolio.ha_position_manager import open_ha_position, check_ha_tiered_exits
from data.mt5_connector import MT5Connector, MT5DataFeed, MT5Config, MT5_AVAILABLE
from db.database          import init_db, close_db, DB_PATH
from db.trade_repository  import save_trade, save_snapshot
from knowledge.vector_store import KnowledgeStore
from knowledge.embedder     import KnowledgeEmbedder
from knowledge.retriever    import TradingKnowledgeRetriever
from knowledge.ingest       import IngestionManager
from ai.claude_integration  import router as claude_router, startup as claude_startup, \
                                    shutdown as claude_shutdown, on_candle_hook, on_trade_hook, \
                                    on_trade_close_hook, build_advisor_context, \
                                    trigger_coaching_session
from ai.adversarial_filter  import check_trade as adversarial_check
from ai.news_blackout       import get_blackout_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

RETRAIN_INTERVAL         = 10     # retrain after this many live candles
MIN_CANDLES              = 50    # warm-up: wait for enough history before trading
TRADE_COOLDOWN_S         = 45    # seconds to wait after a close before next entry (scalping = fast)
CONTINUOUS_TRAIN_STEPS   = 32768  # PPO steps per continuous training cycle (aggressive)
CONTINUOUS_TRAIN_INTERVAL_S = 20  # seconds between continuous training cycles (very fast)
AUTO_PRETRAIN_STEPS      = 2_000_000 # steps for auto-pretrain at startup (2M — deep learning)
POST_TRADE_TRAIN_STEPS   = 8192   # PPO steps to run after each closed trade (learn fast)
COACHING_EVERY_N_TRADES  = 5      # UltraThink coaching every 5 trades (was 10)

clients: Set[WebSocket] = set()
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ai", "models")
strategy_mode: str = "AI"  # "AI" or "HA"
trade_mode: str = "PAPER"  # "PAPER", "LIVE", "SIGNAL"
mt5_connector: MT5Connector = None
perf_tracker = PerformanceTracker()
signal_learner = SignalWeightLearner()
trade_outcomes = TradeOutcomeTracker()

# ── Knowledge Base (lazy-loaded) ──
knowledge_store: KnowledgeStore = None
knowledge_embedder: KnowledgeEmbedder = None
knowledge_retriever: TradingKnowledgeRetriever = None
ingestion_manager: IngestionManager = None


# ─── Per-symbol engine ────────────────────────────────────────────────────────
@dataclass
class SymbolEngine:
    symbol: str
    store: CandleStore = field(default_factory=lambda: CandleStore(maxlen=10000))  # ~7 days of 1m data
    portfolio: PaperPortfolio = field(default_factory=PaperPortfolio)
    candle_count: int = 0
    last_trade_time: float = 0.0
    last_ind: dict = field(default_factory=dict)
    prev_regime: str = "UNKNOWN"

    def __post_init__(self):
        model_path = os.path.join(MODEL_DIR, f"ppo_{self.symbol.lower()}.zip")
        self.rl_agent = RLAgent(model_path=model_path)
        if os.path.exists(model_path):
            dummy_env = TradingEnv([])
            try:
                self.rl_agent.load_or_create(dummy_env)
            except Exception:
                pass


engines: dict[str, SymbolEngine] = {
    "XAUUSD":  SymbolEngine("XAUUSD"),
    "USTEC":   SymbolEngine("USTEC"),
}


# ─── Broadcast ───────────────────────────────────────────────────────────────
async def broadcast(msg_type: str, data: dict):
    payload = json.dumps({"type": msg_type, **data})
    dead = set()
    for ws in clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


# ─── AI intelligence helpers ─────────────────────────────────────────────────

def _build_ai_thoughts(signal, rl_action, rl_confidence, rl_internals,
                        regime_state, eng, ind_1m, kb_context=None) -> dict:
    """Build the AI thought stream payload."""
    thoughts = []
    pattern_names = ind_1m.get("pattern_names", [])
    structure_label = ind_1m.get("structure_label", "")

    # Pattern detection thoughts
    for p in pattern_names:
        bullish = p in ("Bullish Engulfing", "Hammer", "Morning Star")
        thoughts.append({"text": f"{p} detected", "type": "pattern",
                         "sentiment": "bull" if bullish else "bear" if p != "Doji" else "neutral"})

    # Structure thought
    if structure_label:
        bullish = structure_label in ("HH/HL", "HH", "HL")
        thoughts.append({"text": f"Structure: {structure_label}", "type": "structure",
                         "sentiment": "bull" if bullish else "bear"})

    # Signal engine thought
    if signal.direction == 1:
        thoughts.append({"text": f"Signals: LONG ({signal.signal_count}/8, {signal.confidence*100:.0f}%)",
                         "type": "signal", "sentiment": "bull"})
    elif signal.direction == -1:
        thoughts.append({"text": f"Signals: SHORT ({signal.signal_count}/8, {signal.confidence*100:.0f}%)",
                         "type": "signal", "sentiment": "bear"})
    else:
        thoughts.append({"text": f"Signals: HOLD (no confluence)", "type": "signal", "sentiment": "neutral"})

    # RL thought
    rl_dir = 0 if rl_action == 0 else (1 if rl_action == 1 else -1)
    rl_label = {0: "HOLD", 1: "LONG", 2: "SHORT"}.get(rl_action, "HOLD")
    if rl_confidence > 0.6:
        thoughts.append({"text": f"RL: {rl_label} ({rl_confidence*100:.0f}% conf)",
                         "type": "rl", "sentiment": "bull" if rl_action == 1 else "bear" if rl_action == 2 else "neutral"})

    # Divergence detection
    signal_dir = signal.direction
    divergence = False
    divergence_detail = ""
    divergence_score = 0.0

    if signal_dir != 0 and rl_dir != 0 and signal_dir != rl_dir:
        divergence = True
        divergence_score = 1.0
        s_label = "LONG" if signal_dir == 1 else "SHORT"
        divergence_detail = f"Signals say {s_label} but RL prefers {rl_label}"
        thoughts.append({"text": f"DIVERGENCE: {divergence_detail}", "type": "divergence", "sentiment": "warn"})
    elif signal_dir != 0 and rl_dir == 0:
        divergence_score = 0.5 * signal.confidence
        if divergence_score > 0.3:
            s_label = "LONG" if signal_dir == 1 else "SHORT"
            divergence_detail = f"Signals say {s_label} but RL abstaining"
            thoughts.append({"text": divergence_detail, "type": "divergence", "sentiment": "warn"})
    elif signal_dir == 0 and rl_dir != 0 and rl_confidence > 0.7:
        divergence_score = 0.5 * rl_confidence
        divergence_detail = f"No signal but RL wants {rl_label}"
        thoughts.append({"text": divergence_detail, "type": "divergence", "sentiment": "warn"})

    # Regime transition
    regime_changed = eng.prev_regime != regime_state.regime.value and eng.prev_regime != "UNKNOWN"
    if regime_changed:
        thoughts.append({
            "text": f"Regime shift: {eng.prev_regime.replace('_',' ')} → {regime_state.regime.value.replace('_',' ')}",
            "type": "regime", "sentiment": "warn",
        })

    # Knowledge base insights
    if kb_context and kb_context.get("relevance_score", 0) > 0.3:
        for insight in kb_context.get("key_insights", [])[:2]:
            sentiment = "bull" if kb_context.get("sentiment_bias", 0) > 0.2 else \
                        "bear" if kb_context.get("sentiment_bias", 0) < -0.2 else "neutral"
            thoughts.append({"text": f"KB: {insight[:70]}", "type": "knowledge", "sentiment": sentiment})
        if kb_context.get("risk_level", 0) > 0.5:
            thoughts.append({"text": "KB WARNING: Elevated risk conditions", "type": "knowledge", "sentiment": "warn"})

    return {
        "thoughts": thoughts,
        "divergence": divergence,
        "divergence_detail": divergence_detail,
        "divergence_score": round(divergence_score, 3),
        "pattern_names": pattern_names,
        "structure_label": structure_label,
        "regime_changed": regime_changed,
        "regime_from": eng.prev_regime if regime_changed else "",
        "regime_to": regime_state.regime.value if regime_changed else "",
    }


def _compute_entry_forecast(eng, direction, regime) -> dict:
    """Estimate win probability based on historical trades in similar conditions."""
    similar = [t for t in eng.portfolio.trades if t.direction == direction]
    if len(similar) < 3:
        return {"win_prob": 0.5, "avg_win_pct": 0, "avg_loss_pct": 0, "expected_value": 0, "sample_size": 0}
    wins = [t for t in similar if t.pnl_usd > 0]
    losses = [t for t in similar if t.pnl_usd <= 0]
    win_prob = len(wins) / len(similar) if similar else 0.5
    avg_win = sum(t.pnl_pct for t in wins) / len(wins) * 100 if wins else 0
    avg_loss = sum(t.pnl_pct for t in losses) / len(losses) * 100 if losses else 0
    expected = win_prob * avg_win + (1 - win_prob) * avg_loss
    return {
        "win_prob": round(win_prob, 3),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "expected_value": round(expected, 2),
        "sample_size": len(similar),
    }


def _compute_correlation() -> dict:
    """Compute rolling 30-bar return correlation between BTC and XAU."""
    try:
        btc_closes = [c["close"] for c in engines["USTEC"].store._candles_1m][-30:]
        xau_closes = [c["close"] for c in engines["XAUUSD"].store._candles_1m][-30:]
        if len(btc_closes) < 20 or len(xau_closes) < 20:
            return {"correlation": 0, "btc_momentum": 0, "xau_momentum": 0}
        btc_rets = np.diff(btc_closes) / np.array(btc_closes[:-1])
        xau_rets = np.diff(xau_closes) / np.array(xau_closes[:-1])
        min_len = min(len(btc_rets), len(xau_rets))
        corr = float(np.corrcoef(btc_rets[-min_len:], xau_rets[-min_len:])[0, 1])
        if np.isnan(corr):
            corr = 0.0
        return {
            "correlation": round(corr, 3),
            "btc_momentum": round(float(np.mean(btc_rets[-5:])) * 100, 4),
            "xau_momentum": round(float(np.mean(xau_rets[-5:])) * 100, 4),
        }
    except Exception:
        return {"correlation": 0, "btc_momentum": 0, "xau_momentum": 0}


# ─── Shared pipeline ─────────────────────────────────────────────────────────
async def on_tick(tick: dict, symbol: str):
    eng = engines[symbol]
    price = tick["price"]
    pos = eng.portfolio.position

    # ── Tick-level SL/TP checking — prevents stop blowthrough ────────────
    if pos is not None:
        # Update trailing stop on every tick
        atr = eng.last_ind.get("atr", price * 0.01)
        eng.portfolio._update_trailing_stop(price, price, atr)

        # Check exits at tick level (not just candle close)
        closed = eng.portfolio.check_exits(price, price, atr=atr)
        if closed:
            eng.last_trade_time = time.time()
            await save_trade(closed)
            await broadcast("trade_close", {
                "symbol":      symbol,
                "pnl_usd":     round(closed.pnl_usd, 2),
                "pnl_pct":     round(closed.pnl_pct * 100, 2),
                "exit_price":  closed.exit_price,
                "exit_reason": closed.exit_reason,
                "direction":   closed.direction,
                "entry_price": closed.entry_price,
                "reasoning":   closed.reasoning,
            })
            if eng.rl_agent.is_trained:
                asyncio.create_task(_post_trade_retrain_task(symbol))

    unrealized = eng.portfolio.unrealized_pnl(price) if eng.portfolio.position else 0
    tick_data = {
        "symbol":         symbol,
        "price":          price,
        "time":           tick["time"],
        "unrealized_pnl": round(unrealized, 2),
    }
    # Send live stop price so frontend can update the chart line
    if eng.portfolio.position:
        tick_data["stop_price"] = round(eng.portfolio.position.stop_price, 2)
        tick_data["trail_active"] = eng.portfolio.position.trail_activated
    await broadcast("tick", tick_data)


async def on_candle_close(candle: dict, symbol: str):
    eng = engines[symbol]
    eng.store.add(candle)
    eng.candle_count += 1

    df_1m  = eng.store.get_df("1m")
    df_5m  = eng.store.get_df("5m")
    df_15m = eng.store.get_df("15m")

    ind_1m  = await asyncio.to_thread(compute_indicators, df_1m)  if df_1m  is not None else {}
    ind_5m  = await asyncio.to_thread(compute_indicators, df_5m)  if df_5m  is not None else {}
    ind_15m = await asyncio.to_thread(compute_indicators, df_15m) if df_15m is not None else {}

    if not ind_1m:
        return

    eng.last_ind = {**ind_1m}  # store for manual trades

    # ── Session awareness ──
    session_info = detect_session()

    # ── Heiken Ashi strategy evaluation (always runs for frontend display) ──
    ha_result = await asyncio.to_thread(evaluate_ha_strategy, list(eng.store._candles_1m), symbol)

    # ── HA-mode: check tiered exits before anything else ──
    if strategy_mode == "HA" and eng.portfolio.position is not None:
        ha_closes = check_ha_tiered_exits(eng.portfolio, candle["high"], candle["low"], symbol)
        for ct in ha_closes:
            eng.last_trade_time = time.time()
            await save_trade(ct)
            perf_tracker.record_trade(ct.pnl_usd, ha_result.signal.confidence,
                                       ct.strategy, session_info.session.value, ct.strategy)
            await broadcast("trade_close", {
                "symbol": symbol, "pnl_usd": round(ct.pnl_usd, 2),
                "pnl_pct": round(ct.pnl_pct * 100, 2),
                "exit_price": ct.exit_price, "exit_reason": ct.exit_reason,
                "direction": ct.direction, "entry_price": ct.entry_price,
                "reasoning": ct.reasoning,
            })

    candle.update({
        "rsi_signal":       ind_1m.get("rsi_signal", 0),
        "macd_signal":      ind_1m.get("macd_signal", 0),
        "bb_signal":        ind_1m.get("bb_signal", 0),
        "ema_signal":       ind_1m.get("ema_signal", 0),
        "adx":              ind_1m.get("adx", 20),
        "atr":              ind_1m.get("atr", candle["close"] * 0.01),
        "vol_ratio":        ind_1m.get("vol_ratio", 1.0),
        "rsi":              ind_1m.get("rsi", 50),
        "bb_pct":           ind_1m.get("bb_pct", 0.5),
        "ema50":            ind_1m.get("ema50", candle["close"]),
        "macd_hist":        ind_1m.get("macd_hist", 0),
        # Scalping indicators for RL env
        "stochrsi_signal":  ind_1m.get("stochrsi_signal", 0),
        "vwap_signal":      ind_1m.get("vwap_signal", 0),
        "supertrend_signal":ind_1m.get("supertrend_signal", 0),
        "roc_signal":       ind_1m.get("roc_signal", 0),
        "candle_signal":    ind_1m.get("candle_signal", 0),
        "structure_signal": ind_1m.get("structure_signal", 0),
        "accel_signal":     ind_1m.get("accel_signal", 0),
    })

    # Check exits first (pass ATR for trailing stop)
    closed_trade = eng.portfolio.check_exits(candle["high"], candle["low"], atr=ind_1m.get("atr", candle["close"] * 0.01))
    if closed_trade:
        eng.last_trade_time = time.time()
        await save_trade(closed_trade)
        from datetime import datetime, timezone
        entry_hour = datetime.fromtimestamp(closed_trade.entry_time, tz=timezone.utc).hour if closed_trade.entry_time else None
        perf_tracker.record_trade(closed_trade.pnl_usd, signal.confidence if hasattr(signal, 'confidence') else 0.5,
                                   closed_trade.strategy or "AI", entry_hour, closed_trade.strategy or "")
        # Learn from trade: which signals predicted correctly?
        if closed_trade.entry_signals:
            signal_learner.record_trade_outcome(closed_trade.entry_signals, closed_trade.pnl_usd, closed_trade.direction)
        trade_outcomes.record(closed_trade.pnl_usd, closed_trade.entry_regime,
                              session_info.session.value, closed_trade.strategy or "")
        _close_data = {
            "symbol":        symbol,
            "pnl_usd":       round(closed_trade.pnl_usd, 2),
            "pnl_pct":       round(closed_trade.pnl_pct * 100, 2),
            "exit_price":    closed_trade.exit_price,
            "exit_reason":   closed_trade.exit_reason,
            "direction":     closed_trade.direction,
            "entry_price":   closed_trade.entry_price,
            "reasoning":     closed_trade.reasoning,
            "entry_signals": closed_trade.entry_signals,
            "entry_regime":  closed_trade.entry_regime,
            "exit_regime":   closed_trade.exit_regime,
            "strategy":      closed_trade.strategy,
            "entry_time":    closed_trade.entry_time,
            "exit_time":     closed_trade.exit_time,
        }
        await broadcast("trade_close", _close_data)
        # Trade post-mortem journal + UltraThink teaching (non-blocking)
        asyncio.create_task(on_trade_close_hook(
            symbol, eng, ind_1m, regime_state, signal, rl_action, rl_confidence,
            _close_data, broadcast, signal_learner=signal_learner
        ))
        # Trigger UltraThink coaching session every N trades
        total = eng.portfolio.metrics().get("total_trades", 0)
        if total > 0 and total % COACHING_EVERY_N_TRADES == 0:
            asyncio.create_task(trigger_coaching_session(
                symbol, eng, ind_1m, regime_state, signal, rl_action, rl_confidence,
                broadcast, signal_learner=signal_learner, trade_outcomes=trade_outcomes
            ))
        # Immediately retrain on recent candles after each trade closes
        if eng.rl_agent.is_trained:
            asyncio.create_task(_post_trade_retrain_task(symbol))

    # Regime
    regime_state  = detect_regime(ind_1m, ind_5m)
    regime_params = get_regime_params(regime_state.regime)

    # ── Knowledge Base retrieval ──
    kb_context = None
    kb_trading_context = None
    if knowledge_retriever and knowledge_store and knowledge_store.total_chunks > 0:
        try:
            raw_results = await asyncio.to_thread(knowledge_retriever.retrieve, ind_1m, regime_state.regime.value)
            kb_context = await knowledge_store.enrich_results(raw_results)
            kb_trading_context = knowledge_retriever.extract_trading_context(kb_context)
        except Exception as e:
            logger.warning(f"[{symbol}] KB retrieval error: {e}")

    # Confluence gate
    signal = evaluate_signals(
        ind_1m, ind_5m, ind_15m,
        min_signals=regime_params["min_signals"],
        min_confidence=0.75,
        preferred_direction=regime_params["preferred_direction"],
        weight_overrides=signal_learner.get_weight_multipliers(),
        regime=regime_state.regime.value,
    )

    # ── KB confidence modifier ──
    if kb_trading_context and kb_trading_context.get("relevance_score", 0) > 0.3:
        modifier = kb_trading_context.get("confidence_modifier", 1.0)
        signal.confidence = min(signal.confidence * modifier, 1.0)
        if kb_trading_context.get("key_insights"):
            signal.reasoning += f" | KB: {kb_trading_context['key_insights'][0][:60]}"

    # RL agent
    rl_action, rl_confidence, rl_internals = 0, 0.0, {
        "action_probs": {"hold": 0.33, "long": 0.33, "short": 0.33},
        "value_estimate": 0.0, "entropy": 1.0,
    }
    if eng.rl_agent.is_trained and eng.portfolio.position is None:
        obs = _build_obs(ind_1m, candle, eng.portfolio, kb_trading_context)
        rl_action, rl_confidence, rl_internals = await asyncio.to_thread(eng.rl_agent.predict, obs)

    price = candle["close"]
    atr   = ind_1m.get("atr", price * 0.01)

    # ── Signal-based early exit ──────────────────────────────────────────
    # If we're in a position and signals flip hard against us → close early
    if eng.portfolio.position is not None and signal.direction != 0 and not closed_trade:
        pos_dir = eng.portfolio.position.direction
        if signal.direction == -pos_dir and signal.confidence >= 0.50:
            logger.info(f"[{symbol}] Signal flip exit: was {pos_dir}, now {signal.direction} (conf={signal.confidence:.2f})")
            closed_trade = eng.portfolio.close_position(price, "SIGNAL")
            if closed_trade:
                eng.last_trade_time = time.time()
                await save_trade(closed_trade)
                await broadcast("trade_close", {
                    "symbol":      symbol,
                    "pnl_usd":     round(closed_trade.pnl_usd, 2),
                    "pnl_pct":     round(closed_trade.pnl_pct * 100, 2),
                    "exit_price":  closed_trade.exit_price,
                    "exit_reason": "SIGNAL",
                    "direction":   closed_trade.direction,
                    "entry_price": closed_trade.entry_price,
                    "reasoning":   f"Signal flipped: {signal.reasoning}",
                })
                if eng.rl_agent.is_trained:
                    asyncio.create_task(_post_trade_retrain_task(symbol))

    # ── Momentum exhaustion exit ─────────────────────────────────────────
    if eng.portfolio.position is not None and not closed_trade:
        accel = ind_1m.get("accel_signal", 0)
        pos_dir = eng.portfolio.position.direction
        # Sharp momentum deceleration against our position
        if pos_dir == 1 and accel < -0.6:
            unrealized = eng.portfolio.unrealized_pnl(price)
            if unrealized > 0:  # only exit if we're in profit
                closed_trade = eng.portfolio.close_position(price, "EXHAUST")
                if closed_trade:
                    eng.last_trade_time = time.time()
                    await save_trade(closed_trade)
                    await broadcast("trade_close", {
                        "symbol": symbol, "pnl_usd": round(closed_trade.pnl_usd, 2),
                        "pnl_pct": round(closed_trade.pnl_pct * 100, 2),
                        "exit_price": closed_trade.exit_price, "exit_reason": "EXHAUST",
                        "direction": closed_trade.direction, "entry_price": closed_trade.entry_price,
                        "reasoning": f"Momentum exhaustion (accel={accel:.2f})",
                    })
        elif pos_dir == -1 and accel > 0.6:
            unrealized = eng.portfolio.unrealized_pnl(price)
            if unrealized > 0:
                closed_trade = eng.portfolio.close_position(price, "EXHAUST")
                if closed_trade:
                    eng.last_trade_time = time.time()
                    await save_trade(closed_trade)
                    await broadcast("trade_close", {
                        "symbol": symbol, "pnl_usd": round(closed_trade.pnl_usd, 2),
                        "pnl_pct": round(closed_trade.pnl_pct * 100, 2),
                        "exit_price": closed_trade.exit_price, "exit_reason": "EXHAUST",
                        "direction": closed_trade.direction, "entry_price": closed_trade.entry_price,
                        "reasoning": f"Momentum exhaustion (accel={accel:.2f})",
                    })

    should_trade    = signal.direction != 0 or (rl_confidence > 0.85 and rl_action != 0)
    final_direction = signal.direction if signal.direction != 0 else (
        1 if rl_action == 1 else (-1 if rl_action == 2 else 0)
    )

    # Guard: need warm-up history + adaptive cooldown per regime
    cooldown_s = regime_params.get("cooldown_s", TRADE_COOLDOWN_S)
    cooldown_ok = (time.time() - eng.last_trade_time) >= cooldown_s
    # ── Multi-timeframe confluence check ──
    mtf = check_mtf_confluence(ind_1m, ind_5m, ind_15m, final_direction) if final_direction != 0 else {"aligned": True, "score": 0}

    # ── Trade outcome filter (skip regimes/sessions that historically lose) ──
    outcome_check = trade_outcomes.should_trade(regime_state.regime.value, session_info.session.value)

    # ── News blackout check ──
    blackout = get_blackout_manager().check_blackout()
    blackout_ok = not blackout.active or blackout.zone == "CAUTION"
    blackout_size_mod = blackout.size_modifier if blackout.active else 1.0

    # ── R:R GATE: minimum 1.5:1 reward-to-risk ──────────────────────
    rr_ok = True
    if should_trade and final_direction != 0:
        sl_dist = atr * regime_params["stop_atr_mult"]
        tp_dist = atr * regime_params["tp_atr_mult"]
        rr_ratio = tp_dist / (sl_dist + 1e-9)
        if rr_ratio < 1.5:
            rr_ok = False
            logger.info(f"[{symbol}] Trade BLOCKED: R:R too low ({rr_ratio:.2f}:1, need 1.5:1)")

    if should_trade and final_direction != 0 and eng.portfolio.position is None \
            and eng.candle_count >= MIN_CANDLES and cooldown_ok \
            and mtf["aligned"] and outcome_check["ok"] and blackout_ok and rr_ok:

        reasoning = signal.reasoning
        if rl_confidence > 0.85 and signal.direction == 0:
            reasoning = f"RL OVERRIDE (conf={rl_confidence:.2f}) | {reasoning}"
        reasoning += f" | MTF:{mtf['score']:.0%}"
        if blackout.zone == "CAUTION":
            reasoning += f" | CAUTION: {blackout.event_name} in {blackout.minutes_until:.0f}m"

        # ── Adversarial filter (Claude argues against this trade) ──
        adv_size_mod = 1.0
        try:
            adv_ctx = build_advisor_context(symbol, eng, ind_1m, regime_state, signal, rl_action, rl_confidence)
            sl_price = price - (atr * regime_params["stop_atr_mult"]) * final_direction
            tp_price = price + (atr * regime_params["tp_atr_mult"]) * final_direction
            adv_result = await adversarial_check(
                adv_ctx,
                "LONG" if final_direction == 1 else "SHORT",
                price, sl_price, tp_price,
                get_blackout_manager()._status and [] or [],
            )
            if not adv_result.approved:
                logger.info(f"[{symbol}] Trade BLOCKED by adversarial filter: {adv_result.reasoning_summary}")
                await broadcast("claude_trade_explanation", {
                    "symbol": symbol, "action": "BLOCKED",
                    "explanation": f"Adversarial filter (risk {adv_result.risk_score}/10): {adv_result.counter_arguments}",
                    "timestamp": time.time(),
                })
                # Skip this trade entry entirely
                should_trade = False
            else:
                adv_size_mod = adv_result.size_modifier
                reasoning += f" | ADV:{adv_result.risk_score}/10"
        except Exception as e:
            logger.warning(f"Adversarial filter error (allowing trade): {e}")

    if should_trade and final_direction != 0 and eng.portfolio.position is None \
            and eng.candle_count >= MIN_CANDLES and cooldown_ok \
            and mtf["aligned"] and outcome_check["ok"] and blackout_ok and rr_ok:

        # Dynamic sizing: scale by confidence + anti-martingale + blackout + adversarial
        conf = signal.confidence if signal.direction != 0 else rl_confidence
        conf_scale = max(0.6, min(1.4, (conf - 0.5) * 2 + 0.6))
        streak = perf_tracker.current_streak
        am_mult = anti_martingale_size(1.0, max(0, streak), max(0, -streak))
        final_size_mult = regime_params["size_mult"] * conf_scale * am_mult * outcome_check["confidence_mult"] * blackout_size_mod * adv_size_mod

        pos = eng.portfolio.open_position(
            direction=final_direction,
            price=price,
            atr=atr,
            stop_atr_mult=regime_params["stop_atr_mult"],
            tp_atr_mult=regime_params["tp_atr_mult"],
            size_mult=final_size_mult,
            reasoning=reasoning,
        )
        if pos:
            # Store entry context for trade attribution
            pos.entry_signals = {k: round(v, 3) for k, v in signal.raw_signals.items()}
            pos.entry_regime = regime_state.regime.value
            pos.entry_rl_action = rl_action
            pos.entry_rl_confidence = round(rl_confidence, 3)
            pos.strategy = signal.strategy_winner or signal.strategy
            await broadcast("trade_open", {
                "symbol":      symbol,
                "direction":   final_direction,
                "entry_price": price,
                "stop_price":  round(pos.stop_price, 2),
                "tp_price":    round(pos.tp_price, 2),
                "size_usd":    round(pos.size_usd, 2),
                "reasoning":   reasoning,
                "confidence":  round(signal.confidence if signal.direction != 0 else rl_confidence, 3),
                "regime":      regime_state.regime.value,
                "strategy":    signal.strategy,
                "time":        candle["open_time"] // 1000,
            })
            # Claude trade explanation (non-blocking)
            asyncio.create_task(on_trade_hook(
                symbol, eng, ind_1m, regime_state, signal, rl_action, rl_confidence,
                "LONG" if final_direction == 1 else "SHORT",
                price, pos.stop_price, pos.tp_price, broadcast
            ))

    # ── HA-mode: entry logic ──
    if strategy_mode == "HA" and ha_result.signal.direction != 0 \
            and eng.portfolio.position is None \
            and eng.candle_count >= MIN_CANDLES \
            and (time.time() - eng.last_trade_time) >= 60:
        ha_sig = ha_result.signal
        # Get the indecision candle for stop placement
        indecision = ha_result.ha_candles[ha_sig.indecision_idx] if ha_sig.indecision_idx >= 0 else None
        pos = open_ha_position(
            eng.portfolio, ha_sig.direction, price, symbol,
            indecision_candle=indecision,
            reasoning=f"HA {ha_sig.pattern}: {ha_sig.reasoning}",
        )
        if pos:
            pos.entry_regime = regime_state.regime.value
            pos.strategy = f"HA_{ha_sig.pattern}"
            await broadcast("trade_open", {
                "symbol": symbol, "direction": ha_sig.direction,
                "entry_price": price,
                "stop_price": round(pos.stop_price, 2),
                "tp_price": round(pos.tp3_price, 2),
                "size_usd": round(pos.size_usd, 2),
                "reasoning": f"HA {ha_sig.pattern}: {ha_sig.reasoning}",
                "confidence": round(ha_sig.confidence, 3),
                "regime": regime_state.regime.value,
                "strategy": f"HA_{ha_sig.pattern}",
                "time": candle["open_time"] // 1000,
                "ha_tiers": {
                    "tp1": round(pos.tp1_price, 2),
                    "tp2": round(pos.tp2_price, 2),
                    "tp3": round(pos.tp3_price, 2),
                },
            })

    # ── AI Status for chart overlay ──
    _cooldown_remaining = max(0, cooldown_s - (time.time() - eng.last_trade_time))
    if eng.portfolio.position:
        _ai_state, _ai_reason = "in_position", f"In {'LONG' if eng.portfolio.position.direction == 1 else 'SHORT'}"
    elif eng.portfolio._halted:
        _ai_state, _ai_reason = "halted", "Trading halted (drawdown limit)"
    elif eng.candle_count < MIN_CANDLES:
        _ai_state, _ai_reason = "scanning", f"Warming up ({eng.candle_count}/{MIN_CANDLES} candles)"
    elif not cooldown_ok:
        _ai_state, _ai_reason = "cooldown", f"Cooldown ({int(_cooldown_remaining)}s left)"
    elif signal.direction != 0 and signal.confidence >= 0.65:
        _ai_state, _ai_reason = "signal_detected", f"{'LONG' if signal.direction==1 else 'SHORT'} signal ({signal.confidence:.0%})"
    elif signal.direction != 0:
        _ai_state, _ai_reason = "scanning", f"Low confidence ({signal.confidence:.0%})"
    else:
        _ai_state, _ai_reason = "scanning", "No signal alignment"

    await broadcast("candle_close", {
        "symbol": symbol,
        "candle": {
            "time":   candle["open_time"] // 1000,
            "open":   candle["open"],
            "high":   candle["high"],
            "low":    candle["low"],
            "close":  price,
            "volume": candle["volume"],
        },
        "indicators": {
            "rsi":         round(ind_1m.get("rsi", 50), 2),
            "rsi_signal":  round(ind_1m.get("rsi_signal", 0), 3),
            "macd_hist":   round(ind_1m.get("macd_hist", 0), 4),
            "macd_signal": round(ind_1m.get("macd_signal", 0), 3),
            "bb_pct":      round(ind_1m.get("bb_pct", 0.5), 3),
            "bb_signal":   round(ind_1m.get("bb_signal", 0), 3),
            "ema9":        round(ind_1m.get("ema9", price), 2),
            "ema21":       round(ind_1m.get("ema21", price), 2),
            "ema50":       round(ind_1m.get("ema50", price), 2),
            "ema_signal":  round(ind_1m.get("ema_signal", 0), 3),
            "adx":         round(ind_1m.get("adx", 20), 1),
            "atr":         round(atr, 2),
            "vol_ratio":   round(ind_1m.get("vol_ratio", 1), 2),
            # Scalping indicators
            "stochrsi_k":       round(ind_1m.get("stochrsi_k", 50), 1),
            "stochrsi_signal":  round(ind_1m.get("stochrsi_signal", 0), 3),
            "vwap":             round(ind_1m.get("vwap", price), 2),
            "vwap_signal":      round(ind_1m.get("vwap_signal", 0), 3),
            "supertrend_dir":   ind_1m.get("supertrend_dir", 0),
            "supertrend_signal":round(ind_1m.get("supertrend_signal", 0), 3),
            "momentum":         round(ind_1m.get("roc", 0), 4),
            "candle_signal":    round(ind_1m.get("candle_signal", 0), 3),
            "structure_signal": round(ind_1m.get("structure_signal", 0), 3),
            "structure_label":  ind_1m.get("structure_label", ""),
            # Drawing levels
            "bb_upper":         round(ind_1m.get("bb_upper", price * 1.01), 2),
            "bb_lower":         round(ind_1m.get("bb_lower", price * 0.99), 2),
            "bb_mid":           round(ind_1m.get("bb_mid", price), 2),
        },
        "regime":            regime_state.regime.value,
        "regime_confidence": round(regime_state.confidence, 2),
        "signal": {
            "direction":    signal.direction,
            "confidence":   round(signal.confidence, 3),
            "signal_count": signal.signal_count,
            "reasoning":    signal.reasoning,
            "raw":          {k: round(v, 3) for k, v in signal.raw_signals.items()},
        },
        "rl": {
            "action":         rl_action,
            "confidence":     round(rl_confidence, 3),
            "trained":        eng.rl_agent.is_trained,
            "training_steps": eng.rl_agent.training_steps,
        },
        # ── Enriched AI data for dashboard visualizations ──
        "signal_confluence": {
            "bull_score":           signal.bull_score,
            "bear_score":           signal.bear_score,
            "bull_count":           signal.bull_count,
            "bear_count":           signal.bear_count,
            "weighted_signals":     signal.weighted_signals,
            "strategy_winner":      signal.strategy_winner,
            "trend_confidence":     signal.trend_confidence,
            "reversion_confidence": signal.reversion_confidence,
        },
        "regime_features": {
            "scores":     regime_state.scores,
            "volatility": regime_state.volatility,
            "bb_width":   round(regime_state.bb_width, 4),
        },
        "rl_internals": rl_internals,
        # ── Wave 2: AI intelligence data ──
        "ai_thoughts": _build_ai_thoughts(signal, rl_action, rl_confidence, rl_internals,
                                           regime_state, eng, ind_1m, kb_trading_context),
        "entry_forecast": _compute_entry_forecast(eng, final_direction, regime_state.regime.value)
                          if signal.direction != 0 else None,
        "cross_symbol": _compute_correlation(),
        # ── Heiken Ashi strategy data (always sent for frontend) ──
        "ha_strategy": {
            "ha_candles": [
                {"time": c["time"], "open": c["ha_open"], "high": c["ha_high"],
                 "low": c["ha_low"], "close": c["ha_close"]}
                for c in ha_result.ha_candles
            ],
            "classifications": ha_result.classifications,
            "current_type": ha_result.current_type,
            "trend_color": ha_result.trend_color,
            "consecutive_count": ha_result.consecutive_count,
            "signal": {
                "direction": ha_result.signal.direction,
                "pattern": ha_result.signal.pattern,
                "confidence": round(ha_result.signal.confidence, 3),
                "reasoning": ha_result.signal.reasoning,
                "trend_length": ha_result.signal.trend_length,
            },
            "tp_status": {
                "stage": eng.portfolio.position.tp_stage if eng.portfolio.position else 0,
                "tp1_hit": (eng.portfolio.position.tp_stage >= 1) if eng.portfolio.position else False,
                "tp2_hit": (eng.portfolio.position.tp_stage >= 2) if eng.portfolio.position else False,
            },
        },
        "strategy_mode": strategy_mode,
        "trade_mode": trade_mode,
        "mt5_connected": mt5_connector.is_connected if mt5_connector else False,
        # ── AI Status for chart overlay ──
        "ai_status": {
            "state": _ai_state,
            "reason": _ai_reason,
            "cooldown_remaining": round(_cooldown_remaining, 0),
            "candle_warmup": {"current": eng.candle_count, "required": MIN_CANDLES},
            "signal_strength": round(signal.confidence, 3),
            "bull_score": signal.bull_score,
            "bear_score": signal.bear_score,
            "mtf_aligned": mtf.get("aligned", True) if 'mtf' in dir() else True,
            "mtf_score": mtf.get("score", 0) if 'mtf' in dir() else 0,
            "mtf_details": mtf.get("details", "") if 'mtf' in dir() else "",
        },
        # ── Session awareness ──
        "session": {
            "name": session_info.session.value,
            "label": session_info.label,
            "hours_remaining": session_info.hours_remaining,
            "volatility_mult": session_info.volatility_mult,
            "description": session_info.description,
        },
        # ── Performance analytics ──
        "analytics": perf_tracker.get_full_analytics(),
        # ── Knowledge Base context ──
        "kb_context": {
            "relevance_score": round(kb_trading_context["relevance_score"], 3),
            "risk_level": round(kb_trading_context["risk_level"], 3),
            "sentiment_bias": round(kb_trading_context["sentiment_bias"], 3),
            "key_insights": kb_trading_context.get("key_insights", []),
            "pattern_matches": kb_trading_context.get("pattern_matches", []),
            "confidence_modifier": round(kb_trading_context["confidence_modifier"], 3),
            "source_pdfs": [f"{c.get('source_pdf','')}:p{c.get('page_num','')}" for c in (kb_context or [])[:3]],
        } if kb_trading_context else None,
    })

    # ── Claude advisory hook (non-blocking) ──
    asyncio.create_task(on_candle_hook(
        symbol, eng, ind_1m, regime_state, signal,
        rl_action, rl_confidence, broadcast
    ))

    # Update prev regime for transition detection
    eng.prev_regime = regime_state.regime.value

    metrics = eng.portfolio.metrics()
    await save_snapshot(metrics)
    await broadcast("portfolio_update", {"symbol": symbol, **metrics})

    if eng.candle_count >= RETRAIN_INTERVAL and eng.rl_agent.is_trained:
        eng.candle_count = 0
        asyncio.create_task(_retrain_task(symbol))


def _annotate_candles(candles: list[dict]) -> list[dict]:
    """Compute indicators on raw candles so RL env gets real signal features."""
    from data.candle_store import CandleStore
    store = CandleStore(maxlen=500)
    annotated = []
    for c in candles:
        store.add(c)
        df = store.get_df("1m")
        ind = compute_indicators(df) if df is not None else {}
        c.update({
            "rsi_signal": ind.get("rsi_signal", 0), "macd_signal": ind.get("macd_signal", 0),
            "bb_signal": ind.get("bb_signal", 0), "ema_signal": ind.get("ema_signal", 0),
            "adx": ind.get("adx", 20), "atr": ind.get("atr", c["close"] * 0.01),
            "vol_ratio": ind.get("vol_ratio", 1.0), "rsi": ind.get("rsi", 50),
            "bb_pct": ind.get("bb_pct", 0.5), "ema50": ind.get("ema50", c["close"]),
            "macd_hist": ind.get("macd_hist", 0),
            "stochrsi_signal": ind.get("stochrsi_signal", 0),
            "vwap_signal": ind.get("vwap_signal", 0),
            "supertrend_signal": ind.get("supertrend_signal", 0),
            "roc_signal": ind.get("roc_signal", 0),
            "candle_signal": ind.get("candle_signal", 0),
            "structure_signal": ind.get("structure_signal", 0),
            "accel_signal": ind.get("accel_signal", 0),
        })
        annotated.append(c)
    return annotated


def _build_obs(ind_1m: dict, candle: dict, portfolio: PaperPortfolio, kb_context: dict = None) -> np.ndarray:
    obs = np.zeros(TradingEnv.OBS_SIZE, dtype=np.float32)
    price = candle["close"]

    # Core indicator signals [0-7]
    obs[0] = np.clip(ind_1m.get("rsi_signal", 0), -1, 1)
    obs[1] = np.clip(ind_1m.get("macd_signal", 0), -1, 1)
    obs[2] = np.clip(ind_1m.get("bb_signal", 0), -1, 1)
    obs[3] = np.clip(ind_1m.get("ema_signal", 0), -1, 1)
    obs[4] = np.clip(ind_1m.get("vol_ratio", 1) - 1, -2, 2)
    obs[5] = np.clip(ind_1m.get("adx", 20) / 50 - 1, -1, 1)
    obs[6] = np.clip(ind_1m.get("atr", price * 0.01) / price * 100, 0, 3)
    obs[7] = np.clip(ind_1m.get("rsi", 50) / 50 - 1, -1, 1)

    # Scalping signals [8-14]
    obs[8]  = np.clip(ind_1m.get("stochrsi_signal", 0), -1, 1)
    obs[9]  = np.clip(ind_1m.get("vwap_signal", 0), -1, 1)
    obs[10] = np.clip(ind_1m.get("supertrend_signal", 0), -1, 1)
    obs[11] = np.clip(ind_1m.get("roc_signal", 0), -1, 1)
    obs[12] = np.clip(ind_1m.get("candle_signal", 0), -1, 1)
    obs[13] = np.clip(ind_1m.get("structure_signal", 0), -1, 1)
    obs[14] = np.clip(ind_1m.get("accel_signal", 0), -1, 1)

    # Portfolio state [20-23]
    m = portfolio.metrics()
    obs[20] = portfolio.position.direction if portfolio.position else 0
    obs[21] = np.clip(m["return_pct"] / 100, -1, 1)
    obs[22] = m["win_rate"] / 100
    obs[23] = np.clip(m["total_pnl"] / 1000, -1, 1)

    # Knowledge base features [26-29]
    if kb_context:
        obs[26] = np.clip(kb_context.get("relevance_score", 0), 0, 1)
        obs[27] = np.clip(kb_context.get("risk_level", 0), 0, 1)
        obs[28] = np.clip(kb_context.get("sentiment_bias", 0), -1, 1)
        obs[29] = np.clip((kb_context.get("confidence_modifier", 1.0) - 0.8) / 0.4, 0, 1)

    return obs


async def _retrain_task(symbol: str):
    eng = engines[symbol]
    logger.info(f"[{symbol}] Live retrain starting...")
    candles_list = list(eng.store._candles_1m)
    env = TradingEnv(candles_list)
    await asyncio.to_thread(eng.rl_agent.retrain, env, 2048)
    logger.info(f"[{symbol}] Live retrain done. Total steps: {eng.rl_agent.training_steps:,}")
    await broadcast("training_update", {
        "symbol":         symbol,
        "training_steps": eng.rl_agent.training_steps,
        "trained":        True,
        "training_history": eng.rl_agent.training_history[-50:],
    })


async def _auto_pretrain_task():
    """Auto-pretrain both models at startup if no saved model exists."""
    import random
    await asyncio.sleep(5)  # let feeds connect first

    for sym in ["XAUUSD", "USTEC"]:
        eng = engines[sym]
        if eng.rl_agent.is_trained:
            logger.info(f"[{sym}] Pre-trained model already loaded, skipping auto-pretrain.")
            continue

        # Wait until we have enough candles
        for _ in range(30):
            if eng.store.count() >= 100:
                break
            await asyncio.sleep(2)

        candles = list(eng.store._candles_1m)
        if len(candles) < 100:
            logger.warning(f"[{sym}] Not enough candles for pretrain ({len(candles)}), skipping.")
            continue

        # Annotate candles with indicators (critical — RL env reads these fields)
        logger.info(f"[{sym}] Computing indicators on {len(candles)} candles for pretrain...")
        candles = await asyncio.to_thread(_annotate_candles, candles)

        logger.info(f"[{sym}] Auto-pretrain starting on {len(candles)} candles ({AUTO_PRETRAIN_STEPS:,} steps, 8 parallel envs)...")
        # Use vectorized pretrain for 8x faster learning
        dummy_env = TradingEnv(candles)
        eng.rl_agent.load_or_create(dummy_env)
        await asyncio.to_thread(eng.rl_agent.pretrain, candles, AUTO_PRETRAIN_STEPS)
        logger.info(f"[{sym}] Auto-pretrain done. Steps trained: {eng.rl_agent.training_steps:,}")
        await broadcast("training_update", {
            "symbol": sym,
            "training_steps": eng.rl_agent.training_steps,
            "trained": True,
        })


async def _continuous_train_loop():
    """
    24/7 continuous training — runs forever.
    Every CONTINUOUS_TRAIN_INTERVAL_S seconds:
      1. Combine ALL candles from BOTH symbols (cross-symbol learning — normalized signals are scale-invariant)
      2. Train each model on the combined dataset
      3. Broadcast training stats to dashboard
    """
    import random

    while True:
        await asyncio.sleep(CONTINUOUS_TRAIN_INTERVAL_S)

        btc_candles = [c for c in engines["USTEC"].store._candles_1m if c.get("atr")]
        xau_candles = [c for c in engines["XAUUSD"].store._candles_1m  if c.get("atr")]
        combined    = btc_candles + xau_candles

        if len(combined) < 150:
            continue

        random.shuffle(combined)  # mix BTC and XAU patterns for richer training

        for sym, eng in engines.items():
            if not eng.rl_agent.is_trained:
                continue
            try:
                # Vectorized continuous training (4 parallel envs)
                await asyncio.to_thread(eng.rl_agent.retrain_vectorized, combined, CONTINUOUS_TRAIN_STEPS)
                logger.info(f"[{sym}] Continuous train done. Total steps: {eng.rl_agent.training_steps:,}")
                await broadcast("training_update", {
                    "symbol":         sym,
                    "training_steps": eng.rl_agent.training_steps,
                    "trained":        True,
                })
            except Exception as e:
                logger.warning(f"[{sym}] Continuous train error: {e}")


async def _post_trade_retrain_task(symbol: str):
    """Mini-retrain right after a trade closes — learns from the most recent experience."""
    eng = engines[symbol]
    candles = list(eng.store._candles_1m)
    if len(candles) < 100:
        return
    env = TradingEnv(candles)
    await asyncio.to_thread(eng.rl_agent.retrain, env, POST_TRADE_TRAIN_STEPS)
    await broadcast("training_update", {
        "symbol":         symbol,
        "training_steps": eng.rl_agent.training_steps,
        "trained":        True,
        "training_history": eng.rl_agent.training_history[-50:],
    })


# ─── App lifespan ────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global knowledge_store, knowledge_embedder, knowledge_retriever, ingestion_manager
    await init_db()

    # ── Knowledge Base initialization ──
    kb_index_dir = os.path.join(os.path.dirname(__file__), "knowledge", "data")
    os.makedirs(kb_index_dir, exist_ok=True)
    knowledge_store = KnowledgeStore(kb_index_dir, DB_PATH)
    knowledge_embedder = KnowledgeEmbedder()
    knowledge_retriever = TradingKnowledgeRetriever(knowledge_store, knowledge_embedder)
    ingestion_manager = IngestionManager(knowledge_store, knowledge_embedder, broadcast_fn=broadcast)
    await knowledge_store.init_tables()
    kb_index_path = os.path.join(kb_index_dir, "faiss.index")
    if os.path.exists(kb_index_path):
        await asyncio.to_thread(knowledge_store.load)
        logger.info(f"Knowledge base loaded: {knowledge_store.get_stats()}")
    else:
        logger.info("Knowledge base empty — POST /knowledge/ingest to load PDFs")

    # ── MT5 connection (required for both XAUUSD and USTEC) ──
    global mt5_connector
    if MT5_AVAILABLE:
        mt5_connector = MT5Connector(MT5Config(
            max_risk_pct=1.0,        # 1% risk per trade
            max_daily_loss_pct=5.0,  # 5% daily loss limit → halt
            max_positions=2,          # 1 per symbol (XAUUSD + USTEC)
            lot_size=0.01,           # Micro lot default
        ))
        mt5_ok = await asyncio.to_thread(mt5_connector.connect)
    else:
        mt5_ok = False

    feeds = []

    # ── XAU feed: MT5 > Twelve Data > YFinance ──
    twelve_key = os.getenv("TWELVE_API_KEY", "")
    if mt5_ok:
        xau_feed = MT5DataFeed(
            mt5_connector, "XAUUSD",
            on_tick=lambda t: on_tick(t, "XAUUSD"),
            on_candle_close=lambda c: on_candle_close(c, "XAUUSD"),
        )
        logger.info("XAU feed: MT5 broker data (real-time)")
    elif twelve_key:
        xau_feed = TwelveDataFeed(
            on_tick=lambda t: on_tick(t, "XAUUSD"),
            on_candle_close=lambda c: on_candle_close(c, "XAUUSD"),
        )
        logger.info("XAU feed: Twelve Data XAUUSD spot")
    else:
        xau_feed = YFinanceFeed(
            on_tick=lambda t: on_tick(t, "XAUUSD"),
            on_candle_close=lambda c: on_candle_close(c, "XAUUSD"),
            symbol="XAUUSD",
        )
        logger.info("XAU feed: YFinance GC=F (futures fallback)")
    feeds.append(xau_feed)

    # ── USTEC (NAS100) feed: MT5 only ──
    ustec_feed = None
    if mt5_ok and mt5_connector._symbol_map.get("USTEC"):
        ustec_feed = MT5DataFeed(
            mt5_connector, "USTEC",
            on_tick=lambda t: on_tick(t, "USTEC"),
            on_candle_close=lambda c: on_candle_close(c, "USTEC"),
        )
        logger.info("USTEC feed: MT5 broker data (real-time)")
        feeds.append(ustec_feed)
    elif mt5_ok:
        logger.warning("USTEC symbol not found on MT5 — check your broker's symbol name")
    else:
        logger.warning("MT5 not connected — USTEC feed unavailable")

    # ── Pre-load XAU history ──
    if mt5_ok:
        logger.info("Loading XAU history from MT5 broker...")
        xau_history = await xau_feed.fetch_history(count=5000)
        xau_source = "MT5 broker (real-time)"
    elif twelve_key:
        logger.info("Loading XAU history from Twelve Data REST...")
        xau_history = await xau_feed.fetch_history(count=4998)
        xau_source = "Twelve Data"
    else:
        logger.info("Loading XAU history from YFinance...")
        try:
            xau_history = await asyncio.wait_for(xau_feed.fetch_history(count=5000), timeout=60)
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"XAU history fetch failed ({e})")
            xau_history = []
        xau_source = "YFinance"

    for c in xau_history:
        engines["XAUUSD"].store.add(c)
    engines["XAUUSD"].candle_count = len(xau_history)
    logger.info(f"Loaded {len(xau_history)} XAU 1m candles")

    # ── Pre-load USTEC history ──
    if ustec_feed:
        logger.info("Loading USTEC history from MT5 broker...")
        try:
            ustec_history = await ustec_feed.fetch_history(count=5000)
        except Exception as e:
            logger.warning(f"USTEC history fetch failed: {e}")
            ustec_history = []
        for c in ustec_history:
            engines["USTEC"].store.add(c)
        engines["USTEC"].candle_count = len(ustec_history)
        logger.info(f"Loaded {len(ustec_history)} USTEC 1m candles")

    # ── Claude advisory layer ──
    await claude_startup(broadcast)

    tasks = [asyncio.create_task(f.start()) for f in feeds]
    tasks.extend([
        asyncio.create_task(_auto_pretrain_task()),
        asyncio.create_task(_continuous_train_loop()),
    ])
    logger.info(f"Backend started — XAU ({xau_source}) + USTEC (MT5) + Claude advisory")
    yield
    await claude_shutdown()
    for t in tasks:
        t.cancel()
    await close_db()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(claude_router)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    logger.info(f"Dashboard connected ({len(clients)} clients)")

    # Send current state for both symbols on connect
    for sym, eng in engines.items():
        # Portfolio metrics
        m = eng.portfolio.metrics()
        await ws.send_text(json.dumps({"type": "portfolio_update", "symbol": sym, **m}))

        # RL training state — so dashboard doesn't show "Training..." when model is loaded
        await ws.send_text(json.dumps({
            "type":           "training_update",
            "symbol":         sym,
            "training_steps": eng.rl_agent.training_steps,
            "trained":        eng.rl_agent.is_trained,
        }))

        # Historical candles for the chart
        chart_candles = eng.store.get_chart_candles()
        if chart_candles:
            await ws.send_text(json.dumps({
                "type":    "candle_history",
                "symbol":  sym,
                "candles": chart_candles,
            }))
            logger.info(f"Sent {len(chart_candles)} {sym} candles to new client")

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        clients.discard(ws)


@app.get("/")
async def root():
    return {"status": "ok", "dashboard": "http://localhost:5173"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "symbols": {
            sym: {
                "candles":    eng.store.count(),
                "rl_trained": eng.rl_agent.is_trained,
                "halted":     eng.portfolio._halted,
            }
            for sym, eng in engines.items()
        }
    }


@app.get("/status")
async def get_status():
    """Return full system status for autopilot monitoring."""
    return {
        "engines": {
            sym: {
                "equity": eng.portfolio.equity,
                "total_trades": len(eng.portfolio.trades),
                "win_rate": eng.portfolio.metrics().get("win_rate", 0),
                "drawdown_pct": eng.portfolio.metrics().get("drawdown_pct", 0),
                "training_steps": eng.rl_agent.training_steps,
                "candle_count": eng.candle_count,
                "halted": eng.portfolio._halted,
                "position": bool(eng.portfolio.position),
            }
            for sym, eng in engines.items()
        },
        "strategy_mode": strategy_mode,
        "kb_loaded": knowledge_store is not None and knowledge_store.total_chunks > 0,
        "kb_chunks": knowledge_store.total_chunks if knowledge_store else 0,
    }

@app.get("/analytics")
async def get_analytics():
    """Return full performance analytics."""
    return perf_tracker.get_full_analytics()

@app.get("/session")
async def get_session():
    """Return current trading session info."""
    s = detect_session()
    return {"session": s.session.value, "label": s.label,
            "hours_remaining": s.hours_remaining, "description": s.description}

@app.get("/strategy")
async def get_strategy():
    return {"mode": strategy_mode}


@app.post("/strategy/{mode}")
async def set_strategy(mode: str):
    global strategy_mode
    mode = mode.upper()
    if mode not in ("AI", "HA"):
        from fastapi import HTTPException
        raise HTTPException(400, "Mode must be AI or HA")
    strategy_mode = mode
    await broadcast("strategy_mode", {"mode": strategy_mode})
    logger.info(f"Strategy mode changed to: {strategy_mode}")
    return {"ok": True, "mode": strategy_mode}


@app.get("/trade-mode")
async def get_trade_mode():
    return {
        "mode": trade_mode,
        "mt5_connected": mt5_connector.is_connected if mt5_connector else False,
        "mt5_account": mt5_connector.get_account_info() if mt5_connector and mt5_connector.is_connected else None,
    }

@app.post("/trade-mode/{mode}")
async def set_trade_mode(mode: str):
    global trade_mode
    mode = mode.upper()
    if mode not in ("PAPER", "LIVE", "SIGNAL"):
        from fastapi import HTTPException
        raise HTTPException(400, "Mode must be PAPER, LIVE, or SIGNAL")
    if mode == "LIVE" and (not mt5_connector or not mt5_connector.is_connected):
        from fastapi import HTTPException
        raise HTTPException(400, "MT5 not connected — cannot switch to LIVE mode")
    trade_mode = mode
    await broadcast("trade_mode", {"mode": trade_mode, "symbol": "XAUUSD"})
    logger.info(f"Trade mode changed to: {trade_mode}")
    return {"ok": True, "mode": trade_mode}

@app.post("/trade/{symbol}/long")
async def manual_long(symbol: str):
    from fastapi import HTTPException
    eng = engines.get(symbol)
    if not eng:
        raise HTTPException(status_code=404, detail="Unknown symbol")
    price = eng.store.last_close()
    if not price:
        raise HTTPException(status_code=400, detail="No price data yet")
    if eng.portfolio.position is not None:
        raise HTTPException(status_code=400, detail="Already in a position")
    atr = eng.last_ind.get("atr", price * 0.01)
    pos = eng.portfolio.open_position(
        direction=1, price=price, atr=atr, reasoning="MANUAL LONG"
    )
    if pos:
        await broadcast("trade_open", {
            "symbol":      symbol,
            "direction":   1,
            "entry_price": price,
            "stop_price":  round(pos.stop_price, 2),
            "tp_price":    round(pos.tp_price, 2),
            "size_usd":    round(pos.size_usd, 2),
            "reasoning":   "MANUAL LONG",
            "confidence":  1.0,
            "regime":      "MANUAL",
            "time":        int(time.time()),
        })
    return {"ok": True}


@app.post("/trade/{symbol}/short")
async def manual_short(symbol: str):
    from fastapi import HTTPException
    eng = engines.get(symbol)
    if not eng:
        raise HTTPException(status_code=404, detail="Unknown symbol")
    price = eng.store.last_close()
    if not price:
        raise HTTPException(status_code=400, detail="No price data yet")
    if eng.portfolio.position is not None:
        raise HTTPException(status_code=400, detail="Already in a position")
    atr = eng.last_ind.get("atr", price * 0.01)
    pos = eng.portfolio.open_position(
        direction=-1, price=price, atr=atr, reasoning="MANUAL SHORT"
    )
    if pos:
        await broadcast("trade_open", {
            "symbol":      symbol,
            "direction":   -1,
            "entry_price": price,
            "stop_price":  round(pos.stop_price, 2),
            "tp_price":    round(pos.tp_price, 2),
            "size_usd":    round(pos.size_usd, 2),
            "reasoning":   "MANUAL SHORT",
            "confidence":  1.0,
            "regime":      "MANUAL",
            "time":        int(time.time()),
        })
    return {"ok": True}


@app.post("/trade/{symbol}/close")
async def manual_close(symbol: str):
    from fastapi import HTTPException
    eng = engines.get(symbol)
    if not eng:
        raise HTTPException(status_code=404, detail="Unknown symbol")
    price = eng.store.last_close()
    if not price or eng.portfolio.position is None:
        raise HTTPException(status_code=400, detail="No open position")
    closed = eng.portfolio.close_position(price, "MANUAL")
    if closed:
        eng.last_trade_time = time.time()
        await save_trade(closed)
        await broadcast("trade_close", {
            "symbol":      symbol,
            "pnl_usd":     round(closed.pnl_usd, 2),
            "pnl_pct":     round(closed.pnl_pct * 100, 2),
            "exit_price":  closed.exit_price,
            "exit_reason": "MANUAL",
            "direction":   closed.direction,
            "entry_price": closed.entry_price,
            "reasoning":   closed.reasoning,
        })
    return {"ok": True}


# ── Backtesting + Optimization endpoints ─────────────────────────────────────

@app.post("/backtest")
async def api_backtest(
    stop_atr: float = 0.7, tp_atr: float = 1.8,
    min_signals: int = 3, min_confidence: float = 0.65,
    symbol: str = "XAUUSD",
):
    """Run backtest on cached candle data."""
    from ai.backtester import run_backtest, BacktestConfig
    from data.history_cache import load_from_cache, save_to_cache

    # Get candles from store or cache
    eng = engines.get(symbol)
    candles = list(eng.store._candles_1m) if eng else []
    if not candles:
        candles = load_from_cache(symbol)
    if len(candles) < 100:
        from fastapi import HTTPException
        raise HTTPException(400, f"Not enough candles for backtest ({len(candles)})")

    # Save to cache for future use
    save_to_cache(symbol, candles)

    config = BacktestConfig(
        stop_atr_mult=stop_atr, tp_atr_mult=tp_atr,
        min_signals=min_signals, min_confidence=min_confidence,
    )
    result = await asyncio.to_thread(run_backtest, candles, config)
    return result.to_dict()

@app.post("/optimize")
async def api_optimize(symbol: str = "XAUUSD", max_combos: int = 30):
    """Run parameter optimization via grid search backtesting."""
    from ai.optimizer import optimize_parameters
    from data.history_cache import load_from_cache, save_to_cache

    eng = engines.get(symbol)
    candles = list(eng.store._candles_1m) if eng else []
    if not candles:
        candles = load_from_cache(symbol)
    if len(candles) < 200:
        from fastapi import HTTPException
        raise HTTPException(400, f"Not enough candles ({len(candles)})")

    save_to_cache(symbol, candles)
    result = await asyncio.to_thread(optimize_parameters, candles, max_combos)
    return result

@app.post("/backtest/learn")
async def api_backtest_learn(
    symbol: str = "XAUUSD",
    candle_count: int = 5000,
    teach_every_n: int = 1,
    max_teach: int = 200,
):
    """
    Run backtest → feed every trade through UltraThink → store lessons.
    This rapidly trains the bot's memory on historical data.
    """
    from ai.ultra_backtester import ultra_learn_from_backtest, UltraBacktestConfig
    from data.history_cache import load_from_cache, save_to_cache

    candles = []
    if mt5_connector and mt5_connector.is_connected:
        candles = mt5_connector.get_history(symbol, count=candle_count)
    if not candles:
        eng = engines.get(symbol)
        if eng:
            candles = list(eng.store._candles_1m)
    if not candles:
        candles = load_from_cache(symbol)
    if len(candles) < 200:
        from fastapi import HTTPException
        raise HTTPException(400, f"Not enough candles ({len(candles)}). Need 200+, got {len(candles)}")

    save_to_cache(symbol, candles)

    config = UltraBacktestConfig(
        teach_every_n=teach_every_n,
        max_teach_trades=max_teach,
        use_regime=True,
        use_trailing_stop=True,
    )
    result = await ultra_learn_from_backtest(candles, config, symbol=symbol)
    return result


@app.get("/cache/info")
async def cache_info():
    """Get info about cached historical data."""
    from data.history_cache import get_cache_info
    return {
        sym: get_cache_info(sym)
        for sym in ["XAUUSD", "USTEC"]
    }


# ── Knowledge Base endpoints ─────────────────────────────────────────────────

@app.post("/knowledge/ingest")
async def kb_ingest(directory: str = None, force: bool = False, ocr: bool = True):
    """
    Start PDF ingestion. Two-phase: text PDFs first (fast), then OCR scanned PDFs (slow).
    - force=true: clear existing KB and re-ingest everything
    - ocr=false: skip scanned PDFs, only process text-based ones (fast, ~1 min)
    - ocr=true (default): also OCR Arabic/scanned PDFs (slow, ~30-60 min)
    """
    if ingestion_manager is None:
        from fastapi import HTTPException
        raise HTTPException(500, "Knowledge base not initialized")
    default_path = r"C:\Users\xmrx\Downloads\Telegram Desktop\pdf"
    result = await ingestion_manager.ingest_directory(directory or default_path, force=force, ocr=ocr)
    return result


@app.get("/knowledge/status")
async def kb_status():
    """Return KB stats."""
    if knowledge_store is None:
        return {"loaded": False, "total_pdfs": 0, "total_chunks": 0}
    stats = knowledge_store.get_stats()
    stats["loaded"] = True
    if ingestion_manager:
        stats["ingesting"] = ingestion_manager._ingesting
        stats["progress"] = ingestion_manager._progress
    return stats


@app.get("/knowledge/search")
async def kb_search(query: str, top_k: int = 5):
    """Manual semantic search against KB."""
    if knowledge_retriever is None or knowledge_store.total_chunks == 0:
        from fastapi import HTTPException
        raise HTTPException(400, "Knowledge base is empty — POST /knowledge/ingest first")
    raw = await asyncio.to_thread(
        lambda: knowledge_store.search(knowledge_embedder.embed_query(query), top_k)
    )
    results = await knowledge_store.enrich_results(raw)
    return {"results": results, "query": query}


@app.post("/knowledge/ingest-file")
async def kb_ingest_file(path: str):
    """Ingest a single PDF file."""
    if ingestion_manager is None:
        from fastapi import HTTPException
        raise HTTPException(500, "Knowledge base not initialized")
    return await ingestion_manager.ingest_single(path)
