"""
UltraThink Rapid Learning Backtester — XAUUSD focused.

Runs the standard backtester with regime detection + multi-timeframe indicators,
then feeds every trade through UltraThink's ultra_teach() to rapidly populate
UltraMemory with hundreds of lessons in minutes instead of days of live trading.

Architecture:
  1. Fetch 5000+ historical 1m candles from MT5 (or cache)
  2. Resample 1m → 5m and 15m for multi-timeframe indicators
  3. Pre-compute indicators for all three timeframes
  4. Run candle-by-candle backtest with regime detection
  5. Feed each trade through ultra_teach() asynchronously
  6. Aggregate lessons and run a final ultra_coach() session
"""
import asyncio
import logging
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from ai.backtester import BacktestConfig, BacktestTrade, BacktestResult, _precompute_indicators
from ai.regime_detector import detect_regime, get_regime_params
from ai.signal_engine import evaluate_signals

logger = logging.getLogger(__name__)


@dataclass
class UltraBacktestConfig(BacktestConfig):
    """Extended config for UltraThink learning backtest."""
    teach_every_n: int = 1          # Feed every Nth trade to UltraThink (1 = all)
    max_teach_trades: int = 200     # Cap UltraThink calls to control cost
    teach_concurrency: int = 5      # Parallel UltraThink calls
    use_regime: bool = True         # Use regime-specific params
    use_trailing_stop: bool = True  # Trailing stop after profit threshold
    trail_trigger_atr: float = 0.4  # Start trailing after this many ATR in profit
    trail_distance_atr: float = 0.2 # Trail distance in ATR


def _resample_candles(candles_1m: list[dict], target_minutes: int) -> list[dict]:
    """
    Resample 1m candles to 5m or 15m using pandas.
    Returns list of dicts with OHLCV data, aligned to 1m index for lookup.
    """
    if not candles_1m or target_minutes <= 1:
        return candles_1m

    df = pd.DataFrame(candles_1m)
    if "time" not in df.columns and "open_time" in df.columns:
        df["time"] = df["open_time"].astype(int) // 1000
    df.index = pd.to_datetime(df["time"], unit="s", utc=True)

    # Remove duplicates
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]

    rule = f"{target_minutes}min"
    resampled = df.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
    }).dropna()

    # Convert back to list of dicts
    result = []
    for ts, row in resampled.iterrows():
        result.append({
            "time": int(ts.timestamp()),
            "open_time": int(ts.timestamp()) * 1000,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "tick_volume": int(row["tick_volume"]),
        })
    return result


def _build_tf_lookup(candles_1m: list[dict], indicators_tf: list[dict],
                     tf_candles: list[dict]) -> list[dict]:
    """
    Build a lookup so for each 1m candle index, we can find the latest
    completed higher-TF indicator dict.
    Returns list same length as candles_1m.
    """
    if not tf_candles or not indicators_tf:
        return [{}] * len(candles_1m)

    # Build time → indicator mapping for the higher TF
    tf_time_to_ind = {}
    for i, c in enumerate(tf_candles):
        if i < len(indicators_tf) and indicators_tf[i]:
            tf_time_to_ind[c.get("time", 0)] = indicators_tf[i]

    # Sort TF times
    tf_times = sorted(tf_time_to_ind.keys())
    if not tf_times:
        return [{}] * len(candles_1m)

    # For each 1m candle, find the latest completed higher-TF bar
    result = []
    tf_idx = 0
    for c in candles_1m:
        t = c.get("time", c.get("open_time", 0))
        if isinstance(t, (int, float)) and t > 1e12:
            t = t // 1000  # ms to s

        # Advance tf_idx to the latest TF bar at or before this 1m time
        while tf_idx < len(tf_times) - 1 and tf_times[tf_idx + 1] <= t:
            tf_idx += 1

        if tf_idx < len(tf_times) and tf_times[tf_idx] <= t:
            result.append(tf_time_to_ind[tf_times[tf_idx]])
        else:
            result.append({})

    return result


def run_ultra_backtest(candles: list[dict],
                       config: UltraBacktestConfig = None) -> BacktestResult:
    """
    Enhanced backtest with regime detection + multi-timeframe indicators.
    Returns BacktestResult with enriched trade data for UltraThink feeding.
    """
    config = config or UltraBacktestConfig()
    start_time = time.time()
    n_candles = len(candles)

    logger.info(f"UltraBacktest: {n_candles} candles, regime={config.use_regime}, "
                f"trailing={config.use_trailing_stop}")

    # ── Step 1: Pre-compute 1m indicators ──────────────────────────────────
    logger.info("  Computing 1m indicators...")
    ind_1m_all = _precompute_indicators(candles)

    # ── Step 2: Resample and compute 5m/15m indicators ─────────────────────
    logger.info("  Resampling to 5m/15m and computing indicators...")
    candles_5m = _resample_candles(candles, 5)
    candles_15m = _resample_candles(candles, 15)

    ind_5m_raw = _precompute_indicators(candles_5m) if candles_5m else []
    ind_15m_raw = _precompute_indicators(candles_15m) if candles_15m else []

    # Build lookup tables: 1m index → latest 5m/15m indicator
    ind_5m_lookup = _build_tf_lookup(candles, ind_5m_raw, candles_5m)
    ind_15m_lookup = _build_tf_lookup(candles, ind_15m_raw, candles_15m)

    logger.info(f"  Indicators ready in {time.time() - start_time:.1f}s "
                f"(5m: {len(candles_5m)} bars, 15m: {len(candles_15m)} bars)")

    # ── Step 3: Run candle-by-candle backtest ──────────────────────────────
    result = BacktestResult(final_equity=config.initial_equity)
    equity = config.initial_equity
    peak_equity = equity

    position = 0
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0
    trail_price = 0.0  # trailing stop level
    entry_idx = 0
    entry_regime = ""
    entry_signals = {}
    entry_reasoning = ""
    entry_strategy = ""
    cooldown_remaining = 0

    trades = []
    equity_curve = []
    returns = []
    current_wins = 0
    current_losses = 0
    best_consecutive_wins = 0
    worst_consecutive_losses = 0
    max_dd = 0.0

    n_ind = len(ind_1m_all)
    for i in range(50, min(n_candles, n_ind)):
        candle = candles[i]
        ind_1m = ind_1m_all[i]
        if not ind_1m:
            continue

        ind_5m = ind_5m_lookup[i] if i < len(ind_5m_lookup) else {}
        ind_15m = ind_15m_lookup[i] if i < len(ind_15m_lookup) else {}

        price = candle["close"]
        high = candle["high"]
        low = candle["low"]
        atr = ind_1m.get("atr", price * 0.01)

        # Detect regime for this candle
        regime_state = detect_regime(ind_1m, ind_5m) if config.use_regime else None
        regime_str = regime_state.regime.value if regime_state else ""

        # ── Check exits ────────────────────────────────────────────────
        if position != 0:
            # Update trailing stop if enabled and in profit
            if config.use_trailing_stop and trail_price != 0:
                if position == 1:
                    new_trail = high - config.trail_distance_atr * atr
                    if new_trail > trail_price:
                        trail_price = new_trail
                    # Use the tighter of trail and original stop
                    effective_stop = max(stop_price, trail_price)
                else:
                    new_trail = low + config.trail_distance_atr * atr
                    if new_trail < trail_price:
                        trail_price = new_trail
                    effective_stop = min(stop_price, trail_price)
            else:
                effective_stop = stop_price

            # Check if we should start trailing
            if config.use_trailing_stop and trail_price == 0:
                profit_atr = position * (price - entry_price) / (atr + 1e-9)
                if profit_atr >= config.trail_trigger_atr:
                    if position == 1:
                        trail_price = price - config.trail_distance_atr * atr
                    else:
                        trail_price = price + config.trail_distance_atr * atr

            hit_stop = (position == 1 and low <= effective_stop) or \
                       (position == -1 and high >= effective_stop)
            hit_tp = (position == 1 and high >= tp_price) or \
                     (position == -1 and low <= tp_price)

            if hit_tp or hit_stop:
                if hit_tp:
                    exit_price = tp_price
                    exit_reason = "TP"
                elif trail_price != 0 and effective_stop != stop_price:
                    exit_price = effective_stop
                    exit_reason = "TRAIL"
                else:
                    exit_price = effective_stop
                    exit_reason = "SL"

                risk_dist = abs(stop_price - entry_price) + 1e-9
                size = equity * config.risk_pct / risk_dist
                pnl = position * (exit_price - entry_price) * size
                pnl = max(pnl, -equity * config.risk_pct)
                pnl_pct = pnl / equity * 100

                equity += pnl
                returns.append(pnl / config.initial_equity)

                trade = BacktestTrade(
                    direction=position, entry_price=entry_price, exit_price=exit_price,
                    entry_idx=entry_idx, exit_idx=i, pnl=pnl, pnl_pct=pnl_pct,
                    exit_reason=exit_reason, bars_held=i - entry_idx,
                )
                # Attach extra data for UltraThink (stored as attributes)
                trade.entry_regime = entry_regime
                trade.exit_regime = regime_str
                trade.entry_signals = entry_signals
                trade.entry_reasoning = entry_reasoning
                trade.strategy = entry_strategy
                trades.append(trade)

                if pnl > 0:
                    current_wins += 1; current_losses = 0
                    best_consecutive_wins = max(best_consecutive_wins, current_wins)
                else:
                    current_losses += 1; current_wins = 0
                    worst_consecutive_losses = max(worst_consecutive_losses, current_losses)

                position = 0
                trail_price = 0.0
                cooldown_remaining = config.cooldown_candles

                if config.max_trades > 0 and len(trades) >= config.max_trades:
                    break

        # ── Check entry ────────────────────────────────────────────────
        if position == 0 and cooldown_remaining <= 0:
            # Get regime-specific params
            if regime_state and config.use_regime:
                rp = get_regime_params(regime_state.regime)
                min_sigs = rp.get("min_signals", config.min_signals)
                min_conf = max(config.min_confidence, 0.50)
                sl_mult = rp.get("stop_atr_mult", config.stop_atr_mult)
                tp_mult = rp.get("tp_atr_mult", config.tp_atr_mult)
                pref_dir = rp.get("preferred_direction", 0)
            else:
                min_sigs = config.min_signals
                min_conf = config.min_confidence
                sl_mult = config.stop_atr_mult
                tp_mult = config.tp_atr_mult
                pref_dir = 0

            signal = evaluate_signals(
                ind_1m, ind_5m, ind_15m,
                min_signals=min_sigs,
                min_confidence=min_conf,
                preferred_direction=pref_dir,
                regime=regime_str,
            )

            if signal.direction != 0 and signal.confidence >= min_conf:
                position = signal.direction
                entry_price = price
                stop_price = price - position * sl_mult * atr
                tp_price = price + position * tp_mult * atr
                trail_price = 0.0
                entry_idx = i
                entry_regime = regime_str
                entry_signals = dict(signal.raw_signals)
                entry_reasoning = signal.reasoning
                entry_strategy = signal.strategy

        cooldown_remaining = max(0, cooldown_remaining - 1)
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        max_dd = max(max_dd, dd)

        if i % 20 == 0:
            equity_curve.append({"idx": i, "equity": round(equity, 2)})

    # ── Compute stats ──────────────────────────────────────────────────────
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    total_win = sum(t.pnl for t in wins)
    total_loss = abs(sum(t.pnl for t in losses))

    result.total_trades = len(trades)
    result.winning_trades = len(wins)
    result.losing_trades = len(losses)
    result.win_rate = (len(wins) / len(trades) * 100) if trades else 0
    result.total_pnl = equity - config.initial_equity
    result.avg_win = (total_win / len(wins)) if wins else 0
    result.avg_loss = (total_loss / len(losses)) if losses else 0
    result.profit_factor = (total_win / total_loss) if total_loss > 0 else 999
    result.max_drawdown_pct = max_dd
    result.final_equity = equity
    result.return_pct = (equity / config.initial_equity - 1) * 100
    result.trades = trades
    result.equity_curve = equity_curve
    result.best_trade = max((t.pnl for t in trades), default=0)
    result.worst_trade = min((t.pnl for t in trades), default=0)
    result.consecutive_wins = best_consecutive_wins
    result.consecutive_losses = worst_consecutive_losses
    result.avg_bars_held = (sum(t.bars_held for t in trades) / len(trades)) if trades else 0
    result.duration_seconds = time.time() - start_time

    if returns:
        ret_arr = np.array(returns)
        result.sharpe_ratio = float(np.mean(ret_arr) / (np.std(ret_arr) + 1e-9) * np.sqrt(252 * 390))

    logger.info(
        f"UltraBacktest: {result.total_trades} trades, WR={result.win_rate:.1f}%, "
        f"PF={result.profit_factor:.2f}, P&L=${result.total_pnl:.2f} ({result.return_pct:.1f}%), "
        f"MaxDD={result.max_drawdown_pct:.1f}%, Sharpe={result.sharpe_ratio:.2f} "
        f"[{result.duration_seconds:.1f}s]"
    )

    return result


async def ultra_learn_from_backtest(
    candles: list[dict],
    config: UltraBacktestConfig = None,
    symbol: str = "XAUUSD",
) -> dict:
    """
    The main UltraThink learning pipeline:
    1. Run enhanced backtest with regime + multi-TF
    2. Feed trades through ultra_teach() in batches
    3. Run a final ultra_coach() session with all results
    4. Return summary of what was learned

    This is the function that makes the bot learn from hundreds of
    historical trades in minutes instead of days of live trading.
    """
    from ai.claude_advisor import ClaudeAdvisor, AdvisorContext
    from ai.claude_integration import _apply_ultra_signal_adjustments
    from ai.smart_filters import SignalWeightLearner
    from ai.ultra_memory import get_ultra_memory

    config = config or UltraBacktestConfig()
    start_time = time.time()

    logger.info(f"=== UltraThink Rapid Learning: {symbol} ({len(candles)} candles) ===")

    # ── Step 1: Run the enhanced backtest ──────────────────────────────────
    bt_result = await asyncio.to_thread(run_ultra_backtest, candles, config)

    if bt_result.total_trades == 0:
        return {
            "ok": False,
            "error": "No trades generated by backtest",
            "backtest": bt_result.to_dict(),
        }

    logger.info(f"Backtest produced {bt_result.total_trades} trades — feeding to UltraThink...")

    # ── Step 2: Select trades to teach ─────────────────────────────────────
    all_trades = bt_result.trades
    teach_trades = []
    for idx, t in enumerate(all_trades):
        if idx % config.teach_every_n == 0:
            teach_trades.append(t)
    teach_trades = teach_trades[:config.max_teach_trades]

    logger.info(f"Selected {len(teach_trades)} trades for UltraThink teaching "
                f"(every {config.teach_every_n}, max {config.max_teach_trades})")

    # ── Step 3: Feed trades through ultra_teach() ──────────────────────────
    advisor = ClaudeAdvisor()
    signal_learner = SignalWeightLearner()
    memory = get_ultra_memory()
    teach_results = []
    teach_errors = 0

    semaphore = asyncio.Semaphore(config.teach_concurrency)

    async def teach_one(trade: BacktestTrade, trade_num: int) -> dict:
        nonlocal teach_errors
        async with semaphore:
            trade_data = {
                "symbol": symbol,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl_usd": trade.pnl,
                "pnl_pct": trade.pnl_pct,
                "exit_reason": trade.exit_reason,
                "entry_time": 0,
                "exit_time": trade.bars_held * 60,  # approximate seconds
                "entry_regime": getattr(trade, "entry_regime", ""),
                "exit_regime": getattr(trade, "exit_regime", ""),
                "entry_signals": getattr(trade, "entry_signals", {}),
                "reasoning": getattr(trade, "entry_reasoning", ""),
                "strategy": getattr(trade, "strategy", ""),
            }

            ctx = AdvisorContext(
                symbol=symbol,
                regime=getattr(trade, "entry_regime", ""),
                price=trade.entry_price,
                atr=abs(trade.exit_price - trade.entry_price) * 0.5,
            )

            try:
                result = await advisor.ultra_teach(trade_data, ctx)
                teaching = result.get("teaching", {})

                # Apply signal adjustments
                adjustments = teaching.get("signal_adjustments", {})
                if adjustments:
                    _apply_ultra_signal_adjustments(signal_learner, adjustments)

                # Store lesson in memory (store_lesson expects teaching + trade_data)
                memory.store_lesson(teaching, trade_data)

                if trade_num % 10 == 0:
                    logger.info(f"  Taught trade {trade_num}/{len(teach_trades)} "
                                f"(grade: {teaching.get('trade_grade', '?')})")

                return {"ok": True, "teaching": teaching}

            except Exception as e:
                teach_errors += 1
                logger.warning(f"  UltraThink teach error on trade {trade_num}: {e}")
                return {"ok": False, "error": str(e)}

    # Run teaching in parallel batches
    tasks = [teach_one(t, i + 1) for i, t in enumerate(teach_trades)]
    teach_results = await asyncio.gather(*tasks)

    successful_teaches = sum(1 for r in teach_results if r.get("ok"))
    logger.info(f"Teaching complete: {successful_teaches}/{len(teach_trades)} successful, "
                f"{teach_errors} errors")

    # ── Step 4: Run final coaching session ─────────────────────────────────
    logger.info("Running final UltraThink coaching session...")

    # Build trade entries for coaching (same format as live journal)
    recent_entries = []
    for t in all_trades[-30:]:
        recent_entries.append({
            "symbol": symbol,
            "direction": t.direction,
            "pnl_usd": t.pnl,
            "exit_reason": t.exit_reason,
            "regime": getattr(t, "entry_regime", ""),
            "strategy": getattr(t, "strategy", ""),
        })

    signal_report = signal_learner.get_report()

    # Build outcome report
    from ai.smart_filters import TradeOutcomeTracker
    outcome_tracker = TradeOutcomeTracker()
    for t in all_trades:
        outcome_tracker.record(
            pnl=t.pnl,
            regime=getattr(t, "entry_regime", ""),
            strategy=getattr(t, "strategy", ""),
        )
    outcome_report = outcome_tracker.get_report()

    ctx = AdvisorContext(symbol=symbol, regime="", price=0, atr=0)

    try:
        coach_result = await advisor.ultra_coach(
            recent_entries, signal_report, outcome_report, ctx
        )
        coaching = coach_result.get("coaching", {})

        # Apply coaching signal adjustments
        coach_adjustments = coaching.get("signal_adjustments", {})
        if coach_adjustments:
            _apply_ultra_signal_adjustments(signal_learner, coach_adjustments)

        # Store coaching in memory
        memory.store_coaching(coaching)

        logger.info(f"Coaching complete: grade={coaching.get('overall_grade', '?')}")
    except Exception as e:
        logger.error(f"Coaching session failed: {e}")
        coaching = {"error": str(e)}

    # ── Step 5: Compile results ────────────────────────────────────────────
    total_time = time.time() - start_time

    # Collect all grades
    grades = []
    for r in teach_results:
        if r.get("ok"):
            g = r.get("teaching", {}).get("trade_grade", "")
            if g:
                grades.append(g)

    grade_counts = {}
    for g in grades:
        grade_counts[g] = grade_counts.get(g, 0) + 1

    summary = {
        "ok": True,
        "symbol": symbol,
        "total_time_s": round(total_time, 1),
        "backtest": {
            "total_trades": bt_result.total_trades,
            "win_rate": round(bt_result.win_rate, 1),
            "profit_factor": round(bt_result.profit_factor, 2),
            "total_pnl": round(bt_result.total_pnl, 2),
            "return_pct": round(bt_result.return_pct, 2),
            "max_drawdown_pct": round(bt_result.max_drawdown_pct, 2),
            "sharpe_ratio": round(bt_result.sharpe_ratio, 3),
        },
        "learning": {
            "trades_taught": successful_teaches,
            "teach_errors": teach_errors,
            "grade_distribution": grade_counts,
            "signal_adjustments_applied": len(signal_learner.get_report()),
        },
        "coaching": {
            "overall_grade": coaching.get("overall_grade", "?"),
            "top_improvements": coaching.get("top_3_improvements", []),
            "summary": coaching.get("coaching_summary", "")[:500],
        },
        "signal_weights_after": signal_learner.get_report(),
    }

    logger.info(f"=== UltraThink Learning Complete ===")
    logger.info(f"  Time: {total_time:.1f}s | Trades: {bt_result.total_trades} | "
                f"Taught: {successful_teaches} | Grade distribution: {grade_counts}")

    return summary
