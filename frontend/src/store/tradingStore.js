import { create } from 'zustand'

// Remove duplicate timestamps (lightweight-charts requires strictly ascending)
function dedup(sorted) {
  const seen = new Set()
  return sorted.filter(c => seen.has(c.time) ? false : seen.add(c.time))
}

const SYMBOLS = ['XAUUSD', 'USTEC']

const defaultSymbolState = () => ({
  // Live price
  price: null,
  priceTime: null,
  unrealizedPnl: 0,

  // Candles (for chart)
  candles: [],
  candlesVersion: 0,    // Bumped only by setCandles (bulk history load) — chart watches for setData
  latestCandle: null,   // Set by setCandleClose — chart watches for incremental update()
  latestEmaPoint: null, // {time, ema9, ema21, ema50} — chart EMA series watches for update()

  // Indicators
  indicators: {},

  // Regime
  regime: 'UNKNOWN',
  regimeConfidence: 0,

  // AI signals
  signal: null,
  rlState: { action: 0, confidence: 0, trained: false },

  // ── Enriched AI data (from backend Phase 1) ──
  signalConfluence: null,    // { bull_score, bear_score, weighted_signals, strategy_winner, ... }
  regimeFeatures: null,      // { scores: {per-regime}, volatility, bb_width }
  rlInternals: null,         // { action_probs: {hold,long,short}, value_estimate, entropy }
  signalHistory: [],         // Rolling buffer of last 50 signal snapshots for sparklines
  confidenceHistory: [],     // Rolling buffer of {time, signalConf, rlConf, divergence} for timeline

  // ── Wave 2: AI intelligence data ──
  aiThoughts: null,          // Latest ai_thoughts payload
  thoughtHistory: [],        // Rolling buffer of last 30 thought entries
  entryForecast: null,       // { win_prob, avg_win_pct, avg_loss_pct, expected_value, sample_size }
  crossSymbol: null,         // { correlation, btc_momentum, xau_momentum }
  trainingHistory: [],       // [{steps, timestamp, mean_reward}]
  patternMarkers: [],        // Chart markers for candle patterns
  regimeMarkers: [],         // Chart markers for regime changes
  prevRegime: null,          // For detecting regime transitions

  // ── Heiken Ashi strategy ──
  haCandles: [],             // Last 20 HA candles for chart overlay
  haClassifications: [],     // HA candle type strings
  haSignal: null,            // { direction, pattern, confidence, reasoning, trend_length }
  haCurrentType: '',         // "STRONG_BULL", "INDECISION", etc.
  haTrendColor: '',          // "GREEN", "RED", "INDECISION"
  haConsecutiveCount: 0,
  haTpStatus: null,          // { stage, tp1_hit, tp2_hit }

  // Smart Money Structure (FVG, BOS, CHoCH, Order Blocks)
  structureZones: null,       // { fvgs: [], bos: [], order_blocks: [], swing_highs: [], swing_lows: [] }

  // Knowledge Base
  kbContext: null,            // { relevance_score, risk_level, sentiment_bias, key_insights, ... }
  session: null,              // { name, label, hours_remaining, volatility_mult, description }
  analytics: null,            // { calibration, strategy_comparison, hourly_heatmap, pattern_accuracy, streaks }
  aiStatus: null,             // { state, reason, cooldown_remaining, candle_warmup, signal_strength, bull_score, bear_score }

  // Claude Advisory
  claudeAnalysis: null,        // Latest market analysis from Claude
  claudeEventRisk: null,       // Event risk assessment
  claudeExplanation: null,     // Latest trade explanation
  claudeAlerts: [],            // Recent alerts

  // Open position
  position: null,

  // Trade log
  trades: [],

  // Portfolio metrics
  metrics: {
    equity: 10000,
    return_pct: 0,
    total_pnl: 0,
    total_trades: 0,
    win_rate: 0,
    wins: 0,
    losses: 0,
    drawdown_pct: 0,
    sharpe: 0,
    halted: false,
  },

  // Equity curve history
  equityHistory: [],

  // EMA lines (kept for symbol-switch restore; only appended, never full setData)
  emaLines: { ema9: [], ema21: [], ema50: [] },

  // Trade markers
  tradeMarkers: [],
})

export const useTradingStore = create((set, get) => ({
  connected: false,
  activeSymbol: 'XAUUSD',
  strategyMode: 'AI',  // 'AI' or 'HA'
  setStrategyMode: (mode) => set({ strategyMode: mode }),
  tradeMode: 'PAPER',  // 'PAPER' or 'LIVE' or 'SIGNAL'
  setTradeMode: (mode) => set({ tradeMode: mode }),
  mt5Connected: false,
  setMt5Connected: (v) => set({ mt5Connected: v }),
  showHaOverlay: false,
  setShowHaOverlay: (v) => set({ showHaOverlay: v }),

  // Per-symbol state
  symbols: Object.fromEntries(SYMBOLS.map(s => [s, defaultSymbolState()])),

  setConnected: (v) => set({ connected: v }),
  setActiveSymbol: (s) => set({ activeSymbol: s }),

  // Global RL training steps per symbol (updated by training_update WS messages)
  rlTrainingSteps: { XAUUSD: 0, USTEC: 0 },
  setTrainingUpdate: (symbol, steps) => set(state => ({
    rlTrainingSteps: { ...state.rlTrainingSteps, [symbol]: steps },
  })),
  setTrainingHistory: (symbol, history) => get()._patch(symbol, (s) => ({
    ...s,
    trainingHistory: history,
  })),

  // ── Claude advisory setters ──
  setClaudeAnalysis: (symbol, data) => get()._patch(symbol, (s) => ({
    ...s,
    claudeAnalysis: data,
  })),
  setClaudeEventRisk: (symbol, data) => get()._patch(symbol, (s) => ({
    ...s,
    claudeEventRisk: data,
  })),
  setClaudeExplanation: (symbol, data) => get()._patch(symbol, (s) => ({
    ...s,
    claudeExplanation: data,
  })),
  addClaudeAlert: (symbol, alert) => get()._patch(symbol, (s) => ({
    ...s,
    claudeAlerts: [...(s.claudeAlerts || []).slice(-49), alert],
  })),

  // ── Helpers that patch just one symbol's slice ──────────────────────────
  _patch: (symbol, updater) => set((state) => ({
    symbols: {
      ...state.symbols,
      [symbol]: updater(state.symbols[symbol] ?? defaultSymbolState()),
    },
  })),

  setTick: (symbol, data) => get()._patch(symbol, (s) => {
    const updated = {
      ...s,
      price: data.price,
      priceTime: data.time,
      unrealizedPnl: data.unrealized_pnl ?? 0,
    }
    // Live trailing stop update — patch the position's stop_price from tick
    if (s.position && data.stop_price != null) {
      updated.position = { ...s.position, stop_price: data.stop_price, trail_active: data.trail_active }
    }
    return updated
  }),

  // ── Single-patch batch for all candle_close data ────────────────────────
  // Replaces: addCandle + setIndicators + setRegime + setSignal + setRlState + updateEma
  // Result: 1 Zustand dispatch instead of 6 → 6× fewer re-renders per minute
  setCandleClose: (symbol, msg) => get()._patch(symbol, (s) => {
    const {
      candle, indicators = {}, regime, regime_confidence, signal, rl,
      signal_confluence, regime_features, rl_internals,
      ai_thoughts, entry_forecast, cross_symbol,
    } = msg

    // Append or update candle in rolling history
    const idx = s.candles.findIndex(c => c.time === candle.time)
    let newCandles
    if (idx >= 0) {
      newCandles = [...s.candles]
      newCandles[idx] = candle
    } else {
      newCandles = dedup([...s.candles.slice(-499), candle].sort((a, b) => a.time - b.time))
    }

    // Append EMA points
    const pushEma = (arr, val) =>
      val != null ? [...arr.slice(-499), { time: candle.time, value: val }] : arr

    // Rolling signal history (last 50)
    const newSignalHistory = signal?.raw
      ? [...s.signalHistory.slice(-49), { time: candle.time, ...signal.raw }]
      : s.signalHistory

    // Rolling confidence history (last 50) — now includes divergence
    const divergence = ai_thoughts?.divergence_score ?? 0
    const newConfHistory = [
      ...s.confidenceHistory.slice(-49),
      { time: candle.time, signalConf: signal?.confidence ?? 0, rlConf: rl?.confidence ?? 0, divergence },
    ]

    // Thought history (last 30) — accumulate thoughts over time
    const newThoughtHistory = ai_thoughts?.thoughts?.length
      ? [...s.thoughtHistory.slice(-29), ...ai_thoughts.thoughts.map(t => ({ ...t, time: candle.time }))]
        .slice(-30)
      : s.thoughtHistory

    // Pattern markers for chart
    const patternNames = ai_thoughts?.pattern_names ?? indicators?.pattern_names ?? []
    const newPatternMarkers = patternNames.length > 0
      ? [...s.patternMarkers.slice(-49), {
          time: candle.time,
          position: patternNames.some(p => ['Bullish Engulfing','Hammer','Morning Star'].includes(p)) ? 'belowBar' : 'aboveBar',
          color: patternNames.some(p => ['Bullish Engulfing','Hammer','Morning Star'].includes(p)) ? '#fbbf24' : '#f87171',
          shape: 'diamond',
          text: patternNames.join(' + '),
          size: 1,
        }]
      : s.patternMarkers

    // Regime change markers
    const regimeChanged = s.prevRegime && s.prevRegime !== regime && s.prevRegime !== 'UNKNOWN'
    const newRegimeMarkers = regimeChanged
      ? [...s.regimeMarkers.slice(-49), { time: candle.time, from: s.prevRegime, to: regime }]
      : s.regimeMarkers

    return {
      ...s,
      candles: newCandles,
      latestCandle: candle,
      latestEmaPoint: indicators.ema9 != null
        ? { time: candle.time, ema9: indicators.ema9, ema21: indicators.ema21, ema50: indicators.ema50 }
        : s.latestEmaPoint,
      indicators,
      regime,
      regimeConfidence: regime_confidence,
      signal,
      rlState: rl,
      // Wave 1 enriched data
      signalConfluence: signal_confluence ?? s.signalConfluence,
      regimeFeatures: regime_features ?? s.regimeFeatures,
      rlInternals: rl_internals ?? s.rlInternals,
      signalHistory: newSignalHistory,
      confidenceHistory: newConfHistory,
      // Wave 2 AI intelligence data
      aiThoughts: ai_thoughts ?? s.aiThoughts,
      thoughtHistory: newThoughtHistory,
      entryForecast: entry_forecast,  // null when no signal
      crossSymbol: cross_symbol ?? s.crossSymbol,
      patternMarkers: newPatternMarkers,
      regimeMarkers: newRegimeMarkers,
      prevRegime: regime,
      // Heiken Ashi strategy data
      haCandles: msg.ha_strategy?.ha_candles ?? s.haCandles,
      haClassifications: msg.ha_strategy?.classifications ?? s.haClassifications,
      haSignal: msg.ha_strategy?.signal ?? s.haSignal,
      haCurrentType: msg.ha_strategy?.current_type ?? s.haCurrentType,
      haTrendColor: msg.ha_strategy?.trend_color ?? s.haTrendColor,
      haConsecutiveCount: msg.ha_strategy?.consecutive_count ?? s.haConsecutiveCount,
      haTpStatus: msg.ha_strategy?.tp_status ?? s.haTpStatus,
      // Smart Money Structure
      structureZones: msg.structure_zones ?? s.structureZones,
      // Knowledge Base
      kbContext: msg.kb_context ?? s.kbContext,
      // Session + Analytics + AI Status + Trade Mode
      session: msg.session ?? s.session,
      analytics: msg.analytics ?? s.analytics,
      aiStatus: msg.ai_status ?? s.aiStatus,
      emaLines: {
        ema9:  pushEma(s.emaLines.ema9,  indicators.ema9),
        ema21: pushEma(s.emaLines.ema21, indicators.ema21),
        ema50: pushEma(s.emaLines.ema50, indicators.ema50),
      },
    }
  }),

  // Bulk history load — bumps candlesVersion so chart calls setData exactly once
  setCandles: (symbol, candles) => get()._patch(symbol, (s) => ({
    ...s,
    candles: dedup([...candles].sort((a, b) => a.time - b.time)),
    candlesVersion: s.candlesVersion + 1,
    latestCandle: null,
    latestEmaPoint: null,
  })),

  setPosition: (symbol, pos) => get()._patch(symbol, (s) => ({ ...s, position: pos })),
  clearPosition: (symbol) => get()._patch(symbol, (s) => ({ ...s, position: null })),

  addTrade: (symbol, trade) => get()._patch(symbol, (s) => ({
    ...s,
    trades: [trade, ...s.trades].slice(0, 100),
  })),

  setMetrics: (symbol, m) => get()._patch(symbol, (s) => ({
    ...s,
    metrics: m,
    equityHistory: [...s.equityHistory.slice(-499), { time: Date.now(), equity: m.equity ?? 10000 }],
  })),

  addTradeMarker: (symbol, marker) => get()._patch(symbol, (s) => ({
    ...s,
    tradeMarkers: [...s.tradeMarkers, marker],
  })),

  // ── Shortcut: active symbol's state ─────────────────────────────────────
  active: () => get().symbols[get().activeSymbol] ?? defaultSymbolState(),
}))
