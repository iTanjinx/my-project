import { useEffect, useRef, useState } from 'react'
import { createChart, CrosshairMode } from 'lightweight-charts'
import { useTradingStore } from '../store/tradingStore'
import AiChartOverlay from './chart/AiChartOverlay'
import StructureOverlay from './chart/StructureOverlay'

const REGIME_BG_COLORS = {
  TRENDING_UP:   'rgba(34,197,94,0.04)',
  TRENDING_DOWN: 'rgba(239,68,68,0.04)',
  RANGING:       'rgba(245,158,11,0.03)',
  BREAKOUT:      'rgba(167,139,250,0.04)',
  TRANSITION:    'rgba(148,163,184,0.02)',
  UNKNOWN:       'transparent',
}

export default function CandleChart() {
  const containerRef   = useRef(null)
  const chartRef       = useRef(null)
  const seriesRef      = useRef(null)
  const volumeRef      = useRef(null)
  const ema9Ref        = useRef(null)
  const ema21Ref       = useRef(null)
  const ema50Ref       = useRef(null)
  const posLineRef     = useRef(null)
  const slLineRef      = useRef(null)
  const tpLineRef      = useRef(null)
  const tp1LineRef     = useRef(null)
  const tp2LineRef     = useRef(null)
  const tp3LineRef     = useRef(null)
  const prevStopRef    = useRef(null)
  const prevSymbolRef  = useRef(null)

  // Toolbar state
  const [showEma9, setShowEma9]   = useState(true)
  const [showEma21, setShowEma21] = useState(true)
  const [showEma50, setShowEma50] = useState(true)
  const [showVol, setShowVol]     = useState(true)

  const [showSMC, setShowSMC] = useState(true)
  const { activeSymbol, symbols, showHaOverlay, setShowHaOverlay } = useTradingStore()
  const s = symbols[activeSymbol] ?? {}
  const {
    candles = [], candlesVersion = 0,
    latestCandle, latestEmaPoint,
    price, priceTime,
    emaLines = { ema9: [], ema21: [], ema50: [] },
    tradeMarkers = [],
    patternMarkers = [],
    regimeMarkers = [],
    position, regime = 'UNKNOWN',
    haCandles = [],
    signal, aiStatus, session, kbContext, signalConfluence,
    indicators, structureZones,
  } = s

  // ─── Init chart once ────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      width:  containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
      layout: { background: { color: '#0a0f1e' }, textColor: '#64748b', fontSize: 11 },
      grid:   { vertLines: { color: 'rgba(30,41,59,0.5)' }, horzLines: { color: 'rgba(30,41,59,0.5)' } },
      crosshair: { mode: CrosshairMode.Normal, vertLine: { color: 'rgba(148,163,184,0.15)' }, horzLine: { color: 'rgba(148,163,184,0.15)' } },
      timeScale: { borderColor: 'rgba(51,65,85,0.5)', timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor: 'rgba(51,65,85,0.5)' },
    })

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e', downColor: '#ef4444',
      borderUpColor: '#22c55e', borderDownColor: '#ef4444',
      wickUpColor: '#22c55e', wickDownColor: '#ef4444',
    })

    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceScaleId: '',
    })
    volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.78, bottom: 0 } })

    const ema9  = chart.addLineSeries({ color: '#f59e0b', lineWidth: 1, priceLineVisible: false, lastValueVisible: false })
    const ema21 = chart.addLineSeries({ color: '#60a5fa', lineWidth: 1, priceLineVisible: false, lastValueVisible: false })
    const ema50 = chart.addLineSeries({ color: '#a78bfa', lineWidth: 1, priceLineVisible: false, lastValueVisible: false })

    chartRef.current    = chart
    seriesRef.current   = candleSeries
    volumeRef.current   = volumeSeries
    ema9Ref.current     = ema9
    ema21Ref.current    = ema21
    ema50Ref.current    = ema50

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({
          width:  containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    }
    window.addEventListener('resize', handleResize)
    return () => { window.removeEventListener('resize', handleResize); chart.remove() }
  }, [])

  // ─── Toggle EMA visibility ─────────────────────────────────────────────
  useEffect(() => { ema9Ref.current?.applyOptions({ visible: showEma9 }) }, [showEma9])
  useEffect(() => { ema21Ref.current?.applyOptions({ visible: showEma21 }) }, [showEma21])
  useEffect(() => { ema50Ref.current?.applyOptions({ visible: showEma50 }) }, [showEma50])
  useEffect(() => { volumeRef.current?.applyOptions({ visible: showVol }) }, [showVol])

  // ─── Regime background color (overlay div, not chart bg — chart needs solid color) ──
  const [regimeBg, setRegimeBg] = useState('transparent')
  useEffect(() => {
    setRegimeBg(REGIME_BG_COLORS[regime] ?? 'transparent')
  }, [regime])

  // ─── Clear chart when symbol switches ───────────────────────────────────
  useEffect(() => {
    if (!seriesRef.current) return
    if (prevSymbolRef.current === activeSymbol) return
    seriesRef.current.setData([])
    seriesRef.current.setMarkers([])
    volumeRef.current?.setData([])
    ema9Ref.current?.setData([])
    ema21Ref.current?.setData([])
    ema50Ref.current?.setData([])
    for (const ref of [posLineRef, slLineRef, tpLineRef, tp1LineRef, tp2LineRef, tp3LineRef]) {
      if (ref.current) {
        try { seriesRef.current.removePriceLine(ref.current) } catch {}
        ref.current = null
      }
    }
    // Reset price scale so it auto-fits to the new symbol's range
    chartRef.current?.priceScale('right').applyOptions({ autoScale: true })
    prevStopRef.current = null
    prevBBRef.current = null
    prevSymbolRef.current = activeSymbol
  }, [activeSymbol])

  // ─── Full data load ─────────────────────────────────────────────────────
  useEffect(() => {
    if (!seriesRef.current || candles.length === 0) return

    // Use HA candles if overlay is active and we have them
    const displayCandles = showHaOverlay && haCandles.length > 0 ? haCandles : candles

    const seen = new Set()
    const clean = displayCandles.slice().sort((a, b) => a.time - b.time)
      .filter(c => seen.has(c.time) ? false : seen.add(c.time))
    seriesRef.current.setData(clean)

    // Update candle colors for HA mode
    if (showHaOverlay && haCandles.length > 0) {
      seriesRef.current.applyOptions({
        upColor: '#06b6d4', downColor: '#f97316',
        borderUpColor: '#06b6d4', borderDownColor: '#f97316',
        wickUpColor: '#06b6d4', wickDownColor: '#f97316',
      })
    } else {
      seriesRef.current.applyOptions({
        upColor: '#22c55e', downColor: '#ef4444',
        borderUpColor: '#22c55e', borderDownColor: '#ef4444',
        wickUpColor: '#22c55e', wickDownColor: '#ef4444',
      })
    }

    // Volume only for regular candles
    if (!showHaOverlay) {
      volumeRef.current?.setData(
        clean.map(c => ({
          time: c.time, value: c.volume ?? 0,
          color: c.close >= c.open ? '#22c55e20' : '#ef444420',
        }))
      )
    }
    ema9Ref.current?.setData(emaLines.ema9)
    ema21Ref.current?.setData(emaLines.ema21)
    ema50Ref.current?.setData(emaLines.ema50)

    // Auto-fit price scale + time range to new data
    chartRef.current?.timeScale().fitContent()
  }, [candlesVersion, activeSymbol, showHaOverlay, haCandles])  // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Incremental candle update ──────────────────────────────────────────
  useEffect(() => {
    if (!seriesRef.current || !latestCandle) return
    // Guard: ensure time is a valid number (fixes "Cannot update oldest data" crash)
    const t = typeof latestCandle.time === 'number' ? latestCandle.time : Number(latestCandle.time)
    if (!t || isNaN(t)) return
    const safeCandle = { ...latestCandle, time: t }
    try {
      seriesRef.current.update(safeCandle)
      volumeRef.current?.update({
        time: t, value: safeCandle.volume ?? 0,
        color: safeCandle.close >= safeCandle.open ? '#22c55e20' : '#ef444420',
      })
    } catch (e) {
      // Suppress lightweight-charts errors on stale/duplicate candles
      if (!e.message?.includes('Cannot update oldest')) console.warn('Chart update:', e.message)
    }
  }, [latestCandle])

  // ─── Incremental EMA update ─────────────────────────────────────────────
  useEffect(() => {
    if (!latestEmaPoint) return
    const { time: rawTime, ema9, ema21, ema50 } = latestEmaPoint
    const time = typeof rawTime === 'number' ? rawTime : Number(rawTime)
    if (!time || isNaN(time)) return
    try {
      if (ema9  != null) ema9Ref.current?.update({ time, value: ema9 })
      if (ema21 != null) ema21Ref.current?.update({ time, value: ema21 })
      if (ema50 != null) ema50Ref.current?.update({ time, value: ema50 })
    } catch (e) { /* suppress stale data errors */ }
  }, [latestEmaPoint])

  // ─── Live tick ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!seriesRef.current || !price || !priceTime || candles.length === 0) return
    const last = candles[candles.length - 1]
    if (!last) return
    seriesRef.current.update({
      time: last.time, open: last.open,
      high: Math.max(last.high, price), low: Math.min(last.low, price),
      close: price,
    })
  }, [price, priceTime])  // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Trade + pattern markers ────────────────────────────────────────────
  useEffect(() => {
    if (!seriesRef.current) return
    // Merge trade markers and pattern markers, sorted by time
    const all = [...tradeMarkers, ...patternMarkers]
      .sort((a, b) => a.time - b.time)
    seriesRef.current.setMarkers(all)
  }, [tradeMarkers, patternMarkers])

  // ─── Position price lines (entry + SL + TP) ────────────────────────────
  useEffect(() => {
    if (!seriesRef.current) return

    // Clean up all position lines
    for (const ref of [posLineRef, slLineRef, tpLineRef, tp1LineRef, tp2LineRef, tp3LineRef]) {
      if (ref.current) {
        try { seriesRef.current.removePriceLine(ref.current) } catch {}
        ref.current = null
      }
    }
    prevStopRef.current = null

    if (!position?.entry_price) return

    const isLong = position.direction === 1

    // Entry price line
    posLineRef.current = seriesRef.current.createPriceLine({
      price: position.entry_price,
      color: isLong ? '#22c55e' : '#ef4444',
      lineWidth: 1, lineStyle: 2, axisLabelVisible: true,
      title: isLong ? '▲ ENTRY' : '▼ ENTRY',
    })

    // Stop loss line
    if (position.stop_price) {
      slLineRef.current = seriesRef.current.createPriceLine({
        price: position.stop_price,
        color: position.trail_active ? '#f59e0b' : '#ef4444',
        lineWidth: 1, lineStyle: 1, axisLabelVisible: true,
        title: position.trail_active ? 'TRAIL' : 'SL',
      })
      prevStopRef.current = position.stop_price
    }

    // Take profit line(s)
    if (position.ha_tiers) {
      // HA strategy: tiered TPs
      if (position.ha_tiers.tp1) {
        tp1LineRef.current = seriesRef.current.createPriceLine({
          price: position.ha_tiers.tp1, color: '#4ade80', lineWidth: 1, lineStyle: 1,
          axisLabelVisible: true, title: 'TP1 (40%)',
        })
      }
      if (position.ha_tiers.tp2) {
        tp2LineRef.current = seriesRef.current.createPriceLine({
          price: position.ha_tiers.tp2, color: '#22d3ee', lineWidth: 1, lineStyle: 1,
          axisLabelVisible: true, title: 'TP2 (35%)',
        })
      }
      if (position.ha_tiers.tp3) {
        tp3LineRef.current = seriesRef.current.createPriceLine({
          price: position.ha_tiers.tp3, color: '#a78bfa', lineWidth: 1, lineStyle: 1,
          axisLabelVisible: true, title: 'TP3 (25%)',
        })
      }
    } else if (position.tp_price) {
      // AI strategy: single TP
      tpLineRef.current = seriesRef.current.createPriceLine({
        price: position.tp_price, color: '#22c55e', lineWidth: 1, lineStyle: 1,
        axisLabelVisible: true, title: 'TP',
      })
    }
  }, [position?.entry_price, position?.direction])  // eslint-disable-line react-hooks/exhaustive-deps

  // ─── Update SL line on tick (trailing stop moves) ─────────────────────
  useEffect(() => {
    if (!seriesRef.current || !position?.stop_price) return
    // Only recreate if price actually changed
    if (prevStopRef.current === position.stop_price) return
    prevStopRef.current = position.stop_price

    if (slLineRef.current) {
      try { seriesRef.current.removePriceLine(slLineRef.current) } catch {}
    }
    slLineRef.current = seriesRef.current.createPriceLine({
      price: position.stop_price,
      color: position.trail_active ? '#f59e0b' : '#ef4444',
      lineWidth: 1, lineStyle: 1, axisLabelVisible: true,
      title: position.trail_active ? 'TRAIL' : 'SL',
    })
  }, [position?.stop_price, position?.trail_active])

  // ─── AI Drawing: Bollinger Bands + VWAP + Structure label ──────────────
  const bbUpperRef = useRef(null)
  const bbLowerRef = useRef(null)
  const vwapLineRef = useRef(null)
  const prevBBRef = useRef(null)

  useEffect(() => {
    if (!seriesRef.current || !indicators) return
    const { bb_upper, bb_lower, vwap, structure_label } = indicators
    if (!bb_upper || !bb_lower) return

    // Skip if unchanged
    const bbKey = `${bb_upper}-${bb_lower}-${vwap}`
    if (prevBBRef.current === bbKey) return
    prevBBRef.current = bbKey

    // Remove old lines
    for (const ref of [bbUpperRef, bbLowerRef, vwapLineRef]) {
      if (ref.current) {
        try { seriesRef.current.removePriceLine(ref.current) } catch {}
        ref.current = null
      }
    }

    // BB Upper (resistance zone)
    bbUpperRef.current = seriesRef.current.createPriceLine({
      price: bb_upper, color: 'rgba(239,68,68,0.3)', lineWidth: 1, lineStyle: 3,
      axisLabelVisible: false, title: '',
    })

    // BB Lower (support zone)
    bbLowerRef.current = seriesRef.current.createPriceLine({
      price: bb_lower, color: 'rgba(34,197,94,0.3)', lineWidth: 1, lineStyle: 3,
      axisLabelVisible: false, title: '',
    })

    // VWAP (dynamic S/R)
    if (vwap) {
      vwapLineRef.current = seriesRef.current.createPriceLine({
        price: vwap, color: 'rgba(96,165,250,0.4)', lineWidth: 1, lineStyle: 1,
        axisLabelVisible: false, title: '',
      })
    }
  }, [indicators?.bb_upper, indicators?.bb_lower, indicators?.vwap])

  const isXau = activeSymbol === 'XAUUSD'
  const label = isXau ? 'XAU/USD · Gold Spot' : 'NAS100 · US Tech 100'

  const REGIME_COLORS = {
    TRENDING_UP: '#22c55e', TRENDING_DOWN: '#ef4444', RANGING: '#f59e0b',
    BREAKOUT: '#a78bfa', TRANSITION: '#64748b', UNKNOWN: '#475569',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* ── Toolbar ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6, padding: '4px 12px',
        borderBottom: '1px solid var(--border-subtle)',
        background: 'var(--surface-1)',
      }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: '#cbd5e1', marginRight: 4 }}>{label}</span>
        <span style={{
          fontSize: 9, padding: '1px 6px', borderRadius: 'var(--radius-full)',
          background: 'rgba(34,197,94,0.1)', color: '#22c55e', fontWeight: 600,
        }}>
          1m · LIVE
        </span>

        {/* Regime badge */}
        <span style={{
          fontSize: 9, padding: '1px 8px', borderRadius: 'var(--radius-full)', marginLeft: 4,
          background: `${REGIME_COLORS[regime] ?? '#475569'}12`,
          color: REGIME_COLORS[regime] ?? '#475569',
          fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em',
        }}>
          {regime?.replace(/_/g, ' ')}
        </span>

        <div style={{ flex: 1 }} />

        {/* EMA toggles */}
        <TogglePill label="EMA9" color="#f59e0b" active={showEma9} onClick={() => setShowEma9(!showEma9)} />
        <TogglePill label="EMA21" color="#60a5fa" active={showEma21} onClick={() => setShowEma21(!showEma21)} />
        <TogglePill label="EMA50" color="#a78bfa" active={showEma50} onClick={() => setShowEma50(!showEma50)} />
        <TogglePill label="Vol" color="#64748b" active={showVol} onClick={() => setShowVol(!showVol)} />
        <div style={{ width: 1, height: 14, background: 'var(--border-default)', margin: '0 2px' }} />
        <TogglePill label="HA" color="#06b6d4" active={showHaOverlay} onClick={() => setShowHaOverlay(!showHaOverlay)} />
        <TogglePill label="SMC" color="#f59e0b" active={showSMC} onClick={() => setShowSMC(!showSMC)} />
      </div>

      <div style={{ flex: 1, position: 'relative' }}>
        {/* Regime background overlay */}
        <div style={{
          position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 0,
          background: regimeBg, transition: 'background 1.5s ease',
        }} />
        <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative', zIndex: 1 }} />
        {/* Smart Money Structure — FVG, BOS, CHoCH, Order Blocks */}
        <StructureOverlay
          chartRef={chartRef}
          seriesRef={seriesRef}
          structureZones={structureZones}
          visible={showSMC}
        />
        {/* AI Chart Overlay — status HUD, signal bar, reasoning ticker */}
        <AiChartOverlay
          aiStatus={aiStatus}
          signal={signal}
          session={session}
          kbContext={kbContext}
          signalConfluence={signalConfluence}
          indicators={indicators}
        />
      </div>
    </div>
  )
}

function TogglePill({ label, color, active, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex', alignItems: 'center', gap: 4,
        padding: '2px 8px', border: 'none', cursor: 'pointer',
        borderRadius: 'var(--radius-full)', fontSize: 10, fontWeight: 600,
        background: active ? `${color}18` : 'transparent',
        color: active ? color : '#475569',
        transition: 'all 0.2s',
        border: `1px solid ${active ? `${color}30` : 'transparent'}`,
      }}
    >
      <span style={{
        width: 10, height: 2, borderRadius: 1,
        background: active ? color : '#334155',
        transition: 'background 0.2s',
      }} />
      {label}
    </button>
  )
}
