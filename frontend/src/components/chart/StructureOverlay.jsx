/**
 * StructureOverlay — Draws Smart Money Concepts on the chart.
 *
 * Uses lightweight-charts coordinate conversion API to draw:
 *   - FVG (Fair Value Gap) zones as semi-transparent rectangles
 *   - BOS (Break of Structure) as horizontal dashed lines with labels
 *   - CHoCH (Change of Character) as dotted lines with labels
 *   - Order Blocks as filled rectangles
 *   - Swing points as small circles
 *
 * Renders on a <canvas> positioned over the chart container.
 * Re-draws on every animation frame when zones or chart viewport change.
 */
import { useEffect, useRef, useCallback } from 'react'

// ── Colors ──────────────────────────────────────────────────────────────────
const COLORS = {
  BULL_FVG:   { fill: 'rgba(34,197,94,0.10)',  border: 'rgba(34,197,94,0.35)',  label: '#22c55e' },
  BEAR_FVG:   { fill: 'rgba(239,68,68,0.10)',  border: 'rgba(239,68,68,0.35)',  label: '#ef4444' },
  BULL_BOS:   { line: '#22c55e', label: '#22c55e', dash: [6, 3] },
  BEAR_BOS:   { line: '#ef4444', label: '#ef4444', dash: [6, 3] },
  BULL_CHOCH: { line: '#f59e0b', label: '#f59e0b', dash: [3, 3] },
  BEAR_CHOCH: { line: '#f59e0b', label: '#f59e0b', dash: [3, 3] },
  BULL_OB:    { fill: 'rgba(34,197,94,0.06)',  border: 'rgba(34,197,94,0.25)' },
  BEAR_OB:    { fill: 'rgba(239,68,68,0.06)',  border: 'rgba(239,68,68,0.25)' },
  SWING_HIGH: 'rgba(239,68,68,0.5)',
  SWING_LOW:  'rgba(34,197,94,0.5)',
}

export default function StructureOverlay({ chartRef, seriesRef, structureZones, visible }) {
  const canvasRef = useRef(null)
  const rafRef = useRef(null)

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const chart = chartRef?.current
    const series = seriesRef?.current
    if (!canvas || !chart || !series || !structureZones) return

    const ctx = canvas.getContext('2d')
    const { width, height } = canvas
    ctx.clearRect(0, 0, width, height)

    if (!visible) return

    const timeScale = chart.timeScale()

    // Helper: convert time to X pixel
    const timeToX = (t) => {
      const coord = timeScale.timeToCoordinate(t)
      return coord != null ? coord : -999
    }

    // Helper: convert price to Y pixel
    const priceToY = (p) => {
      const coord = series.priceToCoordinate(p)
      return coord != null ? coord : -999
    }

    // Get visible range for culling
    const visRange = timeScale.getVisibleLogicalRange()
    if (!visRange) return

    // ── Draw FVGs ─────────────────────────────────────────────────────────
    const { fvgs = [], bos = [], order_blocks = [], swing_highs = [], swing_lows = [] } = structureZones

    for (const fvg of fvgs) {
      const x1 = timeToX(fvg.time_start)
      const x2 = Math.max(timeToX(fvg.time_end), x1 + 80) // Extend FVG zone to the right
      const xEnd = Math.min(x2 + 200, width) // Extend further but cap at chart edge
      const yTop = priceToY(fvg.top)
      const yBot = priceToY(fvg.bottom)

      if (yTop < -50 || yBot < -50) continue // Off screen

      const colors = fvg.type === 'BULL' ? COLORS.BULL_FVG : COLORS.BEAR_FVG
      const boxHeight = Math.abs(yBot - yTop)
      const yStart = Math.min(yTop, yBot)

      // Fill
      ctx.fillStyle = fvg.filled
        ? fvg.type === 'BULL' ? 'rgba(34,197,94,0.03)' : 'rgba(239,68,68,0.03)'
        : colors.fill
      ctx.fillRect(x1, yStart, xEnd - x1, boxHeight)

      // Border
      ctx.strokeStyle = colors.border
      ctx.lineWidth = 1
      ctx.setLineDash(fvg.filled ? [2, 4] : [])
      ctx.strokeRect(x1, yStart, xEnd - x1, boxHeight)
      ctx.setLineDash([])

      // Label
      ctx.font = '600 9px "Geist Mono", monospace'
      ctx.fillStyle = colors.label
      const label = fvg.filled ? 'FVG (filled)' : `FVG ${fvg.type}`
      ctx.fillText(label, x1 + 4, yStart + 11)
    }

    // ── Draw Order Blocks ─────────────────────────────────────────────────
    for (const ob of order_blocks) {
      const x = timeToX(ob.time)
      const xEnd = Math.min(x + 150, width)
      const yTop = priceToY(ob.top)
      const yBot = priceToY(ob.bottom)

      if (yTop < -50 || yBot < -50) continue

      const colors = ob.type === 'BULL' ? COLORS.BULL_OB : COLORS.BEAR_OB
      const boxHeight = Math.abs(yBot - yTop)
      const yStart = Math.min(yTop, yBot)

      ctx.fillStyle = ob.mitigated ? 'rgba(100,100,100,0.03)' : colors.fill
      ctx.fillRect(x, yStart, xEnd - x, boxHeight)

      ctx.strokeStyle = ob.mitigated ? 'rgba(100,100,100,0.15)' : colors.border
      ctx.lineWidth = 1
      ctx.setLineDash(ob.mitigated ? [2, 4] : [])
      ctx.strokeRect(x, yStart, xEnd - x, boxHeight)
      ctx.setLineDash([])

      // Label
      ctx.font = '600 9px "Geist Mono", monospace'
      ctx.fillStyle = ob.type === 'BULL' ? '#22c55e80' : '#ef444480'
      ctx.fillText(`OB ${ob.type}`, x + 4, yStart + 11)
    }

    // ── Draw BOS / CHoCH lines ────────────────────────────────────────────
    for (const b of bos) {
      const x = timeToX(b.time)
      const xSwing = timeToX(b.swing_time)
      const y = priceToY(b.price)

      if (y < -50 || y > height + 50) continue

      const colors = COLORS[b.type] || COLORS.BULL_BOS

      // Horizontal line from swing origin to break point (and beyond)
      ctx.beginPath()
      ctx.strokeStyle = colors.line
      ctx.lineWidth = 1.5
      ctx.setLineDash(colors.dash)
      ctx.moveTo(Math.max(xSwing, 0), y)
      ctx.lineTo(Math.min(x + 60, width), y)
      ctx.stroke()
      ctx.setLineDash([])

      // Break point marker (small X or arrow)
      ctx.beginPath()
      ctx.arc(x, y, 3, 0, Math.PI * 2)
      ctx.fillStyle = colors.line
      ctx.fill()

      // Label
      ctx.font = '700 10px "Geist Mono", monospace'
      ctx.fillStyle = colors.label
      const isCHoCH = b.type.includes('CHOCH')
      const labelText = isCHoCH ? 'CHoCH' : 'BOS'
      ctx.fillText(labelText, x + 8, y - 5)
    }

    // ── Draw Swing Points ─────────────────────────────────────────────────
    for (const sh of swing_highs) {
      const x = timeToX(sh.time)
      const y = priceToY(sh.price)
      if (y < -20 || y > height + 20) continue

      ctx.beginPath()
      ctx.arc(x, y, sh.broken ? 2 : 3, 0, Math.PI * 2)
      ctx.fillStyle = sh.broken ? 'rgba(239,68,68,0.2)' : COLORS.SWING_HIGH
      ctx.fill()

      if (!sh.broken) {
        // Small triangle above for swing high
        ctx.beginPath()
        ctx.moveTo(x, y - 6)
        ctx.lineTo(x - 3, y - 3)
        ctx.lineTo(x + 3, y - 3)
        ctx.closePath()
        ctx.fillStyle = COLORS.SWING_HIGH
        ctx.fill()
      }
    }

    for (const sl of swing_lows) {
      const x = timeToX(sl.time)
      const y = priceToY(sl.price)
      if (y < -20 || y > height + 20) continue

      ctx.beginPath()
      ctx.arc(x, y, sl.broken ? 2 : 3, 0, Math.PI * 2)
      ctx.fillStyle = sl.broken ? 'rgba(34,197,94,0.2)' : COLORS.SWING_LOW
      ctx.fill()

      if (!sl.broken) {
        // Small triangle below for swing low
        ctx.beginPath()
        ctx.moveTo(x, y + 6)
        ctx.lineTo(x - 3, y + 3)
        ctx.lineTo(x + 3, y + 3)
        ctx.closePath()
        ctx.fillStyle = COLORS.SWING_LOW
        ctx.fill()
      }
    }
  }, [chartRef, seriesRef, structureZones, visible])

  // Resize canvas to match container
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const resize = () => {
      const parent = canvas.parentElement
      if (!parent) return
      const dpr = window.devicePixelRatio || 1
      const w = parent.clientWidth
      const h = parent.clientHeight
      canvas.width = w * dpr
      canvas.height = h * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')
      ctx.scale(dpr, dpr)
      draw()
    }

    resize()
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
  }, [draw])

  // Redraw on data change or chart scroll/zoom
  useEffect(() => {
    const chart = chartRef?.current
    if (!chart) return

    // Subscribe to time scale changes (scroll/zoom) to redraw
    const handler = () => draw()
    chart.timeScale().subscribeVisibleLogicalRangeChange(handler)

    // Also redraw when data changes
    draw()

    return () => {
      try { chart.timeScale().unsubscribeVisibleLogicalRangeChange(handler) } catch {}
    }
  }, [chartRef, draw, structureZones])

  // Continuous RAF for smooth rendering during interactions
  useEffect(() => {
    let running = true
    const loop = () => {
      if (!running) return
      draw()
      rafRef.current = requestAnimationFrame(loop)
    }
    // Only run RAF loop when there are zones to draw
    if (structureZones && visible) {
      loop()
    }
    return () => {
      running = false
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [structureZones, visible, draw])

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 2,
      }}
    />
  )
}
