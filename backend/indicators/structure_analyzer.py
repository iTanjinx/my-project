"""
Smart Money Structure Analyzer — FVGs, BOS, CHoCH, Order Blocks.

Detects institutional price action patterns and sends them to the frontend
for live chart drawing. Runs incrementally on each new candle.

Patterns detected:
  - FVG (Fair Value Gap): 3-candle imbalance zones
  - BOS (Break of Structure): swing high/low violation in trend direction
  - CHoCH (Change of Character): swing break AGAINST prevailing trend
  - OB (Order Block): last opposite candle before an impulse move
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SwingPoint:
    """A confirmed swing high or low."""
    index: int
    time: int
    price: float
    type: str  # "HIGH" or "LOW"
    broken: bool = False  # True after BOS/CHoCH breaks it


@dataclass
class FVG:
    """Fair Value Gap — 3-candle imbalance zone."""
    type: str           # "BULL" or "BEAR"
    top: float          # Upper boundary of the gap
    bottom: float       # Lower boundary of the gap
    time_start: int     # Timestamp of the gap candle (middle)
    time_end: int       # Extends until filled (or current time)
    index: int          # Candle index of the gap
    filled: bool = False
    fill_pct: float = 0.0  # How much of the gap has been filled (0-1)


@dataclass
class BOS:
    """Break of Structure — swing point violated."""
    type: str           # "BULL_BOS", "BEAR_BOS", "BULL_CHOCH", "BEAR_CHOCH"
    price: float        # The swing level that was broken
    time: int           # When the break happened
    index: int
    swing_time: int     # When the original swing was formed


@dataclass
class OrderBlock:
    """Order Block — last opposite candle before impulse."""
    type: str           # "BULL" or "BEAR"
    top: float
    bottom: float
    time: int
    index: int
    mitigated: bool = False  # True if price has returned and tested it


class StructureAnalyzer:
    """
    Incrementally tracks smart money structure across candles.
    Call `update(candle)` on each new candle, then read `.zones` for drawing.

    Keeps a rolling window — old filled/broken zones are pruned automatically.
    """

    def __init__(self, swing_lookback: int = 3, max_zones: int = 30):
        self.swing_lookback = swing_lookback  # Bars left+right to confirm swing
        self.max_zones = max_zones

        self._candles: list = []
        self._swing_highs: List[SwingPoint] = []
        self._swing_lows: List[SwingPoint] = []
        self._trend: str = ""  # "UP", "DOWN", ""

        self.fvgs: List[FVG] = []
        self.bos_events: List[BOS] = []
        self.order_blocks: List[OrderBlock] = []

    def update(self, candle: dict) -> dict:
        """
        Process a new candle. Returns any new zones detected this candle.

        candle: {time, open, high, low, close, volume}
        Returns: {new_fvgs: [], new_bos: [], new_obs: []}
        """
        self._candles.append(candle)
        idx = len(self._candles) - 1
        new_fvgs = []
        new_bos = []
        new_obs = []

        # ── 1. Detect swing points (confirmed after lookback bars) ──
        self._detect_swings(idx)

        # ── 2. Detect FVGs (needs 3 candles) ──
        if idx >= 2:
            fvg = self._detect_fvg(idx)
            if fvg:
                self.fvgs.append(fvg)
                new_fvgs.append(fvg)

        # ── 3. Detect BOS / CHoCH ──
        bos = self._detect_bos(candle, idx)
        if bos:
            self.bos_events.append(bos)
            new_bos.append(bos)
            # Detect order block at the BOS point
            ob = self._detect_order_block(bos, idx)
            if ob:
                self.order_blocks.append(ob)
                new_obs.append(ob)

        # ── 4. Update FVG fill status ──
        self._update_fvg_fills(candle)

        # ── 5. Update order block mitigation ──
        self._update_ob_mitigation(candle)

        # ── 6. Prune old/filled zones ──
        self._prune()

        return {
            "new_fvgs": [self._fvg_to_dict(f) for f in new_fvgs],
            "new_bos": [self._bos_to_dict(b) for b in new_bos],
            "new_obs": [self._ob_to_dict(o) for o in new_obs],
        }

    def get_all_zones(self) -> dict:
        """Return all active (unfilled/unbroken) zones for chart drawing."""
        return {
            "fvgs": [self._fvg_to_dict(f) for f in self.fvgs if not f.filled],
            "bos": [self._bos_to_dict(b) for b in self.bos_events[-self.max_zones:]],
            "order_blocks": [self._ob_to_dict(o) for o in self.order_blocks if not o.mitigated],
            "swing_highs": [
                {"time": sp.time, "price": sp.price, "broken": sp.broken}
                for sp in self._swing_highs[-20:]
            ],
            "swing_lows": [
                {"time": sp.time, "price": sp.price, "broken": sp.broken}
                for sp in self._swing_lows[-20:]
            ],
        }

    # ── Internal detection methods ──────────────────────────────────────────

    def _detect_swings(self, idx: int):
        """Detect swing highs and lows with lookback confirmation."""
        lb = self.swing_lookback
        # We need at least `lb` bars after the candidate to confirm
        confirm_idx = idx - lb
        if confirm_idx < lb:
            return

        candles = self._candles

        # Check if confirm_idx is a swing high
        is_swing_high = True
        is_swing_low = True
        candidate_high = candles[confirm_idx]["high"]
        candidate_low = candles[confirm_idx]["low"]

        for offset in range(1, lb + 1):
            left = confirm_idx - offset
            right = confirm_idx + offset
            if left < 0 or right >= len(candles):
                is_swing_high = False
                is_swing_low = False
                break

            if candles[left]["high"] >= candidate_high or candles[right]["high"] >= candidate_high:
                is_swing_high = False
            if candles[left]["low"] <= candidate_low or candles[right]["low"] <= candidate_low:
                is_swing_low = False

        if is_swing_high:
            sp = SwingPoint(
                index=confirm_idx,
                time=candles[confirm_idx]["time"],
                price=candidate_high,
                type="HIGH",
            )
            # Don't duplicate
            if not self._swing_highs or self._swing_highs[-1].index != confirm_idx:
                self._swing_highs.append(sp)
                self._update_trend()

        if is_swing_low:
            sp = SwingPoint(
                index=confirm_idx,
                time=candles[confirm_idx]["time"],
                price=candidate_low,
                type="LOW",
            )
            if not self._swing_lows or self._swing_lows[-1].index != confirm_idx:
                self._swing_lows.append(sp)
                self._update_trend()

    def _update_trend(self):
        """Determine trend from swing sequence: HH+HL=UP, LH+LL=DOWN."""
        if len(self._swing_highs) >= 2 and len(self._swing_lows) >= 2:
            hh = self._swing_highs[-1].price > self._swing_highs[-2].price
            hl = self._swing_lows[-1].price > self._swing_lows[-2].price
            lh = self._swing_highs[-1].price < self._swing_highs[-2].price
            ll = self._swing_lows[-1].price < self._swing_lows[-2].price

            if hh and hl:
                self._trend = "UP"
            elif lh and ll:
                self._trend = "DOWN"
            # Mixed = keep previous trend (transition)

    def _detect_fvg(self, idx: int) -> Optional[FVG]:
        """
        Fair Value Gap: 3-candle pattern where candle 1 and candle 3
        don't overlap, leaving a gap at candle 2.

        Bullish FVG: candles[i-2].high < candles[i].low
        Bearish FVG: candles[i-2].low > candles[i].high
        """
        c1 = self._candles[idx - 2]
        c2 = self._candles[idx - 1]  # The gap candle
        c3 = self._candles[idx]

        # Bullish FVG: gap above candle 1
        if c1["high"] < c3["low"]:
            gap_size = c3["low"] - c1["high"]
            candle_range = c2["high"] - c2["low"] if c2["high"] != c2["low"] else 1e-9
            # Only significant gaps (at least 30% of middle candle range)
            if gap_size > candle_range * 0.15:
                return FVG(
                    type="BULL",
                    top=c3["low"],
                    bottom=c1["high"],
                    time_start=c2["time"],
                    time_end=c3["time"],
                    index=idx - 1,
                )

        # Bearish FVG: gap below candle 1
        if c1["low"] > c3["high"]:
            gap_size = c1["low"] - c3["high"]
            candle_range = c2["high"] - c2["low"] if c2["high"] != c2["low"] else 1e-9
            if gap_size > candle_range * 0.15:
                return FVG(
                    type="BEAR",
                    top=c1["low"],
                    bottom=c3["high"],
                    time_start=c2["time"],
                    time_end=c3["time"],
                    index=idx - 1,
                )

        return None

    def _detect_bos(self, candle: dict, idx: int) -> Optional[BOS]:
        """
        Break of Structure: current candle breaks a recent swing point.
        - BOS: break in trend direction (continuation)
        - CHoCH: break against trend (character change = potential reversal)
        """
        high = candle["high"]
        low = candle["low"]
        time = candle["time"]

        # Check if we break any recent unbroken swing high
        for sh in reversed(self._swing_highs[-10:]):
            if sh.broken:
                continue
            if high > sh.price:
                sh.broken = True
                if self._trend == "UP":
                    return BOS(type="BULL_BOS", price=sh.price, time=time,
                               index=idx, swing_time=sh.time)
                elif self._trend == "DOWN":
                    return BOS(type="BULL_CHOCH", price=sh.price, time=time,
                               index=idx, swing_time=sh.time)
                else:
                    return BOS(type="BULL_BOS", price=sh.price, time=time,
                               index=idx, swing_time=sh.time)

        # Check if we break any recent unbroken swing low
        for sl in reversed(self._swing_lows[-10:]):
            if sl.broken:
                continue
            if low < sl.price:
                sl.broken = True
                if self._trend == "DOWN":
                    return BOS(type="BEAR_BOS", price=sl.price, time=time,
                               index=idx, swing_time=sl.time)
                elif self._trend == "UP":
                    return BOS(type="BEAR_CHOCH", price=sl.price, time=time,
                               index=idx, swing_time=sl.time)
                else:
                    return BOS(type="BEAR_BOS", price=sl.price, time=time,
                               index=idx, swing_time=sl.time)

        return None

    def _detect_order_block(self, bos: BOS, idx: int) -> Optional[OrderBlock]:
        """
        Order Block: the last opposite candle before the impulse move that broke structure.
        - Bullish OB: last bearish candle before a bullish BOS
        - Bearish OB: last bullish candle before a bearish BOS
        """
        if idx < 5:
            return None

        is_bull = bos.type.startswith("BULL")

        # Walk back to find the last opposite candle
        for i in range(idx - 1, max(idx - 15, 0), -1):
            c = self._candles[i]
            if is_bull and c["close"] < c["open"]:  # bearish candle = bull OB
                return OrderBlock(
                    type="BULL", top=c["high"], bottom=c["low"],
                    time=c["time"], index=i,
                )
            elif not is_bull and c["close"] > c["open"]:  # bullish candle = bear OB
                return OrderBlock(
                    type="BEAR", top=c["high"], bottom=c["low"],
                    time=c["time"], index=i,
                )

        return None

    def _update_fvg_fills(self, candle: dict):
        """Check if current candle fills any active FVGs."""
        for fvg in self.fvgs:
            if fvg.filled:
                continue

            if fvg.type == "BULL":
                # Bullish FVG filled when price drops into the gap
                if candle["low"] <= fvg.top:
                    penetration = min(candle["low"], fvg.top) - fvg.bottom
                    gap_size = fvg.top - fvg.bottom
                    if gap_size > 0:
                        fvg.fill_pct = max(fvg.fill_pct,
                                           1.0 - max(0, candle["low"] - fvg.bottom) / gap_size)
                    if candle["low"] <= fvg.bottom:
                        fvg.filled = True
                        fvg.fill_pct = 1.0
            else:
                # Bearish FVG filled when price rises into the gap
                if candle["high"] >= fvg.bottom:
                    gap_size = fvg.top - fvg.bottom
                    if gap_size > 0:
                        fvg.fill_pct = max(fvg.fill_pct,
                                           1.0 - max(0, fvg.top - candle["high"]) / gap_size)
                    if candle["high"] >= fvg.top:
                        fvg.filled = True
                        fvg.fill_pct = 1.0

    def _update_ob_mitigation(self, candle: dict):
        """Check if current candle mitigates any order blocks."""
        for ob in self.order_blocks:
            if ob.mitigated:
                continue
            if ob.type == "BULL":
                # Bull OB mitigated when price trades through the entire zone
                if candle["low"] <= ob.bottom:
                    ob.mitigated = True
            else:
                # Bear OB mitigated when price trades through the entire zone
                if candle["high"] >= ob.top:
                    ob.mitigated = True

    def _prune(self):
        """Remove old zones to keep memory bounded."""
        # Keep only unfilled FVGs and last N filled ones
        active_fvgs = [f for f in self.fvgs if not f.filled]
        filled_fvgs = [f for f in self.fvgs if f.filled]
        self.fvgs = active_fvgs + filled_fvgs[-5:]  # keep last 5 filled for reference

        # Keep last N BOS events
        self.bos_events = self.bos_events[-self.max_zones:]

        # Keep only unmitigated OBs and last 3 mitigated
        active_obs = [o for o in self.order_blocks if not o.mitigated]
        mitigated_obs = [o for o in self.order_blocks if o.mitigated]
        self.order_blocks = active_obs + mitigated_obs[-3:]

        # Keep swing points bounded
        self._swing_highs = self._swing_highs[-30:]
        self._swing_lows = self._swing_lows[-30:]

    # ── Serialization ───────────────────────────────────────────────────────

    @staticmethod
    def _fvg_to_dict(fvg: FVG) -> dict:
        return {
            "type": fvg.type,
            "top": round(fvg.top, 2),
            "bottom": round(fvg.bottom, 2),
            "time_start": fvg.time_start,
            "time_end": fvg.time_end,
            "filled": fvg.filled,
            "fill_pct": round(fvg.fill_pct, 2),
        }

    @staticmethod
    def _bos_to_dict(bos: BOS) -> dict:
        return {
            "type": bos.type,
            "price": round(bos.price, 2),
            "time": bos.time,
            "swing_time": bos.swing_time,
        }

    @staticmethod
    def _ob_to_dict(ob: OrderBlock) -> dict:
        return {
            "type": ob.type,
            "top": round(ob.top, 2),
            "bottom": round(ob.bottom, 2),
            "time": ob.time,
            "mitigated": ob.mitigated,
        }
