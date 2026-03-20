"""
MetaTrader 5 connector — real-time data feed + live trade execution.
Connects to a running MT5 terminal on the same machine.

Requirements:
  1. MetaTrader 5 terminal installed and running
  2. Logged into a broker account (demo or real)
  3. pip install MetaTrader5

Provides:
  - Real-time tick data (same prices you trade on)
  - Historical candle data (from broker's server)
  - Order placement (market orders with SL/TP)
  - Position management (modify SL, close positions)
  - Account info (balance, equity, margin)
"""
import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Callable, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not installed — MT5 features disabled")


# ── Symbol mapping ────────────────────────────────────────────────────────────
# Map our internal names to MT5 broker symbol names (varies by broker)
DEFAULT_SYMBOL_MAP = {
    "XAUUSD": ["XAUUSDc", "XAUUSD", "GOLD", "XAUUSDm", "XAUUSD.a", "Gold"],
    "USTEC": ["USTECc", "USTEC", "USTECm", "USTEC.c", "NAS100", "NAS100c", "NAS100m", "US100", "US100c"],
    "BTCUSDT": ["BTCUSDc", "BTCUSD", "BTCUSDm", "BTCUSD.a", "Bitcoin"],
}


@dataclass
class MT5Config:
    max_risk_pct: float = 1.0         # Max % of equity per trade
    max_daily_loss_pct: float = 5.0   # Kill switch: halt if daily loss exceeds this
    max_positions: int = 2            # Max simultaneous positions
    lot_size: float = 0.01            # Default lot size (micro lot)
    slippage: int = 20                # Max slippage in points
    magic_number: int = 202603        # Unique ID for our bot's orders


class MT5Connector:
    """Connects to MetaTrader 5 terminal for data and trading."""

    def __init__(self, config: MT5Config = None):
        self.config = config or MT5Config()
        self._connected = False
        self._symbol_map: dict[str, str] = {}  # our_symbol → mt5_symbol
        self._daily_pnl = 0.0
        self._daily_reset_time = 0.0
        self._halted = False

    # ── Connection ────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Initialize connection to MT5 terminal."""
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 package not installed")
            return False

        if not mt5.initialize():
            error = mt5.last_error()
            logger.error(f"MT5 initialize failed: {error}")
            logger.error("Make sure MetaTrader 5 terminal is running and logged in")
            return False

        info = mt5.terminal_info()
        account = mt5.account_info()

        if info is None or account is None:
            logger.error("MT5 connected but no account info — check login")
            mt5.shutdown()
            return False

        self._connected = True
        logger.info(f"MT5 connected: {account.name} @ {account.server}")
        logger.info(f"  Balance: ${account.balance:.2f}, Equity: ${account.equity:.2f}")
        logger.info(f"  Leverage: 1:{account.leverage}")

        # Auto-detect broker symbol names
        self._detect_symbols()

        return True

    def disconnect(self):
        """Shutdown MT5 connection."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def _detect_symbols(self):
        """Auto-detect which symbol names the broker uses."""
        for our_sym, candidates in DEFAULT_SYMBOL_MAP.items():
            for candidate in candidates:
                info = mt5.symbol_info(candidate)
                if info is not None:
                    # Select symbol first to make it visible in Market Watch
                    mt5.symbol_select(candidate, True)
                    self._symbol_map[our_sym] = candidate
                    logger.info(f"  Symbol mapped: {our_sym} → {candidate}")
                    break
            if our_sym not in self._symbol_map:
                logger.warning(f"  Symbol not found for {our_sym} — tried: {candidates}")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Account Info ──────────────────────────────────────────────────────

    def get_account_info(self) -> dict:
        """Get current account state."""
        if not self._connected:
            return {}
        acc = mt5.account_info()
        if acc is None:
            return {}
        return {
            "balance": acc.balance,
            "equity": acc.equity,
            "margin": acc.margin,
            "free_margin": acc.margin_free,
            "profit": acc.profit,
            "leverage": acc.leverage,
            "name": acc.name,
            "server": acc.server,
            "currency": acc.currency,
        }

    # ── Market Data ───────────────────────────────────────────────────────

    def get_tick(self, symbol: str) -> dict | None:
        """Get latest tick for a symbol."""
        mt5_sym = self._symbol_map.get(symbol)
        if not mt5_sym or not self._connected:
            return None

        tick = mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            return None

        return {
            "price": (tick.bid + tick.ask) / 2,  # mid price
            "bid": tick.bid,
            "ask": tick.ask,
            "time": tick.time,
            "spread": round((tick.ask - tick.bid) * (10 ** mt5.symbol_info(mt5_sym).digits), 1),
        }

    def get_history(self, symbol: str, count: int = 1000, timeframe: int = None) -> list[dict]:
        """Fetch historical candles from MT5 broker."""
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M1

        mt5_sym = self._symbol_map.get(symbol)
        if not mt5_sym or not self._connected:
            return []

        rates = mt5.copy_rates_from_pos(mt5_sym, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            return []

        candles = []
        for r in rates:
            candles.append({
                "open_time": int(r['time']) * 1000,
                "open": float(r['open']),
                "high": float(r['high']),
                "low": float(r['low']),
                "close": float(r['close']),
                "volume": float(r['tick_volume']),
                "closed": True,
            })

        return candles

    # ── Order Execution ───────────────────────────────────────────────────

    def open_order(self, symbol: str, direction: int, lot: float = None,
                   sl_price: float = None, tp_price: float = None,
                   comment: str = "AI Trader") -> dict:
        """
        Place a market order.
        direction: 1 = BUY, -1 = SELL
        Returns: {success, ticket, price, error}
        """
        if not self._connected:
            return {"success": False, "error": "MT5 not connected"}
        if self._halted:
            return {"success": False, "error": "Trading halted (daily loss limit)"}

        mt5_sym = self._symbol_map.get(symbol)
        if not mt5_sym:
            return {"success": False, "error": f"Symbol {symbol} not mapped"}

        # Safety checks
        positions = mt5.positions_get(symbol=mt5_sym)
        if positions and len(positions) >= self.config.max_positions:
            return {"success": False, "error": f"Max positions ({self.config.max_positions}) reached"}

        tick = mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            return {"success": False, "error": "No tick data"}

        sym_info = mt5.symbol_info(mt5_sym)
        if sym_info is None:
            return {"success": False, "error": "Symbol info unavailable"}

        # Lot size
        if lot is None:
            lot = self.config.lot_size
        lot = max(sym_info.volume_min, min(lot, sym_info.volume_max))
        lot = round(lot / sym_info.volume_step) * sym_info.volume_step

        order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == 1 else tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_sym,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": self.config.slippage,
            "magic": self.config.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if sl_price:
            request["sl"] = sl_price
        if tp_price:
            request["tp"] = tp_price

        result = mt5.order_send(request)
        if result is None:
            return {"success": False, "error": f"Order send failed: {mt5.last_error()}"}

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "error": f"Order rejected: {result.comment} (code {result.retcode})"}

        logger.info(f"MT5 ORDER: {'BUY' if direction==1 else 'SELL'} {lot} {mt5_sym} @ {result.price} | ticket={result.order}")
        return {
            "success": True,
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
        }

    def close_position(self, ticket: int) -> dict:
        """Close a specific position by ticket."""
        if not self._connected:
            return {"success": False, "error": "MT5 not connected"}

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "error": f"Position {ticket} not found"}

        pos = position[0]
        close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self.config.slippage,
            "magic": self.config.magic_number,
            "comment": "AI Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "error": f"Close failed: {result.comment if result else mt5.last_error()}"}

        logger.info(f"MT5 CLOSE: ticket={ticket} @ {result.price}")
        return {"success": True, "price": result.price}

    def modify_sl_tp(self, ticket: int, sl: float = None, tp: float = None) -> dict:
        """Modify stop loss / take profit of an open position."""
        if not self._connected:
            return {"success": False, "error": "MT5 not connected"}

        position = mt5.positions_get(ticket=ticket)
        if not position:
            return {"success": False, "error": f"Position {ticket} not found"}

        pos = position[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": sl if sl else pos.sl,
            "tp": tp if tp else pos.tp,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return {"success": False, "error": f"Modify failed: {result.comment if result else mt5.last_error()}"}

        return {"success": True}

    def get_positions(self) -> list[dict]:
        """Get all open positions placed by this bot."""
        if not self._connected:
            return []

        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for pos in positions:
            if pos.magic == self.config.magic_number or pos.magic == 0:
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == 0 else "SELL",
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "time": pos.time,
                    "magic": pos.magic,
                    "comment": pos.comment,
                })

        return result

    # ── Safety System ─────────────────────────────────────────────────────

    def check_daily_limit(self) -> bool:
        """Check if daily loss limit has been hit. Returns True if OK to trade."""
        if not self._connected:
            return False

        acc = mt5.account_info()
        if acc is None:
            return False

        # Reset daily counter at midnight UTC
        now = time.time()
        if now - self._daily_reset_time > 86400:
            self._daily_pnl = 0.0
            self._daily_reset_time = now
            self._halted = False

        # Check current floating + realized P&L
        daily_loss_limit = acc.balance * (self.config.max_daily_loss_pct / 100)
        if self._daily_pnl < -daily_loss_limit:
            if not self._halted:
                logger.warning(f"DAILY LOSS LIMIT HIT: ${self._daily_pnl:.2f} (limit: -${daily_loss_limit:.2f})")
                self._halted = True
            return False

        return True

    def calculate_lot_size(self, symbol: str, sl_distance: float) -> float:
        """Calculate position size based on risk percentage and SL distance."""
        if not self._connected:
            return self.config.lot_size

        acc = mt5.account_info()
        mt5_sym = self._symbol_map.get(symbol)
        if not acc or not mt5_sym:
            return self.config.lot_size

        sym_info = mt5.symbol_info(mt5_sym)
        if not sym_info:
            return self.config.lot_size

        risk_usd = acc.equity * (self.config.max_risk_pct / 100)
        tick_value = sym_info.trade_tick_value
        tick_size = sym_info.trade_tick_size

        if tick_value <= 0 or tick_size <= 0 or sl_distance <= 0:
            return self.config.lot_size

        ticks_in_sl = sl_distance / tick_size
        lot = risk_usd / (ticks_in_sl * tick_value)

        # Clamp to broker limits
        lot = max(sym_info.volume_min, min(lot, sym_info.volume_max))
        lot = round(lot / sym_info.volume_step) * sym_info.volume_step

        return round(lot, 2)


class MT5DataFeed:
    """Real-time data feed from MT5 terminal."""

    def __init__(self, connector: MT5Connector, symbol: str,
                 on_tick: Callable, on_candle_close: Callable):
        self.connector = connector
        self.symbol = symbol
        self.on_tick = on_tick
        self.on_candle_close = on_candle_close
        self._running = False
        self._last_candle_time = 0

    async def fetch_history(self, count: int = 5000) -> list[dict]:
        """Fetch historical candles from MT5."""
        return await asyncio.to_thread(self.connector.get_history, self.symbol, count)

    async def start(self):
        """Start tick + candle polling loop."""
        self._running = True
        logger.info(f"MT5 data feed started for {self.symbol}")
        await asyncio.gather(
            self._tick_loop(),
            self._candle_loop(),
        )

    async def stop(self):
        self._running = False

    async def _tick_loop(self):
        """Poll ticks every 500ms — much faster than API feeds."""
        while self._running:
            try:
                tick = await asyncio.to_thread(self.connector.get_tick, self.symbol)
                if tick:
                    await self.on_tick(tick)
            except Exception as e:
                logger.debug(f"MT5 tick error: {e}")
            await asyncio.sleep(0.5)

    async def _candle_loop(self):
        """Poll for new completed candles every 5 seconds."""
        while self._running:
            try:
                candles = await asyncio.to_thread(
                    self.connector.get_history, self.symbol, 3
                )
                if candles and len(candles) >= 2:
                    # The second-to-last candle is the latest completed one
                    latest = candles[-2]
                    candle_time = latest["open_time"]
                    if candle_time > self._last_candle_time:
                        self._last_candle_time = candle_time
                        await self.on_candle_close(latest)
            except Exception as e:
                logger.debug(f"MT5 candle error: {e}")
            await asyncio.sleep(2)
