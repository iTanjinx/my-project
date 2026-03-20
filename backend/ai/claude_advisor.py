"""
Claude AI Advisory Layer — the strategic brain on top of PPO.

Uses Claude Sonnet for real-time market analysis and strategy advice,
Haiku for rapid sentiment classification, and Opus for deep research.

This module does NOT replace the PPO agent — it augments it with:
  • Market context analysis (news, sentiment, macro events)
  • Trade reasoning ("why did we enter?", "should we hold?")
  • Performance review and strategy suggestions
  • Natural language chat interface for the dashboard
  • Alert generation for critical market events
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from ai.ultra_memory import get_ultra_memory

logger = logging.getLogger(__name__)

# ── Model routing ─────────────────────────────────────────────────────────────
MODELS = {
    "fast":    "claude-haiku-4-5",             # Sentiment tagging, quick classification ($1/MTok)
    "smart":   "claude-sonnet-4-6",            # Real-time analysis, trade reasoning ($3/MTok)
    "deep":    "claude-sonnet-4-6",            # Deep research ($3/MTok)
    "ultra":   "claude-opus-4-6",              # UltraThink — adaptive thinking + max effort ($5/$25 MTok)
}

API_URL = "https://api.anthropic.com/v1/messages"
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MAX_TOKENS = {"fast": 512, "smart": 2048, "deep": 4096, "ultra": 16000}

# ── Rate limiting ─────────────────────────────────────────────────────────────
_last_call: dict[str, float] = {"fast": 0, "smart": 0, "deep": 0, "ultra": 0}
_min_interval: dict[str, float] = {"fast": 2.0, "smart": 5.0, "deep": 30.0, "ultra": 10.0}

# ── Cost tracking ─────────────────────────────────────────────────────────────
# Prices per million tokens (input / output) as of March 2026
MODEL_COSTS = {
    "claude-haiku-4-5":   {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-6":  {"input": 3.00, "output": 15.00},
    "claude-opus-4-6":    {"input": 5.00, "output": 25.00},
}

class CostTracker:
    """Tracks API usage costs in real-time."""
    def __init__(self):
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_cost_usd: float = 0.0
        self.calls_by_tier: dict[str, int] = {"fast": 0, "smart": 0, "deep": 0, "ultra": 0}
        self.cost_by_tier: dict[str, float] = {"fast": 0.0, "smart": 0.0, "deep": 0.0, "ultra": 0.0}
        self.session_start: float = time.time()
        self._history: list[dict] = []  # Last 100 calls

    def record(self, tier: str, model: str, input_tokens: int, output_tokens: int):
        costs = MODEL_COSTS.get(model, {"input": 3.0, "output": 15.0})
        input_cost = (input_tokens / 1_000_000) * costs["input"]
        output_cost = (output_tokens / 1_000_000) * costs["output"]
        call_cost = input_cost + output_cost

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += call_cost
        self.calls_by_tier[tier] = self.calls_by_tier.get(tier, 0) + 1
        self.cost_by_tier[tier] = self.cost_by_tier.get(tier, 0) + call_cost

        self._history.append({
            "time": time.time(),
            "tier": tier,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": round(call_cost, 6),
        })
        if len(self._history) > 100:
            self._history = self._history[-100:]

    def get_stats(self) -> dict:
        elapsed_h = max((time.time() - self.session_start) / 3600, 0.01)
        hourly_rate = self.total_cost_usd / elapsed_h
        total_calls = sum(self.calls_by_tier.values())
        return {
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_calls": total_calls,
            "calls_by_tier": self.calls_by_tier,
            "cost_by_tier": {k: round(v, 4) for k, v in self.cost_by_tier.items()},
            "hourly_rate_usd": round(hourly_rate, 4),
            "projected_daily_usd": round(hourly_rate * 24, 2),
            "session_hours": round(elapsed_h, 2),
            "avg_cost_per_call": round(self.total_cost_usd / max(total_calls, 1), 6),
        }

_cost_tracker = CostTracker()

def get_cost_tracker() -> CostTracker:
    return _cost_tracker


@dataclass
class AdvisorContext:
    """Aggregated context passed to Claude for analysis."""
    symbol: str = ""
    price: float = 0.0
    regime: str = ""
    regime_confidence: float = 0.0
    signal_direction: int = 0
    signal_confidence: float = 0.0
    signal_count: int = 0
    rl_action: int = 0
    rl_confidence: float = 0.0
    position: str = "FLAT"
    pnl_usd: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    recent_news: list[dict] = field(default_factory=list)
    sentiment_score: float = 0.0
    indicators: dict = field(default_factory=dict)
    kb_insights: list[str] = field(default_factory=list)

    def to_prompt_context(self) -> str:
        news_text = ""
        if self.recent_news:
            news_items = [f"  - [{n.get('source','?')}] {n.get('headline','')}" for n in self.recent_news[:10]]
            news_text = "\n".join(news_items)

        kb_text = ""
        if self.kb_insights:
            kb_text = "\n".join(f"  - {i}" for i in self.kb_insights[:5])

        return f"""LIVE MARKET STATE:
  Symbol: {self.symbol}
  Price: ${self.price:,.2f}
  Regime: {self.regime} (confidence: {self.regime_confidence:.0%})
  Signal: {'LONG' if self.signal_direction == 1 else 'SHORT' if self.signal_direction == -1 else 'HOLD'} ({self.signal_count} signals, {self.signal_confidence:.0%} conf)
  RL Agent: {['HOLD','LONG','SHORT'][self.rl_action]} ({self.rl_confidence:.0%} conf)
  Position: {self.position}
  Session P&L: ${self.pnl_usd:+,.2f}
  Win Rate: {self.win_rate:.1%} over {self.total_trades} trades

KEY INDICATORS:
  ATR: {self.indicators.get('atr', 'N/A')}
  RSI: {self.indicators.get('rsi', 'N/A')}
  VWAP: {self.indicators.get('vwap', 'N/A')}
  EMA trend: {self.indicators.get('ema_trend', 'N/A')}

MARKET SENTIMENT: {self.sentiment_score:+.2f} (-1 bearish to +1 bullish)

RECENT NEWS:
{news_text or '  (no recent news)'}

KNOWLEDGE BASE INSIGHTS:
{kb_text or '  (no relevant insights)'}"""


class ClaudeAdvisor:
    """Main advisory interface — routes queries to the right Claude model."""

    SYSTEM_PROMPT = """You are an elite XAUUSD/BTCUSDT scalping advisor integrated into a live trading bot.
Your role is to AUGMENT the existing PPO RL agent with strategic market context, NOT to replace it.

Core responsibilities:
1. Analyze market context (news, sentiment, macro events) and their impact on price
2. Identify risks the RL agent might miss (event risk, liquidity gaps, correlations)
3. Provide clear, actionable insights — no vague generalities
4. Alert on critical market conditions (high-impact news, regime changes, unusual volatility)
5. Answer trader questions about the bot's behavior and market conditions

Rules:
- Be concise. This is scalping — seconds matter.
- Give directional bias with confidence: "Bullish XAU (70%): Fed dovish + risk-off flows"
- Flag specific risks: "NFP in 47 min — reduce position size or flatten"
- Never hallucinate data. If uncertain, say so.
- Costs are real — don't waste tokens on filler.
- Format for dashboard display: short sentences, clear structure.
"""

    ULTRATHINK_TRADE_SYSTEM = """You are the world's best quantitative trading coach. You are teaching a live algorithmic trading bot that trades XAU/USD and NAS100 on MetaTrader 5. Your mission: make this bot the most successful auto-trader ever built.

The bot uses:
- 15 technical signals (RSI, MACD, Bollinger, EMA ribbon, Supertrend, Stochastic, ADX, VWAP, volume, structure, etc.) with adaptive weights
- 5 market regime detection (TRENDING_UP, TRENDING_DOWN, RANGING, BREAKOUT, TRANSITION)
- PPO reinforcement learning agent with [512,512,256] network
- Multi-timeframe confluence (1m, 5m, 15m)
- Adversarial filter (argues against bad trades before entry)
- Anti-martingale position sizing

For each closed trade, think DEEPLY through:

1. ENTRY ANALYSIS: Was the entry well-timed? Were the signals genuinely aligned or noise? Which specific signals helped vs hurt? Was the entry on a pullback, breakout, or chasing?

2. EXIT ANALYSIS: Was the exit optimal? SL/TP levels make sense for volatility? Better exit point? Did it leave money on the table or get stopped out prematurely?

3. REGIME FIT: Was this the RIGHT trade for the detected regime? Would a different strategy have worked better? Was the regime correctly identified?

4. SIGNAL WEIGHTS: Which signals should be trusted MORE and which LESS? Be specific with signal names and adjustment amounts.

5. PATTERN RECOGNITION: Name the pattern (e.g. "false breakout in ranging market", "trend continuation after EMA21 pullback", "liquidity grab below support").

6. STRATEGY COACHING: What would the BEST traders in the world do differently here? Teach specific techniques: ideal entry timing, position management, scaling in/out, trailing stops, partial profits.

7. BEST PRACTICES: What rules should the bot ALWAYS follow? What mistakes should it NEVER make? Think about risk management, session timing, spread awareness, news avoidance.

Output ONLY this JSON:
{
  "trade_grade": "A" | "B" | "C" | "D" | "F",
  "entry_quality": <0.0-1.0>,
  "exit_quality": <0.0-1.0>,
  "regime_correct": true | false,
  "signal_adjustments": {
    "<signal_name>": <-0.3 to +0.3 weight adjustment>
  },
  "pattern_name": "<short pattern label>",
  "lessons": ["<lesson 1>", "<lesson 2>", "<lesson 3>"],
  "strategy_tips": ["<pro tip 1>", "<pro tip 2>"],
  "best_practices": ["<rule the bot should always follow>"],
  "avoid_rule": "<IF condition THEN avoid — or null>",
  "repeat_rule": "<IF condition THEN take — or null>",
  "sl_adjustment": "<tighter|wider|good as-is>",
  "tp_adjustment": "<tighter|wider|good as-is>",
  "optimal_entry": "<description of where the perfect entry would have been>",
  "optimal_exit": "<description of where the perfect exit would have been>",
  "focus_area": "<the ONE thing the bot should improve most right now>",
  "reasoning_summary": "<2-3 sentence analysis>"
}

Be BRUTALLY honest. Think like a hedge fund quant reviewing a junior trader's work. The goal is to build the most profitable, disciplined, and intelligent auto-trader ever."""

    ULTRATHINK_COACHING_SYSTEM = """You are the world's best quantitative trading coach. You are reviewing the overall performance of a live algorithmic trading bot and providing comprehensive strategic coaching.

The bot trades XAU/USD and NAS100 on MetaTrader 5 using:
- 15 adaptive technical signals with learned weights
- 5 market regime detection modes
- PPO reinforcement learning with [512,512,256] network
- Multi-timeframe confluence (1m, 5m, 15m)
- Adversarial pre-trade filter
- Anti-martingale sizing

Your job is to provide ELITE-LEVEL coaching that covers:

1. STRATEGY OPTIMIZATION: What strategies work best for each regime? How should the bot adapt its approach based on market conditions? What edge does the data suggest?

2. RISK MANAGEMENT: Position sizing rules, maximum drawdown handling, correlation awareness, session-based risk limits, spread management during volatile periods.

3. TIMING MASTERY: Best times to trade each instrument. When to be aggressive vs defensive. How to handle session opens/closes, news events, liquidity gaps.

4. SIGNAL ENGINEERING: Which signal combinations produce the highest win rates? Which signals are noise? How should signal weights be adjusted based on the regime?

5. EXECUTION EDGE: Entry techniques (limit orders vs market, pullback entries vs breakout entries), partial profit taking, trailing stop strategies, re-entry rules after stops.

6. PSYCHOLOGICAL DISCIPLINE: Anti-tilt rules, max consecutive loss limits, when to stop trading for the day, how to handle drawdowns without revenge trading.

7. ADVANCED TECHNIQUES: Institutional-level concepts the bot should implement: order flow reading, liquidity zone mapping, session pivots, correlation divergences.

Output ONLY this JSON:
{
  "overall_grade": "A" | "B" | "C" | "D" | "F",
  "strongest_edge": "<what the bot does best>",
  "biggest_weakness": "<what costs the most money>",
  "regime_strategies": {
    "TRENDING_UP": "<optimal strategy>",
    "TRENDING_DOWN": "<optimal strategy>",
    "RANGING": "<optimal strategy>",
    "BREAKOUT": "<optimal strategy>",
    "TRANSITION": "<optimal strategy>"
  },
  "signal_adjustments": {"<signal_name>": <-0.3 to +0.3>},
  "risk_rules": ["<rule 1>", "<rule 2>", "<rule 3>"],
  "timing_rules": ["<rule 1>", "<rule 2>"],
  "execution_tips": ["<tip 1>", "<tip 2>", "<tip 3>"],
  "advanced_techniques": ["<technique 1>", "<technique 2>"],
  "top_3_improvements": [
    {"action": "<what to change>", "expected_impact": "<high|medium|low>", "reasoning": "<why>"},
    {"action": "<what to change>", "expected_impact": "<high|medium|low>", "reasoning": "<why>"},
    {"action": "<what to change>", "expected_impact": "<high|medium|low>", "reasoning": "<why>"}
  ],
  "focus_this_week": "<the ONE priority to work on>",
  "coaching_summary": "<3-5 sentence strategic overview>"
}

Think like a world-class hedge fund PM reviewing a systematic strategy. Be specific, data-driven, and actionable."""

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: dict[str, tuple[float, str]] = {}  # key -> (timestamp, response)
        self._cache_ttl = 30  # seconds
        self._conversation_history: list[dict] = []
        self._max_history = 20

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def _call_claude(self, prompt: str, tier: str = "smart",
                           system: str = None, use_cache: bool = True) -> str:
        """Call Claude API with rate limiting and caching."""
        if not API_KEY:
            return "[Claude advisor disabled — set ANTHROPIC_API_KEY env var]"

        # Rate limit
        now = time.time()
        elapsed = now - _last_call.get(tier, 0)
        if elapsed < _min_interval[tier]:
            await asyncio.sleep(_min_interval[tier] - elapsed)
        _last_call[tier] = time.time()

        # Cache check
        cache_key = f"{tier}:{hash(prompt)}"
        if use_cache and cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return cached

        try:
            client = await self._get_client()
            resp = await client.post(
                API_URL,
                headers={
                    "x-api-key": API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": MODELS[tier],
                    "max_tokens": MAX_TOKENS[tier],
                    "system": system or self.SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            if resp.status_code != 200:
                error_body = resp.text[:200]
                logger.error(f"Claude API {resp.status_code}: {error_body}")
                return f"[API error {resp.status_code}]"
            data = resp.json()
            text = data["content"][0]["text"]

            # Track costs from usage data
            usage = data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            _cost_tracker.record(tier, MODELS[tier], input_tokens, output_tokens)

            # Cache it
            self._cache[cache_key] = (time.time(), text)
            return text

        except httpx.TimeoutException:
            logger.warning(f"Claude {tier} call timed out")
            return "[Advisor timeout — market moving fast]"
        except Exception as e:
            logger.error(f"Claude {tier} call failed: {e}")
            return f"[Advisor error: {str(e)[:80]}]"

    async def _call_claude_ultra(self, prompt: str, system: str = None) -> dict:
        """
        UltraThink — Opus 4.6 with adaptive thinking and max effort.
        Returns both the thinking chain and the final answer.
        Uses streaming to avoid HTTP timeout on long reasoning chains.
        """
        if not API_KEY:
            return {"thinking": "", "answer": "[Claude disabled — set ANTHROPIC_API_KEY]", "tokens": {}}

        tier = "ultra"
        now = time.time()
        elapsed = now - _last_call.get(tier, 0)
        if elapsed < _min_interval[tier]:
            await asyncio.sleep(_min_interval[tier] - elapsed)
        _last_call[tier] = time.time()

        # Cache check
        cache_key = f"ultra:{hash(prompt)}"
        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if time.time() - ts < 60:  # 60s cache for ultra
                return json.loads(cached) if isinstance(cached, str) and cached.startswith("{") else {"thinking": "", "answer": cached, "tokens": {}}

        try:
            client = await self._get_client()
            # Use streaming to avoid timeout on long thinking chains
            thinking_text = []
            answer_text = []
            current_block_type = None
            input_tokens = 0
            output_tokens = 0

            async with client.stream(
                "POST", API_URL,
                headers={
                    "x-api-key": API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": MODELS[tier],
                    "max_tokens": MAX_TOKENS[tier],
                    "stream": True,
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "max"},
                    "system": system or self.ULTRATHINK_TRADE_SYSTEM,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120.0,  # Extended timeout for deep reasoning
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    logger.error(f"Claude ultra API {resp.status_code}: {body[:300]}")
                    return {"thinking": "", "answer": f"[API error {resp.status_code}]", "tokens": {}}

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "content_block_start":
                        block = event.get("content_block", {})
                        current_block_type = block.get("type")
                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "thinking_delta":
                            thinking_text.append(delta.get("thinking", ""))
                        elif delta.get("type") == "text_delta":
                            answer_text.append(delta.get("text", ""))
                    elif event_type == "message_start":
                        usage = event.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)
                    elif event_type == "message_delta":
                        usage = event.get("usage", {})
                        output_tokens = usage.get("output_tokens", 0)

            _cost_tracker.record(tier, MODELS[tier], input_tokens, output_tokens)

            result = {
                "thinking": "".join(thinking_text),
                "answer": "".join(answer_text),
                "tokens": {"input": input_tokens, "output": output_tokens},
            }
            self._cache[cache_key] = (time.time(), json.dumps(result))
            return result

        except httpx.TimeoutException:
            logger.warning("Claude ultra call timed out (120s)")
            return {"thinking": "", "answer": "[UltraThink timeout — reasoning took too long]", "tokens": {}}
        except Exception as e:
            logger.error(f"Claude ultra call failed: {e}")
            return {"thinking": "", "answer": f"[UltraThink error: {str(e)[:80]}]", "tokens": {}}

    # ── Core advisory methods ─────────────────────────────────────────────────

    async def analyze_market(self, ctx: AdvisorContext) -> dict:
        """
        Full market analysis — called on each candle or on demand.
        Returns structured analysis for the dashboard.
        """
        prompt = f"""{ctx.to_prompt_context()}

Provide a concise market analysis:
1. BIAS: Overall directional bias with confidence (e.g., "Bullish 65%")
2. KEY FACTORS: Top 2-3 factors driving price right now (1 sentence each)
3. RISKS: Any immediate risks or events to watch (be specific with timing)
4. ADVICE: Should the bot be aggressive, normal, or defensive right now? Why?
5. SIGNAL CHECK: Do you agree with the signal engine's current read? Why/why not?

Keep total response under 200 words. No filler."""

        text = await self._call_claude(prompt, tier="smart")
        return {
            "analysis": text,
            "timestamp": time.time(),
            "model": MODELS["smart"],
            "context_summary": f"{ctx.symbol} @ ${ctx.price:,.2f} | {ctx.regime} | Sentiment: {ctx.sentiment_score:+.2f}",
        }

    async def classify_sentiment(self, headline: str, source: str = "") -> dict:
        """
        Fast sentiment classification using Haiku.
        Returns: score (-1 to +1), impact (low/medium/high), relevance (0-1)
        """
        prompt = f"""Classify this market headline for XAUUSD trading impact.
Source: {source}
Headline: "{headline}"

Respond ONLY as JSON: {{"score": <-1 to 1>, "impact": "<low|medium|high>", "relevance": <0 to 1>, "summary": "<5 words>"}}"""

        text = await self._call_claude(prompt, tier="fast",
                                        system="You are a financial sentiment classifier. Respond only in JSON.")
        try:
            # Parse JSON from response
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return {"score": 0, "impact": "low", "relevance": 0, "summary": "parse_error"}

    async def explain_trade(self, ctx: AdvisorContext, trade_action: str,
                             entry_price: float = 0, sl: float = 0, tp: float = 0) -> str:
        """Explain why a trade was taken or skipped — for the dashboard thought stream."""
        prompt = f"""{ctx.to_prompt_context()}

The bot just {'ENTERED ' + trade_action if trade_action in ('LONG','SHORT') else 'decided to HOLD'}.
{'Entry: $' + f'{entry_price:,.2f}' + f' | SL: ${sl:,.2f} | TP: ${tp:,.2f}' if entry_price else ''}

In 1-2 sentences, explain the reasoning a professional trader would have for this decision.
Focus on what's unique about THIS moment — don't just list indicators."""

        return await self._call_claude(prompt, tier="smart")

    async def check_event_risk(self, ctx: AdvisorContext, upcoming_events: list[dict]) -> dict:
        """Check if upcoming economic events should change trading behavior."""
        events_text = "\n".join(
            f"  - [{e.get('time','?')}] {e.get('event','')} (impact: {e.get('impact','?')})"
            for e in upcoming_events[:10]
        )

        prompt = f"""Current position: {ctx.position} | Symbol: {ctx.symbol} | Price: ${ctx.price:,.2f}

UPCOMING ECONOMIC EVENTS:
{events_text}

Should the bot:
1. Flatten positions before any event? Which one and when?
2. Reduce size? By how much?
3. Widen stops? By how much?
4. Continue normally?

Respond as JSON: {{"action": "<flatten|reduce|widen_stops|continue>", "reason": "<why>", "urgency": "<now|soon|later>", "event": "<which event>"}}"""

        text = await self._call_claude(prompt, tier="smart", use_cache=False)
        try:
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(text)
        except (json.JSONDecodeError, IndexError):
            return {"action": "continue", "reason": "Could not parse response", "urgency": "later", "event": "none"}

    async def ultra_teach(self, trade_data: dict, ctx: AdvisorContext) -> dict:
        """
        UltraThink Trade Teacher — Opus 4.6 adaptive thinking with max effort.
        Analyzes a completed trade and extracts lessons, strategies, and best practices.
        Returns structured teaching analysis + thinking chain.
        """
        duration_s = trade_data.get("exit_time", 0) - trade_data.get("entry_time", 0)
        duration_str = f"{duration_s / 60:.1f} min" if duration_s > 0 else "unknown"

        signals_text = ""
        entry_signals = trade_data.get("entry_signals", {})
        if entry_signals:
            signals_text = "\n".join(f"    {name}: {val:+.3f}" for name, val in entry_signals.items())

        # Inject accumulated memory so UltraThink remembers everything
        memory_ctx = get_ultra_memory().get_context_for_teaching(
            symbol=trade_data.get("symbol", ctx.symbol),
            regime=trade_data.get("entry_regime", ctx.regime),
        )

        prompt = f"""{ctx.to_prompt_context()}

{f"BOT'S ACCUMULATED KNOWLEDGE:{chr(10)}{memory_ctx}" if memory_ctx else ""}

CLOSED TRADE TO ANALYZE:
  Symbol: {trade_data.get('symbol', ctx.symbol)}
  Direction: {'LONG' if trade_data.get('direction') == 1 else 'SHORT'}
  Entry: ${trade_data.get('entry_price', 0):,.2f} → Exit: ${trade_data.get('exit_price', 0):,.2f}
  P&L: ${trade_data.get('pnl_usd', 0):+,.2f} ({trade_data.get('pnl_pct', 0):+.2f}%)
  Exit Reason: {trade_data.get('exit_reason', 'unknown')}
  Duration: {duration_str}
  Entry Regime: {trade_data.get('entry_regime', 'unknown')}
  Exit Regime: {trade_data.get('exit_regime', 'unknown')}
  Strategy: {trade_data.get('strategy', 'default')}

ENTRY SIGNALS AT TIME OF TRADE:
{signals_text or '    (no signal data)'}

ENTRY REASONING: {trade_data.get('reasoning', 'none')}

Consider the bot's accumulated knowledge above — reference past patterns, reinforce proven rules, and build on previous lessons. Analyze this trade through ALL 7 dimensions. Teach the bot everything it needs to know to never make the same mistake again — and to repeat its wins. Output your structured teaching JSON."""

        result = await self._call_claude_ultra(prompt, system=self.ULTRATHINK_TRADE_SYSTEM)

        # Parse the JSON teaching from the answer
        teaching = self._parse_ultra_json(result.get("answer", ""))
        if not teaching:
            answer = result.get("answer", "")
            teaching = {
                "trade_grade": "C",
                "signal_adjustments": {},
                "lessons": [answer[:200] if answer else "Failed to parse UltraThink response"],
                "reasoning_summary": answer[:200] if answer else "Parse error",
            }

        return {
            "teaching": teaching,
            "thinking": result.get("thinking", ""),
            "tokens": result.get("tokens", {}),
            "timestamp": time.time(),
            "model": MODELS["ultra"],
            "tier": "ultra",
        }

    async def ultra_coach(self, recent_trades: list[dict], signal_report: list[dict],
                           outcome_report: dict, ctx: AdvisorContext) -> dict:
        """
        UltraThink Strategy Coach — periodic comprehensive coaching session.
        Reviews overall performance and provides strategic direction.
        Call this every N trades or on demand from the dashboard.
        """
        trades_summary = []
        for t in recent_trades[-30:]:
            trades_summary.append(
                f"  {'WIN' if t.get('pnl_usd', 0) > 0 else 'LOSS'} "
                f"{'LONG' if t.get('direction') == 1 else 'SHORT'} "
                f"{t.get('symbol', '?')} ${t.get('pnl_usd', 0):+.2f} "
                f"| {t.get('exit_reason', '?')} | {t.get('regime', '?')}"
            )

        signal_text = ""
        if signal_report:
            signal_text = "\n".join(
                f"    {s['signal']}: {s['accuracy']}% accurate ({s['total']} trades, weight: {s['weight_mult']}x)"
                for s in signal_report[:15]
            )

        # Inject accumulated memory for coaching context
        memory_ctx = get_ultra_memory().get_context_for_teaching(
            symbol=ctx.symbol,
            regime=ctx.regime,
        )

        prompt = f"""{ctx.to_prompt_context()}

{f"BOT'S ACCUMULATED KNOWLEDGE:{chr(10)}{memory_ctx}" if memory_ctx else ""}

PERFORMANCE DATA (last {len(recent_trades)} trades):
{chr(10).join(trades_summary) or '  (no trades yet)'}

SIGNAL ACCURACY REPORT:
{signal_text or '  (not enough data yet)'}

OUTCOME ANALYSIS:
{json.dumps(outcome_report, indent=2, default=str) if outcome_report else '  (no data)'}

Consider the bot's accumulated knowledge above — its known patterns, proven rules, and past coaching focus areas. Build on what works, fix what doesn't. Conduct a comprehensive coaching session covering strategy, signals, risk, timing, and execution. Output your coaching JSON."""

        result = await self._call_claude_ultra(prompt, system=self.ULTRATHINK_COACHING_SYSTEM)

        coaching = self._parse_ultra_json(result.get("answer", ""))
        if not coaching:
            answer = result.get("answer", "")
            coaching = {
                "overall_grade": "C",
                "signal_adjustments": {},
                "top_3_improvements": [],
                "coaching_summary": answer[:300] if answer else "Failed to parse coaching response",
            }

        return {
            "coaching": coaching,
            "thinking": result.get("thinking", ""),
            "tokens": result.get("tokens", {}),
            "timestamp": time.time(),
            "model": MODELS["ultra"],
            "tier": "ultra",
        }

    @staticmethod
    def _parse_ultra_json(answer: str) -> dict | None:
        """Extract JSON from UltraThink response text."""
        if not answer:
            return None
        try:
            json_start = answer.find("{")
            json_end = answer.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(answer[json_start:json_end])
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    async def review_performance(self, trades: list[dict], equity_curve: list[float]) -> str:
        """Deep performance review — uses Opus for thorough analysis."""
        trades_text = json.dumps(trades[-50:], indent=2, default=str)  # Last 50 trades

        prompt = f"""Here are the last {min(len(trades), 50)} trades from our XAUUSD scalping bot:

{trades_text}

Equity curve (last 100 points): {equity_curve[-100:] if len(equity_curve) > 100 else equity_curve}

Provide a thorough performance review:
1. Win rate, profit factor, and Sharpe assessment
2. Pattern analysis — when does the bot perform best/worst?
3. Regime analysis — which market conditions suit this strategy?
4. Specific parameter suggestions (SL/TP ratios, entry thresholds)
5. Top 3 actionable improvements ranked by expected impact

Be quantitative. Reference specific trades if relevant."""

        return await self._call_claude(prompt, tier="deep")

    # ── Chat interface ────────────────────────────────────────────────────────

    async def chat(self, user_message: str, ctx: AdvisorContext) -> str:
        """Natural language chat — trader asks questions, Claude answers with full context."""
        self._conversation_history.append({"role": "user", "content": user_message})
        if len(self._conversation_history) > self._max_history:
            self._conversation_history = self._conversation_history[-self._max_history:]

        context_block = ctx.to_prompt_context()
        prompt = f"""LIVE CONTEXT:
{context_block}

CONVERSATION:
{json.dumps(self._conversation_history[-10:], indent=2)}

Respond to the trader's latest message. Be direct, specific, and actionable.
If they ask about the bot, reference actual data from the context above.
If they ask about the market, use the news and indicators provided."""

        response = await self._call_claude(prompt, tier="smart", use_cache=False)
        self._conversation_history.append({"role": "assistant", "content": response})
        return response

    # ── Alert generation ──────────────────────────────────────────────────────

    async def generate_alert(self, ctx: AdvisorContext, alert_type: str,
                              details: dict = None) -> dict:
        """Generate a smart alert with context-aware messaging."""
        prompt = f"""{ctx.to_prompt_context()}

ALERT TYPE: {alert_type}
DETAILS: {json.dumps(details or {}, default=str)}

Write a concise alert message (max 2 sentences) that:
1. States what happened
2. States what the trader should know/do

Also classify severity: critical, warning, or info."""

        text = await self._call_claude(prompt, tier="fast")
        return {
            "type": alert_type,
            "message": text,
            "timestamp": time.time(),
            "details": details,
        }

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ── Singleton ─────────────────────────────────────────────────────────────────
_advisor: Optional[ClaudeAdvisor] = None


def get_advisor() -> ClaudeAdvisor:
    global _advisor
    if _advisor is None:
        _advisor = ClaudeAdvisor()
    return _advisor
