"""
UltraThink Persistent Memory — unlimited knowledge accumulation.

The bot NEVER forgets. Every lesson, pattern, rule, signal adjustment,
and coaching insight is stored permanently and fed back into decision-making.

Memory stores:
  1. lessons     — every trade review (raw, never deleted)
  2. patterns    — named recurring patterns with win/loss stats
  3. rules       — avoid/repeat IF/THEN rules with hit counts
  4. signal_log  — per-signal weight adjustment history
  5. coaching    — full coaching session summaries
  6. wisdom      — distilled knowledge (compressed from lessons periodically)
"""
import json
import logging
import os
import time
from collections import defaultdict
from threading import Lock

logger = logging.getLogger(__name__)

MEMORY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "ultra_memory")


class UltraMemory:
    """Persistent, unlimited memory for the UltraThink teaching system."""

    def __init__(self, memory_dir: str = MEMORY_DIR):
        self._dir = memory_dir
        self._lock = Lock()
        os.makedirs(self._dir, exist_ok=True)

        # In-memory caches (loaded from disk on init)
        self.lessons: list[dict] = []
        self.patterns: dict[str, dict] = {}      # pattern_name → {wins, losses, examples, last_seen}
        self.rules: list[dict] = []               # {type: avoid|repeat, rule: str, hits: int, ...}
        self.signal_log: dict[str, list] = defaultdict(list)  # signal_name → [{time, adjustment, reason}]
        self.coaching: list[dict] = []
        self.wisdom: list[dict] = []              # distilled insights
        self.stats: dict = {"total_lessons": 0, "total_coaching": 0, "first_lesson": 0}

        self._load_all()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _path(self, name: str) -> str:
        return os.path.join(self._dir, f"{name}.json")

    def _load_all(self):
        """Load all memory stores from disk."""
        self.lessons = self._load("lessons", [])
        self.patterns = self._load("patterns", {})
        self.rules = self._load("rules", [])
        self.signal_log = defaultdict(list, self._load("signal_log", {}))
        self.coaching = self._load("coaching", [])
        self.wisdom = self._load("wisdom", [])
        self.stats = self._load("stats", {"total_lessons": 0, "total_coaching": 0, "first_lesson": 0})

        total = self.stats.get("total_lessons", len(self.lessons))
        logger.info(
            f"UltraMemory loaded: {total} lessons, {len(self.patterns)} patterns, "
            f"{len(self.rules)} rules, {len(self.coaching)} coaching sessions, "
            f"{len(self.wisdom)} wisdom entries"
        )

    def _load(self, name: str, default):
        path = self._path(name)
        if not os.path.exists(path):
            return default
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")
            return default

    def _save(self, name: str, data):
        try:
            with open(self._path(name), "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save {name}: {e}")

    def _save_all(self):
        self._save("lessons", self.lessons)
        self._save("patterns", self.patterns)
        self._save("rules", self.rules)
        self._save("signal_log", dict(self.signal_log))
        self._save("coaching", self.coaching)
        self._save("wisdom", self.wisdom)
        self._save("stats", self.stats)

    # ── Store trade lesson ───────────────────────────────────────────────────

    def store_lesson(self, teaching: dict, trade_data: dict):
        """Store a trade teaching lesson — NEVER deleted."""
        with self._lock:
            lesson = {
                "id": self.stats["total_lessons"] + 1,
                "timestamp": time.time(),
                "grade": teaching.get("trade_grade", "?"),
                "pattern": teaching.get("pattern_name", ""),
                "lessons": teaching.get("lessons", []),
                "strategy_tips": teaching.get("strategy_tips", []),
                "best_practices": teaching.get("best_practices", []),
                "avoid_rule": teaching.get("avoid_rule"),
                "repeat_rule": teaching.get("repeat_rule"),
                "focus_area": teaching.get("focus_area", ""),
                "optimal_entry": teaching.get("optimal_entry", ""),
                "optimal_exit": teaching.get("optimal_exit", ""),
                "sl_adjustment": teaching.get("sl_adjustment", ""),
                "tp_adjustment": teaching.get("tp_adjustment", ""),
                "signal_adjustments": teaching.get("signal_adjustments", {}),
                "entry_quality": teaching.get("entry_quality", 0),
                "exit_quality": teaching.get("exit_quality", 0),
                "reasoning": teaching.get("reasoning_summary", ""),
                # Trade context
                "symbol": trade_data.get("symbol", ""),
                "direction": trade_data.get("direction", 0),
                "pnl": trade_data.get("pnl_usd", 0),
                "exit_reason": trade_data.get("exit_reason", ""),
                "entry_regime": trade_data.get("entry_regime", ""),
            }
            self.lessons.append(lesson)

            # Update stats
            self.stats["total_lessons"] += 1
            if not self.stats["first_lesson"]:
                self.stats["first_lesson"] = time.time()

            # Update pattern library
            pattern = teaching.get("pattern_name", "")
            if pattern and pattern.lower() not in ("", "none", "n/a", "unknown"):
                self._update_pattern(pattern, trade_data.get("pnl_usd", 0) > 0, lesson)

            # Update rules
            for rule_type in ("avoid_rule", "repeat_rule"):
                rule_text = teaching.get(rule_type)
                if rule_text and str(rule_text).lower() not in ("null", "none", "n/a", ""):
                    self._add_rule(rule_type.replace("_rule", ""), str(rule_text), lesson)

            # Update signal log
            for sig_name, adj in teaching.get("signal_adjustments", {}).items():
                if isinstance(adj, (int, float)):
                    self.signal_log[sig_name].append({
                        "time": time.time(),
                        "adjustment": adj,
                        "grade": teaching.get("trade_grade", "?"),
                        "pnl": trade_data.get("pnl_usd", 0),
                    })

            # Auto-distill every 25 lessons
            if self.stats["total_lessons"] % 25 == 0:
                self._distill_wisdom()

            self._save_all()

    def _update_pattern(self, name: str, is_win: bool, lesson: dict):
        """Track a named pattern with win/loss stats."""
        key = name.lower().strip()
        if key not in self.patterns:
            self.patterns[key] = {
                "name": name,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0.0,
                "examples": [],
                "first_seen": time.time(),
                "last_seen": time.time(),
            }
        p = self.patterns[key]
        if is_win:
            p["wins"] += 1
        else:
            p["losses"] += 1
        p["total_pnl"] += lesson.get("pnl", 0)
        p["last_seen"] = time.time()
        # Keep last 5 examples
        p["examples"].append({
            "id": lesson["id"],
            "grade": lesson["grade"],
            "pnl": lesson["pnl"],
            "regime": lesson.get("entry_regime", ""),
        })
        if len(p["examples"]) > 5:
            p["examples"] = p["examples"][-5:]

    def _add_rule(self, rule_type: str, rule_text: str, lesson: dict):
        """Add or reinforce an avoid/repeat rule."""
        # Check if similar rule exists (simple fuzzy match)
        for existing in self.rules:
            if existing["rule"].lower() == rule_text.lower():
                existing["hits"] += 1
                existing["last_hit"] = time.time()
                existing["last_pnl"] = lesson.get("pnl", 0)
                return

        self.rules.append({
            "type": rule_type,
            "rule": rule_text,
            "hits": 1,
            "created": time.time(),
            "last_hit": time.time(),
            "last_pnl": lesson.get("pnl", 0),
            "from_lesson": lesson["id"],
        })

    # ── Store coaching session ───────────────────────────────────────────────

    def store_coaching(self, coaching: dict):
        """Store a coaching session — NEVER deleted."""
        with self._lock:
            entry = {
                "id": self.stats["total_coaching"] + 1,
                "timestamp": time.time(),
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
                "coaching_summary": coaching.get("coaching_summary", ""),
                "signal_adjustments": coaching.get("signal_adjustments", {}),
            }
            self.coaching.append(entry)
            self.stats["total_coaching"] += 1

            # Add coaching rules to the rules list
            for rule in coaching.get("risk_rules", []):
                if rule:
                    self._add_rule("risk", str(rule), {"id": f"coaching-{entry['id']}", "pnl": 0})
            for rule in coaching.get("timing_rules", []):
                if rule:
                    self._add_rule("timing", str(rule), {"id": f"coaching-{entry['id']}", "pnl": 0})

            self._save_all()

    # ── Wisdom distillation ──────────────────────────────────────────────────

    def _distill_wisdom(self):
        """Compress recent lessons into distilled wisdom entries."""
        recent = self.lessons[-25:]
        if not recent:
            return

        # Aggregate grades
        grades = [l["grade"] for l in recent if l.get("grade")]
        wins = sum(1 for l in recent if l.get("pnl", 0) > 0)
        losses = len(recent) - wins

        # Collect recurring lessons
        all_lessons = []
        for l in recent:
            all_lessons.extend(l.get("lessons", []))

        # Collect best practices
        all_practices = []
        for l in recent:
            all_practices.extend(l.get("best_practices", []))

        # Most common focus areas
        focus_areas = [l.get("focus_area", "") for l in recent if l.get("focus_area")]

        # Signal adjustment trends
        sig_trends = defaultdict(list)
        for l in recent:
            for sig, adj in l.get("signal_adjustments", {}).items():
                if isinstance(adj, (int, float)):
                    sig_trends[sig].append(adj)

        avg_adjustments = {sig: round(sum(vals) / len(vals), 3) for sig, vals in sig_trends.items() if vals}

        entry = {
            "timestamp": time.time(),
            "lesson_range": f"{recent[0]['id']}-{recent[-1]['id']}",
            "win_rate": round(wins / max(wins + losses, 1) * 100, 1),
            "grade_distribution": {g: grades.count(g) for g in set(grades)},
            "top_lessons": list(set(all_lessons))[:10],
            "top_practices": list(set(all_practices))[:5],
            "focus_areas": list(set(focus_areas))[:3],
            "signal_trends": avg_adjustments,
            "total_pnl": round(sum(l.get("pnl", 0) for l in recent), 2),
        }
        self.wisdom.append(entry)
        logger.info(f"UltraMemory: distilled wisdom from lessons {entry['lesson_range']} (WR: {entry['win_rate']}%)")

    # ── Query methods ────────────────────────────────────────────────────────

    def get_context_for_teaching(self, symbol: str = "", regime: str = "") -> str:
        """Build a context string from memory to feed into UltraThink prompts.
        This is how the bot remembers everything it learned."""
        parts = []

        # Recent wisdom
        if self.wisdom:
            latest = self.wisdom[-1]
            parts.append(f"ACCUMULATED WISDOM (from {self.stats['total_lessons']} trades):")
            parts.append(f"  Recent win rate: {latest.get('win_rate', '?')}%")
            if latest.get("top_lessons"):
                parts.append("  Key lessons learned:")
                for l in latest["top_lessons"][:5]:
                    parts.append(f"    - {l}")
            if latest.get("signal_trends"):
                parts.append("  Signal adjustment trends:")
                for sig, adj in latest["signal_trends"].items():
                    direction = "boost" if adj > 0 else "reduce"
                    parts.append(f"    - {sig}: {direction} ({adj:+.3f} avg)")

        # Top patterns for this regime
        relevant_patterns = []
        for key, p in self.patterns.items():
            total = p["wins"] + p["losses"]
            if total >= 2:
                wr = p["wins"] / total * 100
                relevant_patterns.append((key, p, wr, total))

        if relevant_patterns:
            relevant_patterns.sort(key=lambda x: x[3], reverse=True)
            parts.append(f"\nKNOWN PATTERNS ({len(relevant_patterns)} total):")
            for key, p, wr, total in relevant_patterns[:10]:
                parts.append(f"  - {p['name']}: {wr:.0f}% WR ({total} trades, ${p['total_pnl']:+.2f})")

        # Active rules (sorted by hits — most reinforced first)
        active_rules = sorted(self.rules, key=lambda r: r.get("hits", 0), reverse=True)
        if active_rules:
            avoid_rules = [r for r in active_rules if r["type"] == "avoid"][:5]
            repeat_rules = [r for r in active_rules if r["type"] == "repeat"][:5]
            risk_rules = [r for r in active_rules if r["type"] == "risk"][:3]

            if avoid_rules:
                parts.append("\nRULES TO AVOID (learned from losses):")
                for r in avoid_rules:
                    parts.append(f"  - {r['rule']} (confirmed {r['hits']}x)")
            if repeat_rules:
                parts.append("\nRULES TO REPEAT (learned from wins):")
                for r in repeat_rules:
                    parts.append(f"  - {r['rule']} (confirmed {r['hits']}x)")
            if risk_rules:
                parts.append("\nRISK RULES:")
                for r in risk_rules:
                    parts.append(f"  - {r['rule']}")

        # Latest coaching focus
        if self.coaching:
            latest_c = self.coaching[-1]
            parts.append(f"\nLATEST COACHING FOCUS: {latest_c.get('focus_this_week', 'N/A')}")
            parts.append(f"  Strongest edge: {latest_c.get('strongest_edge', 'N/A')}")
            parts.append(f"  Biggest weakness: {latest_c.get('biggest_weakness', 'N/A')}")

            # Regime strategies
            regime_strats = latest_c.get("regime_strategies", {})
            if regime and regime in regime_strats:
                parts.append(f"  Strategy for {regime}: {regime_strats[regime]}")

        return "\n".join(parts) if parts else ""

    def get_recent_lessons(self, limit: int = 50) -> list[dict]:
        return self.lessons[-limit:]

    def get_patterns(self) -> dict:
        return self.patterns

    def get_rules(self) -> list[dict]:
        return sorted(self.rules, key=lambda r: r.get("hits", 0), reverse=True)

    def get_signal_history(self, signal_name: str = None) -> dict:
        if signal_name:
            return {signal_name: self.signal_log.get(signal_name, [])}
        return dict(self.signal_log)

    def get_coaching_sessions(self, limit: int = 10) -> list[dict]:
        return self.coaching[-limit:]

    def get_wisdom(self) -> list[dict]:
        return self.wisdom

    def get_full_stats(self) -> dict:
        """Full memory statistics for the dashboard."""
        total_wins = sum(1 for l in self.lessons if l.get("pnl", 0) > 0)
        total_losses = len(self.lessons) - total_wins
        grade_dist = defaultdict(int)
        for l in self.lessons:
            grade_dist[l.get("grade", "?")] += 1

        return {
            "total_lessons": self.stats["total_lessons"],
            "total_coaching": self.stats["total_coaching"],
            "total_patterns": len(self.patterns),
            "total_rules": len(self.rules),
            "total_wisdom": len(self.wisdom),
            "total_signal_tracked": len(self.signal_log),
            "win_rate": round(total_wins / max(total_wins + total_losses, 1) * 100, 1),
            "grade_distribution": dict(grade_dist),
            "first_lesson": self.stats.get("first_lesson", 0),
            "memory_size_kb": self._get_memory_size_kb(),
            "top_patterns": self._get_top_patterns(5),
            "top_rules": self.get_rules()[:5],
        }

    def _get_memory_size_kb(self) -> float:
        total = 0
        for fname in os.listdir(self._dir):
            fpath = os.path.join(self._dir, fname)
            if os.path.isfile(fpath):
                total += os.path.getsize(fpath)
        return round(total / 1024, 1)

    def _get_top_patterns(self, limit: int) -> list[dict]:
        result = []
        for key, p in self.patterns.items():
            total = p["wins"] + p["losses"]
            if total < 2:
                continue
            result.append({
                "name": p["name"],
                "wins": p["wins"],
                "losses": p["losses"],
                "win_rate": round(p["wins"] / total * 100, 1),
                "total_pnl": round(p["total_pnl"], 2),
                "trades": total,
            })
        result.sort(key=lambda x: x["trades"], reverse=True)
        return result[:limit]


# ── Singleton ────────────────────────────────────────────────────────────────
_memory: UltraMemory | None = None


def get_ultra_memory() -> UltraMemory:
    global _memory
    if _memory is None:
        _memory = UltraMemory()
    return _memory
