"""Turn raw SportyBet events into Win/Draw (1X/X2) and Handicap candidate picks and assemble a parlay."""

from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .client import Selection
from .motivation import TeamMotivation
from .sports import ALLOWED_MARKET_IDS

_HCP_PAIR_RE = re.compile(r"hcp=\s*(-?\d+)\s*:\s*(-?\d+)", re.I)
_HCP_FLOAT_RE = re.compile(r"hcp=\s*([+-]?\d+(?:\.\d+)?)", re.I)
_DESC_PAIR_RE = re.compile(r"\(\s*(-?\d+)\s*:\s*(-?\d+)\s*\)")
_DESC_FLOAT_RE = re.compile(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*\)")


def classify_allowed_outcome(market_id: str, outcome_id: str, outcome_desc: str) -> str | None:
    """Return backed side ('home' or 'away') if this outcome is a valid Win/Draw or Handicap pick, else None."""
    if market_id not in ALLOWED_MARKET_IDS:
        return None
    desc = outcome_desc.strip().lower()
    if market_id == "10":
        if outcome_id == "9" or desc in ("home or draw", "1x"):
            return "home"
        if outcome_id == "11" or desc in ("draw or away", "x2"):
            return "away"
        return None
    if market_id == "14":
        if outcome_id == "1711" or desc.startswith("home"):
            return "home"
        if outcome_id == "1713" or desc.startswith("away"):
            return "away"
        return None
    if market_id == "16":
        if outcome_id == "1714" or desc.startswith("home"):
            return "home"
        if outcome_id == "1715" or desc.startswith("away"):
            return "away"
        return None
    return None


def parse_euro_handicap(specifier: str, outcome_desc: str) -> tuple[int, int] | None:
    """Extract (home_hcp, away_hcp) from 'hcp=0:1' or 'Home (0:1)'."""
    m = _HCP_PAIR_RE.search(specifier or "") or _DESC_PAIR_RE.search(outcome_desc or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def parse_asian_handicap(side: str, specifier: str, outcome_desc: str) -> float | None:
    """Extract the handicap applied to `side` ('home' or 'away')."""
    m_desc = _DESC_FLOAT_RE.search(outcome_desc or "")
    if m_desc:
        return float(m_desc.group(1))
    m_spec = _HCP_FLOAT_RE.search(specifier or "")
    if not m_spec:
        return None
    h_home = float(m_spec.group(1))
    return h_home if side == "home" else -h_home


@dataclass(frozen=True)
class Candidate:
    event_id: str
    home: str
    away: str
    tournament: str
    start_ms: int
    market_id: str
    market_name: str
    specifier: str
    outcome_id: str
    outcome_desc: str
    odds: float
    probability: float
    side: str = "home"  # 'home' | 'away'
    category: str = ""
    tournament_id: str = ""  # sr:tournament:...
    sport_id: str = ""
    home_id: str = ""  # sr:competitor:...
    away_id: str = ""
    # Filled in by motivated.edge
    market_prob: float | None = None
    model_prob: float | None = None
    blended_prob: float | None = None
    edge: float | None = None
    form_home: str = ""
    form_away: str = ""
    mot_home: TeamMotivation | None = None
    mot_away: TeamMotivation | None = None
    table_name: str = ""

    @property
    def backed_team(self) -> str:
        return self.home if self.side == "home" else self.away

    @property
    def backed_motivation(self) -> TeamMotivation | None:
        return self.mot_home if self.side == "home" else self.mot_away

    @property
    def opp_motivation(self) -> TeamMotivation | None:
        return self.mot_away if self.side == "home" else self.mot_home

    @property
    def motivation_score(self) -> float:
        bm = self.backed_motivation
        return bm.score if bm is not None else 0.0

    @property
    def motivation_gap(self) -> float:
        bm, om = self.backed_motivation, self.opp_motivation
        if bm is None or om is None:
            return 0.0
        return round(bm.score - om.score, 3)

    @property
    def motivation_zone(self) -> str:
        bm = self.backed_motivation
        return bm.zone if bm is not None else "unknown"

    @property
    def motivation_summary(self) -> str:
        bm, om = self.backed_motivation, self.opp_motivation
        if bm is None and om is None:
            return "-"
        b_lbl = bm.label if bm else "?"
        o_lbl = om.label if om else "?"
        tag = "H" if self.side == "home" else "A"
        return f"[{tag}] {b_lbl} vs {o_lbl}"

    @property
    def score(self) -> float:
        """Ranking key: combines edge and motivation strength when rated."""
        if self.edge is not None:
            return self.edge + 0.04 * self.motivation_score + 0.02 * max(0.0, self.motivation_gap)
        return self.probability - 1.0

    @property
    def win_prob(self) -> float:
        if self.blended_prob is not None:
            return self.blended_prob
        if self.edge is not None and self.market_prob is not None:
            return self.market_prob * (1.0 + self.edge)
        if self.edge is not None:
            return (self.edge + 1.0) / self.odds
        return self.market_prob if self.market_prob is not None else self.probability

    @property
    def league_label(self) -> str:
        if len(self.tournament) <= 10 and self.category:
            return f"{self.category} {self.tournament}"
        return self.tournament

    @property
    def fixture(self) -> str:
        return f"{self.home} vs {self.away}"

    @property
    def market_label(self) -> str:
        if self.market_id == "10":
            return "Win/Draw (DC)"
        if self.market_id == "16":
            line = self.specifier.split("=", 1)[-1] if self.specifier else ""
            return f"Asian Hcp ({line})" if line else "Asian Hcp"
        if self.market_id == "14":
            line = self.specifier.split("=", 1)[-1] if self.specifier else ""
            return f"Euro Hcp ({line})" if line else "Euro Hcp"
        if self.specifier:
            return f"{self.market_name} ({self.specifier.split('=', 1)[-1]})"
        return self.market_name

    def to_selection(self) -> Selection:
        return Selection(self.event_id, self.market_id, self.outcome_id, self.specifier)


@dataclass(frozen=True)
class OddsSpec:
    min_odds: float = 1.01
    max_odds: float = float("inf")
    total_odds: float | None = None
    tolerance: float = 0.10
    min_legs: int = 1
    max_legs: int = 15
    outcome_filter: str | None = None
    hours_ahead: float | None = 48.0
    window_ms: tuple[int, int] | None = None
    leagues: tuple[str, ...] | None = None
    include_srl: bool = False
    excludes: tuple[str, ...] | None = None
    min_book_prob: float | None = None
    strategy: str = "rollover"  # safest | edge | rollover | motivation | random

    def validate(self) -> None:
        if self.min_odds < 1.0:
            raise ValueError("--min-odds must be >= 1.0")
        if self.max_odds <= self.min_odds:
            raise ValueError("--max-odds must be greater than --min-odds")
        if self.total_odds is not None and self.total_odds <= 1.0:
            raise ValueError("--total-odds must be > 1.0")
        if not (0 < self.tolerance < 1):
            raise ValueError("--tolerance must be between 0 and 1 (e.g. 0.1 for 10%)")
        if self.min_legs < 1 or self.max_legs < self.min_legs:
            raise ValueError("--legs range is invalid")
        if self.strategy not in ("safest", "edge", "rollover", "motivation", "random"):
            raise ValueError("--strategy must be 'safest', 'edge', 'rollover', 'motivation' or 'random'")
        if self.total_odds is not None and self.max_odds != float("inf"):
            lo_band, hi_band = self.total_odds * (1 - self.tolerance), self.total_odds * (1 + self.tolerance)
            reach_hi = self.max_odds ** self.max_legs
            reach_lo = self.min_odds ** self.min_legs
            if reach_hi < lo_band:
                need = math.ceil(math.log(lo_band) / math.log(self.max_odds))
                raise ValueError(
                    f"{self.max_legs} legs at max odds {self.max_odds:.2f} can only reach {reach_hi:.2f}, "
                    f"below the {lo_band:.2f}-{hi_band:.2f} band for {self.total_odds:g}x. "
                    f"Allow at least {need} legs (--legs) or raise --max-odds to "
                    f"{math.exp(math.log(lo_band) / self.max_legs):.2f}+."
                )
            if reach_lo > hi_band:
                raise ValueError(
                    f"{self.min_legs} legs at min odds {self.min_odds:.2f} already exceed {hi_band:.2f}; "
                    f"lower --legs or --min-odds for a {self.total_odds:g}x target."
                )


def event_league(ev: dict[str, Any]) -> tuple[str, str]:
    sport = ev.get("sport") or {}
    category = sport.get("category") or {}
    tournament = ev.get("tournamentName") or (category.get("tournament") or {}).get("name") or ""
    return str(category.get("name") or ""), str(tournament)


def event_tournament_id(ev: dict[str, Any]) -> str:
    if ev.get("tournamentId"):
        return str(ev["tournamentId"])
    sport = ev.get("sport") or {}
    category = sport.get("category") or {}
    tournament = category.get("tournament") or {}
    return str(tournament.get("id") or "")


def is_simulated(category: str, tournament: str) -> bool:
    text = f"{category} {tournament}".lower()
    return "simulated" in text or tournament.rstrip().lower().endswith(" srl")


def league_matches(category: str, tournament: str, leagues: tuple[str, ...] | None, include_srl: bool = False) -> bool:
    haystack = f"{category} {tournament}".lower()
    if not include_srl and is_simulated(category, tournament):
        if not leagues or not any("srl" in n or "simulated" in n for n in leagues):
            return False
    if not leagues:
        return True
    return any(needle in haystack for needle in leagues)


def event_in_window(ev: dict[str, Any], spec: OddsSpec, now_ms: int) -> bool:
    start = int(ev.get("estimateStartTime") or 0)
    if ev.get("status") not in (0, None) or start <= now_ms:
        return False
    if spec.window_ms is not None:
        earliest, horizon = spec.window_ms
        return earliest <= start < horizon
    if spec.hours_ahead is not None:
        return start <= now_ms + int(spec.hours_ahead * 3600 * 1000)
    return True


def extract_candidates(events: Iterable[dict[str, Any]], spec: OddsSpec, now_ms: int | None = None) -> list[Candidate]:
    """Flatten events -> active Win/Draw (1X/X2) and Handicap outcomes inside the spec's per-pick odds range."""
    now_ms = now_ms or int(time.time() * 1000)
    needle = spec.outcome_filter.lower() if spec.outcome_filter else None
    out: list[Candidate] = []

    for ev in events:
        if not event_in_window(ev, spec, now_ms):
            continue
        category, tournament = event_league(ev)
        if not league_matches(category, tournament, spec.leagues, spec.include_srl):
            continue
        if spec.excludes:
            names = f"{category} {tournament} {ev.get('homeTeamName', '')} {ev.get('awayTeamName', '')}".lower()
            if any(x in names for x in spec.excludes):
                continue
        start = int(ev.get("estimateStartTime") or 0)
        sport = ev.get("sport") or {}
        tour_id = event_tournament_id(ev)
        for m in ev.get("markets") or []:
            if m.get("status") not in (0, None) or m.get("banned"):
                continue
            mid = str(m.get("id") or "")
            if mid not in ALLOWED_MARKET_IDS:
                continue
            specifier = str(m.get("specifier") or "")
            for o in m.get("outcomes") or []:
                if o.get("isActive") != 1:
                    continue
                oid = str(o.get("id") or "")
                desc = str(o.get("desc") or "")
                side = classify_allowed_outcome(mid, oid, desc)
                if side is None:
                    continue
                try:
                    odds = float(o["odds"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (spec.min_odds <= odds <= spec.max_odds):
                    continue
                if needle and needle not in desc.lower() and needle not in specifier.lower():
                    continue
                try:
                    prob = float(o.get("probability") or 0)
                except (TypeError, ValueError):
                    prob = 0.0
                if spec.min_book_prob is not None and (prob or 1.0 / odds) < spec.min_book_prob:
                    continue
                out.append(
                    Candidate(
                        event_id=ev["eventId"],
                        home=ev.get("homeTeamName", "?"),
                        away=ev.get("awayTeamName", "?"),
                        tournament=tournament,
                        tournament_id=tour_id,
                        start_ms=start,
                        category=category,
                        market_id=mid,
                        market_name=str(m.get("name") or m.get("desc") or mid),
                        specifier=specifier,
                        outcome_id=oid,
                        outcome_desc=desc,
                        odds=odds,
                        probability=prob or (1.0 / odds),
                        side=side,
                        sport_id=str(sport.get("id") or ""),
                        home_id=str(ev.get("homeTeamId") or ""),
                        away_id=str(ev.get("awayTeamId") or ""),
                    )
                )
    return out


def _pick_for_group(group: list[Candidate], strategy: str, rng: random.Random) -> Candidate:
    if strategy == "random":
        return rng.choice(group)
    if strategy == "motivation":
        return max(group, key=lambda c: (c.motivation_score, c.motivation_gap, c.win_prob, c.score))
    if strategy == "edge":
        return max(group, key=lambda c: (c.score, c.motivation_score, c.probability))
    if strategy == "rollover":
        return max(group, key=lambda c: (c.win_prob + 0.05 * c.motivation_score, c.score))
    return max(group, key=lambda c: (c.probability, c.motivation_score, -c.odds))


def _sort_chosen(chosen: list[Candidate], strategy: str, rng: random.Random) -> None:
    if strategy == "random":
        rng.shuffle(chosen)
    elif strategy == "motivation":
        chosen.sort(key=lambda c: (-c.motivation_score, -c.motivation_gap, -c.win_prob, c.start_ms))
    elif strategy == "edge":
        chosen.sort(key=lambda c: (-c.score, -c.motivation_score, -c.probability, c.start_ms))
    elif strategy == "rollover":
        chosen.sort(key=lambda c: (-(c.win_prob + 0.05 * c.motivation_score), -c.score, c.start_ms))
    else:
        chosen.sort(key=lambda c: (-c.probability, -c.motivation_score, c.start_ms))


def one_per_event(
    cands: list[Candidate],
    strategy: str,
    rng: random.Random,
    spec: OddsSpec | None = None,
) -> list[Candidate]:
    """Keep the single best pick per match according to `strategy`.

    When `spec.total_odds` is set (e.g. a 40x parlay) and each match has multiple handicap/DC lines
    (like +2.5 at 1.19, +1.5 at 1.35, 1X at 1.50, -0.5 at 1.75), picking the 1.19 line on every
    match can leave the entire pool unable to multiply up to `total_odds`. In that case, we shift
    per-match selection toward the per-leg odds needed to reach `total_odds`.
    """
    by_event: dict[str, list[Candidate]] = {}
    for c in cands:
        by_event.setdefault(c.event_id, []).append(c)

    chosen = [_pick_for_group(group, strategy, rng) for group in by_event.values()]
    _sort_chosen(chosen, strategy, rng)

    if spec is not None and spec.total_odds is not None and chosen and strategy != "random":
        lo_band = spec.total_odds * (1.0 - spec.tolerance)
        usable = min(len(chosen), spec.max_legs)
        top_reach = math.prod(sorted((c.odds for c in chosen), reverse=True)[:usable])
        if top_reach < lo_band and usable > 0:
            # The safest/default lines are too short (e.g. 1.20 +2.5 handicaps) to reach the target.
            # Aim for per-leg odds that can reach `total_odds` within `usable` legs.
            req_log = math.log(spec.total_odds) / usable
            rechosen: list[Candidate] = []
            for group in by_event.values():
                best_c = max(
                    group,
                    key=lambda c: (
                        # Prefer picks at or near the required per-leg odds while respecting win_prob & score
                        -abs(math.log(c.odds) - req_log) + 0.35 * c.win_prob + 0.25 * c.score
                    ),
                )
                rechosen.append(best_c)
            chosen = rechosen
            _sort_chosen(chosen, strategy, rng)

    return chosen


def total_odds(picks: Iterable[Candidate]) -> float:
    return math.prod(c.odds for c in picks) if picks else 0.0


def build_slip(
    cands: list[Candidate],
    spec: OddsSpec,
    search_budget: int = 250_000,
    score_fn: Callable[[Candidate], float] | None = None,
) -> list[Candidate]:
    """Pick up to max_legs candidates (already one-per-event) whose product lands nearest total_odds."""
    if not cands:
        return []
    if spec.total_odds is None:
        return cands[:spec.max_legs]

    score_fn = score_fn or (lambda c: c.score)
    target = math.log(spec.total_odds)
    tol = math.log(1 + spec.tolerance)
    logs = [math.log(c.odds) for c in cands]
    scores = [score_fn(c) for c in cands]
    max_log = max(logs)

    best: list[int] = []
    best_key: tuple[int, float] = (2, math.inf)
    budget = search_budget

    def dfs(start: int, chosen: list[int], acc: float, acc_score: float) -> None:
        nonlocal best, best_key, budget
        if budget <= 0:
            return
        budget -= 1
        if len(chosen) >= spec.min_legs:
            err = abs(acc - target)
            key = (0, -acc_score) if err <= tol else (1, err)
            if key < best_key:
                best_key, best = key, list(chosen)
        if len(chosen) == spec.max_legs or acc > target + tol:
            return
        remaining = spec.max_legs - len(chosen)
        if acc + remaining * max_log < target - tol:
            return
        for j in range(start, len(cands)):
            chosen.append(j)
            dfs(j + 1, chosen, acc + logs[j], acc_score + scores[j])
            chosen.pop()
            if budget <= 0:
                return

    dfs(0, [], 0.0, 0.0)
    if not best:
        top = sorted(range(len(cands)), key=lambda i: -logs[i])[: spec.max_legs]
        best = sorted(top)
    return [cands[i] for i in best]
