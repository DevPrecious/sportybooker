"""Turn raw SportyBet events into candidate picks and assemble a slip that fits the odds spec."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .client import Selection


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
    category: str = ""  # country / region, e.g. "England", "International Clubs"
    sport_id: str = ""
    home_id: str = ""  # sr:competitor:...
    away_id: str = ""
    # Filled in by booker.edge when --edge is on
    market_prob: float | None = None  # de-vigged bookmaker probability
    model_prob: float | None = None  # stats-model probability
    edge: float | None = None  # blended_prob * odds - 1
    form_home: str = ""
    form_away: str = ""

    @property
    def score(self) -> float:
        """Ranking key: edge when rated, otherwise a small nudge from bookmaker probability."""
        return self.edge if self.edge is not None else (self.probability - 1.0)

    @property
    def win_prob(self) -> float:
        """Best available estimate of this pick winning: blended model/market when rated, else bookmaker."""
        if self.edge is not None:
            return (self.edge + 1.0) / self.odds
        return self.market_prob if self.market_prob is not None else self.probability

    @property
    def league_label(self) -> str:
        """Tournament name, prefixed with the country when the name alone is too generic ("Cup")."""
        if len(self.tournament) <= 10 and self.category:
            return f"{self.category} {self.tournament}"
        return self.tournament

    @property
    def fixture(self) -> str:
        return f"{self.home} vs {self.away}"

    @property
    def market_label(self) -> str:
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
    tolerance: float = 0.10  # +/- fraction around total_odds that counts as a hit
    min_legs: int = 1
    max_legs: int = 10
    outcome_filter: str | None = None  # case-insensitive substring on outcome desc, e.g. "over 1.5"
    hours_ahead: float | None = 48.0
    window_ms: tuple[int, int] | None = None  # absolute [start, end) kick-off window; overrides hours_ahead
    leagues: tuple[str, ...] | None = None  # lowercase substrings matched against "<country> <tournament>"
    include_srl: bool = False  # keep Simulated Reality League (computer-generated) fixtures
    excludes: tuple[str, ...] | None = None  # reject fixtures whose competition or team names contain any of these
    min_book_prob: float | None = None  # reject outcomes the bookmaker rates below this probability
    strategy: str = "safest"  # safest | edge | random

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
        if self.strategy not in ("safest", "edge", "rollover", "random"):
            raise ValueError("--strategy must be 'safest', 'edge', 'rollover' or 'random'")
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
    """(category, tournament) names, e.g. ("England", "Premier League")."""
    sport = ev.get("sport") or {}
    category = sport.get("category") or {}
    tournament = ev.get("tournamentName") or (category.get("tournament") or {}).get("name") or ""
    return str(category.get("name") or ""), str(tournament)


def is_simulated(category: str, tournament: str) -> bool:
    """SportyBet's 'Simulated Reality League' fixtures are computer-generated, not real matches."""
    text = f"{category} {tournament}".lower()
    return "simulated" in text or tournament.rstrip().lower().endswith(" srl")


def league_matches(category: str, tournament: str, leagues: tuple[str, ...] | None, include_srl: bool = False) -> bool:
    haystack = f"{category} {tournament}".lower()
    if not include_srl and is_simulated(category, tournament):
        # Only let SRL through when the user asked for it by name.
        if not leagues or not any("srl" in n or "simulated" in n for n in leagues):
            return False
    if not leagues:
        return True
    return any(needle in haystack for needle in leagues)


def event_in_window(ev: dict[str, Any], spec: OddsSpec, now_ms: int) -> bool:
    """Not started, and kicking off inside the spec's day window or hours horizon."""
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
    """Flatten events -> active outcomes whose odds fall inside the spec's per-pick range."""
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
        for m in ev.get("markets") or []:
            if m.get("status") not in (0, None) or m.get("banned"):
                continue
            for o in m.get("outcomes") or []:
                if o.get("isActive") != 1:
                    continue
                try:
                    odds = float(o["odds"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (spec.min_odds <= odds <= spec.max_odds):
                    continue
                desc = str(o.get("desc") or "")
                if needle and needle not in desc.lower():
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
                        start_ms=start,
                        category=category,
                        market_id=str(m.get("id")),
                        market_name=str(m.get("name") or m.get("desc") or m.get("id")),
                        specifier=str(m.get("specifier") or ""),
                        outcome_id=str(o.get("id")),
                        outcome_desc=desc,
                        odds=odds,
                        probability=prob or (1.0 / odds),
                        sport_id=str(sport.get("id") or ""),
                        home_id=str(ev.get("homeTeamId") or ""),
                        away_id=str(ev.get("awayTeamId") or ""),
                    )
                )
    return out


def one_per_event(cands: list[Candidate], strategy: str, rng: random.Random) -> list[Candidate]:
    """A slip can't hold two picks on the same match, so keep the best pick per event.

    strategy: 'safest' (highest bookmaker probability), 'edge' (highest model edge),
    'rollover' (highest blended win probability), or 'random'.
    """
    by_event: dict[str, list[Candidate]] = {}
    for c in cands:
        by_event.setdefault(c.event_id, []).append(c)
    chosen: list[Candidate] = []
    for group in by_event.values():
        if strategy == "random":
            chosen.append(rng.choice(group))
        elif strategy == "edge":
            chosen.append(max(group, key=lambda c: (c.score, c.probability)))
        elif strategy == "rollover":
            chosen.append(max(group, key=lambda c: (c.win_prob, c.score)))
        else:
            chosen.append(max(group, key=lambda c: (c.probability, -c.odds)))
    if strategy == "random":
        rng.shuffle(chosen)
    elif strategy == "edge":
        chosen.sort(key=lambda c: (-c.score, -c.probability, c.start_ms))
    elif strategy == "rollover":
        chosen.sort(key=lambda c: (-c.win_prob, -c.score, c.start_ms))
    else:
        chosen.sort(key=lambda c: (-c.probability, c.start_ms))
    return chosen


def total_odds(picks: Iterable[Candidate]) -> float:
    return math.prod(c.odds for c in picks) if picks else 0.0


def build_slip(
    cands: list[Candidate],
    spec: OddsSpec,
    search_budget: int = 250_000,
    score_fn: Callable[[Candidate], float] | None = None,
) -> list[Candidate]:
    """Pick up to max_legs candidates (already one-per-event) whose product lands nearest total_odds.

    Without a target, this just takes the first max_legs picks in the given order.
    With a target, a bounded depth-first search in log space explores combinations; every leg has
    odds > 1 so the accumulated product only ever grows, which lets us prune overshoots early.
    Combinations inside the tolerance band beat any outside it, and among those the highest summed
    `score_fn` wins (default: edge); outside the band the closest product wins.
    """
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
    best_key: tuple[int, float] = (2, math.inf)  # (0=in band, 1=out of band), then -score or error
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
        # Target sits above anything reachable: the closest we can get is the biggest legal product.
        top = sorted(range(len(cands)), key=lambda i: -logs[i])[: spec.max_legs]
        best = sorted(top)
    return [cands[i] for i in best]
