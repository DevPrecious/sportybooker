"""Rate candidate picks against team stats and compute an edge over the bookmaker price.

Football: a recency-weighted Poisson goals model (attack/defence from each side's recent games,
home advantage, head-to-head nudge) prices 1X2, Double Chance, Draw No Bet, Over/Under and BTTS.
Other sports: a recency-weighted form model prices two-way Winner markets.

The model probability is blended with the bookmaker's de-vigged probability (the market is a strong
prior), and edge = blended_prob * odds - 1. Positive edge means the price looks generous.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from .selector import Candidate
from .stats import MatchResult, StatsClient, TeamStats, sr_numeric_id

FOOTBALL_SPORTS = {"sr:sport:1", "sr:sport:137", "sr:sport:29"}  # football, eSoccer, futsal


@dataclass(frozen=True)
class EdgeConfig:
    form_games: int = 10
    model_weight: float = 0.40  # share of the stats model in the blended probability
    home_advantage: float = 1.10  # xG multiplier for the home side (divides the away side)
    decay: float = 0.85  # weight of each older match relative to the next newer one
    h2h_weight: float = 0.20  # how much head-to-head history nudges expected goals / win rate
    min_matches: int = 4  # below this many recent games a team is considered unrated
    max_goals: int = 10
    # Platt-scaling (a, b) fitted by `booker backtest`: p' = sigmoid(a * logit(p) + b). None = raw model.
    calibration: tuple[float, float] | None = None

    def validate(self) -> None:
        if not (0 <= self.model_weight <= 1):
            raise ValueError("--model-weight must be between 0 and 1")
        if self.form_games < 3:
            raise ValueError("--form-games must be at least 3")

    def calibrate(self, p: float) -> float:
        if self.calibration is None:
            return p
        a, b = self.calibration
        p = _clamp(p, 1e-6, 1 - 1e-6)
        z = a * math.log(p / (1 - p)) + b
        return 1 / (1 + math.exp(-z))


CALIBRATION_FILE = "calibration.json"


def load_calibration(cache_dir: Path | None, group: str) -> tuple[float, float] | None:
    """Read the (a, b) saved by `booker backtest --save` for a model group ('football' / 'two_way')."""
    if cache_dir is None:
        return None
    path = cache_dir / CALIBRATION_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = data.get(group)
        return (float(entry["a"]), float(entry["b"])) if entry else None
    except (ValueError, KeyError, TypeError, OSError):
        return None


def model_group(sport_id: str) -> str:
    return "football" if sport_id in FOOTBALL_SPORTS else "two_way"


# ---------------------------------------------------------------------------- helpers
def _weighted_mean(matches: Iterable[MatchResult], fn: Callable[[MatchResult], float], decay: float) -> float | None:
    num = den = 0.0
    w = 1.0
    for m in matches:  # newest first
        num += fn(m) * w
        den += w
        w *= decay
    return num / den if den else None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _pois(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


# ------------------------------------------------------------------- football model
@dataclass(frozen=True)
class GoalModel:
    lam_home: float
    lam_away: float
    max_goals: int = 10

    def matrix(self) -> list[list[float]]:
        ph = [_pois(i, self.lam_home) for i in range(self.max_goals + 1)]
        pa = [_pois(j, self.lam_away) for j in range(self.max_goals + 1)]
        return [[ph[i] * pa[j] for j in range(self.max_goals + 1)] for i in range(self.max_goals + 1)]

    def outcomes(self) -> dict[str, float]:
        m = self.matrix()
        n = self.max_goals + 1
        home = sum(m[i][j] for i in range(n) for j in range(n) if i > j)
        draw = sum(m[i][i] for i in range(n))
        away = sum(m[i][j] for i in range(n) for j in range(n) if i < j)
        total = home + draw + away  # truncation leaves a sliver of mass beyond max_goals
        p0h = sum(m[0][j] for j in range(n))
        p0a = sum(m[i][0] for i in range(n))
        return {
            "home": home / total,
            "draw": draw / total,
            "away": away / total,
            "btts": (1 - p0h - p0a + m[0][0]) / total,
        }

    def over_under(self, line: float) -> tuple[float, float]:
        """P(over), P(under) for a goals line; whole-number lines exclude the push (void) outcome."""
        m = self.matrix()
        n = self.max_goals + 1
        over = under = 0.0
        for i in range(n):
            for j in range(n):
                t = i + j
                if t > line:
                    over += m[i][j]
                elif t < line:
                    under += m[i][j]
        z = over + under
        return (over / z, under / z) if z else (0.5, 0.5)


def fit_goal_model(home: TeamStats, away: TeamStats, cfg: EdgeConfig) -> GoalModel | None:
    if len(home.recent) < cfg.min_matches or len(away.recent) < cfg.min_matches:
        return None
    gf_h = _weighted_mean(home.recent, lambda m: m.goals_for(home.uid), cfg.decay)
    ga_h = _weighted_mean(home.recent, lambda m: m.goals_against(home.uid), cfg.decay)
    gf_a = _weighted_mean(away.recent, lambda m: m.goals_for(away.uid), cfg.decay)
    ga_a = _weighted_mean(away.recent, lambda m: m.goals_against(away.uid), cfg.decay)
    if None in (gf_h, ga_h, gf_a, ga_a):
        return None

    lam_h = ((gf_h + ga_a) / 2) * cfg.home_advantage
    lam_a = ((gf_a + ga_h) / 2) / cfg.home_advantage

    if len(home.h2h) >= 3:
        h2h_h = _weighted_mean(home.h2h, lambda m: m.goals_for(home.uid), cfg.decay)
        h2h_a = _weighted_mean(home.h2h, lambda m: m.goals_for(away.uid), cfg.decay)
        if h2h_h is not None and h2h_a is not None:
            lam_h = (1 - cfg.h2h_weight) * lam_h + cfg.h2h_weight * h2h_h
            lam_a = (1 - cfg.h2h_weight) * lam_a + cfg.h2h_weight * h2h_a

    return GoalModel(_clamp(lam_h, 0.15, 5.0), _clamp(lam_a, 0.15, 5.0), cfg.max_goals)


_OU_RE = re.compile(r"^(over|under)\s+([\d.]+)$", re.I)
_TOTAL_RE = re.compile(r"total=([\d.]+)")


def football_outcome_prob(c: Candidate, model: GoalModel) -> float | None:
    desc = c.outcome_desc.strip().lower()
    o = model.outcomes()
    if c.market_id == "1":
        return {"home": o["home"], "1": o["home"], "draw": o["draw"], "x": o["draw"],
                "away": o["away"], "2": o["away"]}.get(desc)
    if c.market_id == "10":
        if "home" in desc and "draw" in desc:
            return o["home"] + o["draw"]
        if "home" in desc and "away" in desc:
            return o["home"] + o["away"]
        if "draw" in desc and "away" in desc:
            return o["draw"] + o["away"]
        return None
    if c.market_id == "11":
        z = o["home"] + o["away"]
        return {"home": o["home"] / z, "away": o["away"] / z}.get(desc)
    if c.market_id == "18":
        m = _OU_RE.match(desc)
        spec = _TOTAL_RE.search(c.specifier or "")
        line = float(spec.group(1)) if spec else (float(m.group(2)) if m else None)
        if not m or line is None:
            return None
        over, under = model.over_under(line)
        return over if m.group(1).lower() == "over" else under
    if c.market_id == "29":
        return {"yes": o["btts"], "no": 1 - o["btts"]}.get(desc)
    return None


# ------------------------------------------------------------------- two-way model
def form_win_prob(home: TeamStats, away: TeamStats, cfg: EdgeConfig) -> float | None:
    """P(home wins) for sports without draws, from recency-weighted form + head-to-head."""
    if len(home.recent) < cfg.min_matches or len(away.recent) < cfg.min_matches:
        return None
    pts = {"W": 1.0, "D": 0.5, "L": 0.0}
    f_h = _weighted_mean(home.recent, lambda m: pts[m.outcome_for(home.uid)], cfg.decay)
    f_a = _weighted_mean(away.recent, lambda m: pts[m.outcome_for(away.uid)], cfg.decay)
    if f_h is None or f_a is None:
        return None
    p = 0.5 + 0.5 * (f_h - f_a) + 0.04  # modest generic home-court edge
    if len(home.h2h) >= 3:
        h2h = _weighted_mean(home.h2h, lambda m: pts[m.outcome_for(home.uid)], cfg.decay)
        if h2h is not None:
            p = (1 - cfg.h2h_weight) * p + cfg.h2h_weight * h2h
    return _clamp(p, 0.05, 0.95)


def two_way_outcome_prob(c: Candidate, p_home: float) -> float | None:
    if c.market_id not in ("186", "219", "1", "11"):
        return None
    desc = c.outcome_desc.strip().lower()
    return {"home": p_home, "1": p_home, "away": 1 - p_home, "2": 1 - p_home}.get(desc)


# ------------------------------------------------------------------------ rating
def market_probability(c: Candidate) -> float:
    """SportyBet ships a de-vigged `probability` per outcome; fall back to 1/odds if missing."""
    p = c.probability
    if 0 < p < 1 and p * c.odds <= 1.05:
        return p
    return 1.0 / c.odds


def rate_candidates(
    cands: list[Candidate],
    stats: StatsClient,
    cfg: EdgeConfig,
    max_matches: int | None = 60,
) -> tuple[list[Candidate], dict[str, int]]:
    """Attach market/model probabilities and edge to candidates. Returns (rated, summary counts)."""
    cfg.validate()
    by_event: dict[str, list[Candidate]] = {}
    for c in cands:
        by_event.setdefault(c.event_id, []).append(c)

    events = sorted(by_event.values(), key=lambda g: g[0].start_ms)
    if max_matches is not None:
        events = events[:max_matches]

    fixtures: dict[str, tuple[int, int]] = {}
    for group in events:
        h, a = sr_numeric_id(group[0].home_id), sr_numeric_id(group[0].away_id)
        if h and a:
            fixtures[group[0].event_id] = (h, a)

    fetched = stats.fixtures_stats(list(fixtures.values()), n=cfg.form_games)

    summary = {"events": len(events), "rated": 0, "unrated": 0, "no_model_market": 0}
    out: list[Candidate] = []
    rated_events: set[str] = set()
    for group in events:
        pair = fixtures.get(group[0].event_id)
        home_stats, away_stats = fetched.get(pair, (None, None)) if pair else (None, None)
        model: GoalModel | None = None
        p_home: float | None = None
        is_football = group[0].sport_id in FOOTBALL_SPORTS
        if home_stats and away_stats:
            if is_football:
                model = fit_goal_model(home_stats, away_stats, cfg)
            else:
                p_home = form_win_prob(home_stats, away_stats, cfg)
        strength = 0.0
        if home_stats and away_stats:
            strength = min(len(home_stats.recent), len(away_stats.recent)) / cfg.form_games
        w = cfg.model_weight * _clamp(strength, 0.0, 1.0)

        for c in group:
            p_mkt = market_probability(c)
            p_model: float | None = None
            if model is not None:
                p_model = football_outcome_prob(c, model)
            elif p_home is not None:
                p_model = two_way_outcome_prob(c, p_home)

            if p_model is None:
                if model is not None or p_home is not None:
                    summary["no_model_market"] += 1
                out.append(replace(c, market_prob=p_mkt,
                                   form_home=home_stats.form_string if home_stats else "",
                                   form_away=away_stats.form_string if away_stats else ""))
                continue

            p_model = cfg.calibrate(p_model)
            blended = w * p_model + (1 - w) * p_mkt
            out.append(replace(
                c,
                market_prob=p_mkt,
                model_prob=p_model,
                edge=blended * c.odds - 1,
                form_home=home_stats.form_string,
                form_away=away_stats.form_string,
            ))
            rated_events.add(c.event_id)

    summary["rated"] = len(rated_events)
    summary["unrated"] = len(events) - len(rated_events)
    return out, summary
