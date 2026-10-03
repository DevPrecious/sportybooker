"""Rate Win/Draw (1X/X2) and Handicap picks using team form, head-to-head, and league table motivation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from .motivation import (
    LeagueTable,
    TeamMotivation,
    evaluate_team_motivation,
    parse_season_tables,
    select_table_for_fixture,
)
from .selector import Candidate, parse_asian_handicap, parse_euro_handicap
from .stats import MatchResult, StatsClient, TeamStats, sr_numeric_id

FOOTBALL_SPORTS = {"sr:sport:1", "sr:sport:137", "sr:sport:29"}
CALIBRATION_FILE = "motivated_calibration.json"


@dataclass(frozen=True)
class EdgeConfig:
    form_games: int = 10
    model_weight: float = 0.45
    home_advantage: float = 1.10
    decay: float = 0.85
    h2h_weight: float = 0.20
    motivation_boost: float = 0.12  # xG multiplier shift per unit of motivation gap (mot_home - mot_away)
    min_matches: int = 4
    min_table_games: int = 2
    max_goals: int = 10
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


def load_calibration(cache_dir: Path | None, group: str = "football") -> tuple[float, float] | None:
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


def _weighted_mean(matches: Iterable[MatchResult], fn: Callable[[MatchResult], float], decay: float) -> float | None:
    num = den = 0.0
    w = 1.0
    for m in matches:
        num += fn(m) * w
        den += w
        w *= decay
    return num / den if den else None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _pois(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


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
        total = home + draw + away
        return {
            "home": home / total,
            "draw": draw / total,
            "away": away / total,
        }

    def double_chance(self, side: str) -> float:
        """Win or Draw probability for 'home' (1X) or 'away' (X2)."""
        o = self.outcomes()
        return (o["home"] + o["draw"]) if side == "home" else (o["draw"] + o["away"])

    def euro_handicap(self, side: str, home_hcp: int, away_hcp: int) -> float:
        """3-Way European Handicap win probability after adding `home_hcp` to Home and `away_hcp` to Away."""
        m = self.matrix()
        n = self.max_goals + 1
        total = sum(m[i][j] for i in range(n) for j in range(n))
        if total <= 0:
            return 0.5
        if side == "home":
            won = sum(m[i][j] for i in range(n) for j in range(n) if (i + home_hcp) > (j + away_hcp))
        else:
            won = sum(m[i][j] for i in range(n) for j in range(n) if (j + away_hcp) > (i + home_hcp))
        return won / total

    def _asian_single_line(self, side: str, team_hcp: float) -> float:
        """Win probability for a half-goal or whole-goal Asian Handicap line (whole-goal pushes excluded)."""
        m = self.matrix()
        n = self.max_goals + 1
        won = lost = 0.0
        for i in range(n):
            for j in range(n):
                diff = (i - j + team_hcp) if side == "home" else (j - i + team_hcp)
                if diff > 1e-9:
                    won += m[i][j]
                elif diff < -1e-9:
                    lost += m[i][j]
        z = won + lost
        return won / z if z > 0 else 0.5

    def asian_handicap(self, side: str, team_hcp: float) -> float:
        """Asian Handicap win probability for `side` given `team_hcp` (supports .0, .25, .5, .75 lines)."""
        rem = abs(team_hcp) % 1.0
        if abs(rem - 0.25) < 1e-6 or abs(rem - 0.75) < 1e-6:
            return 0.5 * (
                self._asian_single_line(side, team_hcp - 0.25)
                + self._asian_single_line(side, team_hcp + 0.25)
            )
        return self._asian_single_line(side, team_hcp)


def fit_goal_model(
    home: TeamStats,
    away: TeamStats,
    cfg: EdgeConfig,
    mot_home: TeamMotivation | None = None,
    mot_away: TeamMotivation | None = None,
) -> GoalModel | None:
    """Fit Poisson expected goals from recent form, head-to-head, and table motivation disparity."""
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

    # Apply table motivation adjustment when standings data is available
    if mot_home is not None and mot_away is not None and cfg.motivation_boost > 0:
        delta = mot_home.score - mot_away.score
        lam_h *= 1.0 + cfg.motivation_boost * delta
        lam_a *= 1.0 - cfg.motivation_boost * delta

    return GoalModel(_clamp(lam_h, 0.15, 5.0), _clamp(lam_a, 0.15, 5.0), cfg.max_goals)


def outcome_prob(c: Candidate, model: GoalModel) -> float | None:
    """Price Double Chance (1X/X2), European 3-Way Handicap, and Asian Handicap outcomes."""
    if c.market_id == "10":
        return model.double_chance(c.side)
    if c.market_id == "14":
        pair = parse_euro_handicap(c.specifier, c.outcome_desc)
        if pair is None:
            return None
        return model.euro_handicap(c.side, pair[0], pair[1])
    if c.market_id == "16":
        team_hcp = parse_asian_handicap(c.side, c.specifier, c.outcome_desc)
        if team_hcp is None:
            return None
        return model.asian_handicap(c.side, team_hcp)
    return None


def market_probability(c: Candidate) -> float:
    p = c.probability
    if 0 < p < 1 and p * c.odds <= 1.05:
        return p
    return 1.0 / c.odds


def rate_candidates(
    cands: list[Candidate],
    stats: StatsClient,
    cfg: EdgeConfig,
    max_matches: int | None = 100,
) -> tuple[list[Candidate], dict[str, int]]:
    """Attach form, league table motivation, model probability, and edge to each candidate."""
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

    fetched_stats = stats.fixtures_stats(list(fixtures.values()), n=cfg.form_games)

    # Resolve season_id for each event so we can batch-fetch league tables
    event_seasons: dict[str, int] = {}
    for group in events:
        eid = group[0].event_id
        pair = fixtures.get(eid)
        h_stats, a_stats = fetched_stats.get(pair, (None, None)) if pair else (None, None)
        sid = stats.resolve_season_id(group[0].tournament_id, h_stats, a_stats)
        if sid:
            event_seasons[eid] = sid

    raw_tables_by_season = stats.fetch_season_tables_batch(list(event_seasons.values()))
    parsed_tables: dict[int, list[LeagueTable]] = {
        sid: parse_season_tables(sid, raw) for sid, raw in raw_tables_by_season.items()
    }

    summary = {
        "events": len(events),
        "rated": 0,
        "unrated": 0,
        "with_table": 0,
        "motivated_matches": 0,
    }
    out: list[Candidate] = []
    rated_events: set[str] = set()
    table_events: set[str] = set()
    motivated_events: set[str] = set()

    for group in events:
        eid = group[0].event_id
        pair = fixtures.get(eid)
        home_stats, away_stats = fetched_stats.get(pair, (None, None)) if pair else (None, None)

        mot_home: TeamMotivation | None = None
        mot_away: TeamMotivation | None = None
        table_name = ""
        sid = event_seasons.get(eid)
        if sid and pair and sid in parsed_tables:
            tbl = select_table_for_fixture(parsed_tables[sid], pair[0], pair[1])
            if tbl is not None:
                table_name = tbl.name
                mot_home = evaluate_team_motivation(tbl, pair[0], min_games=cfg.min_table_games)
                mot_away = evaluate_team_motivation(tbl, pair[1], min_games=cfg.min_table_games)
                if mot_home is not None or mot_away is not None:
                    table_events.add(eid)
                if (mot_home and mot_home.is_motivated) or (mot_away and mot_away.is_motivated):
                    motivated_events.add(eid)

        model: GoalModel | None = None
        if home_stats and away_stats:
            model = fit_goal_model(home_stats, away_stats, cfg, mot_home=mot_home, mot_away=mot_away)

        strength = 0.0
        if home_stats and away_stats:
            strength = min(len(home_stats.recent), len(away_stats.recent)) / cfg.form_games
        w = cfg.model_weight * _clamp(strength, 0.0, 1.0)

        for c in group:
            p_mkt = market_probability(c)
            p_model = outcome_prob(c, model) if model is not None else None
            if p_model is None:
                out.append(replace(
                    c,
                    market_prob=p_mkt,
                    form_home=home_stats.form_string if home_stats else "",
                    form_away=away_stats.form_string if away_stats else "",
                    mot_home=mot_home,
                    mot_away=mot_away,
                    table_name=table_name,
                ))
                continue

            p_model = cfg.calibrate(p_model)
            blended = w * p_model + (1 - w) * p_mkt
            # Compare blended win probability to the bookmaker's de-vigged baseline probability p_mkt
            # so SportyBet's 7-9% handicap/DC overround doesn't artificially penalize the (1 - w) * p_mkt prior.
            edge = (blended / p_mkt - 1.0) if p_mkt > 0 else (blended * c.odds - 1.0)
            out.append(replace(
                c,
                market_prob=p_mkt,
                model_prob=p_model,
                blended_prob=blended,
                edge=edge,
                form_home=home_stats.form_string if home_stats else "",
                form_away=away_stats.form_string if away_stats else "",
                mot_home=mot_home,
                mot_away=mot_away,
                table_name=table_name,
            ))
            rated_events.add(eid)

    summary["rated"] = len(rated_events)
    summary["unrated"] = len(events) - len(rated_events)
    summary["with_table"] = len(table_events)
    summary["motivated_matches"] = len(motivated_events)
    return out, summary
