"""`motivated backtest`: replay finished matches on Win/Draw (1X/X2) and Handicap markets."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .client import REGIONS, SportyBetClient, SportyBetError
from .edge import CALIBRATION_FILE, EdgeConfig, fit_goal_model
from .selector import OddsSpec, event_in_window, event_league, league_matches
from .sports import resolve_excludes, resolve_leagues, resolve_markets, resolve_sport
from .stats import MatchResult, StatsClient, TeamStats, sr_numeric_id

BACKTEST_MARKETS = (
    "1X (Home/Draw)",
    "X2 (Draw/Away)",
    "AH Home -0.5",
    "AH Away +0.5",
    "AH Home +0.5",
    "AH Away -0.5",
    "AH Home -1.5",
    "AH Away +1.5",
    "AH Home +1.5",
    "AH Away -1.5",
    "EH Home (1:0)",
    "EH Away (0:1)",
    "EH Home (0:1)",
    "EH Away (1:0)",
)


@dataclass(frozen=True)
class Sample:
    match_id: int
    uts: int
    market: str
    p: float
    won: bool


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motivated backtest",
        description="Replay recent finished matches on Win/Draw (1X/X2) and Handicap lines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sport", "-s", default="football")
    p.add_argument("--region", "-r", default="ng", choices=REGIONS)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--teams", type=int, default=160)
    p.add_argument("--samples", type=int, default=600)
    p.add_argument("--league", "-L", default=None)
    p.add_argument("--tier", default=None)
    p.add_argument("--exclude", "-X", default="default")
    p.add_argument("--form-games", type=int, default=10)
    p.add_argument("--no-h2h", action="store_true")
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--json", action="store_true")
    return p


def discover_teams(client: SportyBetClient, sport_id: str, spec: OddsSpec, max_teams: int, max_pages: int = 15) -> list[int]:
    now_ms = int(time.time() * 1000)
    uids: dict[int, None] = {}
    for ev in client.iter_upcoming_events(sport_id, resolve_markets(None, sport_id)[:1], max_pages=max_pages):
        if not event_in_window(ev, spec, now_ms):
            continue
        cat, tour = event_league(ev)
        if not league_matches(cat, tour, spec.leagues, spec.include_srl):
            continue
        if spec.excludes:
            names = f"{cat} {tour} {ev.get('homeTeamName', '')} {ev.get('awayTeamName', '')}".lower()
            if any(x in names for x in spec.excludes):
                continue
        for key in ("homeTeamId", "awayTeamId"):
            uid = sr_numeric_id(str(ev.get(key) or ""))
            if uid:
                uids[uid] = None
        if len(uids) >= max_teams:
            break
    return list(uids)[:max_teams]


def collect_histories(stats: StatsClient, uids: list[int], depth: int = 30) -> dict[int, list[MatchResult]]:
    with ThreadPoolExecutor(max_workers=stats.workers) as pool:
        return dict(zip(uids, pool.map(lambda u: stats.last_matches(u, depth), uids)))


def finished_matches(histories: dict[int, list[MatchResult]], since_uts: int, until_uts: int) -> list[MatchResult]:
    seen: dict[int, MatchResult] = {}
    for matches in histories.values():
        for m in matches:
            if since_uts <= m.uts <= until_uts and m.home_uid in histories and m.away_uid in histories:
                seen[m.match_id] = m
    return sorted(seen.values(), key=lambda m: -m.uts)


def prior(matches: list[MatchResult], before_uts: int, n: int) -> tuple[MatchResult, ...]:
    return tuple(m for m in matches if m.uts < before_uts)[:n]


def match_samples(m: MatchResult, home: TeamStats, away: TeamStats, cfg: EdgeConfig) -> list[Sample]:
    model = fit_goal_model(home, away, cfg)
    if model is None:
        return []
    hg, ag = m.home_goals, m.away_goals
    diff = hg - ag
    truth = {
        "1X (Home/Draw)": hg >= ag,
        "X2 (Draw/Away)": ag >= hg,
        "AH Home -0.5": diff > 0,
        "AH Away +0.5": diff <= 0,
        "AH Home +0.5": diff >= 0,
        "AH Away -0.5": diff < 0,
        "AH Home -1.5": diff >= 2,
        "AH Away +1.5": diff <= 1,
        "AH Home +1.5": diff >= -1,
        "AH Away -1.5": diff <= -2,
        "EH Home (1:0)": hg + 1 > ag,
        "EH Away (0:1)": ag + 1 > hg,
        "EH Home (0:1)": hg > ag + 1,
        "EH Away (1:0)": ag > hg + 1,
    }
    probs = {
        "1X (Home/Draw)": model.double_chance("home"),
        "X2 (Draw/Away)": model.double_chance("away"),
        "AH Home -0.5": model.asian_handicap("home", -0.5),
        "AH Away +0.5": model.asian_handicap("away", 0.5),
        "AH Home +0.5": model.asian_handicap("home", 0.5),
        "AH Away -0.5": model.asian_handicap("away", -0.5),
        "AH Home -1.5": model.asian_handicap("home", -1.5),
        "AH Away +1.5": model.asian_handicap("away", 1.5),
        "AH Home +1.5": model.asian_handicap("home", 1.5),
        "AH Away -1.5": model.asian_handicap("away", -1.5),
        "EH Home (1:0)": model.euro_handicap("home", 1, 0),
        "EH Away (0:1)": model.euro_handicap("away", 0, 1),
        "EH Home (0:1)": model.euro_handicap("home", 0, 1),
        "EH Away (1:0)": model.euro_handicap("away", 1, 0),
    }
    return [Sample(m.match_id, m.uts, k, probs[k], truth[k]) for k in BACKTEST_MARKETS]


def evaluate(
    matches: list[MatchResult],
    histories: dict[int, list[MatchResult]],
    cfg: EdgeConfig,
    h2h_lookup=None,
) -> list[Sample]:
    out: list[Sample] = []
    for m in matches:
        h_recent = prior(histories[m.home_uid], m.uts, cfg.form_games)
        a_recent = prior(histories[m.away_uid], m.uts, cfg.form_games)
        h2h = prior(h2h_lookup(m.home_uid, m.away_uid), m.uts, 10) if h2h_lookup else ()
        home = TeamStats(m.home_uid, h_recent, h2h)
        away = TeamStats(m.away_uid, a_recent, h2h)
        out.extend(match_samples(m, home, away, cfg))
    return out


def brier(samples: list[Sample]) -> float:
    return sum((s.p - s.won) ** 2 for s in samples) / len(samples) if samples else float("nan")


def baseline_brier(samples: list[Sample]) -> float:
    if not samples:
        return float("nan")
    rate = sum(s.won for s in samples) / len(samples)
    return sum((rate - s.won) ** 2 for s in samples) / len(samples)


def fit_platt(samples: list[Sample], iters: int = 400, lr: float = 0.05) -> tuple[float, float]:
    xs = [math.log(max(1e-6, min(1 - 1e-6, s.p)) / (1 - max(1e-6, min(1 - 1e-6, s.p)))) for s in samples]
    ys = [1.0 if s.won else 0.0 for s in samples]
    a, b = 1.0, 0.0
    n = len(xs)
    for _ in range(iters):
        ga = gb = 0.0
        for x, y in zip(xs, ys):
            q = 1 / (1 + math.exp(-(a * x + b)))
            ga += (q - y) * x
            gb += q - y
        a -= lr * ga / n
        b -= lr * gb / n
    return a, b


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *a: None) if args.json else (lambda *a: print(*a, file=sys.stderr))
    try:
        sport_id = resolve_sport(args.sport)
        leagues = resolve_leagues(args.league, args.tier)
        excludes = resolve_excludes(args.exclude)
        spec = OddsSpec(hours_ahead=None, leagues=tuple(leagues) if leagues else None,
                        excludes=tuple(excludes) if excludes else None)
        cfg = EdgeConfig(form_games=args.form_games)
        cache = None if args.no_cache else Path(".booker_cache")
        stats = StatsClient(cache_dir=cache)
        client = SportyBetClient(region=args.region)

        log(f"Backtesting Win/Draw & Handicap markets for {args.sport}...")
        uids = discover_teams(client, sport_id, spec, args.teams)
        histories = collect_histories(stats, uids)
        now = int(time.time())
        matches = finished_matches(histories, now - args.days * 86400, now)[: args.samples]
        if not matches:
            print("Nothing to evaluate. Widen --days or loosen --league.")
            return 2

        h2h_lookup = None
        if not args.no_h2h:
            pairs = list({(m.home_uid, m.away_uid) for m in matches})
            with ThreadPoolExecutor(max_workers=stats.workers) as pool:
                h2h = dict(zip(pairs, pool.map(lambda p: stats.head_to_head(*p), pairs)))
            h2h_lookup = lambda a, b: h2h.get((a, b), [])

        samples = evaluate(matches, histories, cfg, h2h_lookup)
        if not samples:
            print("Not enough prior history to rate matches.")
            return 2

        a, b = fit_platt(samples)
        by_market: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            by_market[s.market].append(s)

        rated_matches = len({s.match_id for s in samples})
        first = datetime.fromtimestamp(min(s.uts for s in samples), tz=timezone.utc).date()
        last = datetime.fromtimestamp(max(s.uts for s in samples), tz=timezone.utc).date()

        if args.save and cache is not None:
            path = cache / CALIBRATION_FILE
            data = {"football": {"a": a, "b": b, "matches": rated_matches, "fitted": datetime.now().isoformat(timespec="seconds")}}
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        if args.json:
            print(json.dumps({
                "matches": rated_matches,
                "from": first.isoformat(),
                "to": last.isoformat(),
                "platt": {"a": a, "b": b},
                "markets": {
                    k: {
                        "n": len(v),
                        "hitRate": sum(s.won for s in v) / len(v),
                        "avgPredicted": sum(s.p for s in v) / len(v),
                        "brier": brier(v),
                        "baselineBrier": baseline_brier(v),
                    }
                    for k, v in by_market.items()
                },
            }, indent=2))
            return 0

        print(f"\nWin/Draw & Handicap Backtest: {rated_matches} matches ({first} to {last})\n")
        print(f"{'Market':<18} {'n':>5} {'Hit rate':>9} {'Model avg':>10} {'Brier':>7} {'Base':>7}  verdict")
        for k in BACKTEST_MARKETS:
            v = by_market.get(k)
            if not v:
                continue
            br, base = brier(v), baseline_brier(v)
            verdict = "beats base rate" if br < base - 0.002 else ("matches base rate" if br <= base + 0.002 else "worse")
            print(f"{k:<18} {len(v):>5} {sum(s.won for s in v)/len(v):>9.1%} {sum(s.p for s in v)/len(v):>10.1%} {br:>7.3f} {base:>7.3f}  {verdict}")
        print(f"\nPlatt fit: a={a:.3f}, b={b:+.3f}")
        return 0
    except (SportyBetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
