"""`booker backtest`: replay finished matches to see whether the model's probabilities are honest.

For every finished match in the window we rebuild each side's history *as it stood before kick-off*,
run the same model the live picker uses, and compare its probabilities with what happened.
Historical bookmaker odds aren't available, so this measures calibration and hit rate by probability
band (the question "when the model says 80%, does it win 80%?"), not edge against the closing price.
"""

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
from .edge import CALIBRATION_FILE, FOOTBALL_SPORTS, EdgeConfig, fit_goal_model, form_win_prob, model_group
from .selector import OddsSpec, event_in_window, event_league, league_matches
from .sports import resolve_excludes, resolve_leagues, resolve_markets, resolve_sport
from .stats import MatchResult, StatsClient, TeamStats, sr_numeric_id

FOOTBALL_MARKETS = ("Home", "Draw", "Away", "1X", "X2", "Over 1.5", "Over 2.5", "Under 2.5", "Under 3.5",
                    "Under 4.5", "BTTS Yes", "BTTS No")


@dataclass(frozen=True)
class Sample:
    match_id: int
    uts: int
    market: str
    p: float  # model probability
    won: bool


# ---------------------------------------------------------------------- collection
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="booker backtest",
        description="Replay recent finished matches through the edge model and report calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sport", "-s", default="football", help="Sport name or id")
    p.add_argument("--region", "-r", default="ng", choices=REGIONS, help="SportyBet site used to discover teams")
    p.add_argument("--days", type=int, default=30, help="How far back to evaluate finished matches")
    p.add_argument("--teams", type=int, default=160, help="Max teams to pull history for (from upcoming fixtures)")
    p.add_argument("--samples", type=int, default=600, help="Max finished matches to evaluate (most recent first)")
    p.add_argument("--league", "-L", default=None, help="Restrict team discovery to these leagues")
    p.add_argument("--tier", default=None, help="League bundle(s): top, big5, europe")
    p.add_argument("--exclude", "-X", default="default", help="Reject competitions/teams containing this text")
    p.add_argument("--form-games", type=int, default=10, help="Recent matches per team fed to the model")
    p.add_argument("--no-h2h", action="store_true", help="Skip head-to-head lookups (faster, slightly different model)")
    p.add_argument("--save", action="store_true", help="Save the fitted calibration so the edge model applies it")
    p.add_argument("--no-cache", action="store_true", help="Don't cache stats responses")
    p.add_argument("--json", action="store_true", help="JSON output")
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


# ---------------------------------------------------------------------- evaluation
def football_samples(m: MatchResult, home: TeamStats, away: TeamStats, cfg: EdgeConfig) -> list[Sample]:
    model = fit_goal_model(home, away, cfg)
    if model is None:
        return []
    o = model.outcomes()
    hg, ag, tot = m.home_goals, m.away_goals, m.home_goals + m.away_goals
    truth = {
        "Home": hg > ag, "Draw": hg == ag, "Away": hg < ag,
        "1X": hg >= ag, "X2": hg <= ag,
        "Over 1.5": tot > 1.5, "Over 2.5": tot > 2.5, "Under 2.5": tot < 2.5,
        "Under 3.5": tot < 3.5, "Under 4.5": tot < 4.5,
        "BTTS Yes": hg > 0 and ag > 0, "BTTS No": not (hg > 0 and ag > 0),
    }
    probs = {
        "Home": o["home"], "Draw": o["draw"], "Away": o["away"],
        "1X": o["home"] + o["draw"], "X2": o["draw"] + o["away"],
        "Over 1.5": model.over_under(1.5)[0], "Over 2.5": model.over_under(2.5)[0],
        "Under 2.5": model.over_under(2.5)[1], "Under 3.5": model.over_under(3.5)[1],
        "Under 4.5": model.over_under(4.5)[1],
        "BTTS Yes": o["btts"], "BTTS No": 1 - o["btts"],
    }
    return [Sample(m.match_id, m.uts, k, probs[k], truth[k]) for k in FOOTBALL_MARKETS]


def two_way_samples(m: MatchResult, home: TeamStats, away: TeamStats, cfg: EdgeConfig) -> list[Sample]:
    if m.home_goals == m.away_goals:
        return []
    p = form_win_prob(home, away, cfg)
    if p is None:
        return []
    return [Sample(m.match_id, m.uts, "Home", p, m.home_goals > m.away_goals),
            Sample(m.match_id, m.uts, "Away", 1 - p, m.home_goals < m.away_goals)]


def evaluate(
    matches: list[MatchResult],
    histories: dict[int, list[MatchResult]],
    cfg: EdgeConfig,
    is_football: bool,
    h2h_lookup=None,
) -> list[Sample]:
    out: list[Sample] = []
    for m in matches:
        h_recent = prior(histories[m.home_uid], m.uts, cfg.form_games)
        a_recent = prior(histories[m.away_uid], m.uts, cfg.form_games)
        h2h = prior(h2h_lookup(m.home_uid, m.away_uid), m.uts, 10) if h2h_lookup else ()
        home = TeamStats(m.home_uid, h_recent, h2h)
        away = TeamStats(m.away_uid, a_recent, h2h)
        out.extend(football_samples(m, home, away, cfg) if is_football else two_way_samples(m, home, away, cfg))
    return out


# ---------------------------------------------------------------------- metrics
def brier(samples: list[Sample]) -> float:
    return sum((s.p - s.won) ** 2 for s in samples) / len(samples) if samples else float("nan")


def log_loss(samples: list[Sample]) -> float:
    eps = 1e-6
    return -sum(math.log(max(eps, s.p if s.won else 1 - s.p)) for s in samples) / len(samples) if samples else float("nan")


def baseline_brier(samples: list[Sample]) -> float:
    """Brier of always predicting the market's own base rate: the model must beat this to be useful."""
    if not samples:
        return float("nan")
    rate = sum(s.won for s in samples) / len(samples)
    return sum((rate - s.won) ** 2 for s in samples) / len(samples)


def calibration_table(samples: list[Sample], edges=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)) -> list[dict]:
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        bucket = [s for s in samples if lo <= s.p < hi]
        if not bucket:
            continue
        rows.append({
            "band": f"{lo:.0%}-{min(hi, 1):.0%}",
            "n": len(bucket),
            "predicted": sum(s.p for s in bucket) / len(bucket),
            "actual": sum(s.won for s in bucket) / len(bucket),
        })
    return rows


def fit_platt(samples: list[Sample], iters: int = 400, lr: float = 0.05) -> tuple[float, float]:
    """Fit p' = sigmoid(a*logit(p) + b) by gradient descent on log loss. (1, 0) means already calibrated."""
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


def apply_platt(samples: list[Sample], a: float, b: float) -> list[Sample]:
    cfg = EdgeConfig(calibration=(a, b))
    return [Sample(s.match_id, s.uts, s.market, cfg.calibrate(s.p), s.won) for s in samples]


# ---------------------------------------------------------------------- command
def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *a: None) if args.json else (lambda *a: print(*a, file=sys.stderr))
    try:
        sport_id = resolve_sport(args.sport)
        is_football = sport_id in FOOTBALL_SPORTS
        leagues = resolve_leagues(args.league, args.tier)
        excludes = resolve_excludes(args.exclude)
        spec = OddsSpec(hours_ahead=None, leagues=tuple(leagues) if leagues else None,
                        excludes=tuple(excludes) if excludes else None)
        cfg = EdgeConfig(form_games=args.form_games)
        cache = None if args.no_cache else Path(".booker_cache")
        stats = StatsClient(cache_dir=cache)
        client = SportyBetClient(region=args.region)

        log(f"Backtest {args.sport}: discovering teams from upcoming fixtures"
            + (f" ({args.league or ''}{' tier=' + args.tier if args.tier else ''})" if (args.league or args.tier) else "") + "...")
        uids = discover_teams(client, sport_id, spec, args.teams)
        log(f"{len(uids)} teams; pulling last 30 results each...")
        histories = collect_histories(stats, uids)

        now = int(time.time())
        matches = finished_matches(histories, now - args.days * 86400, now)[: args.samples]
        log(f"{len(matches)} finished matches in the last {args.days} days with both teams' histories available")
        if not matches:
            print("Nothing to evaluate. Widen --days, raise --teams, or loosen --league/--exclude.")
            return 2

        h2h_lookup = None
        if not args.no_h2h:
            pairs = list({(m.home_uid, m.away_uid) for m in matches})
            log(f"Fetching head-to-head for {len(pairs)} pairings...")
            with ThreadPoolExecutor(max_workers=stats.workers) as pool:
                h2h = dict(zip(pairs, pool.map(lambda p: stats.head_to_head(*p), pairs)))
            h2h_lookup = lambda a, b: h2h.get((a, b), [])

        samples = evaluate(matches, histories, cfg, is_football, h2h_lookup)
        if not samples:
            print("Model could not rate any match (too little prior history). Try a bigger --days window.")
            return 2

        a, b = fit_platt(samples)
        calibrated = apply_platt(samples, a, b)
        by_market: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            by_market[s.market].append(s)

        rated_matches = len({s.match_id for s in samples})
        first = datetime.fromtimestamp(min(s.uts for s in samples), tz=timezone.utc).date()
        last = datetime.fromtimestamp(max(s.uts for s in samples), tz=timezone.utc).date()
        report = {
            "sport": sport_id,
            "group": model_group(sport_id),
            "matches": rated_matches,
            "from": first.isoformat(),
            "to": last.isoformat(),
            "overall": {"brier": brier(samples), "logLoss": log_loss(samples), "baselineBrier": baseline_brier(samples)},
            "markets": {k: {"n": len(v), "brier": brier(v), "baselineBrier": baseline_brier(v),
                            "hitRate": sum(s.won for s in v) / len(v), "avgPredicted": sum(s.p for s in v) / len(v)}
                        for k, v in by_market.items()},
            "calibration": calibration_table(samples),
            "platt": {"a": a, "b": b, "brierAfter": brier(calibrated), "logLossAfter": log_loss(calibrated)},
            "calibrationAfter": calibration_table(calibrated),
            "saved": False,
        }

        if args.save and cache is not None:
            path = cache / CALIBRATION_FILE
            data = {}
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except ValueError:
                    data = {}
            data[report["group"]] = {"a": a, "b": b, "matches": rated_matches, "fitted": datetime.now().isoformat(timespec="seconds")}
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            report["saved"] = True

        if args.json:
            print(json.dumps(report, indent=2))
            return 0

        print(f"\nModel backtest: {rated_matches} matches, {first} to {last}, {len(samples)} market predictions\n")
        print(f"{'Market':<10} {'n':>5} {'Hit rate':>9} {'Model avg':>10} {'Brier':>7} {'Base':>7}  verdict")
        for k in (FOOTBALL_MARKETS if is_football else ("Home", "Away")):
            r = report["markets"].get(k)
            if not r:
                continue
            verdict = "beats base rate" if r["brier"] < r["baselineBrier"] - 0.002 else (
                "no better than base rate" if r["brier"] <= r["baselineBrier"] + 0.002 else "WORSE than base rate")
            print(f"{k:<10} {r['n']:>5} {r['hitRate']:>9.1%} {r['avgPredicted']:>10.1%} {r['brier']:>7.3f} {r['baselineBrier']:>7.3f}  {verdict}")

        print(f"\nCalibration (all markets): when the model says X, how often does it happen?")
        print(f"{'Model band':<12} {'n':>5} {'predicted':>10} {'actual':>8} {'gap':>7}")
        for row in report["calibration"]:
            gap = row["actual"] - row["predicted"]
            print(f"{row['band']:<12} {row['n']:>5} {row['predicted']:>10.1%} {row['actual']:>8.1%} {gap:>+7.1%}")

        over = [r for r in report["calibration"] if r["predicted"] >= 0.7]
        if over:
            n = sum(r["n"] for r in over)
            pred = sum(r["predicted"] * r["n"] for r in over) / n
            act = sum(r["actual"] * r["n"] for r in over) / n
            print(f"\nPicks the model rated 70%+: predicted {pred:.1%}, actually won {act:.1%} ({n} predictions).")

        print(f"\nPlatt fit: a={a:.3f}, b={b:+.3f} (a<1 = model overconfident, shrink toward 50%; a=1,b=0 = already calibrated)")
        print(f"Brier {report['overall']['brier']:.4f} -> {report['platt']['brierAfter']:.4f} after calibration "
              f"(base rate {report['overall']['baselineBrier']:.4f})")
        if report["saved"]:
            print(f"Saved to {cache / CALIBRATION_FILE}; --edge and rollover now apply it automatically.")
        else:
            print("Re-run with --save to have the edge model apply this correction.")
        return 0
    except (SportyBetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
