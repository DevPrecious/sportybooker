"""Command-line entry point: `python -m motivated --leg 1.25 --parlay 3`."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from .client import REGIONS, SportyBetClient, SportyBetError
from .edge import EdgeConfig, load_calibration, rate_candidates
from .motivation import evaluate_team_motivation, parse_season_tables
from .selector import (
    Candidate,
    OddsSpec,
    build_slip,
    event_in_window,
    event_league,
    event_tournament_id,
    extract_candidates,
    league_matches,
    one_per_event,
    total_odds,
)
from .sports import resolve_excludes, resolve_leagues, resolve_markets, resolve_sport
from .stats import StatsClient

LEG_HEADROOM = 0.20
MAX_AUTO_LEGS = 25
FOCUS_CHOICES = ("any", "title", "europe", "promotion", "relegation", "mismatch")


def configure_pick_mode(args: argparse.Namespace) -> None:
    """Apply table-motivation-first defaults unless overridden by CLI flags."""
    if args.strategy is None:
        args.strategy = "rollover"
    if args.min_edge is None:
        args.min_edge = 0.0
    if args.min_prob is None:
        args.min_prob = 0.55
    if args.min_book_prob is None:
        args.min_book_prob = 0.50
    if args.min_motivation is None:
        args.min_motivation = 0.42
    if args.min_mot_gap is None:
        args.min_mot_gap = 0.15 if args.focus == "mismatch" else -0.10
    if args.exclude is None:
        args.exclude = "default"
    if args.parlay is None and args.total_odds is None:
        args.parlay = 3.0


def auto_legs(min_odds: float, max_odds: float, total: float, tolerance: float) -> str:
    lo_band = total * (1 - tolerance)
    hi_band = total * (1 + tolerance)
    n_min = max(1, math.ceil(math.log(lo_band) / math.log(max_odds)))
    n_max = max(n_min, math.floor(math.log(hi_band) / math.log(min_odds)))
    n_max = min(MAX_AUTO_LEGS, n_max)
    return f"{n_min}-{n_max}"


def apply_odds_shortcuts(args: argparse.Namespace) -> None:
    if args.parlay is not None and args.total_odds is not None and args.parlay != args.total_odds:
        raise ValueError("pass --parlay or --total-odds, not both")
    if args.parlay is not None:
        args.total_odds = args.parlay

    if args.leg is not None:
        if args.leg <= 1.0:
            raise ValueError("--leg must be > 1.0")
        if args.min_odds is None:
            args.min_odds = args.leg
        if args.max_odds is None:
            args.max_odds = max(args.min_odds + 0.01, args.leg * (1 + LEG_HEADROOM))

    if args.min_odds is None:
        args.min_odds = 1.18
    if args.max_odds is None:
        args.max_odds = 2.00

    if args.legs is None:
        if args.total_odds is not None:
            args.legs = auto_legs(args.min_odds, args.max_odds, args.total_odds, args.tolerance)
        else:
            args.legs = "1-8"


def _parse_legs(value: str) -> tuple[int, int]:
    if "-" in value:
        lo, hi = value.split("-", 1)
        return int(lo), int(hi)
    n = int(value)
    return n, n


def filter_by_motivation(
    cands: list[Candidate],
    min_motivation: float,
    min_mot_gap: float,
    focus: str = "any",
    allow_no_table: bool = False,
) -> list[Candidate]:
    """Keep only candidates whose backed team passes the table-motivation gates."""
    out: list[Candidate] = []
    for c in cands:
        bm = c.backed_motivation
        if bm is None:
            if allow_no_table:
                out.append(c)
            continue
        if bm.score < min_motivation:
            continue
        if c.motivation_gap < min_mot_gap:
            continue
        if focus == "title" and bm.zone != "title":
            continue
        if focus == "europe" and bm.zone != "europe":
            continue
        if focus == "promotion" and bm.zone not in ("promotion", "europe"):
            continue
        if focus == "relegation" and bm.zone != "relegation":
            continue
        if focus == "mismatch":
            if bm.zone not in ("title", "europe", "promotion", "relegation"):
                continue
            if c.motivation_gap < max(0.15, min_mot_gap):
                continue
        out.append(c)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motivated",
        description="Pick SportyBet Win/Draw (1X/X2) and Handicap selections backed by league table motivations, "
                    "then generate a SportyBet booking code.",
        epilog="Subcommands: 'motivated rollover' builds a daily ~3 odds rollover slip; "
               "'motivated tables' prints live league tables with motivation ratings; "
               "'motivated backtest' evaluates Win/Draw and Handicap calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sport", "-s", default="football", help="Sport name or id (football, futsal, esoccer)")
    p.add_argument("--region", "-r", default="ng", choices=REGIONS, help="SportyBet country site")
    p.add_argument("--leg", type=float, default=None, metavar="ODDS",
                   help="Odds you want on each pick, e.g. 1.25. Uses that price up to +20%%")
    p.add_argument("--parlay", type=float, default=None, metavar="ODDS",
                   help="Target combined odds for the slip, e.g. 3 or 10. Same as --total-odds")
    p.add_argument("--min-odds", type=float, default=None, help="Lowest odds allowed for a single pick (default 1.18)")
    p.add_argument("--max-odds", type=float, default=None, help="Highest odds allowed for a single pick (default 2.00)")
    p.add_argument("--total-odds", "-t", type=float, default=None, help="Target combined odds for the slip")
    p.add_argument("--tolerance", type=float, default=0.10, help="Acceptable +/- fraction around --parlay / --total-odds")
    p.add_argument("--legs", "-l", default=None, help="Number of picks, exact ('4') or range ('2-6')")
    p.add_argument("--markets", "-m", default=None,
                   help="Comma list of allowed markets: dc (Win/Draw 1X/X2), ah (Asian Handicap), eh (Euro 3-Way Handicap), or handicap")
    p.add_argument("--league", "-L", default=None,
                   help="Comma list of leagues by alias (epl, laliga, seriea, bundesliga, ligue1, efl, ucl, uel...) or text")
    p.add_argument("--tier", default=None, help="League bundle(s): top, big5, europe")
    p.add_argument("--exclude", "-X", default=None,
                   help="Comma list of text to reject in competition or team names ('default' skips youth/reserves/women/friendlies)")
    p.add_argument("--min-book-prob", type=float, default=None, help="Minimum bookmaker probability per pick (default 0.50)")
    p.add_argument("--min-prob", type=float, default=None, help="Minimum blended win probability per pick (default 0.55)")
    p.add_argument("--list-leagues", action="store_true", help="Print leagues with events in the window and exit")
    p.add_argument("--include-srl", action="store_true", help="Include Simulated Reality League fixtures")
    p.add_argument("--outcome", "-o", default=None, help="Keep only outcomes matching this substring (e.g. 'Home or Draw', '+1.5')")
    p.add_argument("--hours", type=float, default=48, help="Only events kicking off within N hours (0 = no limit)")
    p.add_argument("--today", action="store_true", help="Only events kicking off today (local time)")
    p.add_argument("--tomorrow", action="store_true", help="Only events kicking off tomorrow (local time)")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD", help="Only events on this local calendar day")
    p.add_argument("--time", default=None, metavar="HH:MM-HH:MM", help="Kick-off time range within the chosen day")
    p.add_argument("--prefer-early", action="store_true", help="Fill the slip from the earliest kick-offs in the window")
    p.add_argument("--strategy", choices=("rollover", "edge", "motivation", "safest", "random"), default=None,
                   help="rollover = highest win prob + motivation; motivation = highest table urgency first; "
                        "edge = best model edge; safest = bookmaker prob")
    p.add_argument("--seed", type=int, default=None, help="Seed for --strategy random")
    p.add_argument("--max-pages", type=int, default=15, help="Max event pages (100 events each) to scan")

    m = p.add_argument_group("table motivation & edge model")
    m.add_argument("--focus", choices=FOCUS_CHOICES, default="any",
                   help="Filter by motivation type: any (Title/Europe/Promo/Relegation), title, europe, promotion, "
                        "relegation, or mismatch (motivated team vs less-motivated opponent)")
    m.add_argument("--min-motivation", type=float, default=None,
                   help="Minimum table motivation score (0..1) for the backed team (default 0.42)")
    m.add_argument("--min-mot-gap", type=float, default=None,
                   help="Minimum motivation advantage (backed - opponent, default -0.10; 0.15 for --focus mismatch)")
    m.add_argument("--motivation-boost", type=float, default=0.12,
                   help="xG multiplier boost per unit of table motivation advantage")
    m.add_argument("--min-table-games", type=int, default=2,
                   help="Minimum league matches played before full table motivation applies")
    m.add_argument("--allow-no-table", action="store_true",
                   help="Keep knockout/cup fixtures that don't have a league standings table")
    m.add_argument("--min-edge", type=float, default=None, help="Minimum model edge over bookmaker odds (default +1%%)")
    m.add_argument("--model-weight", type=float, default=0.45, help="Share of the stats+motivation model in blended prob")
    m.add_argument("--form-games", type=int, default=10, help="Recent matches per team fed to the model")
    m.add_argument("--max-matches", type=int, default=120, help="Max fixtures to pull stats & tables for (0 = all)")
    m.add_argument("--allow-unrated", action="store_true", help="Keep picks the Poisson model couldn't rate")
    m.add_argument("--no-cache", action="store_true", help="Don't cache stats/table responses in .booker_cache/")
    m.add_argument("--no-calibration", action="store_true", help="Ignore saved Platt calibration")

    p.add_argument("--book", action="store_true", help="Create the SportyBet booking code (default is dry run)")
    p.add_argument("--force", action="store_true", help="Book even if total odds missed --total-odds by > --tolerance")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    p.add_argument("--list-sports", action="store_true", help="Print live sport list and exit")
    return p


def _parse_time_range(spec: str) -> tuple[time, time]:
    lo, _, hi = spec.partition("-")
    start = time.fromisoformat(lo.strip())
    end = time.fromisoformat(hi.strip()) if hi.strip() else time(23, 59, 59)
    if end == start:
        raise ValueError(f"--time range {spec!r} is empty; give an end time after the start")
    return start, end


def _day_window_ms(day: date, time_range: str | None = None) -> tuple[int, int]:
    if time_range:
        t0, t1 = _parse_time_range(time_range)
        end_day = day + timedelta(days=1) if t1 < t0 else day
        start = datetime.combine(day, t0).astimezone()
        end = datetime.combine(end_day, t1).astimezone() + timedelta(seconds=1)
    else:
        start = datetime.combine(day, datetime.min.time()).astimezone()
        end = datetime.combine(day + timedelta(days=1), datetime.min.time()).astimezone()
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _pick_day(args: argparse.Namespace) -> date | None:
    if args.date:
        return date.fromisoformat(args.date)
    if getattr(args, "tomorrow", False):
        return date.today() + timedelta(days=1)
    if getattr(args, "today", False) or args.time:
        return date.today()
    return None


def _fmt_time(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone().strftime("%a %d %b %H:%M")


def _pct(x: float | None) -> str:
    return f"{x * 100:5.1f}%" if x is not None else "   -  "


def _print_slip(picks: list[Candidate], show_edge: bool = True) -> None:
    if not picks:
        print("No selections matched your criteria.")
        return
    w_fix = max(len(c.fixture) for c in picks)
    w_lg = min(24, max(len(c.league_label) for c in picks))
    w_mkt = max(len(c.market_label) for c in picks)
    w_out = max(len(c.outcome_desc) for c in picks)
    w_mot = max(len(c.motivation_summary) for c in picks)
    w_mot = max(w_mot, len("Table Motivation (Backed vs Opp)"))

    header = (
        f"{'#':>2}  {'Kick-off':<16} {'League':<{w_lg}}  {'Fixture':<{w_fix}}  "
        f"{'Market':<{w_mkt}}  {'Pick':<{w_out}}  {'Odds':>6}  "
        f"{'Table Motivation (Backed vs Opp)':<{w_mot}}"
    )
    if show_edge:
        header += f"  {'Mot':>4}  {'Form (H / A)':<15}  {'Book%':>6}  {'Model%':>6}  {'Edge':>7}"
    print(header)
    print("-" * len(header))
    for i, c in enumerate(picks, 1):
        line = (
            f"{i:>2}  {_fmt_time(c.start_ms):<16} {c.league_label[:w_lg]:<{w_lg}}  {c.fixture:<{w_fix}}  "
            f"{c.market_label:<{w_mkt}}  {c.outcome_desc:<{w_out}}  {c.odds:>6.2f}  "
            f"{c.motivation_summary:<{w_mot}}"
        )
        if show_edge:
            mot_s = f"{c.motivation_score:.2f}" if c.backed_motivation else "  - "
            form = f"{c.form_home or '-'} / {c.form_away or '-'}"
            edge = f"{c.edge * 100:+6.1f}%" if c.edge is not None else "    -  "
            line += f"  {mot_s:>4}  {form:<15}  {_pct(c.market_prob):>6}  {_pct(c.model_prob):>6}  {edge:>7}"
        print(line)
    print("-" * 20)
    summary = f"Legs: {len(picks)}   Total odds: {total_odds(picks):.2f}"
    rated = [c for c in picks if c.edge is not None]
    if show_edge and rated:
        summary += f"   Avg edge: {sum(c.edge for c in rated) / len(rated) * 100:+.1f}%"
    motivated = [c for c in picks if c.backed_motivation is not None]
    if motivated:
        avg_mot = sum(c.motivation_score for c in motivated) / len(motivated)
        summary += f"   Avg motivation: {avg_mot:.2f}"
    print(summary)


def _pick_to_dict(c: Candidate) -> dict:
    bm = c.backed_motivation
    om = c.opp_motivation
    return {
        "eventId": c.event_id,
        "fixture": c.fixture,
        "tournament": c.tournament,
        "country": c.category,
        "table": c.table_name or None,
        "startTime": datetime.fromtimestamp(c.start_ms / 1000, tz=timezone.utc).isoformat(),
        "marketId": c.market_id,
        "market": c.market_label,
        "specifier": c.specifier,
        "outcomeId": c.outcome_id,
        "pick": c.outcome_desc,
        "backedSide": c.side,
        "backedTeam": c.backed_team,
        "odds": c.odds,
        "bookProb": c.market_prob,
        "modelProb": c.model_prob,
        "edge": c.edge,
        "formHome": c.form_home or None,
        "formAway": c.form_away or None,
        "motivation": {
            "backed": {
                "pos": bm.pos,
                "totalTeams": bm.total_teams,
                "points": bm.points,
                "played": bm.played,
                "remaining": bm.remaining,
                "zone": bm.zone,
                "score": bm.score,
                "label": bm.label,
                "detail": bm.detail,
            } if bm else None,
            "opponent": {
                "pos": om.pos,
                "totalTeams": om.total_teams,
                "points": om.points,
                "played": om.played,
                "remaining": om.remaining,
                "zone": om.zone,
                "score": om.score,
                "label": om.label,
                "detail": om.detail,
            } if om else None,
            "gap": c.motivation_gap,
        },
    }


def tables_main(argv: list[str]) -> int:
    """`python -m motivated tables`: inspect live league standings and computed team motivations."""
    p = argparse.ArgumentParser(
        prog="motivated tables",
        description="Inspect live league tables and computed team motivation scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--league", "-L", default="epl", help="League alias or substring (e.g. epl, laliga, seriea, bundesliga, ucl)")
    p.add_argument("--region", "-r", default="ng", choices=REGIONS)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    client = SportyBetClient(region=args.region)
    stats = StatsClient(cache_dir=None if args.no_cache else Path(".booker_cache"))
    leagues = resolve_leagues(args.league, None)
    spec = OddsSpec(hours_ahead=None, leagues=tuple(leagues) if leagues else None)

    seen_seasons: dict[int, str] = {}
    for ev in client.iter_upcoming_events("sr:sport:1", ["10"], max_pages=10):
        cat, tour = event_league(ev)
        if not league_matches(cat, tour, spec.leagues, False):
            continue
        tid_str = event_tournament_id(ev)
        sid = stats.resolve_season_id(tid_str)
        if sid and sid not in seen_seasons:
            seen_seasons[sid] = f"{cat} - {tour}"
        if len(seen_seasons) >= 5:
            break

    if not seen_seasons:
        print(f"No active league tables found matching {args.league!r}.", file=sys.stderr)
        return 2

    raw_batch = stats.fetch_season_tables_batch(list(seen_seasons.keys()))
    out_json = []
    for sid, label in seen_seasons.items():
        tables = parse_season_tables(sid, raw_batch.get(sid))
        for tbl in tables:
            if args.json:
                rows_data = []
                for r in tbl.rows:
                    mot = evaluate_team_motivation(tbl, r.uid, min_games=1)
                    rows_data.append({
                        "pos": r.pos,
                        "team": r.name,
                        "played": r.played,
                        "points": r.points,
                        "gd": r.gd,
                        "promotion": r.promo_name or None,
                        "motivationZone": mot.zone if mot else None,
                        "motivationScore": mot.score if mot else None,
                        "motivationLabel": mot.label if mot else None,
                        "motivationDetail": mot.detail if mot else None,
                    })
                out_json.append({"seasonId": sid, "table": tbl.name, "round": tbl.current_round, "maxRounds": tbl.max_rounds, "rows": rows_data})
            else:
                print(f"\n=== {tbl.name} (Round {tbl.current_round}/{tbl.max_rounds}) ===")
                print(f"{'Pos':>3}  {'Team':<24} {'P':>3} {'W':>3} {'D':>3} {'L':>3} {'GD':>4} {'Pts':>4}  {'Zone':<11} {'Mot':>5}  {'Motivation Detail'}")
                print("-" * 105)
                for r in tbl.rows:
                    mot = evaluate_team_motivation(tbl, r.uid, min_games=1)
                    z = mot.zone if mot else "-"
                    sc = f"{mot.score:.2f}" if mot else "  - "
                    det = mot.detail if mot else ""
                    print(f"{r.pos:>3}  {r.name[:24]:<24} {r.played:>3} {r.wins:>3} {r.draws:>3} {r.losses:>3} {r.gd:>+4} {r.points:>4}  {z:<11} {sc:>5}  {det}")
    if args.json:
        print(json.dumps(out_json, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("rollover", "ladder"):
        from .rollover import main as rollover_main

        return rollover_main(argv[1:])
    if argv and argv[0] == "backtest":
        from .backtest import main as backtest_main

        return backtest_main(argv[1:])
    if argv and argv[0] in ("tables", "table", "standings"):
        return tables_main(argv[1:])

    args = build_parser().parse_args(argv)
    client = SportyBetClient(region=args.region)

    try:
        if args.list_sports:
            for s in client.list_sports():
                print(f"{s.id:<22} {s.name:<28} {s.event_count:>5} events")
            return 0

        sport_id = resolve_sport(args.sport)
        configure_pick_mode(args)
        apply_odds_shortcuts(args)

        day = _pick_day(args)
        window = _day_window_ms(day, args.time) if day else None

        min_legs, max_legs = _parse_legs(args.legs)
        spec = OddsSpec(
            min_odds=args.min_odds,
            max_odds=args.max_odds,
            total_odds=args.total_odds,
            tolerance=args.tolerance,
            min_legs=min_legs,
            max_legs=max_legs,
            outcome_filter=args.outcome,
            hours_ahead=None if args.hours <= 0 else args.hours,
            window_ms=window,
            leagues=tuple(leagues) if (leagues := resolve_leagues(args.league, args.tier)) else None,
            include_srl=args.include_srl,
            excludes=tuple(ex) if (ex := resolve_excludes(args.exclude)) else None,
            min_book_prob=args.min_book_prob,
            strategy=args.strategy,
        )
        spec.validate()
        if args.min_prob is not None and not (0 <= args.min_prob < 1):
            raise ValueError("--min-prob must be between 0 and 1")
        if not (0 <= args.min_motivation <= 1):
            raise ValueError("--min-motivation must be between 0 and 1")

        market_ids = resolve_markets(args.markets, sport_id)

        if args.list_leagues:
            now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            counts: dict[tuple[str, str], int] = {}
            for ev in client.iter_upcoming_events(sport_id, market_ids[:1], max_pages=args.max_pages):
                if event_in_window(ev, spec, now_ms):
                    key = event_league(ev)
                    if league_matches(*key, spec.leagues, spec.include_srl):
                        counts[key] = counts.get(key, 0) + 1
            for (cat, tour), n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"{n:>3}  {cat} | {tour}")
            print(f"\n{sum(counts.values())} events in {len(counts)} leagues.", file=sys.stderr)
            return 0

        if not args.json:
            print(
                f"Scanning {args.sport} ({sport_id}) on sportybet.com/{args.region} "
                f"| Win/Draw & Handicap markets ({','.join(market_ids)}) "
                f"| odds {spec.min_odds:.2f}-{spec.max_odds:.2f}"
                + (f" | target {spec.total_odds:.2f} +/-{spec.tolerance:.0%}" if spec.total_odds else "")
                + (f" | kick-offs on {day.isoformat()}" + (f" {args.time}" if args.time else "") if window else "")
                + (f" | focus: {args.focus} (min mot {args.min_motivation:.2f}, gap >= {args.min_mot_gap:+.2f})")
                + (f" | min edge {args.min_edge:+.0%}" if args.min_edge is not None else "")
                + (f" | min prob {args.min_prob:.0%}" if args.min_prob is not None else ""),
                file=sys.stderr,
            )

        events = list(client.iter_upcoming_events(sport_id, market_ids, max_pages=args.max_pages))
        cands = extract_candidates(events, spec)
        n_matches = len({c.event_id for c in cands})
        if not args.json:
            print(
                f"{len(events)} events scanned, {len(cands)} Win/Draw & Handicap outcomes in range across {n_matches} matches",
                file=sys.stderr,
            )

        if cands:
            cache_dir = None if args.no_cache else Path(".booker_cache")
            calib = None if args.no_calibration else load_calibration(cache_dir)
            cfg = EdgeConfig(
                form_games=args.form_games,
                model_weight=args.model_weight,
                motivation_boost=args.motivation_boost,
                min_table_games=args.min_table_games,
                calibration=calib,
            )
            stats = StatsClient(cache_dir=cache_dir)
            max_matches = None if args.max_matches <= 0 else args.max_matches
            if not args.json:
                print(
                    f"Pulling form, head-to-head, and league tables for {min(max_matches or n_matches, n_matches)} fixtures...",
                    file=sys.stderr,
                )
            cands, summary = rate_candidates(cands, stats, cfg, max_matches=max_matches)
            if not args.json:
                print(
                    f"Model rated {summary['rated']} fixtures; {summary['with_table']} resolved league standings "
                    f"({summary['motivated_matches']} with active table motivation)",
                    file=sys.stderr,
                )
            if not args.allow_unrated:
                cands = [c for c in cands if c.edge is not None]
            cands = filter_by_motivation(
                cands,
                min_motivation=args.min_motivation,
                min_mot_gap=args.min_mot_gap,
                focus=args.focus,
                allow_no_table=args.allow_no_table,
            )
            if args.min_edge is not None:
                cands = [c for c in cands if c.edge is not None and c.edge >= args.min_edge]
            if args.min_prob is not None:
                cands = [c for c in cands if c.win_prob >= args.min_prob]

        rng = random.Random(args.seed)
        pool = one_per_event(cands, spec.strategy, rng, spec=spec)
        if args.prefer_early:
            pool.sort(key=lambda c: c.start_ms)
            picks = build_slip(pool, spec, score_fn=lambda c: -c.start_ms)
        elif spec.strategy == "rollover":
            picks = build_slip(pool, spec, score_fn=lambda c: math.log(max(c.win_prob, 1e-9)) + 0.08 * c.motivation_score)
        elif spec.strategy == "motivation":
            picks = build_slip(pool, spec, score_fn=lambda c: c.motivation_score + 0.5 * max(0.0, c.motivation_gap))
        else:
            picks = build_slip(pool, spec)

        if not args.json:
            note = f"{len(pool)} motivated matches left in the pool after filters"
            if spec.total_odds is not None and pool:
                mean_log = sum(math.log(c.odds) for c in pool) / len(pool)
                note += f"; at ~{math.exp(mean_log):.2f} avg odds you need ~{math.ceil(math.log(spec.total_odds) / mean_log)} legs for {spec.total_odds:g}x"
            print(note, file=sys.stderr)

        p_model = math.prod(c.win_prob for c in picks) if picks else None
        p_book = (
            math.prod((c.market_prob if c.market_prob is not None else c.probability) for c in picks)
            if picks else None
        )
        got = total_odds(picks) if picks else 0.0
        result: dict = {
            "sport": sport_id,
            "region": args.region,
            "focus": args.focus,
            "legs": len(picks),
            "totalOdds": round(got, 2) if picks else None,
            "avgEdge": (sum(c.edge for c in picks if c.edge is not None) / len(picks))
            if picks and all(c.edge is not None for c in picks) else None,
            "avgMotivation": (sum(c.motivation_score for c in picks) / len(picks)) if picks else None,
            "modelWinProb": p_model,
            "bookWinProb": p_book,
            "expectedReturn": (p_model * got - 1) if p_model is not None else None,
            "selections": [_pick_to_dict(c) for c in picks],
            "booking": None,
        }

        off_target = False
        if picks and spec.total_odds is not None:
            got = total_odds(picks)
            off_target = abs(got / spec.total_odds - 1) > spec.tolerance
            if off_target and not args.json:
                print(
                    f"Warning: closest achievable total was {got:.2f}, outside +/-{spec.tolerance:.0%} of {spec.total_odds:.2f}. "
                    f"Try widening --hours, lowering --min-motivation / --min-edge, or removing --tier.",
                    file=sys.stderr,
                )

        if picks and args.book and off_target and not args.force:
            if not args.json:
                _print_slip(picks, show_edge=True)
                print("\nNot booked: total odds missed the target. Re-run with --force to book it anyway.")
            else:
                result["error"] = "total odds outside tolerance; not booked (use --force)"
                print(json.dumps(result, indent=2))
            return 3

        if picks and args.book:
            booking = client.book([c.to_selection() for c in picks])
            result["booking"] = {
                "shareCode": booking.share_code,
                "shareUrl": booking.share_url,
                "deadline": datetime.fromtimestamp(booking.deadline_ms / 1000, tz=timezone.utc).isoformat()
                if booking.deadline_ms else None,
                "raw": booking.raw,
            }

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            _print_slip(picks, show_edge=True)
            if picks and p_model is not None:
                print(
                    f"Parlay win chance: model {p_model:.0%} vs book {p_book:.0%}   "
                    f"Expected return: {p_model * got - 1:+.1%} per unit staked"
                )
            if result["booking"]:
                b = result["booking"]
                print(f"\nBooking code: {b['shareCode']}")
                print(f"Load it here:  {b['shareUrl']}")
                if b["deadline"]:
                    print(f"Valid until:   {b['deadline']}")
            elif picks:
                print("\nDry run - re-run with --book to generate a SportyBet booking code.")
        return 0 if picks else 2

    except (SportyBetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
