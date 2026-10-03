"""Command-line entry point: `python -m booker --leg 1.20 --parlay 10`."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import date, datetime, time, timedelta, timezone

from pathlib import Path

from .client import REGIONS, SportyBetClient, SportyBetError
from .edge import EdgeConfig, load_calibration, model_group, rate_candidates
from .selector import (
    Candidate,
    OddsSpec,
    build_slip,
    event_in_window,
    event_league,
    extract_candidates,
    league_matches,
    one_per_event,
    total_odds,
)
from .sports import resolve_excludes, resolve_leagues, resolve_markets, resolve_sport
from .stats import StatsClient


# Real football only. eSoccer/futsal share the Poisson pricer but are too noisy for stats-first defaults.
STATS_FIRST_SPORTS = {"sr:sport:1"}
# --leg 1.20 keeps picks at that price, not 1.80s: allow this much room above the asked price.
LEG_HEADROOM = 0.20
MAX_AUTO_LEGS = 16


def configure_pick_mode(args: argparse.Namespace, sport_id: str) -> bool:
    """Apply football stats-first defaults. Returns whether the edge model will run.

    Football is no longer an odds-range filler: unless the user explicitly asks for
    `--strategy safest` or `random`, every pick must be rated and clear edge / probability gates.
    Other sports keep the old safest-by-price default.
    """
    stats_first = sport_id in STATS_FIRST_SPORTS and args.strategy not in ("safest", "random")
    if stats_first:
        if args.strategy is None:
            args.strategy = "rollover"
        if args.min_edge is None:
            args.min_edge = 0.02
        if args.min_prob is None:
            args.min_prob = 0.55
        if args.min_book_prob is None:
            args.min_book_prob = 0.50
        if args.exclude is None:
            args.exclude = "default"
        if args.tier is None and not args.league:
            args.tier = "top"
        if args.parlay is None and args.total_odds is None:
            args.parlay = 3.0
    elif args.strategy is None:
        args.strategy = "safest"

    use_edge = bool(args.edge or args.min_edge is not None or args.strategy in ("edge", "rollover"))
    if use_edge and args.strategy == "safest":
        args.strategy = "edge"
    return use_edge


def stats_first_markets(market_ids: list[str], sport_id: str, markets_arg: str | None, use_edge: bool) -> list[str]:
    """Drop BTTS unless the user asked for it: the football backtest prices GG/NG worse than the base rate."""
    if markets_arg is None and use_edge and sport_id in STATS_FIRST_SPORTS:
        return [m for m in market_ids if m != "29"]
    return market_ids


def auto_legs(min_odds: float, max_odds: float, total: float, tolerance: float) -> str:
    """Leg range that can actually hit `total` given the per-pick odds band."""
    lo_band = total * (1 - tolerance)
    hi_band = total * (1 + tolerance)
    n_min = max(1, math.ceil(math.log(lo_band) / math.log(max_odds)))
    n_max = max(n_min, math.floor(math.log(hi_band) / math.log(min_odds)))
    n_max = min(MAX_AUTO_LEGS, n_max)
    avg = math.exp(0.5 * (math.log(min_odds) + math.log(max_odds)))
    need = max(n_min, min(n_max, math.ceil(math.log(total) / math.log(avg))))
    lo = max(n_min, need - 2)
    hi = min(n_max, max(need + 1, lo))
    if lo > hi:
        lo, hi = n_min, max(n_min, n_max)
    return f"{lo}-{hi}"


def apply_odds_shortcuts(args: argparse.Namespace) -> None:
    """Fill min/max/total/legs from `--leg` and `--parlay` so the long flags are optional."""
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
        args.min_odds = 1.20
    if args.max_odds is None:
        args.max_odds = 2.00

    if args.legs is None:
        if args.total_odds is not None:
            args.legs = auto_legs(args.min_odds, args.max_odds, args.total_odds, args.tolerance)
        else:
            args.legs = "1-8"


def _parse_legs(value: str) -> tuple[int, int]:
    """'5' -> (5, 5); '3-6' -> (3, 6)."""
    if "-" in value:
        lo, hi = value.split("-", 1)
        return int(lo), int(hi)
    n = int(value)
    return n, n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="booker",
        description="Pick SportyBet selections and generate a booking code. "
                    "Usual call: python -m booker --leg 1.20 --parlay 10. "
                    "Football defaults to the stats model (form + xG + head-to-head).",
        epilog="Subcommands: 'booker rollover' builds a daily ~3 odds rollover slip; "
               "'booker backtest' checks the model's accuracy on recent finished matches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sport", "-s", default="football", help="Sport name or id (e.g. football, basketball, tennis, 5)")
    p.add_argument("--region", "-r", default="ng", choices=REGIONS, help="SportyBet country site")
    p.add_argument("--leg", type=float, default=None, metavar="ODDS",
                   help="Odds you want on each pick, e.g. 1.20. Uses that price up to +20%% "
                        "unless you also pass --min-odds / --max-odds")
    p.add_argument("--parlay", type=float, default=None, metavar="ODDS",
                   help="Target combined odds for the slip, e.g. 10. Same as --total-odds. "
                        "Also picks a sensible --legs range if you omit it")
    p.add_argument("--min-odds", type=float, default=None, help="Lowest odds allowed for a single pick (default 1.20, or --leg)")
    p.add_argument("--max-odds", type=float, default=None, help="Highest odds allowed for a single pick (default 2.00, or --leg + 20%%)")
    p.add_argument("--total-odds", "-t", type=float, default=None,
                   help="Target combined odds for the whole slip (same as --parlay)")
    p.add_argument("--tolerance", type=float, default=0.10, help="Acceptable +/- fraction around --parlay / --total-odds")
    p.add_argument("--legs", "-l", default=None, help="Number of picks, exact ('5') or a range ('3-8'). "
                   "Omitted: derived from --parlay and --leg, otherwise 1-8")
    p.add_argument("--markets", "-m", default=None,
                   help="Comma list of markets (1x2, ou, dc, gg, dnb, winner, or numeric ids). Default depends on sport")
    p.add_argument("--league", "-L", default=None,
                   help="Comma list of leagues, by alias (laliga, epl, efl, ucl, uel, seriea, bundesliga, ligue1...) "
                        "or any text matched against '<country> <tournament>' (e.g. 'england premier', 'norway')")
    p.add_argument("--tier", default=None,
                   help="League bundle(s) to include: top, big5, europe (comma list; combines with --league)")
    p.add_argument("--exclude", "-X", default=None,
                   help="Comma list of text to reject in competition or team names; 'default' = "
                        "reserves, youth (u17-u23), women, amateur, serie d, MLS Next Pro. Football stats mode "
                        "applies 'default' automatically")
    p.add_argument("--min-book-prob", type=float, default=None,
                   help="Drop outcomes the bookmaker rates below this probability, e.g. 0.5. "
                        "Football stats mode defaults to 0.50")
    p.add_argument("--min-prob", type=float, default=None,
                   help="Drop picks whose blended win probability is below this. "
                        "Football stats mode defaults to 0.55")
    p.add_argument("--list-leagues", action="store_true",
                   help="Print the leagues with events in the selected time window and exit")
    p.add_argument("--include-srl", action="store_true",
                   help="Include Simulated Reality League (computer-simulated) fixtures, which are skipped by default")
    p.add_argument("--outcome", "-o", default=None,
                   help="Only keep outcomes whose label contains this text, e.g. 'Over 1.5', 'Home', 'Yes'")
    p.add_argument("--hours", type=float, default=48, help="Only events kicking off within this many hours (0 = no limit)")
    p.add_argument("--today", action="store_true", help="Only events kicking off today (local time); overrides --hours")
    p.add_argument("--tomorrow", action="store_true", help="Only events kicking off tomorrow (local time); overrides --hours")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                   help="Only events kicking off on this local calendar day; overrides --hours")
    p.add_argument("--time", default=None, metavar="HH:MM-HH:MM",
                   help="Kick-off time range within the chosen day (local), e.g. 12:00-23:59 or just 18:00. "
                        "Defaults the day to today if none given")
    p.add_argument("--prefer-early", action="store_true",
                   help="Fill the slip from the earliest kick-offs in the window (instead of the highest-edge combo). "
                        "Use with --time when you want the 1am slate in, not 8pm games that merely start after 1am")
    p.add_argument("--strategy", choices=("safest", "edge", "rollover", "random"), default=None,
                   help="safest = highest bookmaker probability; edge = best model edge; "
                        "rollover = highest blended win chance (best for parlays that land); "
                        "random = shuffle. Football defaults to rollover; other sports default to safest")
    p.add_argument("--seed", type=int, default=None, help="Seed for --strategy random")
    p.add_argument("--max-pages", type=int, default=15, help="Max event pages (100 events each) to scan")

    e = p.add_argument_group("edge model (team form + head-to-head from Sportradar stats)")
    e.add_argument("--edge", action="store_true", help="Rate picks with the stats model and show edge")
    e.add_argument("--min-edge", type=float, default=None,
                   help="Drop picks whose edge is below this (e.g. 0.02 = +2%%). Implies --edge. "
                        "Football stats mode defaults to +2%%")
    e.add_argument("--model-weight", type=float, default=0.40,
                   help="Share of the stats model vs the bookmaker's own probability in the blend")
    e.add_argument("--form-games", type=int, default=10, help="Recent matches per team to feed the model")
    e.add_argument("--max-matches", type=int, default=120,
                   help="Max fixtures to pull stats for (soonest first); 0 = all matches in range")
    e.add_argument("--allow-unrated", action="store_true",
                   help="Keep picks the model couldn't rate (no stats / unsupported market) in the pool")
    e.add_argument("--no-cache", action="store_true", help="Don't cache stats responses in .booker_cache/")
    e.add_argument("--no-calibration", action="store_true",
                   help="Ignore the calibration saved by 'booker backtest --save'")
    p.add_argument("--book", action="store_true", help="Actually create the booking code (default is dry run)")
    p.add_argument("--force", action="store_true",
                   help="Book even if the slip's total odds missed --total-odds by more than --tolerance")
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a table")
    p.add_argument("--list-sports", action="store_true", help="Print the live sport list and exit")
    return p


def _parse_time_range(spec: str) -> tuple[time, time]:
    """'12:00-23:59' -> (time(12, 0), time(23, 59)); '18:00' alone means from 18:00 to end of day.

    An end at or before the start (e.g. '23:00-02:00') runs past midnight into the next day.
    """
    lo, _, hi = spec.partition("-")
    start = time.fromisoformat(lo.strip())
    end = time.fromisoformat(hi.strip()) if hi.strip() else time(23, 59, 59)
    if end == start:
        raise ValueError(f"--time range {spec!r} is empty; give an end time after the start")
    return start, end


def _day_window_ms(day: date, time_range: str | None = None) -> tuple[int, int]:
    """[start, end) of `day` in local time as epoch ms; `time_range` ('HH:MM-HH:MM') narrows it."""
    if time_range:
        t0, t1 = _parse_time_range(time_range)
        end_day = day + timedelta(days=1) if t1 < t0 else day
        start = datetime.combine(day, t0).astimezone()
        end = datetime.combine(end_day, t1).astimezone() + timedelta(seconds=1)
    else:
        start = datetime.combine(day, datetime.min.time()).astimezone()
        end = datetime.combine(day + timedelta(days=1), datetime.min.time()).astimezone()
    return int(start.timestamp() * 1000), int(end.timestamp() * 1000)


def _warn_skipped_early_games(events, pool, picks, window, args) -> None:
    """If the window starts at 1am but the slip is all 8pm games, say why the early slate was dropped."""
    if not window:
        return
    earliest, _horizon = window
    early_cut = earliest + 4 * 3600 * 1000
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    early_events = []
    for ev in events:
        start = int(ev.get("estimateStartTime") or 0)
        if ev.get("status") not in (0, None) or start <= now_ms:
            continue
        if earliest <= start < min(early_cut, _horizon):
            early_events.append(ev)
    if not early_events:
        return
    pool_ids = {c.event_id for c in pool}
    dropped = [ev for ev in early_events if ev.get("eventId") not in pool_ids]
    if dropped:
        names = ", ".join(
            f"{ev.get('homeTeamName', '?')} vs {ev.get('awayTeamName', '?')}" for ev in dropped[:4]
        )
        extra = f" (+{len(dropped) - 4} more)" if len(dropped) > 4 else ""
        why = []
        if getattr(args, "min_edge", None) is not None:
            why.append("--min-edge")
        if getattr(args, "min_book_prob", None) is not None:
            why.append("--min-book-prob")
        if getattr(args, "exclude", None):
            why.append("--exclude")
        why.append("odds range")
        print(
            f"Note: {len(dropped)} kick-off(s) in the first hours of the window didn't make the pool "
            f"({', '.join(why)}): {names}{extra}.",
            file=sys.stderr,
        )
        if getattr(args, "min_edge", None) is not None:
            print("      Drop --min-edge (or use --min-edge 0) to keep those early games even if the model disagrees.",
                  file=sys.stderr)
    if picks and not getattr(args, "prefer_early", False):
        first = min(c.start_ms for c in picks)
        if first >= early_cut:
            print(
                f"Note: window starts {_fmt_time(earliest)} but the slip's earliest pick is {_fmt_time(first)}. "
                f"The builder ranked later games higher on edge. Pass --prefer-early to fill from the 1am slate first.",
                file=sys.stderr,
            )


def _pick_day(args: argparse.Namespace) -> date | None:
    """Which local calendar day the user asked for, or None for a rolling --hours window."""
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


def _print_slip(picks: list[Candidate], show_edge: bool) -> None:
    if not picks:
        print("No selections matched your criteria.")
        return
    w_fix = max(len(c.fixture) for c in picks)
    w_lg = min(28, max(len(c.league_label) for c in picks))
    w_mkt = max(len(c.market_label) for c in picks)
    w_out = max(len(c.outcome_desc) for c in picks)
    header = (f"{'#':>2}  {'Kick-off':<16} {'League':<{w_lg}}  {'Fixture':<{w_fix}}  {'Market':<{w_mkt}}  "
              f"{'Pick':<{w_out}}  {'Odds':>6}")
    if show_edge:
        header += f"  {'Form (H / A)':<15}  {'Book%':>6}  {'Model%':>6}  {'Edge':>7}"
    print(header)
    print("-" * len(header))
    for i, c in enumerate(picks, 1):
        line = (f"{i:>2}  {_fmt_time(c.start_ms):<16} {c.league_label[:w_lg]:<{w_lg}}  {c.fixture:<{w_fix}}  "
                f"{c.market_label:<{w_mkt}}  {c.outcome_desc:<{w_out}}  {c.odds:>6.2f}")
        if show_edge:
            form = f"{c.form_home or '-'} / {c.form_away or '-'}"
            edge = f"{c.edge * 100:+6.1f}%" if c.edge is not None else "    -  "
            line += f"  {form:<15}  {_pct(c.market_prob):>6}  {_pct(c.model_prob):>6}  {edge:>7}"
        print(line)
    print("-" * 20)
    summary = f"Legs: {len(picks)}   Total odds: {total_odds(picks):.2f}"
    rated = [c for c in picks if c.edge is not None]
    if show_edge and rated:
        summary += f"   Avg edge: {sum(c.edge for c in rated) / len(rated) * 100:+.1f}%"
        if len(rated) < len(picks):
            summary += f"   ({len(picks) - len(rated)} unrated)"
    print(summary)


def _pick_to_dict(c: Candidate) -> dict:
    return {
        "eventId": c.event_id,
        "fixture": c.fixture,
        "tournament": c.tournament,
        "country": c.category,
        "startTime": datetime.fromtimestamp(c.start_ms / 1000, tz=timezone.utc).isoformat(),
        "marketId": c.market_id,
        "market": c.market_label,
        "specifier": c.specifier,
        "outcomeId": c.outcome_id,
        "pick": c.outcome_desc,
        "odds": c.odds,
        "bookProb": c.market_prob,
        "modelProb": c.model_prob,
        "edge": c.edge,
        "formHome": c.form_home or None,
        "formAway": c.form_away or None,
    }


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("rollover", "ladder"):
        from .rollover import main as rollover_main

        return rollover_main(argv[1:])
    if argv and argv[0] == "backtest":
        from .backtest import main as backtest_main

        return backtest_main(argv[1:])
    args = build_parser().parse_args(argv)
    client = SportyBetClient(region=args.region)

    try:
        if args.list_sports:
            for s in client.list_sports():
                print(f"{s.id:<22} {s.name:<28} {s.event_count:>5} events")
            return 0

        sport_id = resolve_sport(args.sport)
        use_edge = configure_pick_mode(args, sport_id)
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

        market_ids = stats_first_markets(resolve_markets(args.markets, sport_id), sport_id, args.markets, use_edge)

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
            print(f"\n{sum(counts.values())} events in {len(counts)} leagues. "
                  f"Filter with --league \"<text>\" matched against '<country> <tournament>'.", file=sys.stderr)
            return 0

        if not args.json:
            print(f"Scanning {args.sport} ({sport_id}) on sportybet.com/{args.region} | markets {','.join(market_ids)} "
                  f"| odds {spec.min_odds:.2f}-{spec.max_odds:.2f}"
                  + (f" | target {spec.total_odds:.2f} +/-{spec.tolerance:.0%}" if spec.total_odds else "")
                  + (f" | kick-offs on {day.isoformat()}" + (f" {args.time}" if args.time else "") if window else "")
                  + (f" | leagues: {args.league or ''}{' tier=' + args.tier if args.tier else ''}" if (args.league or args.tier) else "")
                  + (f" | exclude: {args.exclude}" if args.exclude else "")
                  + (f" | book prob >= {args.min_book_prob:.0%}" if args.min_book_prob is not None else "")
                  + (f" | min edge {args.min_edge:+.0%}" if args.min_edge is not None else "")
                  + (f" | min model prob {args.min_prob:.0%}" if args.min_prob is not None else ""),
                  file=sys.stderr)

        events = list(client.iter_upcoming_events(sport_id, market_ids, max_pages=args.max_pages))
        cands = extract_candidates(events, spec)
        n_matches = len({c.event_id for c in cands})
        if not args.json:
            print(f"{len(events)} events scanned, {len(cands)} outcomes in range across {n_matches} matches",
                  file=sys.stderr)

        if use_edge and cands:
            cache_dir = None if args.no_cache else Path(".booker_cache")
            calib = None if args.no_calibration else load_calibration(cache_dir, model_group(sport_id))
            cfg = EdgeConfig(form_games=args.form_games, model_weight=args.model_weight, calibration=calib)
            stats = StatsClient(cache_dir=cache_dir)
            if calib and not args.json:
                print(f"Applying saved calibration (a={calib[0]:.2f}, b={calib[1]:+.2f}) from 'booker backtest --save'",
                      file=sys.stderr)
            max_matches = None if args.max_matches <= 0 else args.max_matches
            if not args.json:
                print(f"Pulling form + head-to-head for {min(max_matches or n_matches, n_matches)} fixtures...",
                      file=sys.stderr)
            cands, summary = rate_candidates(cands, stats, cfg, max_matches=max_matches)
            if not args.json:
                print(f"Model rated {summary['rated']} fixtures ({summary['unrated']} without enough stats)",
                      file=sys.stderr)
            if not args.allow_unrated:
                cands = [c for c in cands if c.edge is not None]
            if args.min_edge is not None:
                cands = [c for c in cands if c.edge is not None and c.edge >= args.min_edge]
            if args.min_prob is not None:
                cands = [c for c in cands if c.win_prob >= args.min_prob]

        rng = random.Random(args.seed)
        pool = one_per_event(cands, spec.strategy, rng)
        if args.prefer_early:
            pool.sort(key=lambda c: c.start_ms)
            picks = build_slip(pool, spec, score_fn=lambda c: -c.start_ms)
        elif spec.strategy == "rollover":
            picks = build_slip(pool, spec, score_fn=lambda c: math.log(max(c.win_prob, 1e-9)))
        else:
            picks = build_slip(pool, spec)

        if not args.json:
            note = f"{len(pool)} matches left in the pool after filters"
            if spec.total_odds is not None and pool:
                mean_log = sum(math.log(c.odds) for c in pool) / len(pool)
                note += f"; at ~{math.exp(mean_log):.2f} avg odds you need ~{math.ceil(math.log(spec.total_odds) / mean_log)} legs for {spec.total_odds:g}x"
            print(note, file=sys.stderr)
            _warn_skipped_early_games(events, pool, picks, window, args)

        p_model = math.prod(c.win_prob for c in picks) if picks else None
        p_book = (
            math.prod((c.market_prob if c.market_prob is not None else c.probability) for c in picks)
            if picks else None
        )
        got = total_odds(picks) if picks else 0.0
        result: dict = {
            "sport": sport_id,
            "region": args.region,
            "legs": len(picks),
            "totalOdds": round(got, 2) if picks else None,
            "avgEdge": (sum(c.edge for c in picks if c.edge is not None) / len(picks))
            if use_edge and picks and all(c.edge is not None for c in picks) else None,
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
                short = got < spec.total_odds
                fix = ("The pool is too small to multiply that high: widen the time window "
                       "(--hours / --today), lower --min-edge, or drop the target."
                       if short and len(pool) <= spec.max_legs else
                       "Widen the odds range, legs, tolerance, or --max-matches.")
                print(f"Warning: closest achievable total was {got:.2f}, outside +/-{spec.tolerance:.0%} of "
                      f"{spec.total_odds:.2f}. {fix}", file=sys.stderr)

        if picks and args.book and off_target and not args.force:
            if not args.json:
                _print_slip(picks, show_edge=use_edge)
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
            _print_slip(picks, show_edge=use_edge)
            if picks and p_model is not None:
                print(f"Parlay win chance: model {p_model:.0%} vs book {p_book:.0%}   "
                      f"Expected return: {p_model * got - 1:+.1%} per unit staked")
                if p_model < 0.25:
                    print("This slip is a long shot, not a safe parlay. "
                          "For slips that actually land, keep --parlay at 2-3 (not 8-10).",
                          file=sys.stderr)
            if result["booking"]:
                b = result["booking"]
                print(f"\nBooking code: {b['shareCode']}")
                print(f"Load it here:  {b['shareUrl']}")
                if b["deadline"]:
                    print(f"Valid until:   {b['deadline']}")
                unavailable = (b["raw"] or {}).get("unavailableOutcomes")
                if unavailable:
                    print(f"Note: {len(unavailable)} selection(s) were rejected by SportyBet and left off the slip.")
            elif picks:
                print("\nDry run - re-run with --book to generate a SportyBet booking code.")
        return 0 if picks else 2

    except (SportyBetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
