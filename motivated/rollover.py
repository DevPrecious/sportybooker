"""`motivated rollover`: daily rollover slip using only motivated Win/Draw (1X/X2) and Handicap picks."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from .cli import FOCUS_CHOICES, _day_window_ms, _pick_day, _pick_to_dict, _print_slip, filter_by_motivation
from .client import REGIONS, SportyBetClient, SportyBetError
from .edge import EdgeConfig, load_calibration, rate_candidates
from .selector import Candidate, OddsSpec, build_slip, extract_candidates, one_per_event, total_odds
from .sports import resolve_excludes, resolve_leagues, resolve_markets, resolve_sport
from .stats import StatsClient


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="motivated rollover",
        description="Build today's rollover leg from motivated Win/Draw (1X/X2) and Handicap picks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sport", "-s", default="football", help="Sport name or id")
    p.add_argument("--region", "-r", default="ng", choices=REGIONS, help="SportyBet country site")
    p.add_argument("--odds", "-t", type=float, default=3.0, help="Target total odds for the slip")
    p.add_argument("--tolerance", type=float, default=0.08, help="Acceptable +/- fraction around --odds")
    p.add_argument("--legs", "-l", default="2-4", help="Number of picks, exact ('3') or range ('2-4')")
    p.add_argument("--min-odds", type=float, default=1.20, help="Lowest odds allowed for a single pick")
    p.add_argument("--max-odds", type=float, default=2.00, help="Highest odds allowed for a single pick")
    p.add_argument("--min-edge", type=float, default=0.01, help="Every leg must have at least this edge")
    p.add_argument("--min-prob", type=float, default=0.58, help="Every leg must have at least this blended win probability")
    p.add_argument("--min-book-prob", type=float, default=0.52, help="Every leg must have at least this bookmaker probability")
    p.add_argument("--focus", choices=FOCUS_CHOICES, default="any",
                   help="Motivation filter: any, title, europe, promotion, relegation, or mismatch")
    p.add_argument("--min-motivation", type=float, default=0.45, help="Minimum table motivation score for the backed team")
    p.add_argument("--min-mot-gap", type=float, default=-0.05, help="Minimum motivation advantage (backed - opponent)")
    p.add_argument("--motivation-boost", type=float, default=0.12, help="xG multiplier boost from motivation gap")
    p.add_argument("--allow-no-table", action="store_true", help="Allow cup/knockout matches without a standings table")
    p.add_argument("--markets", "-m", default=None, help="Allowed markets: dc, ah, eh, or handicap")
    p.add_argument("--league", "-L", default=None, help="Comma list of leagues")
    p.add_argument("--tier", default=None, help="League bundle(s): top, big5, europe")
    p.add_argument("--exclude", "-X", default="default", help="Reject competitions/teams matching text")
    p.add_argument("--outcome", "-o", default=None, help="Only outcomes whose label contains this text")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD", help="Kick-offs on this local day (default: today)")
    p.add_argument("--tomorrow", action="store_true", help="Kick-offs tomorrow (local time)")
    p.add_argument("--time", default=None, metavar="HH:MM-HH:MM", help="Kick-off time range within the chosen day")
    p.add_argument("--prefer-early", action="store_true", help="Fill the slip from the earliest kick-offs")
    p.add_argument("--hours", type=float, default=None, help="Rolling N-hour window instead of calendar day")
    p.add_argument("--include-srl", action="store_true", help="Include Simulated Reality League fixtures")
    p.add_argument("--model-weight", type=float, default=0.45, help="Share of the model in the blended probability")
    p.add_argument("--form-games", type=int, default=10, help="Recent matches per team fed to the model")
    p.add_argument("--max-matches", type=int, default=0, help="Max fixtures to pull stats & tables for (0 = all)")
    p.add_argument("--max-pages", type=int, default=15, help="Max event pages to scan")
    p.add_argument("--stake", type=float, default=None, help="Starting stake to print the ladder projection")
    p.add_argument("--days", type=int, default=7, help="Ladder length in days")
    p.add_argument("--no-cache", action="store_true", help="Don't cache stats/table responses")
    p.add_argument("--no-calibration", action="store_true", help="Ignore saved Platt calibration")
    p.add_argument("--book", action="store_true", help="Create the SportyBet booking code")
    p.add_argument("--force", action="store_true", help="Book even if total odds missed the target")
    p.add_argument("--json", action="store_true", help="JSON output")
    return p


def parlay_prob(picks: list[Candidate]) -> float:
    return math.prod(c.win_prob for c in picks) if picks else 0.0


def book_parlay_prob(picks: list[Candidate]) -> float:
    return math.prod((c.market_prob if c.market_prob is not None else c.probability) for c in picks) if picks else 0.0


def _print_ladder(stake: float, odds: float, p: float, days: int) -> None:
    print(f"\nLadder from {stake:,.0f} at {odds:.2f}x/day (model win chance {p:.0%} per day):")
    print(f"{'Day':>4}  {'Bankroll':>14}  {'Survive to here':>16}")
    bank = stake
    for d in range(1, days + 1):
        bank *= odds
        print(f"{d:>4}  {bank:>14,.0f}  {p ** d:>15.1%}")


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    client = SportyBetClient(region=args.region)
    try:
        lo, hi = (args.legs.split("-", 1) if "-" in args.legs else (args.legs, args.legs))
        day = _pick_day(args) or date.today()
        window = None if args.hours else _day_window_ms(day, args.time)
        leagues = resolve_leagues(args.league, args.tier)
        excludes = resolve_excludes(args.exclude)
        spec = OddsSpec(
            min_odds=args.min_odds,
            max_odds=args.max_odds,
            total_odds=args.odds,
            tolerance=args.tolerance,
            min_legs=int(lo),
            max_legs=int(hi),
            outcome_filter=args.outcome,
            hours_ahead=args.hours,
            window_ms=window,
            leagues=tuple(leagues) if leagues else None,
            include_srl=args.include_srl,
            excludes=tuple(excludes) if excludes else None,
            min_book_prob=args.min_book_prob,
            strategy="rollover",
        )
        spec.validate()
        sport_id = resolve_sport(args.sport)
        market_ids = resolve_markets(args.markets, sport_id)
        log = (lambda *a: None) if args.json else (lambda *a: print(*a, file=sys.stderr))

        when = f"{day.isoformat()}{' ' + args.time if args.time else ''}" if window else f"next {args.hours:g}h"
        log(
            f"Motivated Rollover for {when}: {args.sport} on sportybet.com/{args.region}"
            f" | markets {','.join(market_ids)} | target {spec.total_odds:.2f} +/-{spec.tolerance:.0%}"
            f" | focus={args.focus} (min mot {args.min_motivation:.2f})"
        )

        events = list(client.iter_upcoming_events(sport_id, market_ids, max_pages=args.max_pages))
        cands = extract_candidates(events, spec)
        n_matches = len({c.event_id for c in cands})
        log(f"{len(events)} events scanned, {len(cands)} outcomes in range across {n_matches} matches")

        if cands:
            cache_dir = None if args.no_cache else Path(".booker_cache")
            calib = None if args.no_calibration else load_calibration(cache_dir)
            cfg = EdgeConfig(
                form_games=args.form_games,
                model_weight=args.model_weight,
                motivation_boost=args.motivation_boost,
                calibration=calib,
            )
            stats = StatsClient(cache_dir=cache_dir)
            cands, summary = rate_candidates(
                cands,
                stats,
                cfg,
                max_matches=None if args.max_matches <= 0 else args.max_matches,
            )
            log(
                f"Model rated {summary['rated']} fixtures; {summary['with_table']} with standings "
                f"({summary['motivated_matches']} motivated)"
            )
            cands = filter_by_motivation(
                cands,
                min_motivation=args.min_motivation,
                min_mot_gap=args.min_mot_gap,
                focus=args.focus,
                allow_no_table=args.allow_no_table,
            )
            cands = [
                c for c in cands
                if c.edge is not None and c.edge >= args.min_edge
                and c.win_prob >= args.min_prob
                and (c.market_prob if c.market_prob is not None else c.probability) >= args.min_book_prob
            ]

        pool = one_per_event(cands, "rollover", random.Random(), spec=spec)
        if args.prefer_early:
            pool.sort(key=lambda c: c.start_ms)
            picks = build_slip(pool, spec, score_fn=lambda c: -c.start_ms)
        else:
            picks = build_slip(pool, spec, score_fn=lambda c: math.log(max(c.win_prob, 1e-9)) + 0.08 * c.motivation_score)
        log(f"{len(pool)} motivated matches qualify")

        got = total_odds(picks)
        off_target = bool(picks) and abs(got / spec.total_odds - 1) > spec.tolerance
        p_model, p_book = parlay_prob(picks), book_parlay_prob(picks)

        result = {
            "date": day.isoformat(),
            "sport": sport_id,
            "focus": args.focus,
            "legs": len(picks),
            "totalOdds": round(got, 2) if picks else None,
            "modelWinProb": p_model if picks else None,
            "bookWinProb": p_book if picks else None,
            "expectedReturn": (p_model * got - 1) if picks else None,
            "offTarget": off_target,
            "selections": [_pick_to_dict(c) for c in picks],
            "booking": None,
        }

        if not picks:
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print("No qualifying motivated picks. Try --hours 48, lower --min-motivation / --min-prob, or widen odds.")
            return 2

        if off_target:
            log(f"Warning: closest achievable total was {got:.2f}, outside +/-{spec.tolerance:.0%} of {spec.total_odds:.2f}.")

        if args.book and not (off_target and not args.force):
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
            return 3 if (args.book and off_target and not args.force) else 0

        _print_slip(picks, show_edge=True)
        print(
            f"Parlay win chance: model {p_model:.0%} vs book {p_book:.0%}   "
            f"Expected return: {p_model * got - 1:+.1%} per unit staked"
        )
        if args.stake:
            _print_ladder(args.stake, got, p_model, args.days)
        if result["booking"]:
            b = result["booking"]
            print(f"\nBooking code: {b['shareCode']}")
            print(f"Load it here:  {b['shareUrl']}")
        elif args.book and off_target:
            print("\nNot booked: total odds missed the target. Re-run with --force to book it anyway.")
            return 3
        else:
            print("\nDry run - re-run with --book to generate a SportyBet booking code.")
        return 0
    except (SportyBetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
