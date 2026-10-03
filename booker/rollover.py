"""`booker rollover`: a small daily parlay (default ~3.0 odds) built for re-staking day after day.

A rollover lives or dies on survival, so this command optimises for the highest model win
probability among slips that land on the target odds, and requires every leg to carry positive edge.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from .client import REGIONS, SportyBetClient, SportyBetError
from .edge import EdgeConfig, load_calibration, model_group, rate_candidates
from .selector import Candidate, OddsSpec, build_slip, extract_candidates, one_per_event, total_odds
from .sports import resolve_excludes, resolve_leagues, resolve_markets, resolve_sport
from .stats import StatsClient


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="booker rollover",
        description="Build today's rollover leg: a ~3 odds slip made of the safest positive-edge picks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sport", "-s", default="football", help="Sport name or id")
    p.add_argument("--region", "-r", default="ng", choices=REGIONS, help="SportyBet country site")
    p.add_argument("--odds", "-t", type=float, default=3.0, help="Target total odds for the slip")
    p.add_argument("--tolerance", type=float, default=0.07, help="Acceptable +/- fraction around --odds")
    p.add_argument("--legs", "-l", default="1-3", help="Number of picks, exact ('2') or range ('1-3')")
    p.add_argument("--min-odds", type=float, default=1.25, help="Lowest odds allowed for a single pick")
    p.add_argument("--max-odds", type=float, default=2.10, help="Highest odds allowed for a single pick")
    p.add_argument("--min-edge", type=float, default=0.02, help="Every leg must have at least this edge (0.02 = +2%%)")
    p.add_argument("--min-prob", type=float, default=0.55, help="Every leg must have at least this blended win probability")
    p.add_argument("--min-book-prob", type=float, default=0.50,
                   help="Every leg must also have at least this bookmaker probability (guards against model overconfidence)")
    p.add_argument("--markets", "-m", default=None,
                   help="Comma list of markets (1x2, ou, dc, gg, dnb, winner, ids). Football default skips GG/NG, "
                        "which the backtest shows the model prices poorly")
    p.add_argument("--league", "-L", default=None, help="Comma list of leagues (aliases or '<country> <tournament>' text)")
    p.add_argument("--tier", default=None, help="League bundle(s): top, big5, europe (combines with --league)")
    p.add_argument("--exclude", "-X", default="default",
                   help="Comma list of text to reject in competition/team names; 'default' = reserves, youth, "
                        "women, amateur, serie d, MLS Next Pro. Pass '' to disable")
    p.add_argument("--outcome", "-o", default=None, help="Only outcomes whose label contains this text")
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD", help="Kick-offs on this local day (default: today)")
    p.add_argument("--tomorrow", action="store_true", help="Kick-offs tomorrow (local time)")
    p.add_argument("--time", default=None, metavar="HH:MM-HH:MM",
                   help="Kick-off time range within the chosen day (local), e.g. 12:00-23:59 or just 18:00")
    p.add_argument("--prefer-early", action="store_true",
                   help="Fill the slip from the earliest kick-offs in the window")
    p.add_argument("--hours", type=float, default=None, help="Use a rolling N-hour window instead of a calendar day")
    p.add_argument("--include-srl", action="store_true", help="Include Simulated Reality League fixtures")
    p.add_argument("--model-weight", type=float, default=0.40, help="Share of the stats model in the blended probability")
    p.add_argument("--form-games", type=int, default=10, help="Recent matches per team fed to the model")
    p.add_argument("--max-matches", type=int, default=0, help="Max fixtures to pull stats for (0 = all)")
    p.add_argument("--max-pages", type=int, default=15, help="Max event pages (100 events each) to scan")
    p.add_argument("--stake", type=float, default=None, help="Starting stake, to print the ladder projection")
    p.add_argument("--days", type=int, default=7, help="Ladder length for the projection")
    p.add_argument("--no-cache", action="store_true", help="Don't cache stats responses")
    p.add_argument("--no-calibration", action="store_true", help="Ignore the calibration saved by 'booker backtest --save'")
    p.add_argument("--book", action="store_true", help="Create the SportyBet booking code (default is dry run)")
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
    from .cli import _day_window_ms, _pick_day, _pick_to_dict, _print_slip

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
        if not (0 < args.min_prob < 1) or not (0 <= args.min_book_prob < 1):
            raise ValueError("--min-prob and --min-book-prob must be between 0 and 1")

        sport_id = resolve_sport(args.sport)
        market_ids = resolve_markets(args.markets, sport_id)
        if args.markets is None:
            market_ids = [m for m in market_ids if m != "29"]
        log = (lambda *a: None) if args.json else (lambda *a: print(*a, file=sys.stderr))

        when = f"{day.isoformat()}{' ' + args.time if args.time else ''}" if window else f"next {args.hours:g}h"
        log(f"Rollover for {when}: {args.sport} on sportybet.com/{args.region}"
            f" | target {spec.total_odds:.2f} +/-{spec.tolerance:.0%} | legs {spec.min_legs}-{spec.max_legs}"
            f" | pick odds {spec.min_odds:.2f}-{spec.max_odds:.2f} | min edge {args.min_edge:+.0%}"
            f" | min prob {args.min_prob:.0%} (book {args.min_book_prob:.0%})"
            + (f" | leagues: {args.league or ''}{' tier=' + args.tier if args.tier else ''}" if (args.league or args.tier) else "")
            + (f" | exclude: {args.exclude}" if args.exclude else ""))

        events = list(client.iter_upcoming_events(sport_id, market_ids, max_pages=args.max_pages))
        cands = extract_candidates(events, spec)
        n_matches = len({c.event_id for c in cands})
        log(f"{len(events)} events scanned, {len(cands)} outcomes in range across {n_matches} matches")

        if cands:
            cache_dir = None if args.no_cache else Path(".booker_cache")
            calib = None if args.no_calibration else load_calibration(cache_dir, model_group(sport_id))
            cfg = EdgeConfig(form_games=args.form_games, model_weight=args.model_weight, calibration=calib)
            stats = StatsClient(cache_dir=cache_dir)
            if calib:
                log(f"Applying saved calibration (a={calib[0]:.2f}, b={calib[1]:+.2f})")
            log(f"Pulling form + head-to-head for {n_matches if args.max_matches <= 0 else min(args.max_matches, n_matches)} fixtures...")
            cands, summary = rate_candidates(cands, stats, cfg, max_matches=None if args.max_matches <= 0 else args.max_matches)
            log(f"Model rated {summary['rated']} fixtures ({summary['unrated']} without enough stats)")
            cands = [c for c in cands
                     if c.edge is not None and c.edge >= args.min_edge
                     and c.win_prob >= args.min_prob
                     and (c.market_prob if c.market_prob is not None else c.probability) >= args.min_book_prob]

        pool = one_per_event(cands, "rollover", random.Random())
        if args.prefer_early:
            pool.sort(key=lambda c: c.start_ms)
            picks = build_slip(pool, spec, score_fn=lambda c: -c.start_ms)
        else:
            picks = build_slip(pool, spec, score_fn=lambda c: math.log(c.win_prob))
        log(f"{len(pool)} matches qualify (edge >= {args.min_edge:+.0%}, blended prob >= {args.min_prob:.0%}, "
            f"book prob >= {args.min_book_prob:.0%})")

        got = total_odds(picks)
        off_target = bool(picks) and abs(got / spec.total_odds - 1) > spec.tolerance
        p_model, p_book = parlay_prob(picks), book_parlay_prob(picks)

        result = {
            "date": day.isoformat(),
            "sport": sport_id,
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
                print("No qualifying picks. Loosen --min-edge / --min-prob, widen the odds range, or drop --league.")
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
        print(f"Parlay win chance: model {p_model:.0%} vs book {p_book:.0%}   "
              f"Expected return: {p_model * got - 1:+.1%} per unit staked")
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
