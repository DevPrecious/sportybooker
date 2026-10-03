import math
import random
import unittest

from booker.selector import Candidate, OddsSpec, build_slip, extract_candidates, one_per_event, total_odds


def cand(event: str, odds: float, prob: float | None = None, market: str = "1", outcome: str = "Home") -> Candidate:
    return Candidate(
        event_id=event, home="A", away="B", tournament="T", start_ms=10_000, market_id=market,
        market_name="1X2", specifier="", outcome_id="1", outcome_desc=outcome, odds=odds,
        probability=prob if prob is not None else 1 / odds,
    )


class ExtractCandidatesTest(unittest.TestCase):
    def _event(self, **overrides):
        ev = {
            "eventId": "sr:match:1", "homeTeamName": "Home FC", "awayTeamName": "Away FC",
            "estimateStartTime": 2_000, "status": 0,
            "markets": [{
                "id": "18", "name": "Over/Under", "specifier": "total=2.5", "status": 0,
                "outcomes": [
                    {"id": "12", "odds": "1.45", "probability": "0.66", "isActive": 1, "desc": "Over 2.5"},
                    {"id": "13", "odds": "2.60", "probability": "0.34", "isActive": 1, "desc": "Under 2.5"},
                    {"id": "99", "odds": "1.50", "probability": "0.60", "isActive": 0, "desc": "Suspended"},
                ],
            }],
        }
        ev.update(overrides)
        return ev

    def test_filters_by_odds_range_and_activity(self):
        spec = OddsSpec(min_odds=1.3, max_odds=1.6, hours_ahead=None)
        out = extract_candidates([self._event()], spec, now_ms=1_000)
        self.assertEqual([c.outcome_desc for c in out], ["Over 2.5"])
        self.assertEqual(out[0].market_label, "Over/Under (2.5)")

    def test_outcome_filter_is_case_insensitive(self):
        spec = OddsSpec(min_odds=1.0, max_odds=5.0, outcome_filter="under", hours_ahead=None)
        out = extract_candidates([self._event()], spec, now_ms=1_000)
        self.assertEqual([c.outcome_desc for c in out], ["Under 2.5"])

    def test_window_limits_to_day(self):
        spec = OddsSpec(min_odds=1.0, max_odds=5.0, window_ms=(2_000, 5_000), hours_ahead=None)
        inside = self._event(eventId="in", estimateStartTime=3_000)
        before = self._event(eventId="before", estimateStartTime=1_500)
        at_end = self._event(eventId="end", estimateStartTime=5_000)
        out = extract_candidates([inside, before, at_end], spec, now_ms=1_000)
        self.assertEqual({c.event_id for c in out}, {"in"})

    def test_league_filter_matches_country_and_tournament(self):
        spec = OddsSpec(min_odds=1.0, max_odds=5.0, hours_ahead=None,
                        leagues=("england premier league", "uefa europa league"))
        eng = self._event(eventId="eng", tournamentName="Premier League",
                          sport={"category": {"name": "England"}})
        rus = self._event(eventId="rus", tournamentName="Premier League",
                          sport={"category": {"name": "Russia"}})
        uel = self._event(eventId="uel", tournamentName="UEFA Europa League",
                          sport={"category": {"name": "International Clubs"}})
        out = extract_candidates([eng, rus, uel], spec, now_ms=1_000)
        self.assertEqual({c.event_id for c in out}, {"eng", "uel"})
        self.assertEqual(next(c for c in out if c.event_id == "eng").category, "England")

    def test_simulated_reality_excluded_by_default(self):
        srl = self._event(eventId="srl", tournamentName="UEFA Europa League SRL",
                          sport={"category": {"name": "Simulated Reality League"}})
        real = self._event(eventId="real", tournamentName="UEFA Europa League",
                           sport={"category": {"name": "International Clubs"}})
        base = dict(min_odds=1.0, max_odds=5.0, hours_ahead=None)
        ids = lambda spec: {c.event_id for c in extract_candidates([srl, real], spec, now_ms=1_000)}
        self.assertEqual(ids(OddsSpec(**base)), {"real"})
        self.assertEqual(ids(OddsSpec(**base, leagues=("uefa europa league",))), {"real"})
        self.assertEqual(ids(OddsSpec(**base, include_srl=True)), {"srl", "real"})
        self.assertEqual(ids(OddsSpec(**base, leagues=("europa league srl",))), {"srl"})

    def test_excludes_and_min_book_prob(self):
        base = dict(min_odds=1.0, max_odds=5.0, hours_ahead=None)
        reserves = self._event(eventId="res", homeTeamName="Boca Juniors Reserve")
        youth = self._event(eventId="u19", tournamentName="Primera Division U19")
        real = self._event(eventId="real")
        out = extract_candidates([reserves, youth, real], OddsSpec(**base, excludes=("reserve", "u19")), now_ms=1_000)
        self.assertEqual({c.event_id for c in out}, {"real"})
        # Over 2.5 has book prob 0.66, Under 2.5 has 0.34
        out = extract_candidates([real], OddsSpec(**base, min_book_prob=0.5), now_ms=1_000)
        self.assertEqual([c.outcome_desc for c in out], ["Over 2.5"])

    def test_skips_started_and_far_future_events(self):
        spec = OddsSpec(min_odds=1.0, max_odds=5.0, hours_ahead=1)
        started = self._event(estimateStartTime=500)
        too_far = self._event(estimateStartTime=1_000 + 2 * 3600 * 1000)
        self.assertEqual(extract_candidates([started, too_far], spec, now_ms=1_000), [])


class FeasibilityTest(unittest.TestCase):
    def test_unreachable_target_rejected_upfront(self):
        with self.assertRaises(ValueError) as ctx:
            OddsSpec(min_odds=1.2, max_odds=1.6, total_odds=20, min_legs=3, max_legs=6).validate()
        self.assertIn("Allow at least 7 legs", str(ctx.exception))
        with self.assertRaises(ValueError):
            OddsSpec(min_odds=2.0, max_odds=3.0, total_odds=3, min_legs=3, max_legs=5).validate()

    def test_reachable_target_passes(self):
        OddsSpec(min_odds=1.3, max_odds=2.2, total_odds=20, min_legs=4, max_legs=8).validate()
        OddsSpec(min_odds=1.2, total_odds=100).validate()  # no max odds -> no check


class SlipBuilderTest(unittest.TestCase):
    def test_one_per_event_keeps_most_probable(self):
        pool = one_per_event([cand("e1", 1.5, 0.6), cand("e1", 1.3, 0.7), cand("e2", 1.8, 0.5)], "safest", random.Random(0))
        self.assertEqual([(c.event_id, c.odds) for c in pool], [("e1", 1.3), ("e2", 1.8)])

    def test_without_target_takes_first_max_legs(self):
        pool = [cand(f"e{i}", 1.5) for i in range(10)]
        picks = build_slip(pool, OddsSpec(max_legs=3))
        self.assertEqual(len(picks), 3)

    def test_hits_target_within_tolerance(self):
        rng = random.Random(42)
        pool = [cand(f"e{i}", round(rng.uniform(1.2, 2.0), 2)) for i in range(40)]
        spec = OddsSpec(total_odds=25.0, tolerance=0.05, min_legs=2, max_legs=8)
        picks = build_slip(pool, spec)
        got = total_odds(picks)
        self.assertTrue(2 <= len(picks) <= 8)
        self.assertLessEqual(abs(got / 25.0 - 1), 0.05, msg=f"got {got}")
        self.assertEqual(len({c.event_id for c in picks}), len(picks))

    def test_returns_closest_when_target_unreachable(self):
        pool = [cand("e1", 1.2), cand("e2", 1.2)]
        picks = build_slip(pool, OddsSpec(total_odds=100.0, max_legs=2))
        self.assertEqual(len(picks), 2)
        self.assertTrue(math.isclose(total_odds(picks), 1.44))

    def test_empty_pool(self):
        self.assertEqual(build_slip([], OddsSpec(total_odds=5.0)), [])


if __name__ == "__main__":
    unittest.main()
