import unittest
from dataclasses import replace

from booker.edge import (
    EdgeConfig,
    GoalModel,
    fit_goal_model,
    football_outcome_prob,
    form_win_prob,
    market_probability,
    rate_candidates,
)
from booker.selector import Candidate, OddsSpec, build_slip
from booker.stats import MatchResult, TeamStats, sr_numeric_id


def cand(**kw) -> Candidate:
    base = dict(
        event_id="sr:match:1", home="H", away="A", tournament="T", start_ms=10_000, market_id="1",
        market_name="1X2", specifier="", outcome_id="1", outcome_desc="Home", odds=2.0, probability=0.5,
        sport_id="sr:sport:1", home_id="sr:competitor:10", away_id="sr:competitor:20",
    )
    base.update(kw)
    return Candidate(**base)


def result(home_uid, away_uid, hg, ag, uts) -> MatchResult:
    return MatchResult(match_id=uts, home_uid=home_uid, away_uid=away_uid, home_goals=hg, away_goals=ag, uts=uts)


def strong_vs_weak():
    # Team 10 wins everything 3-0; team 20 loses everything 0-3.
    strong = TeamStats(10, tuple(result(10, 99, 3, 0, 100 - i) for i in range(8)), ())
    weak = TeamStats(20, tuple(result(20, 99, 0, 3, 100 - i) for i in range(8)), ())
    return strong, weak


class GoalModelTest(unittest.TestCase):
    def test_outcomes_sum_to_one(self):
        o = GoalModel(1.6, 1.1).outcomes()
        self.assertAlmostEqual(o["home"] + o["draw"] + o["away"], 1.0, places=9)
        self.assertGreater(o["home"], o["away"])

    def test_over_under_half_line_sums_to_one(self):
        over, under = GoalModel(1.4, 1.2).over_under(2.5)
        self.assertAlmostEqual(over + under, 1.0)
        self.assertGreater(over, 0.4)

    def test_fit_reflects_form(self):
        strong, weak = strong_vs_weak()
        model = fit_goal_model(strong, weak, EdgeConfig())
        self.assertIsNotNone(model)
        self.assertGreater(model.lam_home, model.lam_away)
        self.assertGreater(model.outcomes()["home"], 0.8)

    def test_fit_needs_enough_matches(self):
        strong, weak = strong_vs_weak()
        thin = TeamStats(20, weak.recent[:2], ())
        self.assertIsNone(fit_goal_model(strong, thin, EdgeConfig()))

    def test_outcome_mapping(self):
        model = GoalModel(1.5, 1.0)
        o = model.outcomes()
        self.assertAlmostEqual(football_outcome_prob(cand(market_id="1", outcome_desc="Home"), model), o["home"])
        self.assertAlmostEqual(football_outcome_prob(cand(market_id="10", outcome_desc="Home or Draw"), model),
                               o["home"] + o["draw"])
        self.assertAlmostEqual(football_outcome_prob(cand(market_id="29", outcome_desc="No"), model), 1 - o["btts"])
        ou = football_outcome_prob(cand(market_id="18", specifier="total=2.5", outcome_desc="Over 2.5"), model)
        self.assertAlmostEqual(ou, model.over_under(2.5)[0])
        self.assertIsNone(football_outcome_prob(cand(market_id="45", outcome_desc="2:1"), model))


class FormModelTest(unittest.TestCase):
    def test_two_way_form(self):
        strong, weak = strong_vs_weak()
        p = form_win_prob(strong, weak, EdgeConfig())
        self.assertGreater(p, 0.9)
        self.assertLessEqual(p, 0.95)


class RatingTest(unittest.TestCase):
    def test_market_probability_prefers_devigged_field(self):
        self.assertEqual(market_probability(cand(odds=2.0, probability=0.48)), 0.48)
        self.assertAlmostEqual(market_probability(cand(odds=2.0, probability=0.0)), 0.5)

    def test_rate_candidates_uses_stub_stats(self):
        strong, weak = strong_vs_weak()

        class StubStats:
            def fixtures_stats(self, fixtures, n=10):
                return {(10, 20): (strong, weak)}

        cands = [cand(outcome_desc="Home", odds=1.5, probability=0.60),
                 cand(outcome_desc="Away", outcome_id="3", odds=6.0, probability=0.15)]
        rated, summary = rate_candidates(cands, StubStats(), EdgeConfig(), max_matches=None)
        self.assertEqual(summary["rated"], 1)
        home, away = rated
        self.assertGreater(home.edge, 0)  # model thinks Home is far likelier than 60%
        self.assertLess(away.edge, 0)
        self.assertEqual(home.form_home, "WWWWWW")
        self.assertEqual(home.form_away, "LLLLLL")

    def test_sr_numeric_id(self):
        self.assertEqual(sr_numeric_id("sr:competitor:4716"), 4716)
        self.assertIsNone(sr_numeric_id("abc"))
        self.assertIsNone(sr_numeric_id(""))


class RolloverTest(unittest.TestCase):
    def test_win_prob_derived_from_edge(self):
        c = cand(odds=2.0, edge=0.1)  # blended prob = 1.1 / 2.0
        self.assertAlmostEqual(c.win_prob, 0.55)
        self.assertEqual(cand(odds=2.0, probability=0.48).win_prob, 0.48)

    def test_rollover_picks_highest_survival_pair(self):
        import math
        import random
        from booker.selector import one_per_event
        # Both pairs land on ~3.0; pair B has the higher blended win probability.
        a = [cand(event_id="a1", odds=1.73, edge=-0.02), cand(event_id="a2", odds=1.73, edge=-0.02)]  # p=0.566 each
        b = [cand(event_id="b1", odds=1.73, edge=0.05), cand(event_id="b2", odds=1.73, edge=0.05)]  # p=0.607 each
        pool = one_per_event(a + b, "rollover", random.Random(0))
        self.assertEqual([c.event_id[0] for c in pool], ["b", "b", "a", "a"])
        spec = OddsSpec(total_odds=3.0, tolerance=0.07, min_legs=2, max_legs=2)
        picks = build_slip(pool, spec, score_fn=lambda c: math.log(c.win_prob))
        self.assertEqual({c.event_id for c in picks}, {"b1", "b2"})


class EdgeAwareSlipTest(unittest.TestCase):
    def test_prefer_early_picks_sooner_kickoffs_inside_band(self):
        late = [cand(event_id=f"l{i}", odds=1.5, edge=0.10, start_ms=9_000_000) for i in range(2)]
        early = [cand(event_id=f"e{i}", odds=1.5, edge=0.01, start_ms=1_000_000) for i in range(2)]
        spec = OddsSpec(total_odds=2.25, tolerance=0.05, min_legs=2, max_legs=2)
        picks = build_slip(late + early, spec, score_fn=lambda c: -c.start_ms)
        self.assertEqual({c.event_id for c in picks}, {"e0", "e1"})

    def test_prefers_higher_edge_inside_tolerance(self):
        # Two ways to make ~2.25: (1.5 x 1.5) with bad edge or (1.5 x 1.5) with good edge.
        good = [cand(event_id=f"g{i}", odds=1.5, edge=0.05) for i in range(2)]
        bad = [cand(event_id=f"b{i}", odds=1.5, edge=-0.05) for i in range(2)]
        pool = bad + good  # bad ones come first so ordering alone wouldn't pick them
        picks = build_slip(pool, OddsSpec(total_odds=2.25, tolerance=0.05, min_legs=2, max_legs=2))
        self.assertEqual({c.event_id for c in picks}, {"g0", "g1"})


if __name__ == "__main__":
    unittest.main()
