import unittest

from motivated.edge import EdgeConfig, GoalModel, fit_goal_model, outcome_prob
from motivated.motivation import TeamMotivation
from motivated.selector import Candidate, classify_allowed_outcome, parse_asian_handicap, parse_euro_handicap
from motivated.sports import resolve_markets
from motivated.stats import MatchResult, TeamStats


def cand(**kw) -> Candidate:
    base = dict(
        event_id="sr:match:1",
        home="Arsenal",
        away="Leeds",
        tournament="Premier League",
        start_ms=10_000,
        market_id="10",
        market_name="Double Chance",
        specifier="",
        outcome_id="9",
        outcome_desc="Home or Draw",
        odds=1.30,
        probability=0.75,
        side="home",
        sport_id="sr:sport:1",
        home_id="sr:competitor:10",
        away_id="sr:competitor:20",
    )
    base.update(kw)
    return Candidate(**base)


def result(home_uid, away_uid, hg, ag, uts) -> MatchResult:
    return MatchResult(match_id=uts, home_uid=home_uid, away_uid=away_uid, home_goals=hg, away_goals=ag, uts=uts)


class MarketsAndOutcomesTest(unittest.TestCase):
    def test_resolve_markets_only_allows_win_draw_and_handicap(self):
        self.assertEqual(resolve_markets(None), ["10", "16", "14"])
        self.assertEqual(resolve_markets("dc,ah"), ["10", "16"])
        self.assertEqual(resolve_markets("handicap"), ["16", "14"])
        with self.assertRaises(ValueError):
            resolve_markets("1x2")
        with self.assertRaises(ValueError):
            resolve_markets("ou")
        with self.assertRaises(ValueError):
            resolve_markets("gg")

    def test_classify_allowed_outcome_rejects_12_and_handicap_draw(self):
        # Double Chance (10): 1X and X2 allowed, 12 ("Home or Away") rejected
        self.assertEqual(classify_allowed_outcome("10", "9", "Home or Draw"), "home")
        self.assertEqual(classify_allowed_outcome("10", "11", "Draw or Away"), "away")
        self.assertIsNone(classify_allowed_outcome("10", "10", "Home or Away"))

        # European 3-Way Handicap (14): Home and Away allowed, Draw rejected
        self.assertEqual(classify_allowed_outcome("14", "1711", "Home (0:1)"), "home")
        self.assertEqual(classify_allowed_outcome("14", "1713", "Away (0:1)"), "away")
        self.assertIsNone(classify_allowed_outcome("14", "1712", "Draw (0:1)"))

        # Asian Handicap (16): Home and Away allowed
        self.assertEqual(classify_allowed_outcome("16", "1714", "Home (-0.5)"), "home")
        self.assertEqual(classify_allowed_outcome("16", "1715", "Away (+0.5)"), "away")

        # Disallowed market
        self.assertIsNone(classify_allowed_outcome("1", "1", "Home"))

    def test_handicap_parsers(self):
        self.assertEqual(parse_euro_handicap("hcp=0:2", "Away (0:2)"), (0, 2))
        self.assertEqual(parse_euro_handicap("hcp=1:0", "Home (1:0)"), (1, 0))
        self.assertEqual(parse_asian_handicap("home", "hcp=-1.5", "Home (-1.5)"), -1.5)
        self.assertEqual(parse_asian_handicap("away", "hcp=-1.5", "Away (+1.5)"), 1.5)
        self.assertEqual(parse_asian_handicap("away", "hcp=0.5", "Away (-0.5)"), -0.5)


class GoalModelHandicapPricingTest(unittest.TestCase):
    def test_double_chance_and_equivalent_handicaps_match(self):
        m = GoalModel(1.6, 1.1)
        p_1x = m.double_chance("home")
        p_ah_plus_half = m.asian_handicap("home", 0.5)
        p_eh_1_0 = m.euro_handicap("home", 1, 0)
        self.assertAlmostEqual(p_1x, p_ah_plus_half, places=6)
        self.assertAlmostEqual(p_1x, p_eh_1_0, places=6)

        p_x2 = m.double_chance("away")
        p_ah_away_plus_half = m.asian_handicap("away", 0.5)
        p_eh_0_1 = m.euro_handicap("away", 0, 1)
        self.assertAlmostEqual(p_x2, p_ah_away_plus_half, places=6)
        self.assertAlmostEqual(p_x2, p_eh_0_1, places=6)

    def test_asian_handicap_complementary_sides_sum_to_one(self):
        m = GoalModel(1.5, 1.2)
        for line in (-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5):
            p_home = m.asian_handicap("home", line)
            p_away = m.asian_handicap("away", -line)
            self.assertAlmostEqual(p_home + p_away, 1.0, places=6)

    def test_motivation_boost_shifts_expected_goals(self):
        home = TeamStats(10, tuple(result(10, 99, 2, 1, 100 - i) for i in range(6)), ())
        away = TeamStats(20, tuple(result(20, 99, 1, 1, 100 - i) for i in range(6)), ())
        cfg = EdgeConfig(motivation_boost=0.15)
        base_model = fit_goal_model(home, away, cfg)

        mot_high = TeamMotivation(10, 2, 20, 60, 28, 10, "title", 0.90, "#2 Title (-1p)", "Chasing 1st")
        mot_low = TeamMotivation(20, 12, 20, 36, 28, 10, "midtable", 0.15, "#12 Mid-table", "Mid-table")
        boosted_model = fit_goal_model(home, away, cfg, mot_home=mot_high, mot_away=mot_low)

        self.assertGreater(boosted_model.lam_home, base_model.lam_home)
        self.assertLess(boosted_model.lam_away, base_model.lam_away)
        self.assertGreater(
            outcome_prob(cand(market_id="16", specifier="hcp=-0.5", outcome_desc="Home (-0.5)", side="home"), boosted_model),
            outcome_prob(cand(market_id="16", specifier="hcp=-0.5", outcome_desc="Home (-0.5)", side="home"), base_model),
        )


if __name__ == "__main__":
    unittest.main()
