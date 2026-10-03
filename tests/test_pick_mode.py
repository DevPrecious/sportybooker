import argparse
import unittest

from booker.cli import apply_odds_shortcuts, auto_legs, configure_pick_mode, stats_first_markets


def args(**kw) -> argparse.Namespace:
    base = dict(
        strategy=None, edge=False, min_edge=None, min_prob=None,
        min_book_prob=None, exclude=None, tier=None, league=None,
        parlay=None, total_odds=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


class FootballStatsModeTest(unittest.TestCase):
    def test_football_defaults_to_stats_first(self):
        a = args()
        self.assertTrue(configure_pick_mode(a, "sr:sport:1"))
        self.assertEqual(a.strategy, "rollover")
        self.assertEqual(a.min_edge, 0.02)
        self.assertEqual(a.min_prob, 0.55)
        self.assertEqual(a.min_book_prob, 0.50)
        self.assertEqual(a.exclude, "default")
        self.assertEqual(a.tier, "top")
        self.assertEqual(a.parlay, 3.0)

    def test_explicit_league_skips_default_tier(self):
        a = args(league="laliga")
        configure_pick_mode(a, "sr:sport:1")
        self.assertIsNone(a.tier)
        self.assertEqual(a.parlay, 3.0)

    def test_explicit_safest_skips_stats_defaults(self):
        a = args(strategy="safest")
        self.assertFalse(configure_pick_mode(a, "sr:sport:1"))
        self.assertEqual(a.strategy, "safest")
        self.assertIsNone(a.min_edge)
        self.assertIsNone(a.min_prob)

    def test_other_sports_stay_safest(self):
        a = args()
        self.assertFalse(configure_pick_mode(a, "sr:sport:5"))
        self.assertEqual(a.strategy, "safest")
        self.assertIsNone(a.min_edge)
        self.assertIsNone(a.exclude)

    def test_explicit_min_edge_kept(self):
        a = args(min_edge=0.05)
        configure_pick_mode(a, "sr:sport:1")
        self.assertEqual(a.min_edge, 0.05)

    def test_min_edge_forces_edge_strategy(self):
        a = args(strategy="safest", min_edge=0.02)
        self.assertTrue(configure_pick_mode(a, "sr:sport:2"))
        self.assertEqual(a.strategy, "edge")

    def test_skips_btts_in_football_stats_mode(self):
        self.assertEqual(
            stats_first_markets(["1", "18", "10", "29", "11"], "sr:sport:1", None, True),
            ["1", "18", "10", "11"],
        )
        self.assertEqual(
            stats_first_markets(["1", "29"], "sr:sport:1", "1x2,gg", True),
            ["1", "29"],
        )
        self.assertEqual(
            stats_first_markets(["1", "29"], "sr:sport:1", None, False),
            ["1", "29"],
        )


def odds_args(**kw) -> argparse.Namespace:
    base = dict(leg=None, parlay=None, min_odds=None, max_odds=None, total_odds=None, legs=None, tolerance=0.10)
    base.update(kw)
    return argparse.Namespace(**base)


class OddsShortcutsTest(unittest.TestCase):
    def test_leg_sets_a_tight_band(self):
        a = odds_args(leg=1.20)
        apply_odds_shortcuts(a)
        self.assertAlmostEqual(a.min_odds, 1.20)
        self.assertAlmostEqual(a.max_odds, 1.44)
        self.assertEqual(a.legs, "1-8")

    def test_parlay_is_total_odds_and_fills_legs(self):
        a = odds_args(leg=1.20, parlay=10)
        apply_odds_shortcuts(a)
        self.assertEqual(a.total_odds, 10)
        lo, hi = (int(x) for x in a.legs.split("-"))
        self.assertGreaterEqual(lo, 1)
        self.assertGreaterEqual(hi, lo)
        self.assertGreaterEqual(1.44 ** hi, 9.0)
        self.assertLessEqual(1.20 ** lo, 11.0)

    def test_explicit_min_max_legs_kept(self):
        a = odds_args(leg=1.20, parlay=10, min_odds=1.25, max_odds=1.80, legs="3-5")
        apply_odds_shortcuts(a)
        self.assertEqual(a.min_odds, 1.25)
        self.assertEqual(a.max_odds, 1.80)
        self.assertEqual(a.legs, "3-5")

    def test_conflicting_parlay_and_total_rejected(self):
        with self.assertRaises(ValueError):
            apply_odds_shortcuts(odds_args(parlay=8, total_odds=10))

    def test_plain_defaults(self):
        a = odds_args()
        apply_odds_shortcuts(a)
        self.assertEqual(a.min_odds, 1.20)
        self.assertEqual(a.max_odds, 2.00)
        self.assertEqual(a.legs, "1-8")

    def test_auto_legs_can_reach_target(self):
        spec = auto_legs(1.20, 1.44, 10, 0.10)
        lo, hi = (int(x) for x in spec.split("-"))
        self.assertLessEqual(1.20 ** lo, 11.0)
        self.assertGreaterEqual(1.44 ** hi, 9.0)


if __name__ == "__main__":
    unittest.main()
