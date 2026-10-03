import random
import unittest

from booker.backtest import (
    Sample,
    baseline_brier,
    brier,
    calibration_table,
    evaluate,
    finished_matches,
    fit_platt,
    football_samples,
    prior,
)
from booker.edge import EdgeConfig
from booker.stats import MatchResult, TeamStats


def res(mid, h, a, hg, ag, uts):
    return MatchResult(match_id=mid, home_uid=h, away_uid=a, home_goals=hg, away_goals=ag, uts=uts)


class DataTest(unittest.TestCase):
    def test_prior_only_uses_matches_before_kickoff(self):
        hist = [res(3, 1, 2, 1, 0, 300), res(2, 1, 2, 1, 0, 200), res(1, 1, 2, 1, 0, 100)]
        self.assertEqual([m.match_id for m in prior(hist, 250, 10)], [2, 1])
        self.assertEqual([m.match_id for m in prior(hist, 250, 1)], [2])

    def test_finished_matches_dedupes_and_requires_both_histories(self):
        shared = res(10, 1, 2, 2, 2, 500)
        histories = {1: [shared, res(11, 1, 9, 1, 0, 400)], 2: [shared]}
        out = finished_matches(histories, 0, 1000)
        self.assertEqual([m.match_id for m in out], [10])  # 11 dropped: team 9 has no history


class EvalTest(unittest.TestCase):
    def test_football_samples_truth_labels(self):
        cfg = EdgeConfig()
        strong = TeamStats(1, tuple(res(i, 1, 9, 3, 0, 100 - i) for i in range(8)), ())
        weak = TeamStats(2, tuple(res(50 + i, 2, 9, 0, 3, 100 - i) for i in range(8)), ())
        m = res(99, 1, 2, 2, 1, 200)
        samples = football_samples(m, strong, weak, cfg)
        truth = {s.market: s.won for s in samples}
        self.assertTrue(truth["Home"] and truth["1X"] and truth["Over 2.5"] and truth["BTTS Yes"])
        self.assertFalse(truth["Draw"] or truth["Away"] or truth["Under 2.5"] or truth["X2"])
        p = {s.market: s.p for s in samples}
        self.assertGreater(p["Home"], 0.7)

    def test_evaluate_uses_pre_match_history(self):
        cfg = EdgeConfig(min_matches=2, form_games=5)
        # Team 1's later matches (after uts 500) are all heavy losses; only earlier wins should be used.
        hist1 = [res(200 + i, 1, 9, 0, 5, 900 - i) for i in range(5)] + [res(100 + i, 1, 9, 2, 0, 400 - i) for i in range(5)]
        hist2 = [res(300 + i, 2, 9, 1, 1, 400 - i) for i in range(5)]
        target = res(1, 1, 2, 1, 0, 500)
        samples = evaluate([target], {1: hist1, 2: hist2}, cfg, is_football=True)
        p_home = next(s.p for s in samples if s.market == "Home")
        self.assertGreater(p_home, 0.5)


class MetricsTest(unittest.TestCase):
    def test_brier_and_baseline(self):
        s = [Sample(1, 0, "Home", 0.9, True), Sample(2, 0, "Home", 0.9, False)]
        self.assertAlmostEqual(brier(s), (0.01 + 0.81) / 2)
        self.assertAlmostEqual(baseline_brier(s), 0.25)

    def test_calibration_table_buckets(self):
        s = [Sample(i, 0, "Home", 0.75, i % 2 == 0) for i in range(10)]
        rows = calibration_table(s)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["band"], "70%-80%")
        self.assertAlmostEqual(rows[0]["actual"], 0.5)

    def test_platt_shrinks_overconfident_model(self):
        rng = random.Random(1)
        # Model says 85% but the truth is a coin flip -> a should drop well below 1.
        s = [Sample(i, 0, "Home", 0.85, rng.random() < 0.5) for i in range(600)]
        a, b = fit_platt(s)
        cfg = EdgeConfig(calibration=(a, b))
        self.assertLess(cfg.calibrate(0.85), 0.65)

    def test_platt_leaves_calibrated_model_alone(self):
        rng = random.Random(2)
        s = [Sample(i, 0, "Home", p, rng.random() < p) for i, p in enumerate(rng.uniform(0.3, 0.9) for _ in range(2000))]
        a, b = fit_platt(s)
        self.assertAlmostEqual(a, 1.0, delta=0.15)
        self.assertAlmostEqual(b, 0.0, delta=0.15)


if __name__ == "__main__":
    unittest.main()
