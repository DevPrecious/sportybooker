import time
import unittest

from motivated.cli import filter_by_motivation
from motivated.motivation import TeamMotivation
from motivated.selector import Candidate, OddsSpec, extract_candidates


def make_mot(uid: int, zone: str, score: float, label: str = "") -> TeamMotivation:
    return TeamMotivation(uid, 2, 20, 50, 25, 13, zone, score, label or zone, zone)


class SelectorAndFilterTest(unittest.TestCase):
    def test_extract_candidates_keeps_only_win_draw_and_handicaps(self):
        now_ms = int(time.time() * 1000)
        events = [
            {
                "eventId": "sr:match:100",
                "homeTeamName": "Arsenal",
                "awayTeamName": "Leeds",
                "homeTeamId": "sr:competitor:42",
                "awayTeamId": "sr:competitor:34",
                "estimateStartTime": now_ms + 3600_000,
                "status": 0,
                "tournamentName": "Premier League",
                "sport": {"id": "sr:sport:1", "category": {"name": "England", "tournament": {"id": "sr:tournament:17"}}},
                "markets": [
                    {
                        "id": "10",
                        "name": "Double Chance",
                        "status": 0,
                        "outcomes": [
                            {"id": "9", "desc": "Home or Draw", "odds": "1.28", "probability": 0.75, "isActive": 1},
                            {"id": "10", "desc": "Home or Away", "odds": "1.22", "probability": 0.78, "isActive": 1},
                            {"id": "11", "desc": "Draw or Away", "odds": "1.85", "probability": 0.51, "isActive": 1},
                        ],
                    },
                    {
                        "id": "14",
                        "name": "Handicap",
                        "specifier": "hcp=0:1",
                        "status": 0,
                        "outcomes": [
                            {"id": "1711", "desc": "Home (0:1)", "odds": "1.95", "probability": 0.48, "isActive": 1},
                            {"id": "1712", "desc": "Draw (0:1)", "odds": "3.60", "probability": 0.25, "isActive": 1},
                            {"id": "1713", "desc": "Away (0:1)", "odds": "1.85", "probability": 0.51, "isActive": 1},
                        ],
                    },
                    {
                        "id": "16",
                        "name": "Asian Handicap",
                        "specifier": "hcp=-0.5",
                        "status": 0,
                        "outcomes": [
                            {"id": "1714", "desc": "Home (-0.5)", "odds": "1.55", "probability": 0.62, "isActive": 1},
                            {"id": "1715", "desc": "Away (+0.5)", "odds": "2.35", "probability": 0.38, "isActive": 1},
                        ],
                    },
                    {
                        "id": "18",
                        "name": "Over/Under",
                        "specifier": "total=2.5",
                        "status": 0,
                        "outcomes": [
                            {"id": "12", "desc": "Over 2.5", "odds": "1.70", "probability": 0.56, "isActive": 1},
                        ],
                    },
                ],
            }
        ]
        spec = OddsSpec(min_odds=1.15, max_odds=2.50, hours_ahead=24)
        cands = extract_candidates(events, spec, now_ms=now_ms)
        descs = [c.outcome_desc for c in cands]
        self.assertIn("Home or Draw", descs)
        self.assertIn("Draw or Away", descs)
        self.assertIn("Home (0:1)", descs)
        self.assertIn("Away (0:1)", descs)
        self.assertIn("Home (-0.5)", descs)
        self.assertIn("Away (+0.5)", descs)
        # Disallowed outcomes must never appear
        self.assertNotIn("Home or Away", descs)
        self.assertNotIn("Draw (0:1)", descs)
        self.assertNotIn("Over 2.5", descs)

    def test_filter_by_motivation_zones_and_mismatch(self):
        title_mot = make_mot(1, "title", 0.88)
        releg_mot = make_mot(2, "relegation", 0.82)
        mid_mot = make_mot(3, "midtable", 0.15)

        c_title_vs_mid = Candidate(
            event_id="e1", home="A", away="B", tournament="T", start_ms=1, market_id="10",
            market_name="DC", specifier="", outcome_id="9", outcome_desc="Home or Draw",
            odds=1.35, probability=0.72, side="home", mot_home=title_mot, mot_away=mid_mot,
        )
        c_mid_vs_title = Candidate(
            event_id="e1", home="A", away="B", tournament="T", start_ms=1, market_id="10",
            market_name="DC", specifier="", outcome_id="11", outcome_desc="Draw or Away",
            odds=1.90, probability=0.50, side="away", mot_home=title_mot, mot_away=mid_mot,
        )
        c_releg_vs_releg = Candidate(
            event_id="e2", home="C", away="D", tournament="T", start_ms=2, market_id="16",
            market_name="AH", specifier="hcp=0.5", outcome_id="1714", outcome_desc="Home (+0.5)",
            odds=1.45, probability=0.66, side="home", mot_home=releg_mot, mot_away=releg_mot,
        )

        all_cands = [c_title_vs_mid, c_mid_vs_title, c_releg_vs_releg]

        kept_any = filter_by_motivation(all_cands, min_motivation=0.45, min_mot_gap=-0.10, focus="any")
        self.assertEqual([c.event_id for c in kept_any], ["e1", "e2"])

        kept_releg = filter_by_motivation(all_cands, min_motivation=0.45, min_mot_gap=-0.10, focus="relegation")
        self.assertEqual([c.event_id for c in kept_releg], ["e2"])

        kept_mismatch = filter_by_motivation(all_cands, min_motivation=0.45, min_mot_gap=0.20, focus="mismatch")
        self.assertEqual([c.event_id for c in kept_mismatch], ["e1"])

    def test_target_aware_one_per_event_reaches_40x_parlay(self):
        import random
        from motivated.cli import auto_legs
        from motivated.selector import build_slip, one_per_event, total_odds

        mot = make_mot(1, "title", 0.85)
        opp = make_mot(2, "midtable", 0.15)
        # 10 events, each offering a short +2.5 handicap at 1.22 (which only multiplies to 1.22^10 = 7.3x)
        # and a Win/Draw (1X) at 1.50 (which can reach ~40x in 9-10 legs).
        cands = []
        for i in range(10):
            cands.append(
                Candidate(
                    event_id=f"ev{i}", home="H", away="A", tournament="T", start_ms=i + 1,
                    market_id="16", market_name="AH", specifier="hcp=2.5", outcome_id="1714",
                    outcome_desc="Home (+2.5)", odds=1.22, probability=0.80, blended_prob=0.82,
                    edge=0.03, side="home", mot_home=mot, mot_away=opp,
                )
            )
            cands.append(
                Candidate(
                    event_id=f"ev{i}", home="H", away="A", tournament="T", start_ms=i + 1,
                    market_id="10", market_name="DC", specifier="", outcome_id="9",
                    outcome_desc="Home or Draw", odds=1.50, probability=0.65, blended_prob=0.68,
                    edge=0.04, side="home", mot_home=mot, mot_away=opp,
                )
            )

        legs_str = auto_legs(1.18, 2.00, 40.0, 0.10)
        lo, hi = (int(x) for x in legs_str.split("-"))
        spec = OddsSpec(min_odds=1.18, max_odds=2.00, total_odds=40.0, tolerance=0.10, min_legs=lo, max_legs=hi)
        pool = one_per_event(cands, "rollover", random.Random(0), spec=spec)
        picks = build_slip(pool, spec)
        got = total_odds(picks)
        self.assertGreaterEqual(got, 36.0)
        self.assertLessEqual(got, 44.0)


if __name__ == "__main__":
    unittest.main()
