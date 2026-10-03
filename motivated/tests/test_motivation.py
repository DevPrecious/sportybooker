import unittest

from motivated.motivation import (
    LeagueTable,
    TableRow,
    evaluate_team_motivation,
    parse_season_tables,
    select_table_for_fixture,
)


def make_sample_league(played: int = 28, max_rounds: int = 38) -> LeagueTable:
    rows = (
        TableRow(1, "Leader FC", 1, played, 20, 5, 3, 60, 20, 40, 65, "Champions League", "promotioncolor_cl"),
        TableRow(2, "Chaser FC", 2, played, 19, 6, 3, 55, 22, 33, 63, "Champions League", "promotioncolor_cl"),
        TableRow(3, "Third United", 3, played, 17, 5, 6, 48, 28, 20, 56, "Champions League", "promotioncolor_cl"),
        TableRow(4, "Fourth City", 4, played, 15, 6, 7, 44, 30, 14, 51, "Champions League", "promotioncolor_cl"),
        TableRow(5, "Fifth Athletic", 5, played, 15, 5, 8, 42, 31, 11, 50, "UEFA Europa League", "promotioncolor_uefa"),
        TableRow(6, "Sixth Rovers", 6, played, 14, 7, 7, 40, 30, 10, 49, "", ""),
        TableRow(10, "Midtable Town", 10, played, 10, 8, 10, 34, 35, -1, 38, "", ""),
        TableRow(11, "Bored Wanderers", 11, played, 10, 7, 11, 32, 35, -3, 37, "", ""),
        TableRow(16, "Nervous Albion", 16, played, 7, 7, 14, 26, 42, -16, 28, "", ""),
        TableRow(17, "Edge Survival", 17, played, 6, 7, 15, 25, 45, -20, 25, "", ""),
        TableRow(18, "Drop Zone FC", 18, played, 6, 6, 16, 24, 48, -24, 24, "Relegation", "promotioncolor_relegation"),
        TableRow(19, "Struggling Utd", 19, played, 5, 6, 17, 21, 50, -29, 21, "Relegation", "promotioncolor_relegation"),
        TableRow(20, "Doomed County", 20, played, 1, 2, 25, 12, 70, -58, 5, "Relegation", "promotioncolor_relegation"),
    )
    return LeagueTable(season_id=100, name="Sample League", max_rounds=max_rounds, current_round=played, rows=rows)


class MotivationEngineTest(unittest.TestCase):
    def test_title_race_detection(self):
        tbl = make_sample_league()
        m1 = evaluate_team_motivation(tbl, 1)
        m2 = evaluate_team_motivation(tbl, 2)
        self.assertEqual(m1.zone, "title")
        self.assertGreater(m1.score, 0.75)
        self.assertIn("Title", m1.label)
        self.assertEqual(m2.zone, "title")
        self.assertGreater(m2.score, 0.75)

    def test_europe_defend_and_chase(self):
        tbl = make_sample_league()
        m4 = evaluate_team_motivation(tbl, 4)  # 4th (51 pts), 1 pt ahead of 5th (UEL) and 2 pts ahead of 6th (outside)
        m5 = evaluate_team_motivation(tbl, 5)  # 5th (50 pts), chasing 4th UCL (51 pts) and defending UEL over 6th (49 pts)
        m6 = evaluate_team_motivation(tbl, 6)  # 6th (49 pts), 1 pt behind 5th (UEL cutoff)
        self.assertEqual(m4.zone, "europe")
        self.assertTrue(m4.is_motivated)
        self.assertEqual(m5.zone, "europe")
        self.assertTrue(m5.is_motivated)
        self.assertEqual(m6.zone, "europe")
        self.assertTrue(m6.is_motivated)
        self.assertIn("Chase", m6.label)

    def test_relegation_and_survival(self):
        tbl = make_sample_league()
        m17 = evaluate_team_motivation(tbl, 17)  # 25 pts, 1 pt above 18th (24 pts)
        m18 = evaluate_team_motivation(tbl, 18)  # 24 pts, 1 pt behind 17th (25 pts)
        self.assertEqual(m17.zone, "relegation")
        self.assertIn("Survival", m17.label)
        self.assertGreater(m17.score, 0.75)
        self.assertEqual(m18.zone, "relegation")
        self.assertIn("Releg", m18.label)
        self.assertGreater(m18.score, 0.80)

    def test_midtable_and_mathematically_relegated_have_low_motivation(self):
        tbl = make_sample_league(played=34, max_rounds=38)  # 4 games (12 pts) left; Doomed has 5 pts vs 25 safe
        mid = evaluate_team_motivation(tbl, 10)
        doomed = evaluate_team_motivation(tbl, 20)
        self.assertEqual(mid.zone, "midtable")
        self.assertFalse(mid.is_motivated)
        self.assertLess(mid.score, 0.25)
        self.assertEqual(doomed.zone, "relegated")
        self.assertFalse(doomed.is_motivated)
        self.assertLess(doomed.score, 0.20)

    def test_early_season_damping(self):
        tbl = make_sample_league(played=1, max_rounds=38)
        m = evaluate_team_motivation(tbl, 1, min_games=2)
        self.assertEqual(m.zone, "early")
        self.assertFalse(m.is_motivated)

    def test_parse_and_select_split_stage_table(self):
        raw = {
            "name": "Serie C",
            "tables": [
                {
                    "name": "Regular Season",
                    "maxrounds": 19,
                    "currentround": 19,
                    "tablerows": [
                        {"pos": 1, "total": 19, "pointsTotal": 35, "team": {"uid": 10, "name": "A"}},
                        {"pos": 2, "total": 19, "pointsTotal": 32, "team": {"uid": 20, "name": "B"}},
                    ],
                },
                {
                    "name": "Main Round Group B",
                    "maxrounds": 6,
                    "currentround": 4,
                    "tablerows": [
                        {
                            "pos": 1,
                            "total": 4,
                            "pointsTotal": 9,
                            "team": {"uid": 10, "name": "A"},
                            "promotion": {"name": "Playoffs", "cssclass": "promotioncolor_playoff"},
                        },
                        {"pos": 2, "total": 4, "pointsTotal": 7, "team": {"uid": 20, "name": "B"}},
                    ],
                },
            ],
        }
        tables = parse_season_tables(140050, raw)
        self.assertEqual(len(tables), 2)
        chosen = select_table_for_fixture(tables, 10, 20)
        self.assertEqual(chosen.name, "Main Round Group B")


if __name__ == "__main__":
    unittest.main()
