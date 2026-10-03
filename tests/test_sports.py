import unittest

from booker.sports import resolve_leagues, resolve_markets, resolve_sport


class ResolveTest(unittest.TestCase):
    def test_sport_aliases(self):
        self.assertEqual(resolve_sport("football"), "sr:sport:1")
        self.assertEqual(resolve_sport("Ice Hockey"), "sr:sport:4")
        self.assertEqual(resolve_sport("5"), "sr:sport:5")
        self.assertEqual(resolve_sport("sr:sport:2"), "sr:sport:2")
        with self.assertRaises(ValueError):
            resolve_sport("quidditch")

    def test_markets(self):
        self.assertEqual(resolve_markets("1x2, ou,29", "sr:sport:1"), ["1", "18", "29"])
        self.assertEqual(resolve_markets(None, "sr:sport:5"), ["186", "1", "18", "219"])
        with self.assertRaises(ValueError):
            resolve_markets("nope", "sr:sport:1")

    def test_tiers_and_excludes(self):
        from booker.sports import DEFAULT_EXCLUDES, resolve_excludes
        top = resolve_leagues(None, "big5")
        self.assertIn("spain laliga", top)
        self.assertEqual(len(top), 5)
        self.assertIn("uefa champions league", resolve_leagues("laliga", "europe"))
        with self.assertRaises(ValueError):
            resolve_leagues(None, "galactic")
        self.assertEqual(resolve_excludes("default"), DEFAULT_EXCLUDES)
        self.assertEqual(resolve_excludes("cup, Friendly"), ["cup", "friendly"])
        self.assertIsNone(resolve_excludes(""))

    def test_leagues(self):
        self.assertIsNone(resolve_leagues(None))
        self.assertIsNone(resolve_leagues(" , "))
        self.assertEqual(resolve_leagues("laliga"), ["spain laliga"])
        self.assertEqual(resolve_leagues("EFL"),
                         ["england championship", "england league one", "england league two"])
        self.assertEqual(resolve_leagues("laliga, europa league, Norway Eliteserien"),
                         ["spain laliga", "uefa europa league", "norway eliteserien"])


if __name__ == "__main__":
    unittest.main()
