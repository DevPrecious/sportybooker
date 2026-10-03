import unittest
from datetime import date, datetime, time

from booker.cli import _day_window_ms, _parse_time_range


def local_ms(day: date, t: time) -> int:
    return int(datetime.combine(day, t).astimezone().timestamp() * 1000)


class WindowTest(unittest.TestCase):
    day = date(2026, 9, 17)

    def test_full_day(self):
        start, end = _day_window_ms(self.day)
        self.assertEqual(start, local_ms(self.day, time(0, 0)))
        self.assertEqual(end, local_ms(date(2026, 9, 18), time(0, 0)))

    def test_time_range_is_inclusive_of_end_minute(self):
        start, end = _day_window_ms(self.day, "12:00-23:59")
        self.assertEqual(start, local_ms(self.day, time(12, 0)))
        self.assertLessEqual(local_ms(self.day, time(23, 59)), end - 1)
        self.assertLess(local_ms(self.day, time(11, 30)), start)

    def test_open_ended_time(self):
        self.assertEqual(_parse_time_range("18:00"), (time(18, 0), time(23, 59, 59)))

    def test_range_past_midnight_runs_into_next_day(self):
        start, end = _day_window_ms(self.day, "23:00-02:00")
        self.assertEqual(start, local_ms(self.day, time(23, 0)))
        self.assertGreater(local_ms(date(2026, 9, 18), time(1, 30)), start)
        self.assertLess(local_ms(date(2026, 9, 18), time(1, 30)), end)
        self.assertGreater(local_ms(date(2026, 9, 18), time(3, 0)), end)

    def test_empty_range_rejected(self):
        with self.assertRaises(ValueError):
            _parse_time_range("20:00-20:00")


if __name__ == "__main__":
    unittest.main()
