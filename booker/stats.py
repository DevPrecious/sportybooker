"""Team form and head-to-head data from Sportradar's public stats feed.

SportyBet's own "Statistics" tab is a Sportradar widget, and SportyBet's event/competitor ids are
Sportradar ids (sr:match:123 -> 123, sr:competitor:456 -> 456). That lets us look up stats for the
exact fixture we're pricing with no fuzzy team-name matching.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

BASE = "https://stats.fn.sportradar.com/common/en/Etc:UTC/gismo"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Referer": "https://www.sportybet.com/",
    "Origin": "https://www.sportybet.com",
}


@dataclass(frozen=True)
class MatchResult:
    match_id: int
    home_uid: int
    away_uid: int
    home_goals: int
    away_goals: int
    uts: int
    neutral: bool = False

    def outcome_for(self, uid: int) -> str:
        """'W', 'D' or 'L' from the perspective of `uid`."""
        if self.home_goals == self.away_goals:
            return "D"
        home_won = self.home_goals > self.away_goals
        return "W" if (home_won == (uid == self.home_uid)) else "L"

    def goals_for(self, uid: int) -> int:
        return self.home_goals if uid == self.home_uid else self.away_goals

    def goals_against(self, uid: int) -> int:
        return self.away_goals if uid == self.home_uid else self.home_goals


@dataclass(frozen=True)
class TeamStats:
    uid: int
    recent: tuple[MatchResult, ...]  # newest first
    h2h: tuple[MatchResult, ...]  # newest first, may be empty

    @property
    def form_string(self) -> str:
        return "".join(m.outcome_for(self.uid) for m in self.recent[:6])


def sr_numeric_id(sr_id: str) -> int | None:
    """'sr:competitor:4716' -> 4716; returns None if it doesn't look like a Sportradar id."""
    if not sr_id:
        return None
    tail = sr_id.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _parse_match(raw: dict[str, Any]) -> MatchResult | None:
    try:
        teams = raw["teams"]
        result = raw.get("result") or {}
        if result.get("home") is None or result.get("away") is None:
            return None
        if raw.get("postponed") or raw.get("canceled") or raw.get("walkover"):
            return None
        return MatchResult(
            match_id=int(raw["_id"]),
            home_uid=int(teams["home"]["uid"]),
            away_uid=int(teams["away"]["uid"]),
            home_goals=int(result["home"]),
            away_goals=int(result["away"]),
            uts=int((raw.get("time") or {}).get("uts") or 0),
            neutral=bool(raw.get("neutralground")),
        )
    except (KeyError, TypeError, ValueError):
        return None


class StatsClient:
    def __init__(
        self,
        cache_dir: Path | None = Path(".booker_cache"),
        ttl_seconds: int = 3 * 3600,
        timeout: float = 15.0,
        workers: int = 6,
    ):
        self.cache_dir = cache_dir
        self.ttl = ttl_seconds
        self.timeout = timeout
        self.workers = workers
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ fetch
    def _cache_path(self, path: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / (hashlib.sha1(path.encode()).hexdigest() + ".json")

    def _get(self, path: str) -> Any:
        cp = self._cache_path(path)
        if cp and cp.exists() and time.time() - cp.stat().st_mtime < self.ttl:
            try:
                return json.loads(cp.read_text(encoding="utf-8"))
            except ValueError:
                pass
        try:
            resp = self.session.get(f"{BASE}/{path}", timeout=self.timeout)
            resp.raise_for_status()
            body = resp.json()
        except (requests.RequestException, ValueError):
            return None
        docs = body.get("doc") or []
        if not docs or docs[0].get("event") == "exception":
            return None
        data = docs[0].get("data")
        if cp:
            try:
                cp.write_text(json.dumps(data), encoding="utf-8")
            except OSError:
                pass
        return data

    # ----------------------------------------------------------------- public
    def last_matches(self, uid: int, n: int = 10) -> list[MatchResult]:
        data = self._get(f"stats_team_lastx/{uid}/{n}")
        if not isinstance(data, dict):
            return []
        matches = [m for m in (_parse_match(r) for r in data.get("matches") or []) if m]
        matches.sort(key=lambda m: -m.uts)
        return matches

    def head_to_head(self, uid_a: int, uid_b: int) -> list[MatchResult]:
        data = self._get(f"stats_team_versusrecent/{uid_a}/{uid_b}")
        if not isinstance(data, dict):
            return []
        matches = [m for m in (_parse_match(r) for r in data.get("matches") or []) if m]
        matches.sort(key=lambda m: -m.uts)
        return matches

    def team_stats(self, uid: int, opponent_uid: int, n: int = 10) -> TeamStats:
        recent = tuple(self.last_matches(uid, n))
        h2h = tuple(self.head_to_head(uid, opponent_uid))
        return TeamStats(uid=uid, recent=recent, h2h=h2h)

    def fixtures_stats(
        self, fixtures: list[tuple[int, int]], n: int = 10
    ) -> dict[tuple[int, int], tuple[TeamStats, TeamStats]]:
        """Fetch stats for many (home_uid, away_uid) pairs concurrently."""
        uniq = list(dict.fromkeys(fixtures))

        def work(pair: tuple[int, int]) -> tuple[tuple[int, int], tuple[TeamStats, TeamStats]]:
            home, away = pair
            home_recent = tuple(self.last_matches(home, n))
            away_recent = tuple(self.last_matches(away, n))
            h2h = tuple(self.head_to_head(home, away))
            return pair, (TeamStats(home, home_recent, h2h), TeamStats(away, away_recent, h2h))

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return dict(pool.map(work, uniq))
