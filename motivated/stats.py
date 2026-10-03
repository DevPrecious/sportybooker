"""Team form, head-to-head, and live season standings from Sportradar's public stats feed."""

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
    season_id: int | None = None
    utid: int | None = None
    tid: int | None = None

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
    seasons_by_utid: dict[int, int] | None = None
    seasons_by_tid: dict[int, int] | None = None

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
        sid = raw.get("_seasonid")
        utid = raw.get("_utid")
        tid = raw.get("_tid")
        return MatchResult(
            match_id=int(raw["_id"]),
            home_uid=int(teams["home"]["uid"]),
            away_uid=int(teams["away"]["uid"]),
            home_goals=int(result["home"]),
            away_goals=int(result["away"]),
            uts=int((raw.get("time") or {}).get("uts") or 0),
            neutral=bool(raw.get("neutralground")),
            season_id=int(sid) if sid else None,
            utid=int(utid) if utid else None,
            tid=int(tid) if tid else None,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _extract_seasons(data: dict[str, Any]) -> tuple[dict[int, int], dict[int, int]]:
    by_utid: dict[int, int] = {}
    by_tid: dict[int, int] = {}
    for utid_str, ut in (data.get("uniquetournaments") or {}).items():
        cs = ut.get("currentseason") or ut.get("seasonid")
        if cs and str(utid_str).isdigit():
            by_utid[int(utid_str)] = int(cs)
    for tid_str, t in (data.get("tournaments") or {}).items():
        cs = t.get("currentseason") or t.get("seasonid")
        if cs and str(tid_str).isdigit():
            by_tid[int(tid_str)] = int(cs)
            utid = t.get("_utid")
            if utid:
                by_utid.setdefault(int(utid), int(cs))
    return by_utid, by_tid


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
        self._season_map_cache: tuple[dict[int, int], dict[int, int]] | None = None
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

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

    def last_matches_with_seasons(self, uid: int, n: int = 10) -> tuple[list[MatchResult], dict[int, int], dict[int, int]]:
        data = self._get(f"stats_team_lastx/{uid}/{n}")
        if not isinstance(data, dict):
            return [], {}, {}
        matches = [m for m in (_parse_match(r) for r in data.get("matches") or []) if m]
        matches.sort(key=lambda m: -m.uts)
        by_utid, by_tid = _extract_seasons(data)
        return matches, by_utid, by_tid

    def last_matches(self, uid: int, n: int = 10) -> list[MatchResult]:
        matches, _, _ = self.last_matches_with_seasons(uid, n)
        return matches

    def head_to_head(self, uid_a: int, uid_b: int) -> list[MatchResult]:
        data = self._get(f"stats_team_versusrecent/{uid_a}/{uid_b}")
        if not isinstance(data, dict):
            return []
        matches = [m for m in (_parse_match(r) for r in data.get("matches") or []) if m]
        matches.sort(key=lambda m: -m.uts)
        return matches

    def tournament_season_maps(self) -> tuple[dict[int, int], dict[int, int]]:
        """Return (utid -> currentseason, tid -> currentseason) from Sportradar's football config tree."""
        if self._season_map_cache is not None:
            return self._season_map_cache
        tree = self._get("config_tree_mini/41/0/1")
        ut_to_season: dict[int, int] = {}
        t_to_season: dict[int, int] = {}
        if isinstance(tree, list) and tree:
            for cat in tree[0].get("realcategories") or []:
                for utid_str, ut in (cat.get("uniquetournaments") or {}).items():
                    cs = ut.get("currentseason") or ut.get("seasonid")
                    if cs and str(utid_str).isdigit():
                        ut_to_season[int(utid_str)] = int(cs)
                for t in cat.get("tournaments") or []:
                    cs = t.get("currentseason") or t.get("seasonid")
                    tid = t.get("_id")
                    if cs and tid:
                        t_to_season[int(tid)] = int(cs)
                        if t.get("_utid"):
                            ut_to_season.setdefault(int(t["_utid"]), int(cs))
        self._season_map_cache = (ut_to_season, t_to_season)
        return self._season_map_cache

    def resolve_season_id(
        self,
        tournament_id_str: str,
        home_stats: TeamStats | None = None,
        away_stats: TeamStats | None = None,
    ) -> int | None:
        """Resolve a SportyBet tournament id ('sr:tournament:17' or 'sr:simple_tournament:27213') to a Sportradar season_id."""
        tid = sr_numeric_id(tournament_id_str)
        is_simple = "simple_tournament" in (tournament_id_str or "")

        # 1. Check team stats' own tournament metadata first (zero network overhead)
        for ts in (home_stats, away_stats):
            if ts is None or tid is None:
                continue
            if is_simple and ts.seasons_by_tid and tid in ts.seasons_by_tid:
                return ts.seasons_by_tid[tid]
            if not is_simple and ts.seasons_by_utid and tid in ts.seasons_by_utid:
                return ts.seasons_by_utid[tid]
            if ts.seasons_by_utid and tid in ts.seasons_by_utid:
                return ts.seasons_by_utid[tid]
            if ts.seasons_by_tid and tid in ts.seasons_by_tid:
                return ts.seasons_by_tid[tid]

        # 2. Check the global Sportradar tournament tree
        if tid is not None:
            ut_map, t_map = self.tournament_season_maps()
            if is_simple:
                sid = t_map.get(tid) or ut_map.get(tid)
            else:
                sid = ut_map.get(tid) or t_map.get(tid)
            if sid:
                return sid

        # 3. Fallback: if both teams recently played in the same domestic season, use that season
        if home_stats and away_stats and home_stats.recent and away_stats.recent:
            h_seasons = {m.season_id for m in home_stats.recent[:4] if m.season_id}
            a_seasons = {m.season_id for m in away_stats.recent[:4] if m.season_id}
            common = h_seasons & a_seasons
            if len(common) == 1:
                return next(iter(common))
        return None

    def season_tables_raw(self, season_id: int) -> dict[str, Any] | None:
        data = self._get(f"stats_season_tables/{season_id}")
        return data if isinstance(data, dict) else None

    def fixtures_stats(
        self, fixtures: list[tuple[int, int]], n: int = 10
    ) -> dict[tuple[int, int], tuple[TeamStats, TeamStats]]:
        """Fetch stats for many (home_uid, away_uid) pairs concurrently."""
        uniq = list(dict.fromkeys(fixtures))

        def work(pair: tuple[int, int]) -> tuple[tuple[int, int], tuple[TeamStats, TeamStats]]:
            home, away = pair
            h_rec, h_ut, h_t = self.last_matches_with_seasons(home, n)
            a_rec, a_ut, a_t = self.last_matches_with_seasons(away, n)
            h2h = tuple(self.head_to_head(home, away))
            return pair, (
                TeamStats(home, tuple(h_rec), h2h, h_ut, h_t),
                TeamStats(away, tuple(a_rec), h2h, a_ut, a_t),
            )

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return dict(pool.map(work, uniq))

    def fetch_season_tables_batch(self, season_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Fetch raw `stats_season_tables/{sid}` for multiple seasons concurrently."""
        uniq = list(dict.fromkeys(s for s in season_ids if s))
        if not uniq:
            return {}

        def work(sid: int) -> tuple[int, dict[str, Any] | None]:
            return sid, self.season_tables_raw(sid)

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return {sid: data for sid, data in pool.map(work, uniq) if data is not None}
