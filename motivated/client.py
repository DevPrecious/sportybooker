"""Thin HTTP client for SportyBet's public (unauthenticated) web endpoints."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

REGIONS = ("ng", "gh", "ke", "tz", "ug", "zm")

_BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.sportybet.com",
    "ClientId": "web",
    "OperId": "2",
    "Platform": "web",
}


class SportyBetError(RuntimeError):
    """Raised when SportyBet returns a non-success bizCode or an HTTP error."""


@dataclass(frozen=True)
class Sport:
    id: str
    name: str
    event_count: int


@dataclass(frozen=True)
class Selection:
    event_id: str
    market_id: str
    outcome_id: str
    specifier: str = ""

    def to_payload(self) -> dict[str, str]:
        return {
            "eventId": self.event_id,
            "marketId": self.market_id,
            "specifier": self.specifier,
            "outcomeId": self.outcome_id,
        }


@dataclass(frozen=True)
class Booking:
    share_code: str
    share_url: str
    deadline_ms: int | None
    raw: dict[str, Any]


class SportyBetClient:
    def __init__(self, region: str = "ng", timeout: float = 20.0, session: requests.Session | None = None):
        region = region.lower()
        if region not in REGIONS:
            raise ValueError(f"Unknown region {region!r}; expected one of {', '.join(REGIONS)}")
        self.region = region
        self.timeout = timeout
        self.base = f"https://www.sportybet.com/api/{region}"
        self.site = f"https://www.sportybet.com/{region}"
        self.session = session or requests.Session()
        self.session.headers.update(_BASE_HEADERS)
        self.session.headers["Referer"] = f"{self.site}/"

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base}{path}"
        try:
            resp = self.session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise SportyBetError(f"{method} {url} failed: {exc}") from exc
        if resp.status_code >= 400:
            raise SportyBetError(f"{method} {url} -> HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise SportyBetError(f"{method} {url} returned non-JSON: {resp.text[:300]}") from exc
        if body.get("bizCode") != 10000:
            raise SportyBetError(f"{method} {url} -> bizCode {body.get('bizCode')}: {body.get('message')}")
        return body.get("data")

    def list_sports(self) -> list[Sport]:
        data = self._request("GET", "/factsCenter/sportList", params={"productId": 3})
        return [Sport(s["id"], s["name"], int(s.get("eventSize", 0))) for s in data or []]

    def iter_upcoming_events(
        self,
        sport_id: str,
        market_ids: list[str],
        page_size: int = 100,
        max_pages: int = 20,
        pause: float = 0.25,
    ) -> Iterator[dict[str, Any]]:
        """Yield raw event dicts (each carrying its `markets` and tournament metadata) page by page."""
        seen: set[str] = set()
        for page in range(1, max_pages + 1):
            data = self._request(
                "GET",
                "/factsCenter/pcUpcomingEvents",
                params={
                    "sportId": sport_id,
                    "marketId": ",".join(market_ids),
                    "pageSize": page_size,
                    "pageNum": page,
                    "option": 1,
                },
            )
            tournaments = (data or {}).get("tournaments") or []
            new = 0
            for t in tournaments:
                t_id = t.get("id")
                t_name = t.get("name")
                for ev in t.get("events") or []:
                    eid = ev.get("eventId")
                    if not eid or eid in seen:
                        continue
                    seen.add(eid)
                    ev.setdefault("tournamentName", t_name)
                    if t_id:
                        ev.setdefault("tournamentId", t_id)
                    new += 1
                    yield ev
            total = int((data or {}).get("totalNum") or 0)
            if new == 0 or len(seen) >= total:
                break
            time.sleep(pause)

    def book(self, selections: list[Selection]) -> Booking:
        """Create a shareable booking code. No login or stake is involved."""
        if not selections:
            raise ValueError("At least one selection is required to book")
        payload = {"selections": [s.to_payload() for s in selections]}
        data = self._request("POST", "/orders/share", json=payload) or {}
        code = data.get("shareCode") or data.get("code") or ""
        url = data.get("shareURL") or data.get("shareUrl") or (f"{self.site}/?shareCode={code}" if code else "")
        deadline = data.get("deadline")
        return Booking(share_code=code, share_url=url, deadline_ms=int(deadline) if deadline else None, raw=data)
