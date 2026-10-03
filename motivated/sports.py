"""Sport, league, and market resolution restricted to Win/Draw (1X/X2) and Handicap markets."""

from __future__ import annotations

from .client import Sport

KNOWN_SPORTS: dict[str, str] = {
    "football": "sr:sport:1",
    "soccer": "sr:sport:1",
    "futsal": "sr:sport:29",
    "esoccer": "sr:sport:137",
}

# Strictly Win or Draw (Double Chance 10) and Handicap (European 14, Asian 16) markets only.
ALLOWED_MARKET_IDS = ("10", "16", "14")
DEFAULT_MARKETS = ["10", "16", "14"]

MARKET_ALIASES: dict[str, list[str]] = {
    "dc": ["10"],
    "double chance": ["10"],
    "win or draw": ["10"],
    "windraw": ["10"],
    "1x": ["10"],
    "x2": ["10"],
    "ah": ["16"],
    "asian handicap": ["16"],
    "asian": ["16"],
    "eh": ["14"],
    "euro handicap": ["14"],
    "european handicap": ["14"],
    "3way handicap": ["14"],
    "handicap": ["16", "14"],
    "hcp": ["16", "14"],
    "all": ["10", "16", "14"],
}

LEAGUE_ALIASES: dict[str, list[str]] = {
    "epl": ["england premier league"],
    "premier league": ["england premier league"],
    "efl": ["england championship", "england league one", "england league two"],
    "championship": ["england championship"],
    "league one": ["england league one"],
    "league two": ["england league two"],
    "laliga": ["spain laliga"],
    "la liga": ["spain laliga"],
    "segunda": ["spain laliga 2", "spain segunda", "spain laliga hypermotion"],
    "seriea": ["italy serie a"],
    "serie a": ["italy serie a"],
    "serieb": ["italy serie b"],
    "bundesliga": ["germany bundesliga"],
    "bundesliga 2": ["germany 2. bundesliga"],
    "ligue1": ["france ligue 1"],
    "ligue 1": ["france ligue 1"],
    "ligue2": ["france ligue 2"],
    "eredivisie": ["netherlands eredivisie"],
    "primeira": ["portugal primeira liga", "portugal liga portugal"],
    "mls": ["usa mls"],
    "ucl": ["uefa champions league"],
    "champions league": ["uefa champions league"],
    "uel": ["uefa europa league"],
    "europa": ["uefa europa league"],
    "europa league": ["uefa europa league"],
    "uecl": ["uefa conference league"],
    "conference league": ["uefa conference league"],
    "nations league": ["uefa nations league"],
    "brasileirao": ["brazil brasileiro serie a"],
}

LEAGUE_TIERS: dict[str, list[str]] = {
    "top": [
        "england premier league", "spain laliga", "italy serie a", "germany bundesliga", "france ligue 1",
        "uefa champions league", "uefa europa league", "uefa conference league", "uefa nations league",
        "netherlands eredivisie", "portugal primeira liga", "portugal liga portugal", "england championship",
        "usa mls", "brazil brasileiro serie a", "argentina primera lpf", "turkiye super lig", "belgium pro league",
        "scotland premiership", "world cup qualification", "european championship qualification",
    ],
    "big5": ["england premier league", "spain laliga", "italy serie a", "germany bundesliga", "france ligue 1"],
    "europe": ["uefa champions league", "uefa europa league", "uefa conference league"],
}

# Only exclude friendlies and youth/reserve exhibitions; real leagues are gated by Sportradar's live table check.
DEFAULT_EXCLUDES = [
    "friendly", "club friendly", "reserve", "u17", "u18", "u19", "youth", "next pro",
]


def _split_tokens(spec: str) -> list[str]:
    return [" ".join(t.strip().lower().split()) for t in spec.split(",") if t.strip()]


def resolve_leagues(spec: str | None, tiers: str | None = None) -> list[str] | None:
    out: list[str] = []
    for token in _split_tokens(spec or ""):
        out.extend(LEAGUE_ALIASES.get(token, [token]))
    for token in _split_tokens(tiers or ""):
        if token not in LEAGUE_TIERS:
            raise ValueError(f"Unknown tier {token!r}; choose from {', '.join(LEAGUE_TIERS)}")
        out.extend(LEAGUE_TIERS[token])
    return list(dict.fromkeys(out)) or None


def resolve_excludes(spec: str | None) -> list[str] | None:
    if not spec:
        return None
    out: list[str] = []
    for token in _split_tokens(spec):
        out.extend(DEFAULT_EXCLUDES if token == "default" else [token])
    return list(dict.fromkeys(out)) or None


def resolve_sport(name_or_id: str, live: list[Sport] | None = None) -> str:
    key = name_or_id.strip().lower()
    if key.startswith("sr:sport:"):
        return key
    if key.isdigit():
        return f"sr:sport:{key}"
    if key in KNOWN_SPORTS:
        return KNOWN_SPORTS[key]
    if live:
        for s in live:
            if s.name.lower() == key:
                return s.id
        for s in live:
            if key in s.name.lower():
                return s.id
    for known, sid in KNOWN_SPORTS.items():
        if key in known:
            return sid
    raise ValueError(f"Unknown sport {name_or_id!r}. Use --list-sports to see what's available.")


def resolve_markets(spec: str | None, sport_id: str = "sr:sport:1") -> list[str]:
    """Resolve markets strictly within Double Chance (10), Asian Handicap (16), and Euro Handicap (14)."""
    if not spec:
        return list(DEFAULT_MARKETS)
    out: list[str] = []
    for raw in spec.split(","):
        token = " ".join(raw.strip().lower().split())
        if not token:
            continue
        if token in ALLOWED_MARKET_IDS:
            out.append(token)
        elif token in MARKET_ALIASES:
            out.extend(MARKET_ALIASES[token])
        else:
            raise ValueError(
                f"Market {raw.strip()!r} is not allowed. This system only supports Win or Draw (dc/10) "
                f"and Handicap (ah/16, eh/14, handicap) picks."
            )
    return list(dict.fromkeys(out))
