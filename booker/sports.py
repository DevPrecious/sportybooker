"""Sport and market name resolution."""

from __future__ import annotations

from .client import Sport

# Snapshot of SportyBet's sportList so name resolution works offline / before a network call.
# `--list-sports` fetches the live list.
KNOWN_SPORTS: dict[str, str] = {
    "football": "sr:sport:1",
    "soccer": "sr:sport:1",
    "basketball": "sr:sport:2",
    "baseball": "sr:sport:3",
    "ice hockey": "sr:sport:4",
    "hockey": "sr:sport:4",
    "tennis": "sr:sport:5",
    "handball": "sr:sport:6",
    "boxing": "sr:sport:10",
    "rugby": "sr:sport:12",
    "american football": "sr:sport:16",
    "nfl": "sr:sport:16",
    "snooker": "sr:sport:19",
    "table tennis": "sr:sport:20",
    "cricket": "sr:sport:21",
    "darts": "sr:sport:22",
    "volleyball": "sr:sport:23",
    "futsal": "sr:sport:29",
    "badminton": "sr:sport:31",
    "esport counter-strike": "sr:sport:109",
    "csgo": "sr:sport:109",
    "esport league of legends": "sr:sport:110",
    "lol": "sr:sport:110",
    "esport dota": "sr:sport:111",
    "dota": "sr:sport:111",
    "mma": "sr:sport:117",
    "esoccer": "sr:sport:137",
    "ebasketball": "sr:sport:153",
    "etennis": "sr:sport:196",
    "vfootball": "sr:sport:202120001",
}

# Market aliases -> SportyBet market id. Only the common ones; raw numeric ids are always accepted.
MARKET_ALIASES: dict[str, str] = {
    "1x2": "1",
    "match": "1",
    "ou": "18",
    "over/under": "18",
    "overunder": "18",
    "totals": "18",
    "dc": "10",
    "double chance": "10",
    "dnb": "11",
    "draw no bet": "11",
    "handicap": "14",
    "ah": "16",
    "asian handicap": "16",
    "gg": "29",
    "btts": "29",
    "gg/ng": "29",
    "cs": "45",
    "correct score": "45",
    "htft": "47",
    "1h1x2": "60",
    "winner": "186",
    "ml": "186",
    "moneyline": "186",
    "winner ot": "219",
    "winner (incl. overtime)": "219",
}

# Sensible default market sets per sport id; anything else falls back to DEFAULT_MARKETS.
SPORT_DEFAULT_MARKETS: dict[str, list[str]] = {
    "sr:sport:1": ["1", "18", "10", "29", "11"],
    "sr:sport:137": ["1", "18", "10", "29"],
    "sr:sport:202120001": ["1", "18", "10", "29"],
    "sr:sport:2": ["219", "18", "1"],
    "sr:sport:153": ["219", "18"],
    "sr:sport:16": ["219", "18"],
    "sr:sport:4": ["1", "18", "219"],
    "sr:sport:6": ["1", "18"],
    "sr:sport:12": ["1", "18"],
}
DEFAULT_MARKETS = ["186", "1", "18", "219"]


# League aliases -> lowercase substrings matched against "<country> <tournament>" as SportyBet names them.
# A list means "any of these" (e.g. efl = all three EFL divisions).
LEAGUE_ALIASES: dict[str, list[str]] = {
    "epl": ["england premier league"],
    "premier league": ["england premier league"],
    "efl": ["england championship", "england league one", "england league two"],
    "championship": ["england championship"],
    "league one": ["england league one"],
    "league two": ["england league two"],
    "fa cup": ["england fa cup"],
    "laliga": ["spain laliga"],
    "la liga": ["spain laliga"],
    "segunda": ["spain laliga 2", "spain segunda"],
    "seriea": ["italy serie a"],
    "serie a": ["italy serie a"],
    "serieb": ["italy serie b"],
    "bundesliga": ["germany bundesliga"],
    "bundesliga 2": ["germany 2. bundesliga"],
    "dfb pokal": ["germany dfb pokal"],
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
    "nba": ["usa nba"],
    "euroleague": ["euroleague"],
}


# Named bundles of leagues for --tier.
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

# Competitions/teams with thin, noisy data that hurt parlay accuracy. Used by --exclude default.
DEFAULT_EXCLUDES = [
    "reserve", "u17", "u18", "u19", "u20", "u21", "u23", "youth", "women",
    "amateur", "serie d", "next pro",
]


def _split_tokens(spec: str) -> list[str]:
    return [" ".join(t.strip().lower().split()) for t in spec.split(",") if t.strip()]


def resolve_leagues(spec: str | None, tiers: str | None = None) -> list[str] | None:
    """'laliga,efl,europa' -> lowercase substrings; unknown tokens are used as raw substrings.

    `tiers` ('top' or 'top,big5') adds the corresponding league bundles.
    """
    out: list[str] = []
    for token in _split_tokens(spec or ""):
        out.extend(LEAGUE_ALIASES.get(token, [token]))
    for token in _split_tokens(tiers or ""):
        if token not in LEAGUE_TIERS:
            raise ValueError(f"Unknown tier {token!r}; choose from {', '.join(LEAGUE_TIERS)}")
        out.extend(LEAGUE_TIERS[token])
    return list(dict.fromkeys(out)) or None


def resolve_excludes(spec: str | None) -> list[str] | None:
    """Comma list of lowercase substrings to reject; 'default' expands to DEFAULT_EXCLUDES."""
    if not spec:
        return None
    out: list[str] = []
    for token in _split_tokens(spec):
        out.extend(DEFAULT_EXCLUDES if token == "default" else [token])
    return list(dict.fromkeys(out)) or None


def resolve_sport(name_or_id: str, live: list[Sport] | None = None) -> str:
    """Turn 'football', 'Ice Hockey', '5', or 'sr:sport:5' into a SportyBet sport id."""
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


def resolve_markets(spec: str | None, sport_id: str) -> list[str]:
    if not spec:
        return SPORT_DEFAULT_MARKETS.get(sport_id, DEFAULT_MARKETS)
    out: list[str] = []
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token.isdigit():
            out.append(token)
        elif token in MARKET_ALIASES:
            out.append(MARKET_ALIASES[token])
        else:
            raise ValueError(f"Unknown market {raw.strip()!r}. Use a numeric SportyBet market id or one of: "
                             + ", ".join(sorted(MARKET_ALIASES)))
    return list(dict.fromkeys(out))
