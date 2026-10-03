"""Table Motivation Engine: evaluate league standings for Title, Europe/Promotion/Playoffs, and Relegation/Survival."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TableRow:
    uid: int
    name: str
    pos: int
    played: int
    wins: int
    draws: int
    losses: int
    gf: int
    ga: int
    gd: int
    points: int
    promo_name: str = ""
    promo_class: str = ""

    @property
    def is_relegation_zone(self) -> bool:
        text = f"{self.promo_name} {self.promo_class}".lower()
        return "relegation" in text or "rel" == self.promo_name.strip().lower()

    @property
    def is_positive_zone(self) -> bool:
        if not (self.promo_name or self.promo_class):
            return False
        return not self.is_relegation_zone


@dataclass(frozen=True)
class LeagueTable:
    season_id: int
    name: str
    max_rounds: int
    current_round: int
    rows: tuple[TableRow, ...]

    def row_for(self, uid: int) -> TableRow | None:
        for r in self.rows:
            if r.uid == uid:
                return r
        return None


@dataclass(frozen=True)
class TeamMotivation:
    uid: int
    pos: int
    total_teams: int
    points: int
    played: int
    remaining: int
    zone: str  # "title" | "europe" | "promotion" | "relegation" | "midtable" | "locked" | "relegated" | "early"
    score: float  # 0.0 to 1.0 intensity of table motivation
    label: str  # concise display tag, e.g. "#2 Title (-2p)" or "#17 Survival (+1p)"
    detail: str  # fuller human-readable explanation

    @property
    def is_motivated(self) -> bool:
        return self.zone in ("title", "europe", "promotion", "relegation") and self.score >= 0.40


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _short_promo_tag(promo_name: str) -> str:
    low = (promo_name or "").lower()
    if "champions league" in low:
        return "UCL"
    if "europa league" in low:
        return "UEL"
    if "conference league" in low:
        return "UECL"
    if "libertadores" in low:
        return "Lib"
    if "sudamericana" in low:
        return "Sud"
    if "playoff" in low:
        return "Playoff"
    if "promotion" in low or "promoted" in low:
        return "Promo"
    if "main round" in low or "championship" in low or "qualified" in low or "next round" in low:
        return "Qual"
    return "TopZone"


def _zone_category(promo_name: str) -> str:
    low = (promo_name or "").lower()
    if any(k in low for k in ("champions league", "europa", "conference", "libertadores", "sudamericana", "caf", "afc")):
        return "europe"
    return "promotion"


def parse_season_tables(season_id: int, raw: dict[str, Any] | None) -> list[LeagueTable]:
    """Parse Sportradar `stats_season_tables/{season_id}` response into `LeagueTable` objects."""
    if not isinstance(raw, dict):
        return []
    out: list[LeagueTable] = []
    for t in raw.get("tables") or []:
        raw_rows = t.get("tablerows") or []
        rows: list[TableRow] = []
        for idx, r in enumerate(raw_rows, 1):
            team = r.get("team") or {}
            uid = team.get("uid")
            if uid is None:
                continue
            promo = r.get("promotion") or {}
            rows.append(
                TableRow(
                    uid=int(uid),
                    name=str(team.get("name") or "?"),
                    pos=int(r.get("pos") or idx),
                    played=int(r.get("total") or 0),
                    wins=int(r.get("winTotal") or 0),
                    draws=int(r.get("drawTotal") or 0),
                    losses=int(r.get("lossTotal") or 0),
                    gf=int(r.get("goalsForTotal") or 0),
                    ga=int(r.get("goalsAgainstTotal") or 0),
                    gd=int(r.get("goalDiffTotal") or 0),
                    points=int(r.get("pointsTotal") or 0),
                    promo_name=str(promo.get("name") or ""),
                    promo_class=str(promo.get("cssclass") or ""),
                )
            )
        if not rows:
            continue
        rows.sort(key=lambda r: r.pos)
        n = len(rows)
        default_rounds = max((r.played for r in rows), default=0)
        if default_rounds <= 6 and n <= 6:
            default_rounds = max(default_rounds, (n - 1) * 2 if n > 1 else 6)
        elif n >= 10:
            default_rounds = max(default_rounds, (n - 1) * 2)
        max_rounds = int(t.get("maxrounds") or default_rounds or 38)
        current_round = int(t.get("currentround") or max((r.played for r in rows), default=0))
        out.append(
            LeagueTable(
                season_id=season_id,
                name=str(t.get("name") or raw.get("name") or f"Season {season_id}"),
                max_rounds=max_rounds,
                current_round=current_round,
                rows=tuple(rows),
            )
        )
    return out


def select_table_for_fixture(
    tables: list[LeagueTable], home_uid: int, away_uid: int
) -> LeagueTable | None:
    """Choose the relevant table for `(home_uid, away_uid)`."""
    if not tables:
        return None
    both = [t for t in tables if t.row_for(home_uid) is not None and t.row_for(away_uid) is not None]
    if both:
        active = [t for t in both if any(r.played > 0 for r in t.rows)]
        return active[-1] if active else both[-1]
    either = [t for t in tables if t.row_for(home_uid) is not None or t.row_for(away_uid) is not None]
    return either[-1] if either else None


def evaluate_team_motivation(
    table: LeagueTable,
    uid: int,
    min_games: int = 2,
) -> TeamMotivation | None:
    """Compute a team's table motivation from the standings (`Title`, `Europe/Promo/Playoff`, `Relegation/Survival`)."""
    row = table.row_for(uid)
    if row is None:
        return None

    rows = table.rows
    n = len(rows)
    if n < 2:
        return None

    max_rounds = max(table.max_rounds, row.played)
    remaining = max(1, max_rounds - row.played)
    max_pts_left = remaining * 3

    effective_min_games = 1 if max_rounds <= 10 else min_games
    if row.played < effective_min_games:
        return TeamMotivation(
            uid=uid,
            pos=row.pos,
            total_teams=n,
            points=row.points,
            played=row.played,
            remaining=remaining,
            zone="early",
            score=0.25,
            label=f"#{row.pos}/{n} Early ({row.played}g)",
            detail=f"Season just started ({row.played} played)",
        )

    progress = _clamp(row.played / max(max_rounds, 1), 0.15, 1.0)
    stage_factor = 1.0 if max_rounds <= 10 else (0.82 + 0.23 * progress)

    leader = rows[0]
    runner_up = rows[1]

    pos_rows = [r for r in rows if r.is_positive_zone]
    rel_rows = [r for r in rows if r.is_relegation_zone]

    if not pos_rows and n >= 4:
        top_k = 1 if n <= 4 else (2 if n <= 8 else 3)
        pos_rows = list(rows[:top_k])
    if not rel_rows and n >= 12 and max_rounds >= 18:
        rel_rows = list(rows[-3 if n >= 18 else -2:])

    # If a large league table (e.g. 36-team UCL/UEL league phase) marks top 24 as playoffs,
    # cap the positive cutoff and chase spots sensibly.
    max_title_pos = 2 if n <= 6 else min(4, max(2, n // 5))
    max_chase_spots = 1 if n <= 8 else 2
    max_survival_spots = 1 if n <= 10 else 2

    candidates: list[tuple[float, str, str, str]] = []

    # ------------------------------------------------------------------ 1. TITLE RACE / LEADER PROTECTION
    title_window = 3 if (max_rounds <= 10 or progress < 0.35) else 6
    if row.pos == 1:
        lead = row.points - runner_up.points
        if lead > max_pts_left and row.played >= 5:
            return TeamMotivation(
                uid=uid,
                pos=row.pos,
                total_teams=n,
                points=row.points,
                played=row.played,
                remaining=remaining,
                zone="locked",
                score=0.15,
                label=f"#1 Clinched (+{lead}p)",
                detail=f"Already clinched 1st place (+{lead} pts with {remaining} left)",
            )
        if lead <= 3:
            raw_score = 0.92 - 0.03 * lead
        elif lead <= 6:
            raw_score = 0.78 - 0.03 * (lead - 3)
        else:
            raw_score = max(0.50, 0.65 - 0.02 * (lead - 6))
        score = _clamp(raw_score * stage_factor, 0.45, 0.98)
        lead_str = "tied" if lead == 0 else f"+{lead}p"
        candidates.append((score, "title", f"#1 Title ({lead_str})", f"Defending 1st place ({lead_str} over #2)"))
    elif row.pos <= max_title_pos:
        gap_to_first = leader.points - row.points
        if gap_to_first <= min(title_window, max_pts_left):
            if gap_to_first <= 3:
                raw_score = 0.90 - 0.03 * gap_to_first
            else:
                raw_score = 0.76 - 0.04 * (gap_to_first - 3)
            score = _clamp(raw_score * stage_factor, 0.45, 0.96)
            gap_str = "tied #1" if gap_to_first == 0 else f"-{gap_to_first}p #1"
            candidates.append((score, "title", f"#{row.pos} Title ({gap_str})", f"Chasing 1st place ({gap_to_first} pts behind #1)"))

    # ------------------------------------------------------------------ 2. EUROPE / PROMOTION / PLAYOFFS
    if pos_rows:
        cutoff_row = max(pos_rows, key=lambda r: r.pos)
        cutoff_pos = cutoff_row.pos
        cutoff_idx = rows.index(cutoff_row)
        first_out = rows[cutoff_idx + 1] if (cutoff_idx + 1) < n else None
        cutoff_tag = _short_promo_tag(cutoff_row.promo_name)
        cutoff_zone = _zone_category(cutoff_row.promo_name)

        if row.pos <= cutoff_pos:
            own_tag = _short_promo_tag(row.promo_name or cutoff_row.promo_name)
            own_zone = _zone_category(row.promo_name or cutoff_row.promo_name)
            cushion = (row.points - first_out.points) if first_out else 0
            if cushion <= max_pts_left:
                if cushion <= 3:
                    raw_score = 0.86 - 0.03 * cushion
                elif cushion <= 6:
                    raw_score = 0.72 - 0.04 * (cushion - 3)
                else:
                    raw_score = max(0.45, 0.56 - 0.02 * (cushion - 6))
                score = _clamp(raw_score * stage_factor, 0.42, 0.94)
                c_str = "tied cut" if cushion == 0 else f"+{cushion}p"
                candidates.append((
                    score,
                    own_zone,
                    f"#{row.pos} {own_tag} ({c_str})",
                    f"Defending {own_tag} spot (#{row.pos}, +{cushion} pts over #{cutoff_pos + 1})",
                ))

            higher_rows = [
                r for r in pos_rows
                if r.pos < row.pos and r.promo_name and r.promo_name != row.promo_name
            ]
            if higher_rows:
                target_row = higher_rows[-1]
                h_gap = target_row.points - row.points
                h_tag = _short_promo_tag(target_row.promo_name)
                h_zone = _zone_category(target_row.promo_name)
                if (row.pos - target_row.pos) <= max_chase_spots and h_gap <= min(4, max_pts_left):
                    raw_score = 0.86 - 0.04 * h_gap
                    score = _clamp(raw_score * stage_factor, 0.48, 0.93)
                    g_str = f"tied #{target_row.pos}" if h_gap == 0 else f"-{h_gap}p {h_tag}"
                    candidates.append((
                        score,
                        h_zone,
                        f"#{row.pos} Chase {h_tag} ({g_str})",
                        f"Chasing {h_tag} spot (#{target_row.pos}, {h_gap} pts back)",
                    ))
        elif (row.pos - cutoff_pos) <= max_chase_spots:
            gap_to_cut = cutoff_row.points - row.points
            chase_window = 3 if (max_rounds <= 10 or progress < 0.35) else 5
            if gap_to_cut <= min(chase_window, max_pts_left):
                if gap_to_cut <= 3:
                    raw_score = 0.84 - 0.04 * gap_to_cut
                else:
                    raw_score = 0.68 - 0.05 * (gap_to_cut - 3)
                score = _clamp(raw_score * stage_factor, 0.44, 0.92)
                g_str = f"tied #{cutoff_pos}" if gap_to_cut == 0 else f"-{gap_to_cut}p {cutoff_tag}"
                candidates.append((
                    score,
                    cutoff_zone,
                    f"#{row.pos} Chase {cutoff_tag} ({g_str})",
                    f"Chasing {cutoff_tag} cutoff (#{cutoff_pos}, {gap_to_cut} pts back)",
                ))

    # ------------------------------------------------------------------ 3. RELEGATION BATTLE / SURVIVAL
    if rel_rows:
        drop_row = min(rel_rows, key=lambda r: r.pos)
        drop_pos = drop_row.pos
        drop_idx = rows.index(drop_row)
        safe_row = rows[drop_idx - 1] if drop_idx >= 1 else rows[0]

        if row.pos >= drop_pos:
            gap_to_safe = safe_row.points - row.points
            if gap_to_safe > max_pts_left and row.played >= 5:
                return TeamMotivation(
                    uid=uid,
                    pos=row.pos,
                    total_teams=n,
                    points=row.points,
                    played=row.played,
                    remaining=remaining,
                    zone="relegated",
                    score=0.10,
                    label=f"#{row.pos} Relegated (-{gap_to_safe}p)",
                    detail=f"Mathematically relegated ({gap_to_safe} pts from safety with {remaining} left)",
                )
            if gap_to_safe <= 3:
                raw_score = 0.92 - 0.03 * gap_to_safe
            elif gap_to_safe <= 6:
                raw_score = 0.78 - 0.04 * (gap_to_safe - 3)
            elif gap_to_safe <= 9:
                raw_score = 0.58 - 0.04 * (gap_to_safe - 6)
            else:
                raw_score = 0.38
            score = _clamp(raw_score * stage_factor, 0.35, 0.97)
            g_str = "tied safe" if gap_to_safe == 0 else f"-{gap_to_safe}p safe"
            candidates.append((
                score,
                "relegation",
                f"#{row.pos} Releg ({g_str})",
                f"Fighting relegation (#{row.pos}, {gap_to_safe} pts from safety #{drop_pos - 1})",
            ))
        elif (drop_pos - row.pos) <= max_survival_spots:
            cushion = row.points - drop_row.points
            surv_window = 3 if (max_rounds <= 10 or progress < 0.35) else 4
            if cushion <= min(surv_window, max_pts_left):
                if cushion <= 2:
                    raw_score = 0.86 - 0.03 * cushion
                else:
                    raw_score = 0.72 - 0.05 * (cushion - 2)
                score = _clamp(raw_score * stage_factor, 0.45, 0.93)
                c_str = "tied drop" if cushion == 0 else f"+{cushion}p drop"
                candidates.append((
                    score,
                    "relegation",
                    f"#{row.pos} Survival ({c_str})",
                    f"Fighting for survival (#{row.pos}, +{cushion} pts above drop zone #{drop_pos})",
                ))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, best_zone, best_label, best_detail = candidates[0]
        return TeamMotivation(
            uid=uid,
            pos=row.pos,
            total_teams=n,
            points=row.points,
            played=row.played,
            remaining=remaining,
            zone=best_zone,
            score=round(best_score, 3),
            label=best_label,
            detail=best_detail,
        )

    return TeamMotivation(
        uid=uid,
        pos=row.pos,
        total_teams=n,
        points=row.points,
        played=row.played,
        remaining=remaining,
        zone="midtable",
        score=0.15,
        label=f"#{row.pos} Mid-table ({row.points}p)",
        detail=f"Mid-table (#{row.pos}/{n}, {row.points} pts) with no immediate table pressure",
    )
