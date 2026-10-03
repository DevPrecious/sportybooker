# motivated

Build and book SportyBet parlays backed by **league table motivations** (Title race, European qualification / Promotion / Playoff spots, and Relegation survival battles), restricted strictly to **Win or Draw (`1X` / `X2`)** and **Handicap (`Asian Handicap` & `European 3-Way Handicap`)** markets.

## How it works

1. **Win or Draw & Handicap Picks Only**:
   - **Double Chance (`10`)**: Only `Home or Draw` (`1X`) and `Draw or Away` (`X2`) are eligible. `Home or Away` (`12`) is strictly excluded.
   - **Asian Handicap (`16`)**: Quarter, half, and whole-goal lines (`-1.5`, `-1.0`, `-0.5`, `0.0`, `+0.5`, `+1.0`, `+1.5`, etc.) for `Home` or `Away`.
   - **European 3-Way Handicap (`14`)**: `Home (X:Y)` and `Away (X:Y)` lines (`0:1`, `1:0`, `0:2`, `2:0`, etc.). Exact-margin `Draw (X:Y)` outcomes are excluded.
   - Any other market (`1X2`, `Over/Under`, `BTTS`, etc.) is rejected.

2. **Live League Table Motivations (Sportradar Standings)**:
   For every upcoming match, `motivated` resolves the competition's current season table from Sportradar (`stats_season_tables/{season_id}`), selects the active stage/group table containing both teams, and evaluates each side's table situation:
   - **Title Race / Leader Protection (`title`)**: Defending 1st place or chasing within striking distance near the top of the table.
   - **European Qualification / Promotion / Playoffs (`europe`, `promotion`)**: Defending a UCL / UEL / UECL / Promotion / Playoff spot with a tight cushion, or chasing the cutoff from just outside.
   - **Relegation Battle / Survival (`relegation`)**: Fighting inside the drop zone while mathematically within reach of safety, or sitting just above the drop line fighting for survival.
   - **Mid-table / Clinched / Relegated (`midtable`, `locked`, `relegated`)**: Assigned low motivation scores (`0.10 - 0.15`) and filtered out so you never back a team with nothing to play for.

3. **Motivation-Adjusted Poisson Pricing**:
   - Team attack/defence rates from recent form and head-to-head history are adjusted by the **table motivation gap** (`mot_home - mot_away` via `--motivation-boost`), shifting expected goals toward the more desperate/motivated side.
   - The resulting Poisson goal matrix prices `1X`/`X2`, Asian Handicap, and European 3-Way Handicap lines and blends them with SportyBet's de-vigged market probability to compute edge and win probability.

## Usage

Run from the workspace root (`c:\Users\olade\Documents\Developments\booker`):

```bash
# Build a ~3x motivated parlay (Win/Draw + Handicap only)
python -m motivated

# Target a specific per-leg price and parlay total, and generate a SportyBet booking code
python -m motivated --leg 1.25 --parlay 3 --book

# Focus only on motivation mismatches (motivated team vs mid-table / unmotivated opponent)
python -m motivated --focus mismatch --parlay 4 --hours 72

# Focus only on Title race or Relegation survival battles
python -m motivated --focus title --hours 72
python -m motivated --focus relegation --hours 72

# Restrict to Asian Handicap or Win/Draw (Double Chance) only
python -m motivated -m ah --parlay 3
python -m motivated -m dc --parlay 2.5

# Daily rollover / ladder mode
python -m motivated rollover --stake 5000 --days 7
python -m motivated rollover --book

# Inspect live league standings and each team's computed motivation score
python -m motivated tables -L "epl,laliga,seriea"

# Backtest Win/Draw and Handicap calibration on recent finished matches
python -m motivated backtest
```

## Key Flags

| Flag | Meaning | Default |
| --- | --- | --- |
| `--focus` | `any`, `title`, `europe`, `promotion`, `relegation`, or `mismatch` | `any` |
| `--min-motivation` | Minimum motivation score (`0..1`) for the backed team | `0.42` |
| `--min-mot-gap` | Minimum motivation gap (`backed - opponent`) | `-0.10` (`+0.15` on `mismatch`) |
| `--motivation-boost` | xG multiplier shift per unit of motivation gap | `0.12` |
| `--markets, -m` | `dc` (`10`), `ah` (`16`), `eh` (`14`), `handicap` (`16,14`) | `10,16,14` |
| `--leg` | Target odds per leg (uses up to +20% headroom) | none |
| `--parlay, -t` | Target combined parlay odds | `3.0` |
| `--strategy` | `rollover`, `motivation`, `edge`, `safest`, `random` | `rollover` |
| `--book` | Generate a shareable SportyBet booking code | off (dry run) |

## Tests

```bash
python -m unittest discover -s motivated/tests -v
```
