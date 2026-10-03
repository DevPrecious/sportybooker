# booker

Pick SportyBet selections that match the **sport** and **odds** you want, then generate a
SportyBet **booking code** you can load on any device.

It uses SportyBet's public web endpoints (the same ones the site itself calls). No login, no
stake, no money moves: a booking code only reserves a betslip. Placing the bet is still a manual
step you do on SportyBet with the code.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Python 3.10+.

## Usage

```bash
# Football, top leagues, stats model. Builds the ~3x slip most likely to land.
python -m booker

# Same idea, but you pick the per-leg price and the parlay total
python -m booker --leg 1.25 --parlay 3

# Create the booking code
python -m booker --leg 1.25 --parlay 3 --book

# Four basketball picks between 1.5 and 2.5, no total target
python -m booker --sport basketball --min-odds 1.5 --max-odds 2.5 --legs 4

# Only "Over 1.5" goals picks, 5 legs, machine-readable output
python -m booker -s football -m ou -o "Over 1.5" --min-odds 1.15 --max-odds 1.4 -l 5 --json

# What sports are available right now?
python -m booker --list-sports

# What football leagues have games today? Then build a slip from just LaLiga + EFL + Europa League
python -m booker -s football --today --list-leagues
python -m booker -s football --today --league "laliga,efl,europa" --min-odds 1.3 --max-odds 2 -t 12 -l 3-6 --edge

# Edge mode: rate every pick with team form + head-to-head, keep only picks the
# model thinks are underpriced (edge >= +2%), then build a ~8x slip from them
python -m booker -s football --min-odds 1.3 --max-odds 1.9 -t 8 -l 3-6 --edge --min-edge 0.02
```

Example output:

```
Scanning football (sr:sport:1) on sportybet.com/ng | markets 1,18,10,29,11 | odds 1.30-1.70 | target 10.00 +/-10%
236 events scanned, 161 outcomes in range across 33 matches
 #  Kick-off         Fixture                              Market            Pick       Odds
 1  Wed 16 Sep 20:00 Coventry City vs Aston Villa         Over/Under (4)    Under 4    1.30
 2  Wed 16 Sep 20:00 Olympiacos vs Jagiellonia Bialystok  1X2               Home       1.42
 ...
Legs: 6   Total odds: 10.02

Booking code: ABC123
Load it here:  https://www.sportybet.com/ng/?shareCode=ABC123
```

### `booker rollover` - daily 3-odds rollover (ladder)

```bash
python -m booker rollover                       # today's football, ~3.0 odds, 1-3 legs, dry run
python -m booker rollover --stake 5000 --days 7 # show the ladder projection from a 5,000 stake
python -m booker rollover -L "laliga,epl,ucl"   # restrict to leagues
python -m booker rollover --odds 2 --legs 1-2   # a 2-odds ladder instead
python -m booker rollover --book                # generate the booking code
```

A rollover lives or dies on survival, so this command differs from the main one:

- The edge model is always on, and **every leg** must pass three gates: edge >= `--min-edge` (+2%),
  blended win probability >= `--min-prob` (55%), and bookmaker probability >= `--min-book-prob` (50%).
  The last gate stops a model that's overconfident on thin data from dragging a 2.05 long shot into your ladder.
- Among slips inside the tolerance band (`--odds` +/- 7%) it picks the one with the **highest
  parlay win probability**, not the highest edge.
- It prints the model's and the bookmaker's win chance for the whole slip, the expected return,
  and (with `--stake`) a day-by-day ladder with the probability of surviving to each rung. Those
  survival numbers are the honest part: a 3-odds rollover at a 42% daily hit rate survives three
  days only 7% of the time.
- Defaults to today's kick-offs; `--tomorrow`, `--date YYYY-MM-DD`, `--time 12:00-23:59` or `--hours N` change the window.

Other flags mirror the main command (`--sport`, `--region`, `--markets`, `--outcome`, `--min-odds`,
`--max-odds`, `--legs`, `--model-weight`, `--force`, `--json`); `booker rollover --help` lists them.

### `booker backtest` - is the model honest?

```bash
python -m booker backtest                      # football, last 30 days, ~200 matches
python -m booker backtest --save               # ...and save a calibration the edge model applies automatically
python -m booker backtest --tier top --days 45 # major leagues only, longer window
python -m booker backtest -s basketball
```

It rebuilds each team's history *as it stood before kick-off* for every finished match in the window,
runs the same model the live picker uses, and reports per market: hit rate vs model average, Brier
score vs the "always predict the base rate" baseline, and a calibration table ("when the model says
70-80%, how often does it actually happen?"). It then fits a Platt-scaling correction
`p' = sigmoid(a*logit(p) + b)`; `--save` writes it to `.booker_cache/calibration.json`, and `--edge`
and `rollover` apply it automatically (`--no-calibration` to ignore).

Historical bookmaker odds aren't available from the feed, so this measures whether the model's
probabilities are truthful, not whether they beat closing prices.

First results (football, 209 matches, Aug-Sep 2026): the model beats the base rate on 1X2, Double
Chance and Over/Under, is roughly calibrated (about 3 points overconfident in the 70-90% band, Platt
a=0.92), and is *worse* than the base rate on BTTS, so `rollover` skips GG/NG by default.

### Options (main command)

| Flag | Meaning | Default |
| --- | --- | --- |
| `--sport, -s` | Sport name or id: `football`, `basketball`, `tennis`, `ice hockey`, `5`, `sr:sport:5`... | `football` |
| `--region, -r` | SportyBet country site: `ng gh ke tz ug zm` | `ng` |
| `--leg` | Odds you want **per pick**, e.g. `1.20`. Uses that price up to +20% | none |
| `--parlay` / `--total-odds, -t` | Target **combined** odds for the slip, e.g. `3`. Also fills `--legs` if you omit it | `3` on football; none otherwise |
| `--min-odds` / `--max-odds` | Override the per-pick range | `1.20` / `2.00`, or the `--leg` band |
| `--tolerance` | Acceptable +/- fraction around the target | `0.10` (10%) |
| `--legs, -l` | Exact (`5`) or range (`3-8`) number of picks | derived from `--parlay` + `--leg`, else `1-8` |
| `--markets, -m` | Comma list: `1x2 ou dc gg dnb handicap winner` or numeric ids | sensible per-sport set |
| `--league, -L` | Comma list of leagues: aliases (`laliga epl efl ucl uel seriea bundesliga ligue1 mls ...`) or any text matched against `<country> <tournament>` (`"england premier"`, `"norway"`) | all |
| `--tier` | League bundle(s): `top`, `big5`, `europe` (comma list, combines with `--league`) | `top` on football; none otherwise |
| `--exclude, -X` | Reject fixtures whose competition or team names contain this text; `default` = reserves, youth (u17-u23), women, amateur, serie d, MLS Next Pro | `default` on football; none otherwise (`default` in `rollover`) |
| `--min-book-prob` | Drop outcomes the bookmaker rates below this probability, e.g. `0.5` | `0.50` on football; none otherwise |
| `--min-prob` | Drop picks whose blended win probability is below this | `0.55` on football; none otherwise |
| `--list-leagues` | Print leagues with events in the selected time window and exit | off |
| `--include-srl` | Include Simulated Reality League (computer-simulated) fixtures, skipped by default | off |
| `--outcome, -o` | Keep only picks whose label contains this text (`Home`, `Over 2.5`, `Yes`) | none |
| `--hours` | Only events starting within N hours (0 = no limit) | `48` |
| `--today` / `--tomorrow` | Only events kicking off today / tomorrow, local time (overrides `--hours`) | off |
| `--date` | Only events kicking off on a given local day, `YYYY-MM-DD` (overrides `--hours`) | none |
| `--time` | Kick-off time range, local: `12:00-23:59`, `18:00` for "from 18:00 on", or `23:00-02:00` to run past midnight into the next day | whole day |
| `--prefer-early` | Fill the slip from the earliest kick-offs in the window (instead of the highest-edge combo later in the day) | off |
| `--strategy` | `safest` (highest bookmaker probability first), `edge` (best model edge first), `rollover` (highest blended win probability) or `random` | `rollover` on football; `safest` otherwise |
| `--seed` | Seed for `--strategy random` | none |
| `--book` | Create the booking code (otherwise dry run) | off |
| `--force` | Book even if total odds missed `--total-odds` by more than `--tolerance` | off |
| `--json` | JSON output | off |

Edge model flags:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--edge` | Rate picks with the stats model and show Form / Book% / Model% / Edge columns | on for football; off otherwise |
| `--min-edge` | Drop picks below this edge, e.g. `0.02` for +2% (implies `--edge`) | `0.02` on football; none otherwise |
| `--model-weight` | Share of the stats model vs the bookmaker's probability in the blend | `0.40` |
| `--form-games` | Recent matches per team fed to the model | `10` |
| `--max-matches` | Max fixtures to pull stats for (soonest kick-offs first); `0` = all | `120` |
| `--allow-unrated` | Keep picks the model couldn't rate in the pool | off |
| `--no-cache` | Don't cache stats in `.booker_cache/` (3h TTL) | off |
| `--no-calibration` | Ignore the correction saved by `booker backtest --save` | off |

Exit codes: `0` slip built, `2` nothing matched, `3` refused to book because total odds missed the target, `1` error.

### Reaching big totals (e.g. 100x)

Legs needed is roughly `log(target) / log(average pick odds)`: ~12 legs at 1.48 average, ~7 at 2.0,
~5 at 2.5. The tool prints this estimate along with how many matches survived your filters. If the
pool is smaller than your minimum legs, or the picks can't multiply up to the target, it prints a
warning and (with `--book`) refuses to book unless you pass `--force`. To grow the pool: widen
`--min-odds/--max-odds`, lower `--min-edge`, use `--max-matches 0`, or add `--hours 72`.

## How it picks

1. Fetches upcoming events for the sport and the requested markets.
2. Keeps every active outcome whose odds fall inside `--min-odds..--max-odds` (and matches `--outcome`, if set).
3. Football (unless you pass `--strategy safest` / `random`) always runs the stats model. It skips
   BTTS by default (the backtest prices GG/NG worse than the base rate), drops youth/reserves/women,
   and keeps only picks that clear `--min-edge`, `--min-prob` and `--min-book-prob`. Other sports
   still need `--edge` / `--min-edge` to rate picks.
4. Collapses to one pick per match (a slip can't hold two picks on the same game), preferring the
   most probable outcome (`safest`), the highest edge (`edge`), or a random one (`random`).
5. If `--parlay` / `--total-odds` is given, a bounded depth-first search over log-odds explores combinations of
   `--legs` picks. Anything inside the tolerance band beats anything outside it; inside the band the
   highest total edge wins, outside it the closest product wins. Without a target it takes the first `--legs` picks.
6. With `--book`, posts the selections to SportyBet and prints the share code and URL.

To get the old football behaviour (bookmaker-probability ranking, no model gates):
`--strategy safest --exclude "" --min-book-prob 0`.

## The edge model

SportyBet's own "Statistics" tab is a Sportradar widget, and SportyBet's event and competitor ids
*are* Sportradar ids. So for every fixture we can pull the exact same data the site shows, with no
team-name matching: each side's last N results (`stats_team_lastx`) and the head-to-head record
(`stats_team_versusrecent`).

- **Football (and eSoccer/futsal):** recency-weighted goals for/against per team give expected
  goals for each side (`(home attack + away defence) / 2`, times a home-advantage factor), nudged
  toward head-to-head scoring if there are 3+ meetings. Independent Poisson distributions then price
  1X2, Double Chance, Draw No Bet, Over/Under (whole-number lines exclude the push) and BTTS.
- **Everything else:** a recency-weighted W/D/L form rate for each side plus a small home edge and
  head-to-head nudge gives P(home win), which prices two-way Winner markets.
- **Blend:** `p = w * model + (1 - w) * bookmaker`, where the bookmaker probability is SportyBet's
  own de-vigged `probability` field. `w` is `--model-weight` scaled down when a team has fewer than
  `--form-games` results. **Edge = p x odds - 1.**

Teams with fewer than 4 recent results, and markets the model doesn't price (correct score,
handicaps, ...), are left unrated and excluded unless `--allow-unrated` is passed.

This is a deliberately simple, transparent model. It has no injuries, lineups, motivation, or
league-strength adjustment, and a 10-game Poisson fit is noisy, so treat Model% as a second opinion
rather than truth. `--model-weight` is the dial: `0` ranks purely by bookmaker probability, `1`
trusts the stats alone. Football's default blend is still 40% model / 60% book, but a pick only
lands on the slip if the model rated it and the price still looks short.

## Tests

```bash
python -m unittest discover -s tests
```

## Caveats

- These are undocumented endpoints (SportyBet and Sportradar); either can change or block them without notice.
- Booking codes expire when the earliest selected event kicks off.
- SportyBet may reject individual selections at booking time (suspended markets); the tool reports
  how many were dropped.
- This tool does not place bets or handle money. Gamble responsibly.
