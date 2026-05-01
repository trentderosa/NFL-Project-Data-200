import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = BASE_DIR / "draft_text.txt"
OUTPUT_FILE = DATA_DIR / "clean_nfl_draft_data.csv"

POSITIONS = [
    "QB", "RB", "WR", "TE", "OT", "OG", "G", "C",
    "DE", "DT", "DL", "NT", "OLB", "LB", "CB", "S", "DB",
    "P", "K", "FB"
]
POSITIONS_SET = set(POSITIONS)

# Pre-compiled patterns for each position, sorted longest-first.
# Sorting prevents "LB" from matching inside "OLB", "CB" from matching as "C", etc.
# Each pattern requires a lowercase letter before the position (end of a name)
# and an uppercase letter after (start of a college name) — exactly the
# concatenation point in the raw text: "...BaileyOLBTexas..." or "...DelaneCBLSU..."
POSITION_PATTERNS = [
    (pos, re.compile(r"(?<=[a-z])" + re.escape(pos) + r"(?=[A-Z])"))
    for pos in sorted(POSITIONS, key=len, reverse=True)
]

# All 32 NFL team names used to cleanly delimit team vs. player in concatenated text
KNOWN_TEAMS = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
    "Los Angeles Chargers", "Los Angeles Rams", "Las Vegas Raiders", "Miami Dolphins",
    "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "Seattle Seahawks",
    "San Francisco 49ers", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders",
]


def draft_score(row):
    round_scores = {1: 10, 2: 7, 3: 5, 4: 3, 5: 2, 6: 1, 7: 1}
    position_bonus = {
        "QB": 4, "WR": 3, "DE": 3, "OLB": 3, "CB": 3, "OT": 3,
        "DT": 2, "DL": 2, "LB": 2, "S": 2,  "TE": 2,
        "RB": 1, "G":  1, "OG": 1, "C":  1,
    }
    return round_scores.get(row["round"], 0) + position_bonus.get(row["position"], 0)


def extract_team_and_player(before):
    """Return (team_name, player_name) from the text that precedes the position marker.

    The raw text concatenates team + player with no separator, e.g.:
        'Las Vegas RaidersFernando Mendoza'
        '(Browns to Chiefs)Kansas City ChiefsMansoor Delane'

    Strategy:
      1. Strip any trade notation like '(Browns to Chiefs)'.
      2. Match a known NFL team name at the start of the cleaned string.
      3. Treat the remainder as the player name.
    """
    # Remove trade notation e.g. "(Browns to Chiefs)"
    clean = re.sub(r"\([^)]+\)", "", before).strip()

    # Primary: team name appears at the start
    for known_team in KNOWN_TEAMS:
        if clean.startswith(known_team):
            remainder = clean[len(known_team):]
            # Match player name; extended pattern handles initials like "T.J."
            matches = re.findall(r"[A-Z][a-zA-Z.']*\s+[A-Z][a-z]+", remainder)
            player = matches[-1].strip() if matches else remainder.strip()
            return known_team, player

    # Fallback: team name appears anywhere in the string (e.g. extra leading text)
    for known_team in KNOWN_TEAMS:
        if known_team in clean:
            idx = clean.index(known_team)
            remainder = clean[idx + len(known_team):]
            matches = re.findall(r"[A-Z][a-zA-Z.']*\s+[A-Z][a-z]+", remainder)
            player = matches[-1].strip() if matches else remainder.strip()
            return known_team, player

    # Last resort: old pattern-based extraction
    matches = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", clean)
    player = matches[-1] if matches else "Unknown"
    return "Unknown", player


def extract_picks(text):
    text = re.sub(r"\s+", " ", text)  # collapse all whitespace to single spaces
    chunks = re.split(r"(RD\d+,\s*PK\d+)", text)
    picks = []

    for i in range(1, len(chunks), 2):
        header = chunks[i]
        content = chunks[i + 1] if i + 1 < len(chunks) else ""

        round_pick = re.search(r"RD(\d+),\s*PK(\d+)", header)
        if not round_pick:
            continue

        round_num = int(round_pick.group(1))
        pick_num  = int(round_pick.group(2))

        # Locate the position marker structurally: it appears concatenated
        # between the player's last name (ends with a lowercase letter) and the
        # college name (starts with an uppercase letter).
        #   "BaileyOLBTexas"   ->  'OLB' at the boundary
        #   "DelaneCBLSU"      ->  'CB' at the boundary (not 'CBL')
        # We check each position as an exact pattern (longest first) so that
        # "OLB" is never misread as "LB" and "CBL" never swallows "CB".
        pos_match = None
        pos_idx   = -1
        for pos, pattern in POSITION_PATTERNS:
            m = pattern.search(content[:300])
            if m:
                pos_match = pos
                pos_idx   = m.start()
                break

        if not pos_match:
            continue

        before = content[:pos_idx]
        after  = content[pos_idx + len(pos_match):]

        team, player = extract_team_and_player(before)

        college_match = re.match(r"([A-Z][a-zA-Z .()]+)", after)
        college = college_match.group(1).strip() if college_match else "Unknown"

        picks.append({
            "round":    round_num,
            "pick":     pick_num,
            "team":     team,
            "player":   player,
            "position": pos_match,
            "college":  college,
        })

    return picks


def main():
    if not INPUT_FILE.exists():
        print(f"File not found: {INPUT_FILE}")
        return

    DATA_DIR.mkdir(exist_ok=True)

    text = INPUT_FILE.read_text(encoding="utf-8")
    picks = extract_picks(text)

    if not picks:
        print("No picks found. Check the formatting of draft_text.txt.")
        return

    df = pd.DataFrame(picks)
    df["draft_score"] = df.apply(draft_score, axis=1)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"SUCCESS: Saved {len(df)} picks to {OUTPUT_FILE}")
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
