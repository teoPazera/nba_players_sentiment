from pathlib import Path
import re

import pandas as pd
import wikipedia
from bs4 import BeautifulSoup

DATA_DIR = Path("data") / "player_data"
SIGNINGS_CSV = DATA_DIR / "signings_with_salaries.csv"
# SIGNINGS_CSV = DATA_DIR / "fox_free_agency_signings_2025.csv"
ALIASES_OUT = DATA_DIR / "player_aliases.csv"


def normalize(s: str) -> str:
    s = str(s).lower()
    # drop common suffixes
    s = re.sub(r"\b(jr|jr\.|sr|sr\.|ii|iii|iv)\b", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def scrape_nicknames_with_wikipedia():
    """Scrape the Wikipedia "List of nicknames in basketball" page."""
    wikipedia.set_lang("en")

    # get the page (the title is the same as the URL slug)
    page = wikipedia.page("List of nicknames in basketball")
    html = page.html()

    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("div", class_="mw-parser-output")
    if content is None:
        raise RuntimeError("Couldn't find mw-parser-output in wikipedia HTML")

    nick_by_player = {}

    # nickname entries live mostly in <li> elements in the main content
    for li in content.find_all("li"):
        text = li.get_text(" ", strip=True)
        if not text:
            continue

        # skip junk that clearly isn't "Player – nickname..."
        if "–" not in text and "-" not in text:
            continue

        # remove citation markers like [1], [12]
        text = re.sub(r"\[\d+\]", "", text)

        # split "Player – nicknames..."
        if "–" in text:
            player_part, nick_part = text.split("–", 1)
        else:
            player_part, nick_part = text.split("-", 1)

        player = player_part.strip()
        nick_part = nick_part.strip()

        # nicknames are often quoted
        nicks = re.findall(r'"([^"]+)"', nick_part)
        nicks = [n.strip() for n in nicks if n.strip()]

        if not nicks:
            # fallback: comma-separated short phrases
            rough = [p.strip() for p in nick_part.split(",")]
            rough = [r for r in rough if 1 <= len(r.split()) <= 4]
            nicks = rough

        if nicks:
            nick_by_player[player] = nicks

    print(f"Scraped {len(nick_by_player)} players with nicknames from Wikipedia")
    return nick_by_player


def build_alias_map(signings_csv: Path | str = SIGNINGS_CSV):
    """Build `data/player_data/player_aliases.csv` from the signings list + Wikipedia nicknames."""
    # Player list from signings CSV.
    players_df = pd.read_csv(signings_csv)

    players_df = players_df.dropna(subset=["player"])
    players_df["norm_player"] = players_df["player"].apply(normalize)

    # map normalized full name -> canonical player name
    norm_to_player = {
        row["norm_player"]: row["player"]
        for _, row in players_df.iterrows()
    }

    # Nicknames from Wikipedia.
    wiki_nicks = scrape_nicknames_with_wikipedia()

    # Alias map.
    alias_to_player = {}

    # full names and unique last names
    last_counts = {}
    for _, row in players_df.iterrows():
        norm_full = row["norm_player"]
        last = norm_full.split()[-1]
        last_counts[last] = last_counts.get(last, 0) + 1

    for _, row in players_df.iterrows():
        canonical = row["player"]
        norm_full = row["norm_player"]
        last = norm_full.split()[-1]

        # full name alias
        alias_to_player[norm_full] = canonical

        # unique last name alias (avoid huge collisions like "Smith")
        if last_counts[last] == 1 and len(last) > 3:
            alias_to_player[last] = canonical

    # Add Wikipedia nicknames for players present in the signings list.
    for wiki_player, nicks in wiki_nicks.items():
        norm_wiki = normalize(wiki_player)
        if norm_wiki not in norm_to_player:
            # ignore retired / non-NBA / not in your CSV
            continue

        canonical = norm_to_player[norm_wiki]
        for nick in nicks:
            norm_nick = normalize(nick)
            if not norm_nick:
                continue
            # skip extra-short nicknames to avoid false positives
            if len(norm_nick) <= 2:
                continue
            alias_to_player[norm_nick] = canonical

    print(f"Total aliases built: {len(alias_to_player)}")

    alias_df = pd.DataFrame(
        [{"alias": a, "player": p} for a, p in alias_to_player.items()]
    )
    ALIASES_OUT.parent.mkdir(parents=True, exist_ok=True)
    alias_df.to_csv(ALIASES_OUT, index=False)
    print(f"Saved alias map to {ALIASES_OUT}")
    return alias_to_player


if __name__ == "__main__":
    build_alias_map()
