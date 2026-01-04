"""Process a single article URL into sentence records with player attribution."""

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import requests
import spacy
from requests import Response
from spacy.matcher import PhraseMatcher

import trafilatura


def normalize(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_alias_map(path: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """Return (alias_to_player, player_to_aliases) from the alias CSV."""
    df = pd.read_csv(path)
    if not {"player", "alias"}.issubset(df.columns):
        raise ValueError("alias CSV must have columns: player, alias")

    alias_to_player: Dict[str, str] = {}
    player_to_aliases: Dict[str, Set[str]] = {}

    for _, row in df.iterrows():
        player = str(row["player"]).strip()
        alias_raw = str(row["alias"]).strip()
        if not player or not alias_raw:
            continue
        alias_norm = normalize(alias_raw)
        if alias_norm:
            alias_to_player[alias_norm] = player
        player_to_aliases.setdefault(player, set()).add(alias_raw)

    # ensure canonical name is present as an alias
    for player in player_to_aliases.keys():
        player_to_aliases[player].add(player)

    return alias_to_player, player_to_aliases


def create_player_matcher(nlp, alias_to_player: Dict[str, str]) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = []
    for alias_norm in alias_to_player.keys():
        if len(alias_norm) <= 2:
            continue  # avoid extremely short tokens
        patterns.append(nlp.make_doc(alias_norm))
    matcher.add("PLAYER", patterns)
    return matcher


def find_players_in_text_spacy(text: str, nlp, matcher, alias_to_player: Dict[str, str]) -> List[str]:
    doc = nlp(text)
    found = []
    for match_id, start, end in matcher(doc):
        span = doc[start:end]
        alias_norm = normalize(span.text)
        player = alias_to_player.get(alias_norm)
        if player:
            found.append(player)
    # dedupe preserving order
    return list(dict.fromkeys(found))


def _try_fetch(url: str, headers: Dict[str, str]) -> Response:
    return requests.get(url, headers=headers, timeout=20, allow_redirects=True)


def _amp_variants(url: str) -> List[str]:
    variants = []
    if "?" in url:
        variants.append(f"{url}&output=amp")
    else:
        variants.append(f"{url}?output=amp")
    if not url.endswith("/"):
        url = url + "/"
    variants.append(url + "amp/")
    return variants


def fetch_text(url: str, suppress_errors: bool = False) -> str:
    """Fetch a URL and extract main text with trafilatura (tries a few AMP variants)."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    attempts = [url, *_amp_variants(url)]
    last_exc: Optional[Exception] = None
    html = ""

    for attempt in attempts:
        try:
            resp = _try_fetch(attempt, headers=headers)
            resp.raise_for_status()
            html = resp.text
            break
        except Exception as exc:
            last_exc = exc
            continue

    if not html and last_exc:
        if suppress_errors:
            logging.warning("Fetch failed for %s: %s", url, last_exc)
            return ""
        raise last_exc

    extracted = trafilatura.extract(html)
    if extracted:
        return extracted
    return html


def process_url(
    url: str,
    alias_csv: Path,
    model: str = "en_core_web_sm",
    suppress_fetch_errors: bool = False,
):
    alias_to_player, player_to_aliases = load_alias_map(alias_csv)
    nlp = spacy.load(model)
    matcher = create_player_matcher(nlp, alias_to_player)

    logging.info("Fetching %s", url)
    text = fetch_text(url, suppress_errors=suppress_fetch_errors)
    if not text:
        return []
    logging.info("Extracted %d chars", len(text))

    doc = nlp(text)
    records = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        matched_players = find_players_in_text_spacy(sent_text, nlp, matcher, alias_to_player)

        other_persons = []
        for ent in sent.ents:
            if ent.label_ != "PERSON":
                continue
            norm = normalize(ent.text)
            if norm in alias_to_player:
                continue  # already captured as a tracked player
            other_persons.append(ent.text)

        if not matched_players and not other_persons:
            # discard sentences without any person/entity
            continue

        records.append(
            {
                "sentence": sent_text,
                "players": "; ".join(matched_players),
                "other_persons": "; ".join(dict.fromkeys(other_persons)),
            }
        )

    return records


def main():
    parser = argparse.ArgumentParser(description="Process a single article URL and assign sentences to players/entities.")
    parser.add_argument("--url", required=True, help="article URL to fetch")
    parser.add_argument("--alias-csv", default="data/player_data/player_aliases.csv", help="path to player alias CSV")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model to load")
    parser.add_argument(
        "--suppress-fetch-errors",
        action="store_true",
        default=True,
        help="Log and skip fetch failures instead of raising (helps with 403/404 in single-run mode). "
        "Use --no-suppress-fetch-errors to raise.",
    )
    parser.add_argument(
        "--no-suppress-fetch-errors",
        dest="suppress_fetch_errors",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = process_url(
        args.url,
        Path(args.alias_csv),
        model=args.model,
        suppress_fetch_errors=args.suppress_fetch_errors,
    )

    if not rows:
        print("No sentences with players/entities found.")
        return

    for rec in rows:
        print("---")
        print(f"players: {rec['players'] or '(none)'}")
        print(f"other_persons: {rec['other_persons'] or '(none)'}")
        print(rec["sentence"])


if __name__ == "__main__":
    main()
