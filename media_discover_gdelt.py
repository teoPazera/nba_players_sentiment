import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


def normalize(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_alias_map(path: Path):
    df = pd.read_csv(path)
    if not {"player", "alias"}.issubset(df.columns):
        raise ValueError("alias CSV must have columns: player, alias")
    alias_map = {}
    for _, row in df.iterrows():
        player = str(row["player"]).strip()
        alias = str(row["alias"]).strip()
        if not player or not alias:
            continue
        alias_map.setdefault(player, set()).add(alias)
    return alias_map


def load_signing_dates(path: Path, fallback_dt: datetime | None = None):
    df = pd.read_csv(path)
    if "player" not in df.columns or "signing_date" not in df.columns:
        raise ValueError("signings CSV must have columns: player, signing_date")
    df = df.dropna(subset=["player"])
    df["signing_date_dt"] = pd.to_datetime(df["signing_date"], errors="coerce")
    signing = {}
    for _, row in df.iterrows():
        player = str(row["player"]).strip()
        dt = row["signing_date_dt"]
        if pd.isna(dt):
            if fallback_dt is None:
                continue
            dt = fallback_dt
        signing[normalize(player)] = dt
    return signing


def build_queries(aliases):
    # Build a few query variants to find free-agency coverage.
    cleaned = []
    for alias in sorted(set(a.strip() for a in aliases if str(a).strip())):
        alias = alias.replace('"', "")
        if not alias:
            continue
        if " " in alias:
            cleaned.append(f"\"{alias}\"")
        else:
            cleaned.append(alias)

    if not cleaned:
        return []

    # GDELT dislikes parentheses around single terms; only wrap when OR-ing.
    if len(cleaned) == 1:
        alias_group = cleaned[0]
    else:
        alias_group = "(" + " OR ".join(cleaned) + ")"

    market_terms = [
        "rumors",
        "\"trade rumors\"",
        "\"showing interest\"",
    ]

    value_terms = [
        "\"max contract\"",
        "overpay",
        "underpay",
        "\"team-friendly\"",
        "worth",
        "deserves",
    ]

    opinion_terms = [
        "analysis",
        "\"scouting report\"",
        "preview",
        "grades",
        "\"stock up\"",
        "\"stock down\"",
    ]

    def or_clause(terms):
        return "(" + " OR ".join(terms) + ")"

    queries = [
        f"{alias_group} AND NBA AND {or_clause(market_terms)}",
        f"{alias_group} AND NBA AND {or_clause(value_terms)}",
        f"{alias_group} AND NBA AND {or_clause(opinion_terms)}",
        f'{alias_group} AND NBA AND ("free agency" OR "free agent" OR offseason)',
    ]

    return queries


def to_gdelt_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_gdelt(query: str, start_dt: datetime, end_dt: datetime, max_records: int):
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": max_records,
        "format": "json",
        "startdatetime": to_gdelt_dt(start_dt),
        "enddatetime": to_gdelt_dt(end_dt),
    }
    r = requests.get(GDELT_DOC_ENDPOINT, params=params, timeout=20)
    if r.status_code != 200:
        logging.warning("GDELT %s %s", r.status_code, r.text[:200])
        return []
    try:
        data = r.json()
    except ValueError:
        logging.warning(
            "GDELT JSON decode failed (status=%s). Body (first 200): %s",
            r.status_code,
            r.text[:200],
        )
        return []

    return data.get("articles", []) or []


def main():
    parser = argparse.ArgumentParser(description="Discover media URLs via GDELT for NBA players.")
    parser.add_argument("--alias-csv", default="data/player_data/player_aliases.csv")
    parser.add_argument("--signings-csv", default="data/player_data/fox_free_agency_signings_2025.csv")
    parser.add_argument("--start-date", default="2025-05-01", help="YYYY-MM-DD window start (May 1 default)")
    parser.add_argument("--max-records", type=int, default=250, help="max records per player query (GDELT cap, per query variant)")
    parser.add_argument("--out-csv", default="data/media_data/discovered_urls_gdelt.csv")
    parser.add_argument("--player", help="only run for this player name (matches alias CSV 'player' col)")
    parser.add_argument("--max-players", type=int, help="limit to first N players (after sorting by name)")
    parser.add_argument("--single-alias-only", action="store_true", help="only run players that have exactly one alias")
    parser.add_argument(
        "--append",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="if output CSV exists, append and dedupe by url (default: append)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_path = Path(args.out_csv)
    existing_df = None
    seen_urls = set()
    if args.append and out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            if "url" in existing_df.columns:
                seen_urls = set(existing_df["url"].dropna())
            logging.info("loaded %d existing rows from %s", len(existing_df), out_path)
        except Exception as exc:
            logging.warning("could not load existing %s (append requested): %s", out_path, exc)

    alias_map = load_alias_map(Path(args.alias_csv))
    start_dt_global = pd.to_datetime(args.start_date)
    fallback_dt = datetime(start_dt_global.year, 9, 1)
    signing_map = load_signing_dates(Path(args.signings_csv), fallback_dt=fallback_dt)

    # allow narrowing to a single player or the first N
    items = sorted(alias_map.items(), key=lambda kv: kv[0].lower())
    if args.player:
        items = [(p, a) for p, a in items if p == args.player]
    if args.max_players:
        items = items[: args.max_players]

    rows = []

    for player, aliases in items:
        alias_list = [str(a).strip() for a in aliases if str(a).strip()]
        if args.single_alias_only and len(set(alias_list)) != 1:
            logging.info("skip %s (multiple aliases, single-alias-only set)", player)
            continue

        norm_player = normalize(player)
        signing_dt = signing_map.get(norm_player)
        if signing_dt is None:
            logging.info("skip %s (no signing date)", player)
            continue

        queries = build_queries(alias_list)
        if not queries:
            logging.info("skip %s (no aliases)", player)
            continue

        start_dt = start_dt_global
        end_dt = signing_dt
        if end_dt <= start_dt:
            logging.info("skip %s (signing before start window)", player)
            continue

        for query in queries:
            logging.info("query %s | %s -> %s | %s", player, start_dt.date(), end_dt.date(), query)
            articles = fetch_gdelt(query, start_dt, end_dt, args.max_records)

            for art in articles:
                url = art.get("url")
                if not url:
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)

                row = {
                    "player": player,
                    "query": query,
                    "url": url,
                    "title": art.get("title"),
                    "domain": art.get("domain"),
                    "seendate": art.get("seendate"),
                    "sourcecountry": art.get("sourcecountry"),
                    "language": art.get("language"),
                    "gdelt_source": art.get("source"),
                    "gdelt_theme": art.get("themes"),
                }
                rows.append(row)

    if not rows:
        logging.info("no articles found")
        return

    new_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    if "url" in combined_df.columns:
        combined_df.drop_duplicates(subset=["url"], inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_path, index=False)
    logging.info(
        "saved %d rows to %s (new %d, appended=%s)",
        len(combined_df),
        out_path,
        len(rows),
        existing_df is not None,
    )


if __name__ == "__main__":
    main()
