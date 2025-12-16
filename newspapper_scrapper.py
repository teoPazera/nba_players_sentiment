import os
import time
import logging
from datetime import datetime
from pathlib import Path
import re

from dotenv import load_dotenv
import requests
import pandas as pd

# ==========================
# CONFIGURATION
# ==========================

# --- CSV paths (relative to project root) ---
PLAYER_ALIASES_PATH = Path("data/player_data/player_aliases.csv")
SIGNINGS_PATH = Path("data/player_data/fox_free_agency_signings_2025.csv")

# --- Column name assumptions ---
ALIASES_PLAYER_KEY_COL = "player"
ALIASES_ALIAS_COL = "alias"

SIGNINGS_PLAYER_KEY_COL = "player_key"
SIGNINGS_DATE_COL = "signing_date"  # e.g. '2025-07-06'

# --- Offseason window defaults ---
DEFAULT_OFFSEASON_YEAR = 2025
FALLBACK_END_MONTH_DAY = "08-31"  # used if signing_date is missing

# May 1 is fixed as the start of the search window within the offseason year
START_MONTH_DAY = "05-01"

# --- Output paths ---
OUTPUT_PATH = Path("data/media_data/news_articles_may_to_signing_2025.csv")
PROCESSED_PLAYERS_PATH = Path(
    "data/media_data/news_articles_processed_players.csv"
)

# --- TheNewsAPI settings (https://www.thenewsapi.com) ---
THENEWS_API_ENDPOINT = "https://api.thenewsapi.com/v1/news/all"

load_dotenv()
THENEWS_API_KEY = os.getenv("the_news_api_token")

# Restrict to sports / NBA-related outlets (domain filter)
NEWS_DOMAINS = [
    "espn.com",
    "nba.com",
    "bleacherreport.com",
    "sports.yahoo.com",
    "cbssports.com",
    "si.com",              # Sports Illustrated
    "theathletic.com",     # often paywalled but metadata is fine
    "nbcsports.com",
    "foxsports.com",
    "sportingnews.com",
]

# Free plan: 100 requests/day, 3 articles per request, so be gentle
REQUESTS_PER_SECOND = 0.8         # ~1 req/sec
MAX_PAGES_PER_QUERY = 2           # 2 pages * 3 = up to 6 articles per query
ARTICLES_PER_PAGE = 3             # free tier limit

# ==========================
# LOGGING
# ==========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ==========================
# EXCEPTIONS
# ==========================

class QuotaExceededError(Exception):
    pass

# ==========================
# HELPER FUNCTIONS
# ==========================

def normalize(s):
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def make_offseason_window(signing_date_str):
    """
    Given a signing date as string or NaN, return (start_iso, end_iso).

    - Start is May 1 of the relevant year.
    - End is signing date if known, otherwise DEFAULT_OFFSEASON_YEAR-08-31.
    Returned as ISO 8601 strings suitable for TheNewsAPI's published_before/after.
    """
    if pd.isna(signing_date_str) or not str(signing_date_str).strip():
        year = DEFAULT_OFFSEASON_YEAR
        end_dt = datetime.strptime(f"{year}-{FALLBACK_END_MONTH_DAY}", "%Y-%m-%d")
    else:
        try:
            signing_dt = pd.to_datetime(signing_date_str).to_pydatetime()
        except Exception as e:
            logging.warning(
                "Could not parse signing_date '%s', using fallback window. Error: %s",
                signing_date_str, e
            )
            signing_dt = datetime.strptime(
                f"{DEFAULT_OFFSEASON_YEAR}-{FALLBACK_END_MONTH_DAY}",
                "%Y-%m-%d",
            )

        year = signing_dt.year
        end_dt = signing_dt

    start_dt = datetime.strptime(f"{year}-{START_MONTH_DAY}", "%Y-%m-%d")

    return start_dt.isoformat(), end_dt.isoformat()


def load_player_aliases(path):
    """
    Load aliases into a dict: normalized_player_key -> set of alias strings.

    Assumes columns [ALIASES_PLAYER_KEY_COL, ALIASES_ALIAS_COL].
    """
    df = pd.read_csv(path)

    if ALIASES_PLAYER_KEY_COL not in df.columns or ALIASES_ALIAS_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{ALIASES_PLAYER_KEY_COL}' and "
            f"'{ALIASES_ALIAS_COL}' in {path}, got {list(df.columns)}"
        )

    alias_map = {}
    for row in df.itertuples(index=False):
        player_key = getattr(row, ALIASES_PLAYER_KEY_COL)
        alias = getattr(row, ALIASES_ALIAS_COL)

        if pd.isna(player_key) or pd.isna(alias):
            continue

        norm_key = normalize(player_key)
        alias = str(alias).strip()
        if not norm_key or not alias:
            continue

        alias_map.setdefault(norm_key, set()).add(alias)

    return alias_map


def load_signing_windows(path):
    """
    Load signing dates and return a dict:
        normalized_player_key -> (start_iso, end_iso)
    """
    df = pd.read_csv(path)

    if SIGNINGS_PLAYER_KEY_COL not in df.columns or SIGNINGS_DATE_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{SIGNINGS_PLAYER_KEY_COL}' and "
            f"'{SIGNINGS_DATE_COL}' in {path}, got {list(df.columns)}"
        )

    windows = {}
    for row in df.itertuples(index=False):
        player_key = getattr(row, SIGNINGS_PLAYER_KEY_COL)
        signing_date = getattr(row, SIGNINGS_DATE_COL)

        if pd.isna(player_key):
            continue

        norm_key = normalize(str(player_key))
        if not norm_key:
            continue

        start_iso, end_iso = make_offseason_window(signing_date)
        windows[norm_key] = (start_iso, end_iso)

    return windows


def build_search_queries_for_player(aliases):
    # Clean and quote aliases
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

    alias_group = "(" + " OR ".join(cleaned) + ")"

    market_terms = [
        "rumors",
        "\"trade rumors\"",
        "\"showing interest\"",
        "\"drawing interest\"",
        "\"linked to\"",
        "\"on the market\"",
        "\"on the block\"",
        "pursuing",
        "targeting",
    ]

    value_terms = [
        "\"max contract\"",
        "supermax",
        "\"offer sheet\"",
        "\"sign-and-trade\"",
        "extension",
        "\"extension talks\"",
        "overpay",
        "underpay",
        "\"team-friendly\"",
        "payday",
        "\"big raise\"",
        "worth",
        "deserves",
    ]

    opinion_terms = [
        "analysis",
        "\"deep dive\"",
        "breakdown",
        "\"scouting report\"",
        "preview",
        "\"season preview\"",
        "\"player preview\"",
        "\"winners and losers\"",
        "grades",
        "\"report card\"",
        "takeaways",
        "\"biggest questions\"",
        "\"rumors tracker\"",
        "\"stock up\"",
        "\"stock down\"",
    ]

    def or_clause(terms):
        return "(" + " OR ".join(terms) + ")"

    queries = [
        f"{alias_group} AND NBA AND {or_clause(market_terms)}",
        f"{alias_group} AND NBA AND {or_clause(value_terms)}",
        f"{alias_group} AND NBA AND {or_clause(opinion_terms)}",
        # one broader offseason/free-agency anchor
        f'{alias_group} AND NBA AND ("free agency" OR "free agent" OR offseason)',
    ]

    return queries


def thenewsapi_query(search, from_iso, to_iso, page=1):
    """
    Call TheNewsAPI /v1/news/all with:
    - search query
    - published_after / published_before (ISO 8601)
    - language=en
    - domains filter (sports/NBA outlets)
    """
    if not THENEWS_API_KEY:
        raise RuntimeError(
            "the_news_api_token environment variable not set. "
            "Export it (or put it in .env) before running this script."
        )

    params = {
        "api_token": THENEWS_API_KEY,
        "search": search,
        "published_after": from_iso,
        "published_before": to_iso,
        "language": "en",
        "domains": ",".join(NEWS_DOMAINS),
        "limit": ARTICLES_PER_PAGE,
        "page": page,
    }

    logging.info(
        "TheNewsAPI request: search=%r, from=%s, to=%s, page=%d",
        search,
        from_iso,
        to_iso,
        page,
    )

    response = requests.get(THENEWS_API_ENDPOINT, params=params, timeout=20)

    if response.status_code == 402:
        logging.error(
            "TheNewsAPI quota/usage limit reached (402). "
            "Stopping further requests."
        )
        raise QuotaExceededError("TheNewsAPI quota/usage limit reached (402).")

    if response.status_code != 200:
        logging.warning(
            "TheNewsAPI error %s: %s", response.status_code, response.text
        )
        return None

    return response.json()


def fetch_articles_for_search(search, from_iso, to_iso):
    """
    Fetch up to MAX_PAGES_PER_QUERY pages of results for a single search query.

    Returns a list of article dicts (raw TheNewsAPI 'data' entries).
    """
    all_articles = []

    for page in range(1, MAX_PAGES_PER_QUERY + 1):
        data = thenewsapi_query(search, from_iso, to_iso, page=page)
        if not data:
            break

        articles = data.get("data", [])
        if not articles:
            break

        all_articles.extend(articles)

        meta = data.get("meta", {}) or {}
        returned = meta.get("returned", len(articles))
        limit = meta.get("limit", ARTICLES_PER_PAGE)

        # stop if we've reached the end
        if returned < limit:
            break

        time.sleep(1.0 / REQUESTS_PER_SECOND)

    return all_articles


def flatten_article(player_key, search_query, aliases, article):
    """
    Convert a single TheNewsAPI article entry + player info into a flat dict.
    """
    categories = article.get("categories")
    if isinstance(categories, list):
        categories_str = ",".join(categories)
    else:
        categories_str = categories

    return {
        "player_key": player_key,
        "search_query": search_query,
        "aliases": "; ".join(sorted(aliases)),
        "uuid": article.get("uuid"),
        "source": article.get("source"),          # domain, e.g. 'espn.com'
        "title": article.get("title"),
        "description": article.get("description"),
        "snippet": article.get("snippet"),
        "url": article.get("url"),
        "image_url": article.get("image_url"),
        "language": article.get("language"),
        "published_at": article.get("published_at"),
        "categories": categories_str,
    }


# ==========================
# MAIN SCRAPER
# ==========================

def main():
    logging.info("Loading player aliases from %s", PLAYER_ALIASES_PATH)
    alias_map = load_player_aliases(PLAYER_ALIASES_PATH)

    logging.info("Loading signing windows from %s", SIGNINGS_PATH)
    signing_windows = load_signing_windows(SIGNINGS_PATH)

    # --- Resume: load processed players (even if they had 0 articles) ---
    processed_players = set()
    if PROCESSED_PLAYERS_PATH.exists():
        logging.info("Loading processed players from %s", PROCESSED_PLAYERS_PATH)
        processed_df = pd.read_csv(PROCESSED_PLAYERS_PATH)
        if "player_key" in processed_df.columns:
            processed_players = set(
                normalize(pk) for pk in processed_df["player_key"].dropna()
            )
        logging.info("Found %d processed players in log.", len(processed_players))

    # --- Resume: load existing articles (for dedup + keeping previous work) ---
    existing_df = None
    seen_uuids = set()

    if OUTPUT_PATH.exists():
        logging.info("Existing results found at %s, loading.", OUTPUT_PATH)
        existing_df = pd.read_csv(OUTPUT_PATH)

        if "uuid" in existing_df.columns:
            seen_uuids = set(str(u) for u in existing_df["uuid"].dropna())

        logging.info("Existing results: %d rows, %d unique uuids.",
                     len(existing_df), len(seen_uuids))

    all_rows = []
    quota_exhausted = False

    for norm_player_key, (start_iso, end_iso) in signing_windows.items():
        if quota_exhausted:
            break

        if norm_player_key in processed_players:
            logging.info(
                "Player_key=%s is already marked processed, skipping.",
                norm_player_key,
            )
            continue

        aliases = alias_map.get(norm_player_key)

        if not aliases:
            logging.info(
                "No aliases found for player_key=%s, skipping", norm_player_key
            )
            # we can still mark them processed so we don't keep re-checking
            processed_players.add(norm_player_key)
            continue

        logging.info(
            "Processing player_key=%s with %d aliases, window %s → %s",
            norm_player_key, len(aliases), start_iso, end_iso
        )

        search_queries = build_search_queries_for_player(aliases)
        if not search_queries:
            logging.info(
                "No usable aliases for player_key=%s after cleaning, skipping",
                norm_player_key,
            )
            processed_players.add(norm_player_key)
            continue

        player_completed = True  # will flip to False only if quota fails mid-way

        for search_q in search_queries:
            if quota_exhausted:
                player_completed = False
                break

            try:
                articles = fetch_articles_for_search(search_q, start_iso, end_iso)
            except QuotaExceededError:
                logging.error(
                    "Quota reached while processing player_key=%s. "
                    "Will save partial results and exit.",
                    norm_player_key,
                )
                quota_exhausted = True
                player_completed = False
                break

            logging.info(
                "Found %d raw articles for player_key=%s query=%r",
                len(articles), norm_player_key, search_q
            )

            for article in articles:
                uuid = article.get("uuid")
                if uuid:
                    uuid_str = str(uuid)
                    if uuid_str in seen_uuids:
                        continue
                    seen_uuids.add(uuid_str)

                all_rows.append(
                    flatten_article(norm_player_key, search_q, aliases, article)
                )

            time.sleep(1.0 / REQUESTS_PER_SECOND)

        # Only mark the player as processed if we finished all their queries
        # without quota being exceeded mid-way.
        if player_completed:
            processed_players.add(norm_player_key)

    # --- Combine old + new article rows ---
    if not all_rows and existing_df is None:
        logging.warning(
            "No new articles found and no previous results. Nothing to save."
        )
    else:
        new_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()

        if existing_df is not None:
            if not new_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = existing_df
        else:
            combined_df = new_df

        if not combined_df.empty:
            subset_cols = [c for c in ["uuid", "player_key", "url"]
                           if c in combined_df.columns]
            if subset_cols:
                combined_df.drop_duplicates(subset=subset_cols, inplace=True)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(OUTPUT_PATH, index=False)

        logging.info(
            "Saved %d rows to %s (quota_exhausted=%s)",
            len(combined_df),
            OUTPUT_PATH,
            quota_exhausted,
        )

    # --- Save processed players list (so 0-article players are remembered too) ---
    if processed_players:
        processed_df_out = pd.DataFrame(
            {"player_key": sorted(processed_players)}
        )
        PROCESSED_PLAYERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        processed_df_out.to_csv(PROCESSED_PLAYERS_PATH, index=False)
        logging.info(
            "Saved %d processed players to %s.",
            len(processed_players),
            PROCESSED_PLAYERS_PATH,
        )


if __name__ == "__main__":
    main()
